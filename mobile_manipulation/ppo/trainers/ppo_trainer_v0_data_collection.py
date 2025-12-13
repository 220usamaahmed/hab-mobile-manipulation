#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import deque
from typing import Dict

import numpy as np
import torch
import tqdm
from gym import spaces
from habitat import Config, RLEnv, logger
from habitat.core.environments import get_env_class
from habitat_baselines import BaseTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
    generate_video,
    get_checkpoint_id,
)
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from mobile_manipulation.common.registry import mm_registry
from mobile_manipulation.common.rollout_storage import RolloutStorage
from mobile_manipulation.ppo.policy import ActorCritic
from mobile_manipulation.utils.common import (
    Timer,
    extract_scalars_from_info,
    get_git_commit_id,
    get_latest_checkpoint,
)
from mobile_manipulation.utils.env_utils import (
    VectorEnv,
    construct_envs,
    make_env_fn,
)
from mobile_manipulation.utils.wrappers import HabitatActionWrapper


from einops import rearrange

import random
from habitat_extensions.utils.viewer import OpenCVViewer
from habitat_extensions.tasks.rearrange.play import get_action_from_key


from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid

import pickle

import numpy as np
import inspect


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
)


def pca_torch(vectors: torch.Tensor, n_components: int = 2):
    """
    Perform PCA using PyTorch (no external libraries like sklearn).

    Args:
        vectors (torch.Tensor): Input data (N, D) where N is the number of samples and D is the dimensionality.
        n_components (int): Number of components to keep (e.g., 2 for 2D visualization).

    Returns:
        torch.Tensor: The reduced data (N, n_components).
    """
    # Center the data (subtract the mean of each feature)
    mean = torch.mean(vectors, dim=0)
    centered_vectors = vectors - mean

    # Compute the covariance matrix
    covariance_matrix = torch.mm(centered_vectors.T, centered_vectors) / (
        vectors.shape[0] - 1
    )

    # Perform SVD
    U, S, V = torch.svd(covariance_matrix)

    # Select the top 'n_components' eigenvectors (principal components)
    principal_components = V[:, :n_components]

    # Project the data onto the top principal components
    reduced_vectors = torch.mm(centered_vectors, principal_components)

    return reduced_vectors


def kmeans_torch(vectors, k, num_iters=100, tol=1e-4):
    N, D = vectors.shape
    indices = torch.randperm(N)[:k]
    centroids = vectors[indices].clone()

    for _ in range(num_iters):
        dists = torch.cdist(vectors, centroids)
        labels = torch.argmin(dists, dim=1)
        new_centroids = torch.stack(
            [
                (
                    vectors[labels == i].mean(dim=0)
                    if (labels == i).any()
                    else centroids[i]
                )
                for i in range(k)
            ]
        )
        if torch.norm(centroids - new_centroids).item() < tol:
            break
        centroids = new_centroids

    dists = torch.cdist(vectors, centroids)
    closest_dists = dists[torch.arange(vectors.shape[0]), labels]
    inertia = torch.sum(closest_dists**2).item()

    return labels, centroids, inertia


def find_best_k(vectors, k_range=range(1, 11)):
    inertias = []
    for k in k_range:
        _, _, inertia = kmeans_torch(vectors, k)
        inertias.append(inertia)

    # Elbow detection (simple numerical version)
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    elbow_k = k_range[
        np.argmin(second_diffs) + 2
    ]  # shift for diff index offset

    return elbow_k, inertias


def auto_kmeans_cluster(vectors: torch.Tensor, max_k: int = 10):
    """
    Automatically chooses number of clusters and runs k-means.

    Args:
        vectors (torch.Tensor): (N, D) trajectory vectors
        max_k (int): Max clusters to try

    Returns:
        labels (torch.Tensor): Cluster assignments
        centers (torch.Tensor): Final centroids
        best_k (int): Chosen number of clusters
    """
    k_range = list(range(1, max_k + 1))
    best_k, inertias = find_best_k(vectors, k_range)
    print(f"Automatically selected k = {best_k}")
    labels, centers, _ = kmeans_torch(vectors, best_k)

    # Visualizing the clusters with PCA (implemented manually)
    reduced_vectors = pca_torch(
        vectors, n_components=2
    )  # Reduce to 2D for visualization

    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced_vectors[:, 0].cpu().numpy(),
        reduced_vectors[:, 1].cpu().numpy(),
        c=labels.cpu().numpy(),
        cmap="Spectral",
        s=10,
    )
    plt.title(f"K-Means Clustering with k={best_k}")
    plt.colorbar(label="Cluster ID")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

    return labels, centers, best_k


def cosine_similarity_matrix_torch(vectors: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity matrix for a set of vectors using PyTorch.

    Parameters:
        vectors (Tensor): A 2D tensor of shape (n_vectors, dimensions)

    Returns:
        Tensor: A 2D tensor of shape (n_vectors, n_vectors) with cosine similarities
    """
    # Normalize each vector (row) to unit length
    norms = torch.norm(vectors, dim=1, keepdim=True)  # shape: (n_vectors, 1)
    normalized_vectors = vectors / (
        norms + 1e-8
    )  # add small value to avoid division by zero

    # Compute cosine similarity as dot product of normalized vectors
    similarity_matrix = torch.matmul(normalized_vectors, normalized_vectors.T)

    return similarity_matrix


def choose_best_traj(estimated_trajectories, grasped=1):

    if len(estimated_trajectories.shape) == 1:
        num_trajs = 1
        best_idx = 0
        best_traj = estimated_trajectories
    else:
        num_trajs = estimated_trajectories.shape[0]
        for traj in range(num_trajs):
            if traj == 0:
                if grasped > 0.3:
                    rel_ee_pos = estimated_trajectories[traj][584:587]
                    rel_rob_pos = estimated_trajectories[traj][580:583]
                else:
                    rel_ee_pos = (
                        estimated_trajectories[traj][584:587]
                        - estimated_trajectories[traj][597:600]
                    )
                    rel_rob_pos = (
                        estimated_trajectories[traj][580:583]
                        - estimated_trajectories[traj][597:600]
                    )
                best_rel_ee_pos = rel_ee_pos
                best_rel_rob_pos = rel_rob_pos
                best_rel_pos = 0.7 * torch.norm(rel_ee_pos) + 0.3 * torch.norm(
                    rel_rob_pos
                )
                best_idx = 0
            else:
                if grasped > 0.3:
                    rel_ee_pos = estimated_trajectories[traj][584:587]
                    rel_rob_pos = estimated_trajectories[traj][580:583]
                else:
                    rel_ee_pos = (
                        estimated_trajectories[traj][584:587]
                        - estimated_trajectories[traj][597:600]
                    )
                    rel_rob_pos = (
                        estimated_trajectories[traj][580:583]
                        - estimated_trajectories[traj][597:600]
                    )
                rel_pos = 0.7 * torch.norm(rel_ee_pos) + 0.3 * torch.norm(
                    rel_rob_pos
                )
                if rel_pos < best_rel_pos:
                    #  best_rel_ee_pos=rel_ee_pos
                    best_rel_pos = rel_pos
                    best_idx = traj
    print("grasped == ", grasped)
    print("best index == ", best_idx)
    print("smallest distance == ", torch.norm(best_rel_rob_pos))
    #   input()

    return best_idx


def imagine_trajectories(
    env, action_trajectories, gripper_is_grasped, render=True, viewer=None
):
    num_trajs = action_trajectories.shape[0]
    num_actions_per_traj = action_trajectories.shape[1]
    initial_robot_pos = env.env._env._sim.robot.base_pos
    initial_robot_ori = env.env._env._sim.robot.base_ori
    initial_qpos = env.env._env._sim.robot.arm_joint_pos
    start_state = (np.array(initial_robot_pos), initial_robot_ori)
    start_state = env.env._env._sim.get_state()
    gripped = False
    for traj in range(num_trajs):
        # env.env._env.reset_to_given_pose(start_state=start_state,qpos=initial_qpos)
        env.env._env._sim.set_state(start_state)
        for act in range(num_actions_per_traj):
            action = action_trajectories[traj][act].cpu().numpy()
            #    print("action == " , action)
            base_action = action[0:2]
            arm_action = action[2:9]
            gripper_action = action[9]
            step_action = {
                "action": "BaseArmGripperAction2",
                "action_args": {
                    "base_action": (base_action),
                    "arm_action": (arm_action),
                    "gripper_action": gripper_action,
                },
                "value": 2.9779255390167236,
            }
            ob, reward, done, info = env.step(step_action)

            if render:
                frame = env.render(
                    "human", overlay_info=False, show_info=False
                )
                key = viewer.imshow(frame[..., :3], delay=10)

        pick_goal = env.env._env._task.pick_goal
        place_goal = env.env._env._task.place_goal
        robot_ee_pos = env.env._env._sim.robot.ee_T.translation
        rob_base_pos = env.env._env._sim.robot.base_pos

        if gripper_is_grasped > 0.5:
            rel_ee_pos = np.array(robot_ee_pos - place_goal)
            rel_base_pos = np.array(rob_base_pos - place_goal)

        else:
            rel_ee_pos = np.array(robot_ee_pos - pick_goal)
            rel_base_pos = np.array(rob_base_pos - pick_goal)
        rel_ee_pos = torch.norm(torch.from_numpy(rel_ee_pos))
        rel_base_pos = torch.norm(torch.from_numpy(rel_base_pos))
        if traj == 0:
            best_rel_ee_pos = rel_ee_pos  # +rel_base_pos
            best_traj = 0
        else:
            # if rel_ee_pos+rel_base_pos< best_rel_ee_pos:
            if rel_ee_pos < best_rel_ee_pos:
                best_rel_ee_pos = rel_ee_pos  # +rel_base_pos
                best_traj = traj

        if env.env._env._sim.gripper.is_grasped:
            gripped = True

        print("finished imagined trajectory number {}".format(traj))
        print("target index == ", env.env._env._task.tgt_idx)
        print("grasped == ", gripper_is_grasped)
        print("press enter for the next trajectory")
    #   input()
    # env.env._env.reset_to_given_pose(start_state=start_state,qpos=initial_qpos)
    env.env._env._sim.set_state(start_state)
    return best_traj, gripped


# Sinusoidal Timestep Embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device)
            * -(torch.log(torch.tensor(10000.0)) / half_dim)
        )
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# Cross-Attention Block
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        b, n, _ = x.shape
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn_scores.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# Transformer Block with Cross Attention
class DiffusionTransformerBlock(nn.Module):
    def __init__(self, dim, cond_dim, heads=8, dim_head=64):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, batch_first=True
        )
        self.cross_attn = CrossAttention(dim, cond_dim, heads, dim_head)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, cond):
        x = self.attn(x)
        x = x + self.cross_attn(self.norm(x), cond)
        return x


# Conditional Diffusion Model
class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self, cond_dim=406, output_dim=206, hidden_dim=256, num_layers=6
    ):
        super().__init__()
        self.input_proj = nn.Linear(output_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t, cond):
        x = self.input_proj(x).unsqueeze(1)  # [B, 1, H]
        t_emb = self.time_mlp(t)  # [B, H]
        cond = self.cond_proj(cond).unsqueeze(0)  # [B, 1, H]
        cond = cond.unsqueeze(0)
        x = x + t_emb.unsqueeze(1)  # add time embedding

        for block in self.transformer_blocks:
            x = block(x, cond)

        return self.output_proj(x.squeeze(1))  # [B, output_dim]


# Noise Scheduler (like DDPM)
class NoiseScheduler:
    def __init__(self, timesteps=500, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().unsqueeze(1)
        sqrt_one_minus_alpha_cumprod = (
            (1.0 - self.alpha_cumprod[t]).sqrt().unsqueeze(1)
        )
        return (
            sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
        )

    def get_loss(self, model, x_start, t, cond):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(
            x_noisy.to(torch.float64),
            t.to(torch.float64),
            cond.to(torch.float64),
        )
        return F.mse_loss(predicted_noise, noise)


# Sampling function
@torch.no_grad()
def sample(model, scheduler, cond, steps=500, random_samples=20):
    model.eval()
    batch_size = random_samples
    x = torch.randn(batch_size, 206).to(cond.device)
    # x=torch.clamp(x, min=-0.2, max=0.2)
    # print(x)
    # input()
    for t in reversed(range(steps)):
        #    print("in diffusion step == " , t)
        t_batch = torch.full(
            (batch_size,), t, dtype=torch.long, device=cond.device
        )
        predicted_noise = model(
            x.to(torch.float64),
            t_batch.to(torch.float64),
            cond.to(torch.float64),
        )

        alpha = scheduler.alphas[t].to(cond.device)
        alpha_hat = scheduler.alpha_cumprod[t].to(cond.device)
        beta = scheduler.betas[t].to(cond.device)

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        x = (1 / alpha.sqrt()) * (
            x - beta / (1 - alpha_hat).sqrt() * predicted_noise
        ) + beta.sqrt() * noise
    return x


def extract_actions_from_diff_output(
    estimated_trajectories, num_predicted_actions=20, act_dim=10, obs_dim=20
):
    if len(estimated_trajectories.shape) == 1:
        num_trajs = 1
        estimated_trajectories = torch.unsqueeze(estimated_trajectories, dim=0)
    else:
        num_trajs = estimated_trajectories.shape[0]
    extracted_action_trajs = None
    for traj in range(num_trajs):
        trajectory = estimated_trajectories[traj]
        trajectory = torch.tensor(trajectory)
        actions_traj = None
        for act in range(num_predicted_actions):
            if act == 0:
                #     print("trajectory[0:act_dim] == " , trajectory)
                #    print("shape ==  " , trajectory.shape)
                actions_traj = torch.unsqueeze(trajectory[0:act_dim], dim=0)
            else:
                actions_traj = torch.cat(
                    (
                        actions_traj,
                        torch.unsqueeze(
                            trajectory[
                                act
                                * (obs_dim + act_dim) : act
                                * (obs_dim + act_dim)
                                + act_dim
                            ],
                            dim=0,
                        ),
                    ),
                    dim=0,
                )

        if traj == 0:
            extracted_action_trajs = torch.unsqueeze(actions_traj, dim=0)
        else:
            extracted_action_trajs = torch.cat(
                (extracted_action_trajs, torch.unsqueeze(actions_traj, dim=0)),
                dim=0,
            )

    return extracted_action_trajs


def extract_actions_from_diff_output_actions_only(
    estimated_trajectories, num_predicted_actions=20, act_dim=10, obs_dim=20
):

    if len(estimated_trajectories.shape) == 1:
        num_trajs = 1
        estimated_trajectories = torch.unsqueeze(estimated_trajectories, dim=0)
    else:
        num_trajs = estimated_trajectories.shape[0]
    extracted_action_trajs = None
    for traj in range(num_trajs):
        trajectory = estimated_trajectories[traj]
        trajectory = torch.tensor(trajectory)
        actions_traj = None
        for act in range(num_predicted_actions):
            if act == 0:
                #     print("trajectory[0:act_dim] == " , trajectory)
                #    print("shape ==  " , trajectory.shape)
                actions_traj = torch.unsqueeze(trajectory[0:act_dim], dim=0)
            else:
                actions_traj = torch.cat(
                    (
                        actions_traj,
                        torch.unsqueeze(
                            trajectory[
                                act * (act_dim) : act * (act_dim) + act_dim
                            ],
                            dim=0,
                        ),
                    ),
                    dim=0,
                )

        if traj == 0:
            extracted_action_trajs = torch.unsqueeze(actions_traj, dim=0)
        else:
            extracted_action_trajs = torch.cat(
                (extracted_action_trajs, torch.unsqueeze(actions_traj, dim=0)),
                dim=0,
            )

    return extracted_action_trajs


def select_traj_from_last_action(estimated_trajectories, grasped):
    if len(estimated_trajectories.shape) == 1:
        num_trajs = 1
        best_idx = 0
    else:
        num_trajs = estimated_trajectories.shape[0]
        for traj in range(num_trajs):
            if traj == 0:
                if grasped > 0.3:
                    rel_ee_pos = estimated_trajectories[traj][200:203]
                    rel_ee_pos = torch.norm(rel_ee_pos)
                else:
                    rel_ee_pos = (
                        estimated_trajectories[traj][200:203]
                        - estimated_trajectories[traj][203:206]
                    )
                    rel_ee_pos = torch.norm(rel_ee_pos)
                best_rel_ee_pos = rel_ee_pos
                best_idx = 0
            else:
                if grasped > 0.3:
                    rel_ee_pos = estimated_trajectories[traj][200:203]
                    rel_ee_pos = torch.norm(rel_ee_pos)
                else:
                    rel_ee_pos = (
                        estimated_trajectories[traj][200:203]
                        - estimated_trajectories[traj][203:206]
                    )
                    rel_ee_pos = torch.norm(rel_ee_pos)
                if rel_ee_pos < best_rel_ee_pos:
                    best_rel_ee_pos = rel_ee_pos
                    best_idx = traj

    return best_idx


class EarlyFusionRGBDResNet(nn.Module):
    def __init__(self, output_dim=128):
        super(EarlyFusionRGBDResNet, self).__init__()

        # Load pretrained ResNet18
        base_model = models.resnet18(pretrained=True)

        # Modify first conv layer to accept 4-channel input
        old_conv1 = base_model.conv1
        new_conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = old_conv1.weight
            new_conv1.weight[:, 3:4, :, :] = old_conv1.weight.mean(
                dim=1, keepdim=True
            )

        base_model.conv1 = new_conv1

        # Use the feature extractor part (remove FC layer)
        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1]
        )  # Up to avgpool

        # Add a projection layer to change output feature size
        self.projection = nn.Linear(512, output_dim)

    def forward(self, rgbd):
        x = self.feature_extractor(rgbd)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        x = self.projection(x)  # (B, output_dim)
        return x


class EarlyFusionRGBDResNet_depth_only(nn.Module):
    def __init__(self, output_dim=128):
        super(EarlyFusionRGBDResNet_depth_only, self).__init__()

        # Load pretrained ResNet18
        base_model = models.resnet18(pretrained=True)

        # Modify first conv layer to accept 4-channel input
        old_conv1 = base_model.conv1
        new_conv1 = nn.Conv2d(
            5, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = old_conv1.weight.mean(
                dim=1, keepdim=True
            )
            new_conv1.weight[:, 3:5, :, :] = old_conv1.weight.mean(
                dim=1, keepdim=True
            )

        base_model.conv1 = new_conv1

        # Use the feature extractor part (remove FC layer)
        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1]
        )  # Up to avgpool

        # Add a projection layer to change output feature size
        self.projection = nn.Linear(512, output_dim)

    def forward(self, rgbd):
        x = self.feature_extractor(rgbd)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        x = self.projection(x)  # (B, output_dim)
        return x


def build_mlp(
    input_size=790, output_size=300, hidden_layers=5, activation=nn.ReLU
):
    """
    Builds an MLP with given input/output size and number of hidden layers.
    The hidden layer sizes decrease linearly from input_size to output_size.

    Returns:
        nn.Sequential model
    """
    # Compute intermediate hidden layer sizes
    sizes = np.linspace(input_size, output_size, hidden_layers + 1).astype(int)

    layers = []
    for i in range(hidden_layers):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < hidden_layers - 1:
            layers.append(activation())

    return nn.Sequential(*layers)


@baseline_registry.register_trainer(name="ppo-v0")
class PPOTrainerV0(BaseTrainer):
    r"""Basic PPO Trainer."""

    envs: VectorEnv
    device: torch.device
    actor_critic: ActorCritic
    optimizer: torch.optim.Adam

    obs_space: spaces.Space
    _obs_batching_cache: ObservationBatchingCache
    action_space: spaces.Space

    def __init__(self, config: Config):
        self.config = config

    # print("in the ppo trainer")
    # input()

    def is_done(self):
        return self.num_steps_done >= self.config.TOTAL_NUM_STEPS

    def percent_done(self):
        return self.num_steps_done / self.config.TOTAL_NUM_STEPS

    def train(self) -> None:
        ppo_cfg = self.config.RL.PPO
        print("in the train function")
        self._init_train()
        self._init_rollouts()
        self.resume()

        if ppo_cfg.use_linear_lr_decay:
            min_lr = ppo_cfg.get("min_lr", 0.0)
            min_lr_ratio = min_lr / ppo_cfg.lr
            lr_lambda = lambda x: max(
                1 - self.percent_done() * (1.0 - min_lr_ratio), min_lr_ratio
            )
            lr_scheduler = LambdaLR(
                optimizer=self.optimizer, lr_lambda=lr_lambda
            )

        while not self.is_done():
            # Rollout and collect transitions
            self.actor_critic.eval()
            for _ in range(ppo_cfg.num_steps):
                self.step()

            with self.timer.timeit("update_model"):
                self.actor_critic.train()
                losses, metrics = self.update()
                self.rollouts.after_update()

            # Logging
            episode_metrics = self.get_episode_metrics()
            if self.num_updates_done % self.config.LOG_INTERVAL == 0:
                self.log(episode_metrics)

            # Tensorboard
            metrics.update(**episode_metrics)
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            if self.should_summarize():
                self.summarize(losses, metrics)
            if self.should_summarize(10):
                self.summarize2()

            # Checkpoint
            if self.should_checkpoint():
                self.count_checkpoints += 1
                self.prev_ckpt_step = self.num_steps_done
                self.save(ckpt_id=self.count_checkpoints)
            if self.should_checkpoint2():
                self.save(ckpt_id=-1)

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

        # Save the last model
        if self.num_steps_done > self.prev_ckpt_step:
            self.count_checkpoints += 1
            self.save(ckpt_id=self.count_checkpoints)

        self.writer.close()
        self.envs.close()

    @torch.no_grad()
    def step(self):
        """Take one rollout step."""
        n_envs = self.envs.num_envs
        print("in stepppppp ")
        input()

        with self.timer.timeit("sample_action"):
            step_batch = self.rollouts.buffers[self.rollouts.step_idx]
            #     print("step batch obs== ", step_batch)
            # Assume that observations are stored at rollouts in the last step
            output_batch = self.actor_critic.act(step_batch)
            actions = output_batch["action"]
            actions = actions.to(device="cpu", non_blocking=True)

        with self.timer.timeit("step_env"):
            for i_env, action in zip(range(n_envs), actions.unbind(0)):
                self.envs.async_step_at(i_env, {"action": action.numpy()})

        with self.timer.timeit("update_rollout"):
            self.rollouts.insert(
                next_recurrent_hidden_states=output_batch.get(
                    "rnn_hidden_states"
                ),
                actions=output_batch["action"],
                action_log_probs=output_batch["action_log_probs"],
                value_preds=output_batch["value"],
            )

        with self.timer.timeit("step_env"):
            results = self.envs.wait_step()
            #   print("results == " , results)
            #  input()
            obs, rews, dones, infos = map(list, zip(*results))
            self.num_steps_done += n_envs

        # self.envs.render("human", delay=10)

        # -------------------------------------------------------------------------- #
        # Reset and deal with truncated episodes
        # -------------------------------------------------------------------------- #
        next_value = None
        are_truncated = [False for _ in range(n_envs)]
        ignore_truncated = self.config.RL.get("IGNORE_TRUNCATED", False)

        if any(dones):
            # Check which envs are truncated
            for i_env in range(n_envs):
                if dones[i_env]:
                    self.envs.async_reset_at(i_env)
                    is_truncated = infos[i_env].get(
                        "is_episode_truncated", False
                    )
                    are_truncated[i_env] = is_truncated

            if ignore_truncated:
                are_truncated = [False for _ in range(n_envs)]

            # Estimate values of actual next obs
            if any(are_truncated):
                next_batch = batch_obs(
                    obs,
                    device=self.device,
                    cache=self._obs_batching_cache,
                )
                next_step_batch = self.rollouts.buffers[
                    self.rollouts.step_idx + 1
                ]
                next_step_batch["observations"] = next_batch
                # Only the really truncated episodes have valid results
                next_step_batch["masks"] = torch.ones_like(
                    next_step_batch["masks"]
                )
                next_value = self.actor_critic.get_value(next_step_batch)

            for i_env in range(n_envs):
                if dones[i_env]:
                    obs[i_env] = self.envs.wait_reset_at(i_env)
        # -------------------------------------------------------------------------- #

        with self.timer.timeit("batch_obs"):
            batch = batch_obs(
                obs, device=self.device, cache=self._obs_batching_cache
            )
            rewards = torch.tensor(rews, dtype=torch.float).unsqueeze(1)
            done_masks = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
            not_done_masks = torch.logical_not(done_masks)
            truncated_masks = torch.tensor(
                are_truncated, dtype=torch.bool
            ).unsqueeze(1)

        with self.timer.timeit("update_stats"):
            for i_env in range(n_envs):
                self.episode_rewards[i_env] += rews[i_env]
                if dones[i_env]:
                    episode_info = self._extract_scalars_from_info(
                        infos[i_env]
                    )
                    episode_info["return"] = self.episode_rewards[i_env].item()
                    self.window_episode_stats.append(episode_info)
                    self.episode_rewards[i_env] = 0.0

        with self.timer.timeit("update_rollout"):
            self.rollouts.insert(
                next_observations=batch,
                rewards=rewards,
                next_masks=not_done_masks,
                next_value_preds=next_value,
                truncated_masks=truncated_masks,
            )
            self.rollouts.advance()

    def update(self):
        """PPO update."""
        ppo_cfg = self.config.RL.PPO
        if ppo_cfg.use_linear_clip_decay:
            clip_param = ppo_cfg.clip_param * max(1 - self.percent_done(), 0.0)
        else:
            clip_param = ppo_cfg.clip_param
        ppo_epoch = ppo_cfg.ppo_epoch

        with torch.no_grad():
            step_batch = self.rollouts.buffers[self.rollouts.step_idx]
            next_value = self.actor_critic.get_value(step_batch)

            # NOTE(jigu): next_value will be stored in the buffer.
            # However, it will be overwritten when next action is taken.
            self.rollouts.compute_returns(
                next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
            )
            advantages = self.rollouts.get_advantages(
                ppo_cfg.use_normalized_advantage
            )

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        num_updates = 0
        num_clipped_epoch = [0 for _ in range(ppo_epoch)]
        num_samples_epoch = [0 for _ in range(ppo_epoch)]

        for i_epoch in range(ppo_epoch):
            if ppo_cfg.use_recurrent_generator:
                data_generator = self.rollouts.recurrent_generator(
                    advantages, ppo_cfg.num_mini_batch
                )
            else:
                data_generator = self.rollouts.feed_forward_generator(
                    advantages, ppo_cfg.mini_batch_size
                )

            for batch in data_generator:
                outputs = self.actor_critic.evaluate_actions(
                    batch, batch["actions"]
                )
                values = outputs["value"]  # [B, 1]
                action_log_probs = outputs["action_log_probs"]  # [B, 1]
                dist_entropy = outputs["dist_entropy"]  # [B, A]

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
                    * batch["advantages"]
                )
                action_loss = -torch.min(surr1, surr2)

                if ppo_cfg.use_clipped_value_loss:
                    value_pred_clipped = batch["value_preds"] + (
                        values - batch["value_preds"]
                    ).clamp(-clip_param, clip_param)
                    value_losses = (values - batch["returns"]).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - batch["returns"]
                    ).pow(2)
                    value_loss = 0.5 * torch.max(
                        value_losses, value_losses_clipped
                    )
                else:
                    value_loss = 0.5 * (batch["returns"] - values).pow(2)

                action_loss = action_loss.mean()
                value_loss = value_loss.mean()
                dist_entropy = dist_entropy.mean()

                self.optimizer.zero_grad()

                # ppo extra metrics
                num_clipped = torch.logical_or(
                    ratio < 1.0 - clip_param,
                    ratio > 1.0 + clip_param,
                ).float()
                num_clipped_epoch[i_epoch] += num_clipped.sum().item()
                num_samples_epoch[i_epoch] += num_clipped.size(0)

                total_loss = (
                    value_loss * ppo_cfg.value_loss_coef
                    + action_loss
                    - dist_entropy * ppo_cfg.entropy_coef
                )
                total_loss.backward()

                if ppo_cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(),
                        ppo_cfg.max_grad_norm,
                    )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                num_updates += 1

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        loss_dict = dict(
            value_loss=value_loss_epoch,
            action_loss=action_loss_epoch,
            dist_entropy=dist_entropy_epoch,
        )

        metric_dict = dict()
        for i_epoch in range(ppo_epoch):
            clip_ratio = num_clipped_epoch[i_epoch] / max(
                num_samples_epoch[i_epoch], 1
            )
            metric_dict[f"clip_ratio_{i_epoch}"] = clip_ratio

        self.num_updates_done += 1
        return loss_dict, metric_dict

    def get_episode_metrics(self):
        if len(self.window_episode_stats) == 0:
            return {}
        # Assume all episodes have the same keys. True for Habitat.
        return {
            k: np.mean([ep_info[k] for ep_info in self.window_episode_stats])
            for k in self.window_episode_stats[0].keys()
        }

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict):
        return extract_scalars_from_info(
            info, blacklist=["terminal_observation"]
        )

    def log(self, metrics: Dict[str, float]):
        wall_time = (time.time() - self.t_start) + self.prev_time
        logger.info(
            "update: {}\tframes: {}\tfps: {:.3f}\tpercent: {:.2f}%".format(
                self.num_updates_done,
                self.num_steps_done,
                self.num_steps_done / wall_time,
                self.percent_done() * 100,
            )
        )
        logger.info(
            "\t".join(
                "{}: {:.3f}s".format(k, v)
                for k, v in self.timer.elapsed_times.items()
            )
        )
        logger.info(
            "  ".join("{}: {:.3f}".format(k, v) for k, v in metrics.items()),
        )

    def summarize(self, losses: Dict[str, float], metrics: Dict[str, float]):
        """Summarize scalars in tensorboard."""
        for k, v in losses.items():
            self.writer.add_scalar(f"losses/{k}", v, self.num_steps_done)
        for k, v in metrics.items():
            self.writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)

    def summarize2(self):
        """Summarize histogram and video in tensorboard."""
        self.writer.add_histogram(
            "value_preds",
            self.rollouts.buffers["value_preds"],
            global_step=self.num_steps_done,
        )
        self.writer.add_histogram(
            "discounted_returns",
            self.rollouts.buffers["returns"],
            global_step=self.num_steps_done,
        )

        video_keys = self.config.get("TB_VIDEO_KEYS", [])
        for key in video_keys:
            video_tensor = self.rollouts.buffers["observations"][key]
            self.writer.add_video(
                key,
                video_tensor.permute(1, 0, 4, 2, 3),
                global_step=self.num_steps_done,
                fps=10,
            )

    def should_summarize(self, mult=1) -> bool:
        if self.config.SUMMARIZE_INTERVAL == -1:
            interval = self.config.LOG_INTERVAL
        else:
            interval = self.config.SUMMARIZE_INTERVAL
        return self.num_updates_done % (interval * mult) == 0

    def should_checkpoint(self) -> bool:
        if self.config.NUM_CHECKPOINTS == -1:
            ckpt_freq = self.config.CHECKPOINT_INTERVAL
        else:
            ckpt_freq = (
                self.config.TOTAL_NUM_STEPS // self.config.NUM_CHECKPOINTS
            )
        return self.num_steps_done >= (self.count_checkpoints + 1) * ckpt_freq

    def should_checkpoint2(self) -> bool:
        """Check whether to save (overwrite) the latest checkpoint."""
        if (
            self.config.NUM_CHECKPOINTS == -1
            or self.config.CHECKPOINT_INTERVAL == -1
        ):
            return False
        return self.num_updates_done % self.config.CHECKPOINT_INTERVAL == 0

    def save_checkpoint(self, ckpt_path):
        wall_time = (time.time() - self.t_start) + self.prev_time
        checkpoint = dict(
            config=self.config,
            state_dict=self.actor_critic.state_dict(),
            optim_state=self.optimizer.state_dict(),
            step=self.num_steps_done,
            wall_time=wall_time,
            num_updates_done=self.num_updates_done,
            count_checkpoints=self.count_checkpoints,
        )
        torch.save(checkpoint, ckpt_path)

    def save(self, ckpt_id):
        if not self.config.CHECKPOINT_FOLDER:
            return
        ckpt_path = os.path.join(
            self.config.CHECKPOINT_FOLDER, f"ckpt.{ckpt_id}.pth"
        )
        self.save_checkpoint(ckpt_path)
        logger.info(
            f"Saved checkpoint to {ckpt_path} at {self.num_steps_done}th step"
        )

    def resume(self):
        if not self.config.CHECKPOINT_FOLDER:
            return
        ckpt_path = get_latest_checkpoint(self.config.CHECKPOINT_FOLDER, False)
        if ckpt_path is None:
            return
        assert os.path.isfile(ckpt_path), ckpt_path
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        logger.info(f"Resume from {ckpt_path}")

        self.actor_critic.load_state_dict(ckpt_dict["state_dict"])
        self.optimizer.load_state_dict(ckpt_dict["optim_state"])

        self.num_steps_done = ckpt_dict["step"]
        self.num_updates_done = ckpt_dict["num_updates_done"]
        self.prev_time = ckpt_dict["wall_time"]
        self.count_checkpoints = ckpt_dict["count_checkpoints"]
        self.prev_ckpt_step = self.num_steps_done

    def _init_envs(self, config: Config, auto_reset_done=False):
        r"""Initialize vectorized environments."""
        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            split_dataset=config.get("SPLIT_DATASET", True),
            workers_ignore_signals=False,
            auto_reset_done=auto_reset_done,
            wrappers=[HabitatActionWrapper],
        )

    def _init_observation_space(self, config: Config):
        if isinstance(self.envs, VectorEnv):
            obs_space = self.envs.observation_spaces[0]
        else:
            env: RLEnv = self.envs[0]
            obs_space = env.observation_space
        self.obs_space = obs_space

    def _init_action_space(self, config: Config):
        if isinstance(self.envs, VectorEnv):
            self.action_space = self.envs.action_spaces[0]
        else:
            env: RLEnv = self.envs[0]
            self.action_space = env.action_space

    def _setup_actor_critic(self, config: Config) -> None:
        r"""Set up actor critic for PPO."""
        policy_cfg = config.RL.POLICY
        # policy = baseline_registry.get_policy(policy_cfg.name)
        policy = mm_registry.get_policy(policy_cfg.name)

        self.actor_critic: ActorCritic = policy.from_config(
            policy_cfg, self.obs_space, self.action_space
        )
        self.actor_critic.to(self.device)

        ppo_cfg = config.RL.PPO
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=ppo_cfg.lr, eps=ppo_cfg.eps
        )

        ckpt_path = self.config.RL.POLICY.get("pretrained_weights", None)

        if ckpt_path:
            assert os.path.isfile(ckpt_path), ckpt_path
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            logger.info("Load checkpoint from {}".format(ckpt_path))
            self.actor_critic.load_state_dict(ckpt_dict["state_dict"])

    def _setup_rollouts(self, config: Config):
        ppo_cfg = config.RL.PPO
        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,  # number of steps for each env
            self.envs.num_envs,
            observation_space=self.obs_space,
            action_space=self.action_space,
            recurrent_hidden_state_size=self.actor_critic.net.rnn_hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        self.rollouts.to(self.device)

    def _init_train(self):
        if self.config.LOG_FILE:
            log_dir = os.path.dirname(self.config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(self.config.LOG_FILE)

        if self.config.VERBOSE:
            logger.info(f"config:\n {self.config}")
            logger.info("commit id: {}".format(get_git_commit_id()))

        if self.config.CHECKPOINT_FOLDER:
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)

        # ---------------------------------------------------------------------------- #
        # Initialization
        # ---------------------------------------------------------------------------- #
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # ---------------------------------------------------------------------------- #
        # NOTE(jigu): workaround from erik, to avoid high gpu memory fragmentation
        env = make_env_fn(
            self.config,
            get_env_class(self.config.ENV_NAME),
            wrappers=[HabitatActionWrapper],
        )
        self.envs = [env]
        self._init_observation_space(self.config)
        self._init_action_space(self.config)
        self._setup_actor_critic(self.config)
        env.close()
        # ---------------------------------------------------------------------------- #

        self._init_envs(self.config)
        self._init_observation_space(self.config)
        self._init_action_space(self.config)
        self._setup_rollouts(self.config)

        if self.config.VERBOSE:
            logger.info(f"actor_critic: {self.actor_critic}")
        logger.info(
            "#parameters: {}".format(
                sum(param.numel() for param in self.actor_critic.parameters())
            )
        )
        logger.info("obs space: {}".format(self.obs_space))
        logger.info("action space: {}".format(self.action_space))

        # ---------------------------------------------------------------------------- #
        # Setup statistic
        # ---------------------------------------------------------------------------- #
        # Current episode rewards (return)
        self.episode_rewards = torch.zeros(self.envs.num_envs, 1)
        # Recent episode stats (each stat is a dict)
        self.window_episode_stats = deque(
            maxlen=self.config.RL.PPO.reward_window_size
        )

        self.t_start = time.time()  # record overall time
        self.timer = Timer()  # record fine-grained scopes
        self.writer = TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=30
        )

        # resumable stats
        self.num_steps_done = 0
        self.num_updates_done = 0
        self.prev_time = 0.0
        self.count_checkpoints = 0
        self.prev_ckpt_step = 0

    def _init_rollouts(self):
        self._obs_batching_cache = ObservationBatchingCache()
        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        self.rollouts.buffers["observations"][0] = batch

    def eval(self):
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        if self.config.LOG_FILE:
            log_dir = os.path.dirname(self.config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(self.config.LOG_FILE)

        writer = TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=30)

        if self.config.EVAL.CKPT_PATH:
            ckpt_path = self.config.EVAL.CKPT_PATH
        else:
            ckpt_path = get_latest_checkpoint(
                self.config.CHECKPOINT_FOLDER, True
            )
            ###  this function finds the latest check-point in the folder
        assert os.path.isfile(ckpt_path), ckpt_path
        ckpt_id = get_checkpoint_id(ckpt_path)
        if ckpt_id is None:
            ckpt_id = -1

        if self.config.EVAL.BATCH_ENVS:
            self._batch_eval_checkpoint(ckpt_path, writer, ckpt_id)
        else:
            self._eval_checkpoint(ckpt_path, writer, ckpt_id)
        writer.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = -1,
    ) -> None:
        # Map location CPU is almost always better than mapping to a CUDA device.
        logger.info(f"Loaded {checkpoint_path}")
        ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
        #   print("check point dict == " , ckpt_dict)
        #  input()
        config = self.config.clone()

        config.defrost()

        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT

        config.freeze()

        if config.VERBOSE:
            logger.info(config)

        env = make_env_fn(
            config,
            get_env_class(config.ENV_NAME),
            wrappers=[HabitatActionWrapper],
        )

        # print(" config == ", config)
        # input()

        self.envs = [env]  ###here it is a single env
        self._init_observation_space(config)
        self._init_action_space(config)
        self._setup_actor_critic(config)
        self.actor_critic.load_state_dict(ckpt_dict["state_dict"])
        #   print(self.actor_critic.net.visual_encoder)
        #    PATH='/home/shokry/hab-mobile-manipulation/rl_trained_encoder/visual_encoder_nav_17_sept.pth'
        #   torch.save(self.actor_critic.net.visual_encoder.state_dict(), PATH)
        #  input()
        self.actor_critic.eval()

        ### check the action space in config

        if config.EVAL.NUM_EPISODES == -1:
            num_eval_episodes = env.number_of_episodes
        else:
            num_eval_episodes = config.EVAL.NUM_EPISODES

        current_episode_reward = 0.0
        all_episode_stats = []
        rgb_frames = []
        failure_episodes = []

        # Initialize policy inputs
        #    env.current_episode.episode_id=0
        #  print("config == " , config)
        #   print("env.current_episode.episode_id == " , env.current_episode.episode_id )
        #   input()

        obs = env.reset()
        # print("observation == ", obs)
        # input()
        self._obs_batching_cache = ObservationBatchingCache()
        batch = batch_obs(
            [obs], device=self.device, cache=self._obs_batching_cache
        )
        buffer = dict(
            recurrent_hidden_states=torch.zeros(
                1,
                self.actor_critic.net.num_recurrent_layers,
                self.actor_critic.net.rnn_hidden_size,
                device=self.device,
            ),
            prev_actions=torch.zeros(
                1,
                *self.action_space.shape,
                device=self.device,
                dtype=torch.float,
            ),
            masks=torch.zeros(
                1,
                1,
                device=self.device,
                dtype=torch.bool,
            ),
        )

        metrics = {}

        prev_actions = torch.zeros(
            1, *self.action_space.shape, device=self.device, dtype=torch.float
        )
        if len(prev_actions.shape) == 1:
            prev_actions = torch.unsqueeze(prev_actions, dim=0)
        pbar = tqdm.tqdm(total=num_eval_episodes)

        saved_eps = 0

        rel_rob_pos_temp = np.array([])
        rob_ori_temp = np.array([])
        rel_ee_pos_temp = np.array([])
        rob_qpos_temp = np.array([])
        rob_base_lin_vel_temp = np.array([])
        rob_base_ang_vel_temp = np.array([])
        rob_grasped_temp = np.array([])
        pick_pos_temp = np.array([])
        robot_head_rgb_temp = np.array([[[]]])
        robot_head_depth_temp = np.array([])
        robot_arm_rgb_temp = np.array([[[]]])
        robot_arm_depth_temp = np.array([])
        prev_actions_temp = np.array([])
        current_actions_temp = np.array([])
        action_to_save_temp = np.array([])
        num_suc_episodes = 0
        dataset_dict = {}

        current_step = 0

        actions_till_now = []
        episode_ids = []

        total_episodes = 0
        total_steps = 0
        hi = 0
        saves = 0

        num_prev_obs = 5
        conditions_dimensionality = 28
        choose_diffusion = False
        depth_only = False
        viewer_ = False
        rl_encoder = False

        if viewer_:
            viewer = OpenCVViewer(config.TASK_CONFIG.TASK.TYPE)

        if depth_only:
            #  save_dir='/home/shokry/hab-mobile-manipulation/diffusion/weights'
            save_dir = "./diffusion_dataset_new/weights_pick_ep5_seed300"
            diffusion_model = (
                ConditionalDiffusionModel().to(device).to(torch.float64)
            )
            # diffusion_model.load_state_dict(torch.load("/home/shokry/hab-mobile-manipulation/diff_transformer_weights/context_model_30000.pth", map_location=device))
            # diffusion_model.load_state_dict(torch.load("/home/shokry/hab-mobile-manipulation/diffusion_dataset_new/model_generate_actions_last_state_all_tasks_lr1_trgt_idx_1_epoch_2010.pth", map_location=device))
            #   diffusion_model.load_state_dict(torch.load("/home/shokry/hab-mobile-manipulation/diffusion_dataset_new/model_generate_actions_last_state_all_tasks_lr1_trgt_idx_1_epoch_950.pth", map_location=device))

            scheduler = NoiseScheduler(timesteps=500)
            diffusion_model.eval()

            mlp = build_mlp(input_size=406, output_size=600, hidden_layers=5)
            #  mlp.load_state_dict(torch.load(f"{save_dir}/pick_only_small_model_120000.pth", map_location=device)) #120000 is good
            mlp.to(device)

            model = EarlyFusionRGBDResNet_depth_only()
            model.eval()

        else:
            save_dir = "./mlp_models/weights"
            mlp = build_mlp(input_size=406, output_size=600, hidden_layers=5)
            #    mlp.load_state_dict(torch.load(f"{save_dir}/model_50.pth", map_location=device))
            mlp.to(device)

            model = EarlyFusionRGBDResNet()
            model.eval()

        print("num_eval_episodes == ", num_eval_episodes)
        # print("press enter to continue")
        #   input()
        while len(all_episode_stats) < num_eval_episodes:
            total_steps += 1
            placed = False
            if len(config.VIDEO_OPTION) > 0:
                #   print("in if len(config.VIDEO_OPTION) > 0: ")
                rgb_frames.append(env.render("human", info=metrics))
            #  print("finished if len(config.VIDEO_OPTION) > 0: ")
            with torch.no_grad():
                # print("in with torch.no_grad() ")
                step_batch = dict(observations=batch, **buffer)
                outputs_batch = self.actor_critic.act(
                    step_batch,
                    deterministic=False,  # config.EVAL.DETERMINISTIC_ACTION
                )
                actions = outputs_batch["action"]
                gripper_is_grasped = env.env._env._sim.gripper.is_grasped

                # print("action by policy == " , actions)
                if actions.shape[-1] == 1:

                    self.possible_velocities = np.array(
                        [
                            [lin_vel, ang_vel]
                            for lin_vel in np.linspace(-0.5, 1.0, 4)
                            for ang_vel in np.linspace(-1.0, 1.0, 5)
                        ]
                    )
                    current_velocity = self.possible_velocities[actions]
                    if gripper_is_grasped:
                        action_to_save = torch.tensor(
                            [
                                [
                                    current_velocity[0] * 1.5,
                                    current_velocity[1] * 1.5,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    1,
                                ]
                            ]
                        )

                    else:
                        action_to_save = torch.tensor(
                            [
                                [
                                    current_velocity[0] * 1.5,
                                    current_velocity[1] * 1.5,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    -1,
                                ]
                            ]
                        )  ## the gripper action should be -1
                # print("current_velocity == " , current_velocity)
                else:
                    action_to_save = torch.clamp(actions, min=-1, max=1)
                    action_to_save[:, 0:2] *= 1.5

            robot_base_pos = env.env._env._sim.robot.base_pos
            robot_base_orientation = env.env._env._sim.robot.base_ori
            robot_qpos = env.env._env._sim.robot.arm_joint_pos
            # robot_ee_T= env.env._env._sim.robot.ee_T
            robot_ee_pos = env.env._env._sim.robot.ee_T.translation

            gripper_is_grasped = env.env._env._sim.gripper.is_grasped
            if gripper_is_grasped:
                grasped = 1
            else:
                grasped = -1

            episode = env.env._env._sim.habitat_config.EPISODE

            pick_goal = env.env._env._task.pick_goal
            place_goal = env.env._env._task.place_goal

            robot_head_rgb = step_batch["observations"]["robot_head_rgb"][0]
            robot_arm_rgb = step_batch["observations"]["robot_arm_rgb"][0]

            robot_head_depth = step_batch["observations"]["robot_head_depth"][
                0
            ]

            robot_arm_depth = step_batch["observations"]["robot_arm_depth"][0]

            if current_step == 0:
                print("robot_base_pos == ", robot_base_pos)
            #    input()

            rel_rob_pos = np.array(robot_base_pos - place_goal)
            rob_ori = np.array([robot_base_orientation])
            rel_ee_pos = np.array(robot_ee_pos - place_goal)
            rob_qpos = np.array(robot_qpos)
            rob_grasped = np.array([grasped])
            pick_pos = np.array(pick_goal - place_goal)

            robot_head_rgb = robot_head_rgb.cpu().numpy()
            robot_head_depth = robot_head_depth.cpu().numpy()
            robot_arm_rgb = robot_arm_rgb.cpu().numpy()
            robot_arm_depth = robot_arm_depth.cpu().numpy()
            action_to_save = action_to_save.cpu().numpy().squeeze(0)

            if current_step == 0:

                rel_rob_pos_temp = [rel_rob_pos]
                rob_ori_temp = [rob_ori]
                rel_ee_pos_temp = [rel_ee_pos]
                rob_qpos_temp = [rob_qpos]
                rob_grasped_temp = [rob_grasped]
                pick_pos_temp = [pick_pos]
                robot_head_depth_temp = [robot_head_depth]
                robot_arm_depth_temp = [robot_arm_depth]
                action_to_save_temp = [
                    action_to_save
                ]  # .cpu().numpy()] #########################

            else:
                rel_rob_pos_temp = np.append(
                    rel_rob_pos_temp, [rel_rob_pos], axis=0
                )
                rob_ori_temp = np.append(rob_ori_temp, [rob_ori], axis=0)
                rel_ee_pos_temp = np.append(
                    rel_ee_pos_temp, [rel_ee_pos], axis=0
                )
                rob_qpos_temp = np.append(rob_qpos_temp, [rob_qpos], axis=0)
                rob_grasped_temp = np.append(
                    rob_grasped_temp, [rob_grasped], axis=0
                )
                pick_pos_temp = np.append(pick_pos_temp, [pick_pos], axis=0)
                robot_head_depth_temp = np.append(
                    robot_head_depth_temp, [robot_head_depth], axis=0
                )
                robot_arm_depth_temp = np.append(
                    robot_arm_depth_temp, [robot_arm_depth], axis=0
                )
                action_to_save_temp = np.append(
                    action_to_save_temp, [action_to_save], axis=0
                )  ######################

            #   print("actions to save== " , action_to_save_temp)
            #   print("actions to save== " , len(action_to_save_temp))
            # print("actions to save last shape == " , action_to_save_temp[len(action_to_save_temp-1)].shape)
            #   input()

            if choose_diffusion:

                state = np.concatenate(
                    (rel_rob_pos, rob_ori, rel_ee_pos, rob_grasped, pick_pos),
                    axis=-1,
                )
                #  print("prev_actions == " , prev_actions)
                # print("state == " , state)
                condition = np.concatenate((state, action_to_save), axis=-1)
                rgbd_head = torch.cat(
                    (
                        torch.from_numpy(robot_head_rgb).permute(2, 0, 1),
                        torch.from_numpy(robot_head_depth).permute(2, 0, 1),
                    ),
                    dim=0,
                )
                rgbd_arm = torch.cat(
                    (
                        torch.from_numpy(robot_arm_rgb).permute(2, 0, 1),
                        torch.from_numpy(robot_arm_depth).permute(2, 0, 1),
                    ),
                    dim=0,
                )
                #  print("rgbd_head shape == " , rgbd_head.shape)
                #  input()

                if current_step == 0:
                    conditions = torch.from_numpy(condition)
                    rgbd_head_seq = torch.unsqueeze(rgbd_head, dim=0)
                    rgbd_arm_seq = torch.unsqueeze(rgbd_arm, dim=0)
                    depth_only_head_seq = torch.unsqueeze(
                        torch.from_numpy(robot_head_depth).permute(2, 0, 1),
                        dim=0,
                    )
                    depth_only_arm_seq = torch.unsqueeze(
                        torch.from_numpy(robot_arm_depth).permute(2, 0, 1),
                        dim=0,
                    )
                else:
                    conditions = torch.cat(
                        (conditions, torch.from_numpy(condition)), 0
                    )

                    rgbd_head_seq = torch.cat(
                        (rgbd_head_seq, torch.unsqueeze(rgbd_head, dim=0)),
                        dim=0,
                    )
                    rgbd_arm_seq = torch.cat(
                        (rgbd_arm_seq, torch.unsqueeze(rgbd_arm, dim=0)), dim=0
                    )
                    depth_only_head_seq = torch.cat(
                        (
                            depth_only_head_seq,
                            torch.unsqueeze(
                                torch.from_numpy(robot_head_depth).permute(
                                    2, 0, 1
                                ),
                                dim=0,
                            ),
                        ),
                        dim=0,
                    )
                    depth_only_arm_seq = torch.cat(
                        (
                            depth_only_arm_seq,
                            torch.unsqueeze(
                                torch.from_numpy(robot_arm_depth).permute(
                                    2, 0, 1
                                ),
                                dim=0,
                            ),
                        ),
                        dim=0,
                    )
                    if current_step >= num_prev_obs:
                        #   print("conditions before == ",conditions )
                        conditions = conditions[
                            -(num_prev_obs * conditions_dimensionality) :
                        ]
                        #       print("conditions shape == " , conditions.shape )
                        rgbd_head_seq = rgbd_head_seq[-num_prev_obs:]
                        rgbd_arm_seq = rgbd_arm_seq[-num_prev_obs:]
                        #    print("rgbd_head_seq after == " , rgbd_head_seq.shape)
                        depth_only_head_seq = depth_only_head_seq[
                            -num_prev_obs:
                        ]
                        depth_only_arm_seq = depth_only_arm_seq[-num_prev_obs:]
                    #   print("depth_only_head_seq after == " , depth_only_head_seq.shape)
                    #  input()

            gripped = False
            if choose_diffusion:

                if current_step < num_prev_obs:
                    step_action = {
                        "action": "BaseArmGripperAction2",
                        "action_args": {
                            "base_action": (0, 0),
                            "arm_action": (0, 0, 0, 0, 0, 0, 0),
                            "gripper_action": 1,
                        },
                        "value": 2.9779255390167236,
                    }
                    ob, reward, done, info = env.step(step_action)
                    # current_step+=1
                    estimated_action_trajs = None
                else:  # imagine_trajectories

                    if (current_step - num_prev_obs) % 20 == 0:
                        print(
                            "Generating action trajectories, current step == ",
                            current_step,
                        )

                        """
                        samples, intermediate=sample_ddpm_context(10, conditions, save_rate=20)
                        estimated_trajs=None
                        for sample in range(samples.shape[0]):
                            estimated_traj=from_img_to_vector(samples[sample], required_length=( num_predicted_acts*  (obs_dim + act_dim)))
                            if sample==0:
                                estimated_trajs=np.expand_dims(estimated_traj,0)
                            else:
                                estimated_trajs=np.concatenate((estimated_trajs, np.expand_dims(estimated_traj,0)), axis=0)
                        
                        """

                        if depth_only:
                            with torch.no_grad():
                                rgbd_head_features = model(
                                    torch.unsqueeze(
                                        torch.reshape(
                                            depth_only_head_seq,
                                            (num_prev_obs, 128, 128),
                                        ),
                                        dim=0,
                                    )
                                )
                                rgbd_arm_features = model(
                                    torch.unsqueeze(
                                        torch.reshape(
                                            depth_only_arm_seq,
                                            (num_prev_obs, 128, 128),
                                        ),
                                        dim=0,
                                    )
                                )
                                features_head_squeezed = torch.squeeze(
                                    torch.reshape(rgbd_head_features, (1, -1))
                                ).to(device)
                                features_arm_squeezed = torch.squeeze(
                                    torch.reshape(rgbd_arm_features, (1, -1))
                                ).to(device)
                                network_input = torch.cat(
                                    (
                                        conditions.to(device),
                                        features_head_squeezed,
                                        features_arm_squeezed,
                                    ),
                                    dim=0,
                                ).to(device)
                        else:
                            with torch.no_grad():
                                rgbd_head_features = model(rgbd_head_seq)
                                rgbd_arm_features = model(rgbd_arm_seq)
                                features_head_squeezed = torch.squeeze(
                                    torch.reshape(rgbd_head_features, (1, -1))
                                ).to(device)
                                features_arm_squeezed = torch.squeeze(
                                    torch.reshape(rgbd_arm_features, (1, -1))
                                ).to(device)
                                network_input = torch.cat(
                                    (
                                        conditions.to(device),
                                        features_head_squeezed,
                                    ),
                                    dim=0,
                                ).to(device)

                        print("network input shape == ", network_input.shape)
                        # mlp.float()
                        ## estimated_trajs=mlp(network_input.float())
                        # cond=network_input
                        diff_output = sample(
                            diffusion_model, scheduler, network_input
                        )
                        similarity = cosine_similarity_matrix_torch(
                            diff_output
                        )
                        print("Cosine Similarity Matrix (PyTorch):")
                        for i in range(similarity.shape[0]):
                            print(similarity[i])
                        estimated_trajs = diff_output[:, :200]
                        last_states = diff_output[:, 200:206]

                        # diff_output = diff_output / (diff_output.norm(dim=1, keepdim=True) + 1e-8)
                        # labels = cluster_with_dbscan(diff_output, eps=1.0, min_samples=10)
                        # output_vectors = torch.randn(500, 206)  # Your diffusion model output
                        #   labels, centers, best_k = auto_kmeans_cluster(diff_output, max_k=12)
                        #  print("labels == " , labels)
                        # print("K == " , best_k)

                        #  print("estimated trajectories shape == ", estimated_trajs.shape)
                        # input()
                        # extracted_action_trajs=extract_actions_from_diff_output(estimated_trajs)
                        extracted_action_trajs = (
                            extract_actions_from_diff_output_actions_only(
                                estimated_trajs
                            )
                        )
                        #  print("extracted_action_trajs shape == " , extracted_action_trajs.shape)
                        #    estimated_action_trajs=torch.squeeze(extracted_action_trajs)
                        # best_traj_index=choose_best_traj(estimated_trajs,grasped=rob_grasped)
                        best_traj_index, gripped = imagine_trajectories(
                            env,
                            extracted_action_trajs,
                            gripper_is_grasped,
                            render=True,
                            viewer=viewer,
                        )
                        #  best_traj_index=random.randint(0, estimated_trajs.shape[0]-1)
                        #  best_traj_index=select_traj_from_last_action(diff_output,rob_grasped)
                        estimated_action_trajs = extracted_action_trajs[
                            best_traj_index
                        ]
                    #  estimated_action_trajs=torch.mean(extracted_action_trajs,0)

                    #  print("best estimated action traj shape == " , estimated_action_trajs.shape)
                    #   input()
                    #   estimated_action_trajs=torch.squeeze(estimated_action_trajs)

                    # print(estimated_action_trajs)
                    # best_traj_index=imagine_trajectories(env , estimated_action_trajs, render=True, viewer=viewer)
                    # best_action_traj=estimated_action_trajs[best_traj_index]
                    # for act in range(best_action_traj.shape[0]):
                    #   for act in range(estimated_action_trajs.shape[0]):
                    # action=best_action_traj[act].cpu().numpy()

                    action = (
                        estimated_action_trajs[
                            (current_step - num_prev_obs) % 20
                        ]
                        .cpu()
                        .numpy()
                    )
                    base_action = action[0:2]
                    arm_action = action[2:9]
                    gripper_action = action[9]
                    step_action = {
                        "action": "BaseArmGripperAction2",
                        "action_args": {
                            "base_action": (base_action),
                            "arm_action": (arm_action),
                            "gripper_action": gripper_action,
                        },
                        "value": 2.9779255390167236,
                    }
            #            print("step action == " , step_action)

            else:
                actions = outputs_batch["action"]
                step_action = actions[0].cpu().numpy()
                action = actions[0].cpu().numpy()

            ##################################################################################################################################
            ##################################################################################################################################

            # print("obs == ", obs)
            # print(dir(env.env._env._sim))
            # print(env.env._env._sim.robot.get_state())
            # print("targets == " , env.env._env._sim.targets.keys()) #'007_tuna_fish_can_:0000'
            # print("target == ", env.env._env._sim.rigid_objs['007_tuna_fish_can_:0000'].transformation) ##habitat_config["EPISODE"]
            # print("episode == " , env.env._env._sim.habitat_config["EPISODE"]['rigid_objs'])
            # print("action == ", actions)
            # print("reward == ", reward)
            #    print("info == ", info)
            # input()

            # step_action = {"action": actions[0].cpu().numpy()}
            #   step_action = actions[0].cpu().numpy()
            #  print("step action == " , step_action)
            obs, reward, done, info = env.step(step_action)
            # input()

            successful_grasp = False
            """
            if info['obj_to_goal_dist'] <0.05 :
                placed=True
             #   print("correctly placed")
             #   input()
            
            if placed and info['gripper_to_resting_dist'] <0.05:
                done=True
                successful_grasp=True
              #  info['rearrange_place_success']=True
                print("Successful placing")
               # input()
            """

            ### added by me , to change the task action to the common base-arm action (10D) to be used as the output of the diffusion model
            current_actions = action

            if actions.shape[1] == 1 and not choose_diffusion:
                if gripper_is_grasped:
                    current_action = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

                    current_action = torch.from_numpy(current_action)
                else:
                    current_action = np.array(
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, -1]]
                    )
                    current_action = torch.from_numpy(current_action)

                current_actions = current_action[0].cpu().numpy()

            # current_actions=  current_action

            #     print("step action == " , current_actions)
            #    print("step == " , current_step)
            #       input()

            #  print("current action == " ,   current_actions)
            if current_step == 0:
                current_actions_temp = [current_actions]
            else:
                current_actions_temp = np.append(
                    current_actions_temp, [current_actions], axis=0
                )
            #   input()
            ##########################################

            current_step += 1

            #  if current_step%20==0:
            #     print("reset episode ?")
            #    x=input()
            #   if x=='y':
            #      print("reseting epsidoe")
            #     done=True

            current_episode_reward += reward
            metrics = self._extract_scalars_from_info(info)

            if viewer_:
                frame = env.render(
                    "human",
                    info=metrics,
                    overlay_info=False,
                    show_info=True,
                )

                key = viewer.imshow(
                    frame[..., :3],
                    delay=1,
                )

            if choose_diffusion:
                prev_actions = np.expand_dims(current_actions, axis=0)
            else:
                prev_actions = actions.cpu()

            if (
                done or current_step > 300
            ):  # or successful_grasp or current_step>1000:# or gripper_is_grasped or gripped: #or gripper_is_grasped or current_step>400:

                episode_stats = metrics.copy()
                episode_stats["return"] = current_episode_reward
                all_episode_stats.append(episode_stats)
                total_episodes += 1
                pbar.update()

                success_measure = self.config.RL.SUCCESS_MEASURE
                if success_measure in info:
                    episode_success = info[success_measure]
                    if not episode_success:
                        failure_episodes.append(env.current_episode.episode_id)
                else:
                    episode_success = -1

                if (
                    episode_success
                ):  # or not episode_success:#or successful_grasp:
                    print("successful trajectory")
                    print("Episode ID == ", env.current_episode.episode_id)
                    print("save episode ?")
                    #  x=input()
                    if True:  # x=='y':
                        print("saving epsiode")
                        done = True
                        episode_ids.append(env.current_episode.episode_id)

                        dataset_dict["{}".format(total_episodes - 1)] = {}
                        dataset_dict["{}".format(total_episodes - 1)][
                            "rel_rob_pos"
                        ] = rel_rob_pos_temp
                        dataset_dict["{}".format(total_episodes - 1)][
                            "rob_ori"
                        ] = rob_ori_temp
                        dataset_dict["{}".format(total_episodes - 1)][
                            "rel_ee_pos"
                        ] = rel_ee_pos_temp
                        dataset_dict["{}".format(total_episodes - 1)][
                            "rob_qpos"
                        ] = rob_qpos_temp
                        dataset_dict["{}".format(total_episodes - 1)][
                            "rob_grasped"
                        ] = rob_grasped_temp
                        dataset_dict["{}".format(total_episodes - 1)][
                            "pick_pos"
                        ] = pick_pos_temp
                        #  dataset_dict['{}'.format(num_suc_episodes)]['robot_head_rgb']=robot_head_rgb_temp
                        dataset_dict["{}".format(total_episodes - 1)][
                            "robot_head_depth"
                        ] = robot_head_depth_temp
                        # dataset_dict['{}'.format(num_suc_episodes)]['robot_arm_rgb']=robot_arm_rgb_temp

                        dataset_dict["{}".format(total_episodes - 1)][
                            "robot_arm_depth"
                        ] = robot_arm_depth_temp
                        #  dataset_dict['{}'.format(num_suc_episodes)]['prev_actions']=prev_actions_temp
                        dataset_dict["{}".format(total_episodes - 1)][
                            "current_actions"
                        ] = action_to_save_temp
                        dataset_dict["{}".format(total_episodes - 1)][
                            "episode_ids"
                        ] = episode_ids

                        # print("dict == " , dataset_dict)
                        #    print("keys == " , dataset_dict.keys())
                        ##    print("episode IDs == " , episode_ids)
                        print("total number of episodes == ", total_episodes)
                        print("successful episodes == ", num_suc_episodes)
                        print("total number of steps == ", total_steps)
                        print(
                            "len(all_episode_stats) == ",
                            len(all_episode_stats),
                        )

                        if saved_eps % 100 == 99:
                            saves += 1
                            if not choose_diffusion:
                                with open(
                                    "./hab-mobile-manipulation/new_dataset_17_sept/nav_ee_noise_0_15_seed_200_random_ep_p{}.pkl".format(
                                        saves
                                    ),
                                    "wb",
                                ) as f:
                                    pickle.dump(dataset_dict, f)
                                    print("saved dataset_dict to file")
                                    dataset_dict = {}
                                #  input()
                        if num_suc_episodes % 100 == 99:
                            saved_eps = 0
                        else:
                            saved_eps += 1

                    rel_rob_pos_temp = np.array([])
                    rob_ori_temp = np.array([])
                    rel_ee_pos_temp = np.array([])
                    rob_qpos_temp = np.array([])
                    rob_base_lin_vel_temp = np.array([])
                    rob_base_ang_vel_temp = np.array([])
                    rob_grasped_temp = np.array([])
                    pick_pos_temp = np.array([])
                    #        robot_head_rgb_temp=np.array([])
                    robot_head_depth_temp = np.array([])
                    #       robot_arm_rgb_temp=np.array([])
                    robot_arm_depth_temp = np.array([])
                    prev_actions_temp = np.array([])
                    current_actions_temp = np.array([])
                    action_to_save_temp = np.array([])

                    num_suc_episodes += 1
                    current_step = 0

                    #   input()

                else:

                    rel_rob_pos_temp = np.array([])
                    rob_ori_temp = np.array([])
                    rel_ee_pos_temp = np.array([])
                    rob_qpos_temp = np.array([])
                    rob_base_lin_vel_temp = np.array([])
                    rob_base_ang_vel_temp = np.array([])
                    rob_grasped_temp = np.array([])
                    pick_pos_temp = np.array([])
                    #     robot_head_rgb_temp=np.array([])
                    robot_head_depth_temp = np.array([])
                    #    robot_arm_rgb_temp=np.array([])
                    robot_arm_depth_temp = np.array([])
                    prev_actions_temp = np.array([])
                    current_actions_temp = np.array([])
                    action_to_save_temp = np.array([])

                    current_step = 0

                ##  episode IDs ==  ['376', '661', '65', '842', '805', '711', '906', '132']
                ##episode IDs ==  ['342', '524', '986', '979', '167', '563', '255', '189', '247', '373', '456', '241', '527', '999', '995', '179']

                """
                video_dir='/home/shokry/hab-mobile-manipulation/diffusion_dataset_new/videos/all_targets_epoch_14000'
                if len(config.VIDEO_OPTION) > 0:
                    print("saving video")
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        #video_dir=config.VIDEO_DIR,
                        video_dir=video_dir,
                        images=rgb_frames,
                        episode_id=env.current_episode.episode_id,
                        checkpoint_idx=total_episodes,
                        metrics={"success": episode_success},
                        tb_writer=writer,
                        fps=30,
                    )
                """

                #  env.current_episode.episode_id=0
                obs = env.reset()
                #  print("env.current_episode.episode_id == " , env.current_episode.episode_id )
                #  input()
                metrics = {}
                current_episode_reward = 0
                rgb_frames = []

            # Update policy inputs
            batch = batch_obs(
                [obs], device=self.device, cache=self._obs_batching_cache
            )
            #    print(" batch == " , batch['robot_arm_depth'].shape)
            #    print("finished batch = batch_obs() ")
            not_done_masks = torch.tensor(
                [[not done]], dtype=torch.bool, device=self.device
            )
            #     print("finished not_done_masks = torch.tensor( ")
            buffer.update(
                recurrent_hidden_states=outputs_batch["rnn_hidden_states"],
                prev_actions=outputs_batch["action"],
                masks=not_done_masks,
            )
        #     print("finished buffer.update(")

        # Logging metrics
        aggregated_stats = {
            k: np.mean([ep_info[k] for ep_info in all_episode_stats])
            for k in all_episode_stats[0].keys()
        }
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        failure_episodes = sorted(failure_episodes)
        failure_episodes_str = ",".join(map(str, failure_episodes))
        logger.info("Failure episodes:\n{}".format(failure_episodes_str))

        # Summarize in tensorboard
        step_id = ckpt_dict.get("step", checkpoint_index)
        for k, v in aggregated_stats.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)
        print("closing env")
        env.close()

    def _batch_eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = -1,
    ) -> None:
        """Evaluate the checkpoint with a batch of envs.
        Videos are not supported for simplicity.
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        logger.info(f"Loaded {checkpoint_path}")
        ckpt_dict = torch.load(checkpoint_path, map_location="cpu")

        config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if config.VERBOSE:
            logger.info(config)

        self._init_envs(config, auto_reset_done=True)
        self._init_observation_space(config)
        self._init_action_space(config)
        self._setup_actor_critic(config)
        self.actor_critic.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic.eval()

        if config.EVAL.NUM_EPISODES == -1:
            num_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            num_eval_episodes = config.EVAL.NUM_EPISODES

        num_envs = self.envs.num_envs
        current_episode_rewards = [0 for _ in range(num_envs)]
        all_episode_stats = dict()
        failure_episodes = []

        # Initialize policy inputs
        obs = self.envs.reset()
        self._obs_batching_cache = ObservationBatchingCache()
        batch = batch_obs(
            obs, device=self.device, cache=self._obs_batching_cache
        )
        buffer = dict(
            recurrent_hidden_states=torch.zeros(
                num_envs,
                self.actor_critic.net.num_recurrent_layers,
                self.actor_critic.net.rnn_hidden_size,
                device=self.device,
            ),
            prev_actions=torch.zeros(
                num_envs,
                *self.action_space.shape,
                device=self.device,
                dtype=torch.float,
            ),
            masks=torch.zeros(
                num_envs,
                1,
                device=self.device,
                dtype=torch.bool,
            ),
        )

        pbar = tqdm.tqdm(total=num_eval_episodes)
        while len(all_episode_stats) < num_eval_episodes:
            current_episodes = self.envs.current_episodes()
            with torch.no_grad():
                step_batch = dict(observations=batch, **buffer)
                outputs_batch = self.actor_critic.act(
                    step_batch, deterministic=config.EVAL.DETERMINISTIC_ACTION
                )
                actions = outputs_batch["action"]
                actions = actions.to(device="cpu", non_blocking=True)

            step_action = [{"action": a.numpy()} for a in actions]
            results = self.envs.step(step_action)
            obs, rewards, dones, infos = zip(*results)

            for i_env in range(num_envs):
                current_episode_rewards[i_env] += rewards[i_env]

                if dones[i_env]:
                    episode_id = current_episodes[i_env].episode_id
                    # print("Episode {} done".format(episode_id))

                    # Ignore if the episode has already been evaluated
                    if episode_id not in all_episode_stats:
                        metrics = self._extract_scalars_from_info(infos[i_env])
                        episode_stats = metrics.copy()
                        episode_stats["return"] = current_episode_rewards[
                            i_env
                        ]
                        all_episode_stats[episode_id] = episode_stats

                        success_measure = self.config.RL.SUCCESS_MEASURE
                        if success_measure in infos[i_env]:
                            episode_success = infos[i_env][success_measure]
                            if not episode_success:
                                failure_episodes.append(episode_id)
                        else:
                            episode_success = -1

                        pbar.update()

                    # Reset episode stats
                    current_episode_rewards[i_env] = 0

            # Update policy inputs
            batch = batch_obs(
                obs, device=self.device, cache=self._obs_batching_cache
            )
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.device,
            )
            buffer.update(
                recurrent_hidden_states=outputs_batch["rnn_hidden_states"],
                prev_actions=outputs_batch["action"],
                masks=not_done_masks,
            )

        # Logging metrics
        episode_ids = list(all_episode_stats.keys())
        stat_keys = list(all_episode_stats[episode_ids[0]].keys())
        aggregated_stats = {
            k: np.mean([ep_info[k] for ep_info in all_episode_stats.values()])
            for k in stat_keys
        }
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        # Logging failure episodes
        failure_episodes = sorted(failure_episodes)
        failure_episodes_str = ",".join(map(str, failure_episodes))
        logger.info("Failure episodes:\n{}".format(failure_episodes_str))

        # Summarize in tensorboard
        step_id = ckpt_dict.get("step", checkpoint_index)
        for k, v in aggregated_stats.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
