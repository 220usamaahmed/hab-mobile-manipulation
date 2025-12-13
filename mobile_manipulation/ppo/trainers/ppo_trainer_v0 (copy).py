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
from habitat_extensions.utils.viewer import OpenCVViewer
from habitat.tasks.utils import cartesian_to_polar
import pickle



from einops import rearrange

import gc

import math






Diffusion_policy=False
Save_data=True

if Diffusion_policy and Save_data:
    print("cannot have both diffusion policy and data saving enabled at the same time")
    input()

device = torch.device("cuda" if torch.cuda.is_available() else torch.device('cpu'))
#device =torch.device( "cpu")

gc.collect()
torch.cuda.empty_cache()
  

def imagine_trajectories(env , action_trajectories,gripper_is_grasped,similarity_vector, render=False, viewer=None):
    num_trajs=action_trajectories.shape[0]
    num_actions_per_traj=action_trajectories.shape[1]
    initial_robot_pos=env.env._env._sim.robot.base_pos
    initial_robot_ori=env.env._env._sim.robot.base_ori
    initial_qpos=env.env._env._sim.robot.arm_joint_pos
    start_state=(np.array(initial_robot_pos), initial_robot_ori)
    start_state=env.env._env._sim.get_state()
    gripped=False
    distance_vector= torch.zeros(num_trajs,device=device)
    for traj in range(num_trajs):
        #env.env._env.reset_to_given_pose(start_state=start_state,qpos=initial_qpos)
        env.env._env._sim.set_state(start_state)
        for act in range(num_actions_per_traj):
            action=action_trajectories[traj][act].detach().cpu().numpy()
           # print("action == " , action)
            base_action=action[0:2]
            arm_action=action[2:9]
            gripper_action=action[9]
            step_action={'action': 'BaseArmGripperAction2', 'action_args': {'base_action': (base_action) , 'arm_action':(arm_action) , 'gripper_action':gripper_action }, 'value': 2.9779255390167236}
            ob, reward, done, info=env.step(step_action)

            if render:
                frame = env.render("human",overlay_info=False,show_info=False)
                key = viewer.imshow(
                    frame[..., :3], delay=10 
                )
            if env.env._env._sim.gripper.is_grasped>0.5:
                if gripper_is_grasped==-1:
                    return traj,True

        pick_goal=env.env._env._task.pick_goal
        place_goal=env.env._env._task.place_goal         
        robot_ee_pos=env.env._env._sim.robot.ee_T.translation
        rob_base_pos=env.env._env._sim.robot.base_pos

        if gripper_is_grasped>0:
            rel_ee_pos= np.array(robot_ee_pos - place_goal)
            rel_base_pos=np.array(rob_base_pos-place_goal)
            
        else:
            rel_ee_pos= np.array(robot_ee_pos - pick_goal)
            rel_base_pos=np.array(rob_base_pos-pick_goal)
        rel_ee_pos=torch.norm(torch.from_numpy(rel_ee_pos))
        rel_base_pos=torch.norm(torch.from_numpy(rel_base_pos))
        distance_vector[traj]=rel_ee_pos
        if traj==0:
            best_rel_ee_pos=rel_ee_pos#+rel_base_pos
            best_traj=0
        else:
            #if rel_ee_pos+rel_base_pos< best_rel_ee_pos:
            if rel_ee_pos< best_rel_ee_pos:
                best_rel_ee_pos=rel_ee_pos#+rel_base_pos
                best_traj=traj

        if env.env._env._sim.gripper.is_grasped :
            gripped=True
            
            

    #    print("finished imagined trajectory number {}".format(traj))
     #   print("target index == " ,env.env._env._task.tgt_idx)
      #  print("grasped == " , gripper_is_grasped)
       # print("press enter for the next trajectory")
     #   input()
   # env.env._env.reset_to_given_pose(start_state=start_state,qpos=initial_qpos)

    weighted_distance=distance_vector*(1-similarity_vector)
   # best_traj=torch.argmin(weighted_distance)
    env.env._env._sim.set_state(start_state)
    return best_traj,gripped









def cosine_similarity_matrix_torch(vectors: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity matrix for a set of vectors using PyTorch.
    
    Parameters:
        vectors (Tensor): A 2D tensor of shape (n_vectors, dimensions)
        
    Returns:
        Tensor: A 2D tensor of shape (n_vectors, n_vectors) with cosine similarities
    """
    vectors=torch.reshape(vectors,(-1,200))
 #   print("vectors shape == " , vectors.shape)
    # Normalize each vector (row) to unit length
    norms = torch.norm(vectors, dim=1, keepdim=True)  # shape: (n_vectors, 1)
    normalized_vectors = vectors / (norms + 1e-8)     # add small value to avoid division by zero
    
    # Compute cosine similarity as dot product of normalized vectors
    similarity_matrix = torch.matmul(normalized_vectors, normalized_vectors.T)
    similarity_vector=torch.sum(similarity_matrix,dim=1)
  #  print("similarity vector shape == " , similarity_vector.shape)
    similarity_summation=torch.sum(similarity_vector)
    similarity_vector=similarity_vector/similarity_summation
    best_idx=torch.argmax(similarity_vector)

    print("simiarity matrix  == ", similarity_matrix)
  #  print("similarity vector  == ", similarity_vector)
  #  print("best index == " , best_idx)
   # print("best similarity == " , similarity_vector[best_idx])
    #print("best similarity matrix == " , similarity_matrix[best_idx])

    return similarity_vector,best_idx






class Flatten(nn.Module):
    r"""Copied from torch 1.9."""
    __constants__ = ["start_dim", "end_dim"]
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return "start_dim={}, end_dim={}".format(self.start_dim, self.end_dim)


class SimpleCNN(nn.ModuleList):
    def __init__(self, in_channels, input_shape, out_channels) -> None:
        super().__init__()

        self.extend(
            [
                nn.Conv2d(in_channels, 32, 8, stride=4),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(True),
                nn.Conv2d(64, 32, 3, stride=1),
                Flatten(),
            ]
        )

        # Infer the final output resolution
        with torch.no_grad():
            x = torch.zeros(1, in_channels, *input_shape)
            dim = self.forward(x).size(-1)
        self.extend([nn.Linear(dim, out_channels), nn.ReLU(True)])

        self.reset_parameters()

    def forward(self, x):
        for m in self:
            x = m(x)
        return x

    def reset_parameters(self):
        for m in self:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)




Feat_ext = SimpleCNN(1, (128, 128), 512).to(device).to(torch.float32)
Feat_ext.load_state_dict(torch.load(
    '/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/visual_encoder.pth',
    map_location='cpu',
    weights_only=True
))

Feat_ext.to(device)
Feat_ext.eval()









class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)                     # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)       # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                      # (d_model/2)

        pe[:, 0::2] = torch.sin(position * div_term)           # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)           # Apply cos to odd indices
        pe = pe.unsqueeze(0)                                   # Shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)  # Not a parameter, but saved with the model

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :].to(x.device)





# Sinusoidal Timestep Embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / half_dim))
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# Cross-Attention Block
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        b, n_decoder, _ = x.shape
        b_context, n_context, _ = context.shape
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, 'b n_decoder (h d) -> b h n_decoder d', h=h)
        k = rearrange(k, 'b n_context (h d) -> b h n_context d', h=h)
        v = rearrange(v, 'b n_context (h d) -> b h n_context d', h=h)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn_scores.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n_decoder d -> b n_decoder (h d)')
        return self.to_out(out)

# Transformer Block with Cross Attention
class DiffusionTransformerBlock(nn.Module):
    def __init__(self, dim, cond_dim, heads=8, dim_head=128):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True,dim_feedforward=256)
     #   self.atten_cond = nn.TransformerEncoderLayer(d_model=cond_dim, nhead=heads, batch_first=True,dim_feedforward=256)
        self.cross_attn = CrossAttention(dim, cond_dim, heads, dim_head)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, cond):
        x = self.attn(x)
      #  cond = self.atten_cond(cond)
        x = self.norm(x + self.cross_attn(x, cond))
        return x

# Conditional Diffusion Model
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, action_dim=10, output_dim=10,sensor_dim=21,depth_features_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()

    #    self.visual_feature_extractor = Feat_ext
        self.action_input_proj = nn.Linear(action_dim , hidden_dim)
        self.visual_obs_projection= nn.Linear(depth_features_dim, hidden_dim)
        self.non_visual_obs_projection= nn.Linear(sensor_dim, hidden_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),

        )
      #  self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear( hidden_dim,action_dim )
        
        self.decoder_position_embedding=SinusoidalPositionalEncoding(hidden_dim, max_len=21)  # Action position embedding
        self.encoder_position_embedding=SinusoidalPositionalEncoding(hidden_dim, max_len=21)  # Sensor position embedding


    def forward(self, visual_obs, non_visual_obs, noisy_action, t):

        batch_size=non_visual_obs.shape[0]
        context_length=non_visual_obs.shape[1]
        


        noisy_action=self.action_input_proj(noisy_action.to(torch.float32))  




        visual_obs=self.visual_obs_projection(visual_obs.to(torch.float32))  
       # print("shape after visual feature projection == " , visual_obs.shape )
        visual_obs=visual_obs.reshape(batch_size, context_length, -1)  


       # print("shape after reshaping back to batch and context length == " , visual_obs.shape )

       # print("initial non visual observations shape == " , non_visual_obs.shape)
        non_visual_obs=self.non_visual_obs_projection(non_visual_obs.to(torch.float32))
      #  print("shape after reshaping back to batch and context length == " , non_visual_obs.shape )

        t=self.time_mlp(t.to(torch.float32))  # Time embedding
        t = t.unsqueeze(1)
        t=t.repeat(batch_size, 1, 1)  # Repeat to match action sequence length
     #   print("t shape after embedding == " , t.shape)

        encoder_input=torch.cat((visual_obs,non_visual_obs,t),dim=1)
       # print("encoder input shape == " , encoder_input.shape)
        encoder_input=self.encoder_position_embedding(encoder_input)  # Apply sensor position embedding

        decoder_input=torch.cat((t,noisy_action),dim=1)
       # print("decoder input shape == " , decoder_input.shape)
        decoder_input = self.decoder_position_embedding(decoder_input)  # Apply action position embedding

        

        for block in self.transformer_blocks:
            decoder_input = block(decoder_input, encoder_input)
        out=self.output_proj(decoder_input)
        out=out[:,1:,:]
     #   print("final out shape == " , out.shape)
        return out             


# Noise Scheduler (like DDPM)
class NoiseScheduler:
    def __init__(self, timesteps=500, beta_start=1e-4, beta_end=0.02):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       # device = torch.device("cpu")
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t=torch.reshape(t, (x_start.shape[0],-1))  # Ensure t is a 1D tensor
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().unsqueeze(1)
        sqrt_one_minus_alpha_cumprod = (1. - self.alpha_cumprod[t]).sqrt().unsqueeze(1)
        x_new=sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    def get_loss(self, model, x_start, t, visual_obs_batch , non_visual_obs_batch): 
        noise = torch.randn_like(x_start).to(torch.float32)
        noisy_action = self.q_sample(x_start, t, noise)
        predicted_noise = model(visual_obs_batch.to(torch.float32), non_visual_obs_batch.to(torch.float32), noisy_action.to(torch.float32), t.to(torch.float32))
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def p_sample(self, model, noisy_action, t, visual_obs , non_visual_obs):

        t.to(device)
        # Predict noise using the model

        predicted_noise = model(visual_obs, non_visual_obs, noisy_action, t )

      #  print("noisy action shape == " , noisy_action_decoder.shape)
       # print("predicted noise shape == " , predicted_noise.shape)
        # Extract coefficients for denoising
        alpha_t = self.alphas[t.to(device).to(torch.long)]#.view(-1,1, 1)
       # alpha_t= alpha_t.repeat(10,20,1)
        #print("alpha t shape == " , alpha_t.shape)
        alpha_cumprod_t = self.alpha_cumprod[t.to(device).to(torch.long)]#.view(-1, 1)
        beta_t = self.betas[t.to(device).to(torch.long)]#.view(-1, 1)
        
        # Compute mean of reverse process
        sqrt_alpha_t = torch.sqrt(alpha_t).to(device)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        
        # Compute mean
        pred_mean = (noisy_action - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t) / sqrt_alpha_t
      #  print("predicted mean shape == " , pred_mean.shape)
        # Add noise for all timesteps except t=0
        if t.min() > 0:
            noise = torch.randn_like(noisy_action)
            pred_mean = pred_mean + torch.sqrt(beta_t) * noise
        
        return pred_mean

    @torch.no_grad()
    def sample(self, model, shape, visual_obs , non_visual_obs , device, num_random_samples=20):
        """
        Generate samples using the reverse diffusion process
        
        Starts from pure noise and gradually denoises to generate data
        
        Args:
            model: Trained diffusion model
            shape: Shape of data to generate (batch_size, 10, 10)
            condition: Conditioning information
            device: Device to run on
        Returns:
            Generated samples
        """
        # Start from pure noise
        noisy_action_decoder = torch.randn(shape, device=device)
       # if len(visual_obs.shape)==4:
        #    visual_obs=visual_obs.unsqueeze(0)  # Add sequence dimension if missing
        visual_obs=visual_obs.repeat(shape[0],1,1)  # Repeat condition for batch size
        non_visual_obs=non_visual_obs.repeat(shape[0],1,1) 


        for t in reversed(range(self.timesteps)):
            # Create timestep tensor
            t_tensor = torch.tensor([t]).to(device).to(torch.float32)#torch.full((shape[0],), t, device=device, dtype=torch.long)
            noisy_action_decoder = self.p_sample(model,  noisy_action_decoder, t_tensor, visual_obs , non_visual_obs)
        return noisy_action_decoder






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

    def is_done(self):
        return self.num_steps_done >= self.config.TOTAL_NUM_STEPS

    def percent_done(self):
        return self.num_steps_done / self.config.TOTAL_NUM_STEPS

    def train(self) -> None:
        ppo_cfg = self.config.RL.PPO

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
                self.count_checkpoints += 1
                self.prev_ckpt_step = self.num_steps_done
                self.save(ckpt_id=self.count_checkpoints)


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

        with self.timer.timeit("sample_action"):
            step_batch = self.rollouts.buffers[self.rollouts.step_idx]
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
      #  checkpoint_path='/home/shokry/hab-mobile-manipulation/data/results/rearrange/skills/tidy_house/ckpt.10.pth'
        ckpt_dict = torch.load(checkpoint_path, map_location="cpu")

        config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.freeze()

        if config.VERBOSE:
            logger.info(config)

        env = make_env_fn(
            config,
            get_env_class(config.ENV_NAME),
            wrappers=[HabitatActionWrapper],
        )

        self.envs = [env]
        self._init_observation_space(config)
        self._init_action_space(config)
        self._setup_actor_critic(config)
        self.actor_critic.load_state_dict(ckpt_dict["state_dict"])
     #   print(self.actor_critic.net.visual_encoder)
      #  PATH='/home/shokry/hab-mobile-manipulation/pretrained_encoder/visual_encoder.pth'
      #  torch.save(self.actor_critic.net.visual_encoder.state_dict(), PATH)
      #  input()
        self.actor_critic.eval()

        if config.EVAL.NUM_EPISODES == -1:
            num_eval_episodes = env.number_of_episodes
        else:
            num_eval_episodes = config.EVAL.NUM_EPISODES

        current_episode_reward = 0.0
        all_episode_stats = []
        rgb_frames = []
        failure_episodes = []
        
        show_viewer=True
        if show_viewer:
            viewer = OpenCVViewer(config.TASK_CONFIG.TASK.TYPE)


        # Initialize policy inputs

        obs = env.reset()
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
        current_step=0
        

        print("num_eval_episodes == ", num_eval_episodes)

        robot_head_depth_temp=np.array([])
        robot_arm_depth_temp=np.array([])
        rob_qpos_temp=np.array([])
        rel_resting_pos_temp=np.array([])
        rel_pick_pos_ee_temp=np.array([])
        rel_place_pos_ee_temp=np.array([])
        rel_pick_pos_base_temp=np.array([])
        rel_place_pos_base_temp=np.array([])
        rel_pick_pos_base_polar_temp=np.array([])
        rel_place_pos_base_polar_temp=np.array([])
        is_holding_temp=np.array([])
        action_to_save_temp=np.array([])

        dataset_dict={}
        episode_ids=[]


        number_of_episodes=0
        num_suc_episodes=0
        total_steps=0
        saves=0


        num_prev_obs=5
        num_predicted_acts=20

        number_of_steps=0
        visual_obs_buffer=[]
        non_visual_obs_buffer=[]

        if Diffusion_policy:
            diffusion_policy=ConditionalDiffusionModel()
            diffusion_policy.load_state_dict(torch.load("/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/weights_diff_transformer_complete_trajs_cnn_encoder_scratch/model_30000.pt",
                                                    map_location=device))
            diffusion_policy.to(device)
            diffusion_policy.eval()
            for p in diffusion_policy.parameters():
                p.requires_grad_(False)
            scheduler = NoiseScheduler()

            print("Loaded Diffusion Policy")
            
        pbar = tqdm.tqdm(total=num_eval_episodes)

        while len(all_episode_stats) < num_eval_episodes:
            print("current episode == ", env.current_episode.episode_id)
            if len(config.VIDEO_OPTION) > 0:
                rgb_frames.append(env.render("human", info=metrics))

            with torch.no_grad():
                step_batch = dict(observations=batch, **buffer)
                outputs_batch = self.actor_critic.act(
                    step_batch, deterministic=config.EVAL.DETERMINISTIC_ACTION
                )
                actions = outputs_batch["action"]
                print("actions before diffusion == " , actions)

            # step_action = {"action": actions[0].cpu().numpy()}
            step_action = actions[0].cpu().numpy()


            gripper_is_grasped= env.env._env._sim.gripper.is_grasped 	

            if actions.shape[-1] == 1:
                self.possible_velocities = np.array(
                [
                    [lin_vel, ang_vel]
                    for lin_vel in np.linspace(-0.5, 1.0, 4)
                    for ang_vel in np.linspace(-1.0, 1.0, 5)
                ]
                )
                current_velocity=self.possible_velocities[actions]
                if gripper_is_grasped:
                    action_to_save=torch.tensor([[current_velocity[0]*3 , current_velocity[1]*3 , 0 , 0, 0 ,0 , 0 ,0 ,0 , 1 ]])

                else:
                    action_to_save=torch.tensor([[current_velocity[0]*3 , current_velocity[1]*3  , 0 , 0, 0 ,0 , 0 ,0 ,0 , -1 ]]) ## the gripper action should be -1
                print("current_velocity == " , current_velocity)
            else:
                action_to_save=torch.clamp(actions, min=-1, max=1)
                action_to_save[:, 0:2]*=1.5
 

            action_to_save=action_to_save.cpu().numpy()
         #   print("action_to_save == " , action_to_save)
          #  print("action to save == " , action_to_save.shape)




            #######################################################
            #############   Data collection code  #################
            ## This data is collected before action execution ##

            robot_base_pos= env.env._env._sim.robot.base_pos
            robot_base_orientation= env.env._env._sim.robot.base_ori
            robot_qpos= env.env._env._sim.robot.arm_joint_pos 	
           # robot_ee_T= env.env._env._sim.robot.ee_T 	
            robot_ee_pos=env.env._env._sim.robot.gripper_T.translation
        
            gripper_is_grasped= env.env._env._sim.gripper.is_grasped 	
            if gripper_is_grasped:
            	grasped=1
            else:
            	grasped=-1


            pick_goal=   env.env._env._task.pick_goal    
            place_goal=   env.env._env._task.place_goal 
            resting_pos=env.env._env._task.resting_position   


            ## rgb and depth images has the format (W,H,C) where C is the number of channels
            ## You will need to use permute function to change it to (C,W,H) to be compatible with pytorch
            robot_head_rgb = obs['robot_head_rgb']
            robot_arm_rgb = obs['robot_arm_rgb']
            robot_head_depth = obs['robot_head_depth']
            robot_arm_depth = obs['robot_arm_depth']

            robot_transform = env.env._env._sim.robot.base_T
            robot_ee_transform=env.env._env._sim.robot.gripper_T
            local_ee_pos_relative_to_base=robot_transform.inverted().transform_point(robot_ee_pos)
          #  abs_ee_pos = env.env._env._sim.robot.ee_transform.translation
            relative_pick_pos_base = robot_transform.inverted().transform_point(pick_goal)
            relative_pick_pos_base_polar=cartesian_to_polar(relative_pick_pos_base[0], relative_pick_pos_base[2])
            relative_pick_pos_ee=robot_ee_transform.inverted().transform_point(pick_goal)

            relative_place_pos_base = robot_transform.inverted().transform_point(place_goal)
            relative_place_pos_base_polar=cartesian_to_polar(relative_place_pos_base[0], relative_place_pos_base[2])
            relative_place_pos_ee = robot_ee_transform.inverted().transform_point(place_goal)
            relative_resting_position=local_ee_pos_relative_to_base-resting_pos
       #     print("image shape == " , torch.from_numpy(robot_head_depth).unsqueeze(0).permute(0,3,1,2).to(device).shape)
            robot_head_depth_features=Feat_ext( torch.from_numpy(robot_head_depth).unsqueeze(0).permute(0,3,1,2).to(device))
            visual_obs_buffer.append(robot_head_depth_features.squeeze(0).cpu().detach().numpy())

            visual_obs_buffer_np=np.array(visual_obs_buffer)

            non_visual_obs_buffer.append( np.concatenate((
                relative_resting_position,
                relative_pick_pos_ee,
                relative_place_pos_ee,
                relative_pick_pos_base_polar,
                relative_place_pos_base_polar,
                robot_qpos,
                np.array([int(env.env._env._sim.gripper.is_grasped)]),
            ),axis=-1))
            non_visual_obs_buffer_np=np.array(non_visual_obs_buffer)


            if number_of_steps==0:
                visual_obs_buffer_np=np.repeat(visual_obs_buffer_np, num_prev_obs, axis=0)
                non_visual_obs_buffer_np=np.repeat(non_visual_obs_buffer_np, num_prev_obs, axis=0)

            elif number_of_steps >= num_prev_obs:
                visual_obs_buffer=visual_obs_buffer[-num_prev_obs: ]
                visual_obs_buffer_np=visual_obs_buffer_np[-num_prev_obs: , ...]

                non_visual_obs_buffer=non_visual_obs_buffer[-num_prev_obs: ]
                non_visual_obs_buffer_np=non_visual_obs_buffer_np[-num_prev_obs: , ...]

       #     print("visual_obs_buffer_np.shape == " , visual_obs_buffer_np.shape)
        #    print("non_visual_obs_buffer_np.shape == " , non_visual_obs_buffer_np.shape)
            if Diffusion_policy:
                if number_of_steps==0 or number_of_steps % 10 ==0:
                    with torch.no_grad():
                        shape = (10,20,10)
                        actions=scheduler.sample(diffusion_policy, shape, torch.from_numpy(visual_obs_buffer_np).to(device).to(torch.float32) , torch.from_numpy(non_visual_obs_buffer_np).to(device).to(torch.float32) , device, num_random_samples=20)
                        #actions[:,:,0:2]*=3.0
                        similarity_vector , best_traj_index = cosine_similarity_matrix_torch(actions)
                    #  if relative_pick_pos_base_polar[0]<1 or relative_place_pos_base_polar[0]<1:
                        best_traj_index,gripped=imagine_trajectories(env , actions,gripper_is_grasped,similarity_vector, render=True, viewer=viewer)
                        estimated_action_trajs=actions[best_traj_index]

                action=estimated_action_trajs[number_of_steps%10 ].detach().cpu().numpy()
            #    print("action == " , action)
                action[0:2]=np.clip(action[0:2],-3,3)  # Clip base actions
                action[2:9]=np.clip(action[2:9],-1,1)  # Clip arm actions
                action[9]=np.clip(action[9],-1,1)  # Clip gripper action 

                base_action=np.clip(action[0:2],-3,3)
            #  print("base action == " , base_action)
                arm_action=np.clip(action[2:9],-1,1)
            # print("arm action == " , arm_action)
                gripper_action=np.clip(action[9],-1,1)

                step_action={'action': 'BaseArmGripperAction2', 'action_args': {'base_action': (base_action) , 'arm_action':(arm_action) , 'gripper_action':gripper_action }, 'value': 2.9779255390167236}
                

            print("step action == " , step_action)
            ob, reward, done, info = env.step(step_action)
           # episode_reward += reward

            print("step number == " , number_of_steps)
            number_of_steps+=1

            metrics = extract_scalars_from_info(info)
            success = metrics.get(config.RL.SUCCESS_MEASURE, -1)

            if number_of_steps%10==0:
                print("reset episode ?")
                x=input()
                if x=='y':
                    print("reseting epsidoe")
                    done=True
            

            if current_step==0:
                robot_head_depth_temp=[robot_head_depth]
                robot_arm_depth_temp=[robot_arm_depth]
                rob_qpos_temp=[robot_qpos]
                rel_resting_pos_temp=[local_ee_pos_relative_to_base-resting_pos]
                rel_pick_pos_ee_temp=[relative_pick_pos_ee]
                rel_place_pos_ee_temp=[relative_place_pos_ee]
                rel_pick_pos_base_polar_temp=[relative_pick_pos_base_polar]
                rel_place_pos_base_polar_temp=[relative_place_pos_base_polar]
                rel_pick_pos_base_temp=[relative_pick_pos_base]
                rel_place_pos_base_temp=[relative_place_pos_base]
                is_holding_temp=[np.array([gripper_is_grasped])]
                action_to_save_temp=action_to_save

            else:
                robot_head_depth_temp=np.append( robot_head_depth_temp,[robot_head_depth],axis=0)
                robot_arm_depth_temp=np.append( robot_arm_depth_temp,[robot_arm_depth],axis=0)
                rob_qpos_temp=np.append( rob_qpos_temp,[robot_qpos],axis=0)
                rel_resting_pos_temp=np.append( rel_resting_pos_temp,[local_ee_pos_relative_to_base-resting_pos],axis=0)
                rel_pick_pos_ee_temp=np.append( rel_pick_pos_ee_temp,[relative_pick_pos_ee],axis=0)
                rel_place_pos_ee_temp=np.append( rel_place_pos_ee_temp,[relative_place_pos_ee],axis=0)
                rel_pick_pos_base_polar_temp=np.append( rel_pick_pos_base_polar_temp,[relative_pick_pos_base_polar],axis=0)
                rel_place_pos_base_polar_temp=np.append( rel_place_pos_base_polar_temp,[relative_place_pos_base_polar],axis=0)
                rel_pick_pos_base_temp=np.append( rel_pick_pos_base_temp,[relative_pick_pos_base],axis=0)
                rel_place_pos_base_temp=np.append( rel_place_pos_base_temp,[relative_place_pos_base],axis=0)
                is_holding_temp=np.append( is_holding_temp,[np.array([gripper_is_grasped])],axis=0)
                action_to_save_temp=np.concatenate( (action_to_save_temp, action_to_save ) , axis=0)




         #   obs, reward, done, info = env.step(step_action)
            current_step+=1

            current_episode_reward += reward
            metrics = self._extract_scalars_from_info(info)
            
            if show_viewer:
                frame = env.render(
                            "human",
                            info=metrics,
                            overlay_info=False,
                            show_info=True,
                        )
                        
                key = viewer.imshow(
                        frame[..., :3],delay=1,
                    )
               # input()

            if done or current_step==config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS:
                episode_stats = metrics.copy()
                episode_stats["return"] = current_episode_reward
                all_episode_stats.append(episode_stats)
                pbar.update()

                success_measure = self.config.RL.SUCCESS_MEASURE
                if success_measure in info:
                    episode_success = info[success_measure]
                    if not episode_success:
                        failure_episodes.append(env.current_episode.episode_id)
                else:
                    episode_success = -1

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames,
                        episode_id=env.current_episode.episode_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={"success": episode_success},
                        tb_writer=writer,
                        fps=30,
                    )

                if episode_success==1:
                    print("successful trajectory")
                    print("Episode ID == " , env.current_episode.episode_id)
                    episode_ids.append(env.current_episode.episode_id)
                    
                    print("robot_head_depth_temp shape == ", robot_head_depth_temp.shape)
                    print("robot_arm_depth_temp shape == ", robot_arm_depth_temp.shape)
                    print("rob_qpos_temp shape == ", rob_qpos_temp.shape)
                    print("rel_resting_pos_temp shape == ", rel_resting_pos_temp.shape)
                    print("rel_pick_pos_ee_temp shape == ", rel_pick_pos_ee_temp.shape)
                    print("rel_place_pos_ee_temp shape == ", rel_place_pos_ee_temp.shape)
                    print("rel_pick_pos_base_polar_temp shape == ", rel_pick_pos_base_polar_temp.shape)
                    print("rel_place_pos_base_polar_temp shape == ", rel_place_pos_base_polar_temp.shape)
                    print("rel_pick_pos_base_temp shape == ", rel_pick_pos_base_temp.shape)
                    print("rel_place_pos_base_temp shape == ", rel_place_pos_base_temp.shape)
                    print("is_holding_temp shape == ", is_holding_temp.shape)
                    print("action_to_save_temp shape == ", action_to_save_temp.shape)
                    
                    dataset_dict['{}'.format(num_suc_episodes)]={}
                    dataset_dict['{}'.format(num_suc_episodes)]['robot_head_depth']=robot_head_depth_temp
                    dataset_dict['{}'.format(num_suc_episodes)]['robot_arm_depth']=robot_arm_depth_temp
                    dataset_dict['{}'.format(num_suc_episodes)]['rob_qpos']=rob_qpos_temp
                    dataset_dict['{}'.format(num_suc_episodes)]['rel_resting_pos']=rel_resting_pos_temp
                    dataset_dict['{}'.format(num_suc_episodes)]['rel_pick_pos_ee']=rel_pick_pos_ee_temp
                    dataset_dict['{}'.format(num_suc_episodes)]['rel_place_pos_ee']=rel_place_pos_ee_temp
                    dataset_dict['{}'.format(num_suc_episodes)]['rel_pick_pos_base']=rel_pick_pos_base_temp
                    dataset_dict['{}'.format(num_suc_episodes)]['rel_place_pos_base']=rel_place_pos_base_temp
                    dataset_dict['{}'.format(num_suc_episodes)]['rel_pick_pos_base_polar']=rel_pick_pos_base_polar_temp
                    dataset_dict['{}'.format(num_suc_episodes)]['rel_place_pos_base_polar']=rel_place_pos_base_polar_temp
                    dataset_dict['{}'.format(num_suc_episodes)]['is_holding']=is_holding_temp
                    dataset_dict['{}'.format(num_suc_episodes)]['action_to_save']=action_to_save_temp
                   
                    robot_head_depth_temp=np.array([])
                    robot_arm_depth_temp=np.array([])
                    rob_qpos_temp=np.array([])
                    rel_resting_pos_temp=np.array([])
                    rob_qpos_temp=np.array([])
                    rel_pick_pos_ee_temp=np.array([])
                    rel_place_pos_ee_temp=np.array([])
                    rel_pick_pos_base_temp=np.array([])
                    rel_place_pos_base_temp=np.array([])
                    rel_pick_pos_base_polar_temp=np.array([])
                    rel_place_pos_base_polar_temp=np.array([])
                    is_holding_temp=np.array([])
                    action_to_save_temp=np.array([])

                    total_steps+=current_step

                    num_suc_episodes+=1
                    print("total number of episodes == " , number_of_episodes+1)
                    print("successful episodes == " , num_suc_episodes )
                    print("total number of steps == " , total_steps)
                    print("len(all_episode_stats) == " , len(all_episode_stats) )

                    if num_suc_episodes%50 == 49 and Save_data:
                        saves+=1
                    
                        with open('/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/nav_only_part_{}.pkl'.format(saves), 'wb') as f:
                            pickle.dump(dataset_dict, f)
                            dataset_dict={} 

                else:
      
                    robot_head_depth_temp=np.array([])
                    robot_arm_depth_temp=np.array([])
                    rob_qpos_temp=np.array([])
                    rel_resting_pos_temp=np.array([])
                    rob_qpos_temp=np.array([])
                    rel_pick_pos_ee_temp=np.array([])
                    rel_place_pos_ee_temp=np.array([])
                    rel_pick_pos_base_temp=np.array([])
                    rel_place_pos_base_temp=np.array([])
                    rel_pick_pos_base_polar_temp=np.array([])
                    rel_place_pos_base_polar_temp=np.array([])
                    is_holding_temp=np.array([])
                    action_to_save_temp=np.array([])


                obs = env.reset()
                current_step=0
                metrics = {}
                current_episode_reward = 0
                rgb_frames = []

            # Update policy inputs
            batch = batch_obs(
                [obs], device=self.device, cache=self._obs_batching_cache
            )
            not_done_masks = torch.tensor(
                [[not done]], dtype=torch.bool, device=self.device
            )
            buffer.update(
                recurrent_hidden_states=outputs_batch["rnn_hidden_states"],
                prev_actions=outputs_batch["action"],
                masks=not_done_masks,
            )

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
