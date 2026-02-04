import argparse
import json
import os
import os.path as osp
import re
from os import path
import magnum as mn
import numpy as np
import torch
from habitat import Config, logger
from habitat_baselines.utils.common import batch_obs, generate_video

import mobile_manipulation.methods.skills
from habitat_extensions.tasks.rearrange import RearrangeRLEnv
from habitat_extensions.tasks.rearrange.play import get_action_from_key
from habitat_extensions.utils.viewer import OpenCVViewer
from habitat_extensions.utils.visualizations.utils import put_info_on_image
from mobile_manipulation.config import get_config
from mobile_manipulation.methods.skill import CompositeSkill
from mobile_manipulation.utils.common import (
    extract_scalars_from_info,
    get_git_commit_id,
    get_run_name,
)
from mobile_manipulation.utils.wrappers import HabitatActionWrapperV1

from mobile_manipulation.transformer_policy.upload_policy_transformer import SkillTransformerPolicyLoader

import sys

from typing import Any, Dict
from habitat.tasks.utils import cartesian_to_polar
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

import numpy as np

from einops import rearrange

import math

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pickle

import random

from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import os

import gc

import importlib.util
import random

from torch.utils.data import IterableDataset

from dataclasses import dataclass
from typing import Optional



device = torch.device("cuda" if torch.cuda.is_available() else torch.device('cpu'))
#device =torch.device( "cpu")

gc.collect()
torch.cuda.empty_cache()
  

current_directory=os.getcwd()


 


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

   # print("simiarity matrix  == ", similarity_matrix)
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
    path.join( current_directory,'check_points/visual_encoder.pth') ,
    map_location='cpu',
    weights_only=True
))

Feat_ext.to(device)
Feat_ext.eval()




number_of_mc_dropout_samples=20

def best_traj_q_value(q_value_network, visual_obs, non_visual_obs_buffer, actions):

    visual_obs=torch.from_numpy(visual_obs).to(device)
    non_visual_obs_buffer=torch.from_numpy(non_visual_obs_buffer).to(device).to(torch.float32)
    if len(visual_obs.shape)==5:
        visual_obs=visual_obs.permute(0,1,4,2,3) 
        # arranged_visual_obs=rearrange(visual_obs, 'b s c h w -> (b s) c h w')

        visual_obs=Feat_ext(rearrange(visual_obs, 'b s c h w -> (b s) c h w'))

    elif len(visual_obs.shape)==4:
        visual_obs=visual_obs.permute(0,3,1,2)
        visual_obs=Feat_ext(visual_obs)

    num_trajectories=actions.shape[0]
    visual_obs=visual_obs.repeat(num_trajectories,1,1)
    non_visual_obs_buffer=non_visual_obs_buffer.repeat(num_trajectories,1,1)


  #  actions=torch.from_numpy(actions).to(device)
 #   print("visual obs buffer shape == " , visual_obs_buffer.shape)
  #  print("non visual obs buffer shape == " , non_visual_obs_buffer.shape)
  #  print("actions shape == " , actions.shape)
    q_values_concatenated=None
    with torch.no_grad():
        for t in range(number_of_mc_dropout_samples):
            q_values=q_value_network(visual_obs, non_visual_obs_buffer, actions)
            q_values=q_values.unsqueeze(-1)
            if q_values_concatenated is None:
                q_values_concatenated=q_values
            else:
                q_values_concatenated=torch.cat((q_values_concatenated,q_values),dim=1)
    q_values_mean=torch.mean(q_values_concatenated,dim=1,keepdim=True)
    q_values_std=torch.std(q_values_concatenated,dim=1,keepdim=True)
    q_values_mean_max=torch.max(q_values_mean,dim=0)
    q_values_mean_min=torch.min(q_values_mean,dim=0)
    q_values_means_nomralized=(q_values_mean - q_values_mean_min.values)/(q_values_mean_max.values - q_values_mean_min.values+1e-8)
    q_values_std_max=torch.max(q_values_std,dim=0)
    q_values_std_min=torch.min(q_values_std,dim=0)
    q_values_std_normalized=(q_values_std - q_values_std_min.values)/(q_values_std_max.values - q_values_std_min.values+1e-8)
    print("q values concatenated == " , q_values_concatenated)
    print("q values mean == " , q_values_mean)
    print("q values std == " , q_values_std)
    #input()
    weighted_q_values=q_values_means_nomralized * (1-q_values_std_normalized)
    print("q values means normalized == " , q_values_means_nomralized)
    print("q values std normalized == " , q_values_std_normalized)
    print("weighted q values == " , weighted_q_values)
    best_traj_idx=torch.argmax(weighted_q_values.squeeze(-1))
    print("best traj index == " , best_traj_idx)
    print("best q value == " , weighted_q_values[best_traj_idx])
    return best_traj_idx
    











@torch.no_grad()
def sample_actions_flow_matching(
    model,
    
    visual_obs: torch.Tensor,
    non_visual_obs: torch.Tensor,
    num_predicted_actions: int = 10,
    device: Optional[torch.device] = None,
    action_len: int = 20,
    action_dim: int = 10,
    num_steps: int = 20,
    method: str = "heun",
):
    """
    Flow Matching sampler (ODE integration):
        dx/dt = v_theta(x, t, cond),  t in [0, 1]
    Start:
        x(0) ~ N(0, I)
    Output:
        x(1)
    """
    model.eval()
    if device is None:
        device = visual_obs.device

    visual_obs = visual_obs.to(device=device, dtype=torch.float32)
    non_visual_obs = non_visual_obs.to(device=device, dtype=torch.float32)
    visual_obs = visual_obs.repeat(num_predicted_actions, 1,1)
    non_visual_obs = non_visual_obs.repeat(num_predicted_actions, 1,1)

    x = torch.randn((num_predicted_actions, action_len, action_dim), device=device, dtype=torch.float32)

    dt = 1.0 / num_steps
    for k in range(num_steps):
        t_k = k * dt
        t_k_tensor = torch.tensor(t_k, device=device, dtype=torch.float32)
        t_k_tensor = t_k_tensor.unsqueeze(0)


        #t_k_tensor = torch.full((num_predicted_actions,), t_k, device=device, dtype=torch.float32)
        #print("t k tensor == " , t_k_tensor)
       # t_k_tensor=t_k_tensor.unsqueeze(-1)

        #print("t_k tensor shape in flow matching sampler == " , t_k_tensor.shape)
        v_k = model(visual_obs, non_visual_obs, x, t_k_tensor)

        if method.lower() == "euler":
            x = x + dt * v_k
        elif method.lower() == "heun":
            x_pred = x + dt * v_k
            t_k1 = (k + 1) * dt
            #t_k1_tensor = torch.full((num_predicted_actions,), t_k1, device=device, dtype=torch.float32)
            t_k1_tensor = torch.tensor( t_k1, device=device, dtype=torch.float32)
            t_k1_tensor = t_k1_tensor.unsqueeze(0)
            v_k1 = model(visual_obs, non_visual_obs, x_pred, t_k1_tensor)
            x = x + 0.5 * dt * (v_k + v_k1)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'euler' or 'heun'.")

    return x





class FlowMatchingScheduler:
    """
    Rectified Flow / Flow Matching training objective.

    We define a path from noise x0 ~ N(0,I) to data x1 (target actions):
        x_t = (1 - t) * x0 + t * x1,   where t in [0,1]

    The target velocity field along this path is:
        v* = d x_t / dt = x1 - x0

    The model is trained to predict v*(x_t, t, cond) via MSE loss.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps  # not strictly needed; kept for potential numerical guards

    def sample_xt_and_v(self, x1: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x1: target actions, shape (B, L, D) or (B, D)
            t: continuous times in [0,1], shape (B,)

        Returns:
            x_t: interpolated actions at time t, same shape as x1
            v:   target velocity, same shape as x1
            x0:  sampled noise start, same shape as x1
        """

        x0 = torch.randn_like(x1)

        # reshape t for broadcasting over x1
        while t.dim() < x1.dim():
            t = t.unsqueeze(-1)

        
        x_t = (1.0 - t) * x0 + t * x1
        
        v = x1 - x0

        
        return x_t, v, x0

    def get_loss(self, model, x1, t, visual_obs_batch, non_visual_obs_batch):
        """
        Model now predicts velocity v at x_t (NOT noise epsilon).
        Signature matches the original diffusion scheduler for minimal code changes.
        """

        x_t, v, _ = self.sample_xt_and_v(x1, t)
        v_pred = model(
            visual_obs_batch.to(torch.float32),
            non_visual_obs_batch.to(torch.float32),
            x_t.to(torch.float32),
            t.to(torch.float32),
        )
        return F.mse_loss(v_pred, v)









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

       # print("initial visual observation shape == " , visual_obs.shape)
        if len(visual_obs.shape)==5:
            visual_obs=visual_obs.permute(0,1,4,2,3) 
           # arranged_visual_obs=rearrange(visual_obs, 'b s c h w -> (b s) c h w')

            visual_obs=Feat_ext(rearrange(visual_obs, 'b s c h w -> (b s) c h w'))

        elif len(visual_obs.shape)==4:
            visual_obs=visual_obs.permute(0,3,1,2)
            visual_obs=Feat_ext(visual_obs)

        visual_obs=visual_obs.reshape(batch_size*context_length, -1)  # Reshape to (batch_size * context_length, feature_dim)



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
        if len(visual_obs.shape)==4:
            visual_obs=visual_obs.unsqueeze(0)  # Add sequence dimension if missing
        visual_obs=visual_obs.repeat(shape[0],1,1,1,1)  # Repeat condition for batch size
        non_visual_obs=non_visual_obs.repeat(shape[0],1,1) 


        for t in reversed(range(self.timesteps)):
            # Create timestep tensor
            t_tensor = torch.tensor([t]).to(device).to(torch.float32)#torch.full((shape[0],), t, device=device, dtype=torch.long)
            noisy_action_decoder = self.p_sample(model,  noisy_action_decoder, t_tensor, visual_obs , non_visual_obs)
        return noisy_action_decoder









@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    d_vis: int = 512
    d_nonvis: int = 21
    d_act: int = 10

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.2

    hist_len: int = 5
    horizon: int = 20

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_epochs: int = 50

    gamma: float = 0.99
    num_action_samples: int = 20
    target_ema_tau: float = 0.005

    log_every: int = 50
    ckpt_every_steps: int = 100
    out_dir: str = "./q_training_runs/run_small_dataset"


class QTransformer(nn.Module):
    def __init__(
        self,
        d_vis: int,
        d_nonvis: int,
        d_act: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        hist_len: int = 5,
        horizon: int = 20,
    ):
        super().__init__()
        self.hist_len = hist_len
        self.horizon = horizon
        self.num_tokens = 2 * hist_len + horizon

        self.vis_proj = nn.Linear(d_vis, d_model)
        self.nonvis_proj = nn.Linear(d_nonvis, d_model)
        self.act_proj = nn.Linear(d_act, d_model)

       # self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        #nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.pos_emb=SinusoidalPositionalEncoding(d_model, max_len=self.num_tokens)
      #  print("d_model in Q transformer == ", d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            #norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.q_head = nn.Sequential(
        #    nn.LayerNorm(d_model),
            nn.Linear(d_model, int(d_model/4)),
            nn.GELU(),
            nn.Linear(int(d_model/4), 1),
        )

    def forward(self, vis_hist: torch.Tensor, nonvis_hist: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        B = vis_hist.shape[0]
    #    print("Visual history shape in Q transformer == ", vis_hist.shape)
     #   print("Non visual history shape in Q transformer == ", nonvis_hist.shape)
        vis_tok = self.vis_proj(vis_hist)
        nonvis_tok = self.nonvis_proj(nonvis_hist)
      #  print("Projected visual history shape in Q transformer == ", vis_tok.shape)
      #  print("Projected non visual history shape in Q transformer == ", nonvis_tok.shape)

        state_tokens = torch.stack([vis_tok, nonvis_tok], dim=2)
       # print("Stacked state tokens shape in Q transformer == ", state_tokens.shape)
        state_tokens = state_tokens.view(B, 2 * self.hist_len, -1)
       # print("Reshaped state tokens shape in Q transformer == ", state_tokens.shape)

       # print("Action sequence shape in Q transformer == ", act_seq.shape)
        act_tokens = self.act_proj(act_seq)
       # print("Projected action sequence shape in Q transformer == ", act_tokens.shape)

        x = torch.cat([state_tokens, act_tokens], dim=1)
       # print("Transformer input sequence shape in Q transformer == ", x.shape)
        x=self.pos_emb(x)
        #x = x + self.pos_emb
      #  x = self.dropout(x)

        h = self.encoder(x)
     #   print("original output of the q transformer == " , h.shape)
        pooled = h.mean(dim=1)
      #  print("pooled output of the q transformer == " , pooled.shape)
        q = self.q_head(pooled).squeeze(-1)
      #  print("final Q values shape in Q transformer == " , q.shape)
       # input()
        return q








def preprocess_config(config_path: str, config: Config):
    config.defrost()

    fileName = osp.splitext(osp.basename(config_path))[0]
    runName = get_run_name()
    substitutes = dict(fileName=fileName, runName=runName)

    config.PREFIX = config.PREFIX.format(**substitutes)
    config.BASE_RUN_DIR = config.BASE_RUN_DIR.format(**substitutes)

    for key in ["LOG_FILE", "VIDEO_DIR"]:
        config[key] = config[key].format(
            prefix=config.PREFIX, baseRunDir=config.BASE_RUN_DIR, **substitutes
        )


def update_ckpt_path(config: Config, seed: int):
    config.defrost()
    for k in config:
        if k == "CKPT_PATH":
            ckpt_path = config[k]
            new_ckpt_path = re.sub(r"seed=[0-9]+", f"seed={seed}", ckpt_path)
            print(f"Update {ckpt_path} to {new_ckpt_path}")
            config[k] = new_ckpt_path
        elif isinstance(config[k], Config):
            update_ckpt_path(config[k], seed)
    config.freeze()


def update_sensor_resolution(config: Config, height, width):
    config.defrost()
    sensor_names = [
        "THIRD_RGB_SENSOR",
        "RGB_SENSOR",
        "DEPTH_SENSOR",
        "SEMANTIC_SENSOR",
    ]
    for name in sensor_names:
        sensor_cfg = config.TASK_CONFIG.SIMULATOR[name]
        sensor_cfg.HEIGHT = height
        sensor_cfg.WIDTH = width
        print(f"Update {name} resolution")
    config.freeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="config_path", type=str, required=True)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    # Episodes
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="whether to shuffle test episodes",
    )
    parser.add_argument(
        "--num-episodes", type=int, help="number of episodes to evaluate"
    )
    parser.add_argument(
        "--episode-ids", type=str, help="episodes ids to evaluate"
    )

    # Save
    parser.add_argument("--save-video", choices=["all", "failure"])
    parser.add_argument("--save-log", action="store_true")

    # Viewer
    parser.add_argument(
        "--viewer", action="store_true", help="enable OpenCV viewer"
    )
    parser.add_argument("--viewer-delay", type=int, default=10)
    parser.add_argument(
        "--play", action="store_true", help="enable input control"
    )

    # Policy
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-seed", type=int)

    # Rendering
    parser.add_argument("--render-mode", type=str, default="human")
    parser.add_argument("--render-info", action="store_true")
    parser.add_argument(
        "--no-rgb", action="store_true", help="disable rgb observations"
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="use high resolution for visualization",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------------- #
    # Configure
    # ---------------------------------------------------------------------------- #
    config = get_config(args.config_path, opts=args.opts)
    preprocess_config(args.config_path, config)
    torch.set_num_threads(1)

    config.defrost()
    if args.split is not None:
        config.TASK_CONFIG.DATASET.SPLIT = args.split
    if not args.shuffle:
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = False
    if args.no_rgb:
        sensors = config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = [
            x for x in sensors if "RGB" not in x
        ]
    config.freeze()

    if args.train_seed is not None:
        update_ckpt_path(config, seed=args.train_seed)

    if args.high_res:
        update_sensor_resolution(config, height=720, width=1080)

    if args.save_log:
        if config.LOG_FILE:
            log_dir = os.path.dirname(config.LOG_FILE)
            os.makedirs(log_dir, exist_ok=True)
            logger.add_filehandler(config.LOG_FILE)
        logger.info(config)
        logger.info("commit id: {}".format(get_git_commit_id()))

    # For reproducibility, just skip other episodes
    if args.episode_ids is not None:
        eval_episode_ids = eval(args.episode_ids)
        eval_episode_ids = [str(x) for x in eval_episode_ids]
    else:
        eval_episode_ids = None

    # ---------------------------------------------------------------------------- #
    # Initialize env
    # ---------------------------------------------------------------------------- #
    env = RearrangeRLEnv(config)
    env = HabitatActionWrapperV1(env)
    env.seed(config.TASK_CONFIG.SEED)


    # -------------------------------------------------------------------------- #
    # Initialize policy
    # -------------------------------------------------------------------------- #


    num_prev_obs=5
    num_predicted_acts=20

    flow_matching_policy=ConditionalDiffusionModel()
    flow_matching_policy.load_state_dict(torch.load(
                                           # path.join( current_directory,"check_points/diffusion_model_epoch_2200.pt"),
                                            path.join( current_directory,"check_points/flow_matching_all_tasks_epoch_530.pt"),
                                            map_location=device))
    flow_matching_policy.to(device)
    flow_matching_policy.eval()
    for p in flow_matching_policy.parameters():
        p.requires_grad_(False)
    scheduler = FlowMatchingScheduler()
    print("Loaded flow matching Policy")


    cfg = TrainConfig()
    q_value_network=  QTransformer(
        d_vis=cfg.d_vis,
        d_nonvis=cfg.d_nonvis,
        d_act=cfg.d_act,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        hist_len=cfg.hist_len,
        horizon=cfg.horizon,
    ).to(cfg.device)

    q_network_ckpt = torch.load(path.join( current_directory,'check_points/ckpt_with_normalization_and_uncertainty_aware_all_data_step_600.pt'))
    q_value_network.load_state_dict(q_network_ckpt["q_state_dict"])
    q_value_network.to(device)
   # q_value_network.eval()

    print("Loaded Q transformer model")

    


    # -------------------------------------------------------------------------- #
    # Main
    # -------------------------------------------------------------------------- #
    num_episodes = env.number_of_episodes
    # num_episodes = len(env.habitat_env.episode_iterator.episodes)
    if args.num_episodes is not None:
        num_episodes = args.num_episodes

    #done, info = True, {}
    done, info = False, {}
    all_episode_stats = []
    episode_reward = 0
    failure_episodes = []

    if args.save_video is not None:
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
    rgb_frames = []

    if args.viewer:
        viewer = OpenCVViewer(config.TASK_CONFIG.TASK.TYPE)

    number_of_episodes=0
    number_of_successful_episodes=0

    for i_ep in range(num_episodes):
        ob = env.reset()
        print("initial robot pos == " , env.env._env._sim.robot.base_pos)

        
        initial_robot_pos = env.env._env._sim.robot.base_pos
        #policy.reset(ob)

        done = False


        episode_reward = 0.0
        info = {}
        rgb_frames = []
        episode_id = env.current_episode.episode_id
        scene_id = env.current_episode.scene_id

        

        
        #print("current episode == " , env.current_episode.target_receptacles[0][1])


        # Skip episode and keep reproducibility
        if eval_episode_ids is not None and episode_id not in eval_episode_ids:
            print("Skip episode", episode_id)
            continue



        number_of_steps=0
        visual_obs_buffer=[]
        non_visual_obs_buffer=[]
        while True:


            # -------------------------------------------------------------------------- #
            # Visualization
            # -------------------------------------------------------------------------- #
            if args.viewer or args.save_video:
                # Add additional info
              #  info["values"] = step_action.get("values")
               # info["value"] = step_action.get("value")
               # info["success_probs"] = step_action.get("success_probs")

                metrics = extract_scalars_from_info(info)
                if args.render_mode == "human":
                    frame = env.render(
                        "human",
                        info=metrics,
                        overlay_info=False,
                        show_info=args.render_info,
                    )
                else:
                    frame = env.render(args.render_mode)
                    if args.render_info:
                        frame = put_info_on_image(
                            frame, info=metrics, overlay=False
                        )
                rgb_frames.append(frame)

            if args.viewer:
                key = viewer.imshow(
                    frame[..., :3], delay=0 if args.play else args.viewer_delay
                )

            if args.play:
                play_action = get_action_from_key(key, "BaseArmGripperAction")
                if play_action is not None:
                    step_action = play_action
            # -------------------------------------------------------------------------- #




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

         #   print("current episode == " , env.current_episode)
         #   print("pick goal == " , pick_goal)
          #  print("place goal == " , place_goal)
          #  input()

           # env.env._env._sim.add_viz_obj(pick_goal)
            #env.env._env._sim.add_viz_obj(place_goal)
            

          #  print("robot pos == " , robot_base_pos)
          #  print("qpos == " , robot_qpos)
          #  print("pick goal == " , pick_goal)
          #  print("place goal == " , place_goal)
          #  print("receptacle == " , env.current_episode.target_receptacles[0][1])





            ## rgb and depth images has the format (W,H,C) where C is the number of channels
            ## You will need to use permute function to change it to (C,W,H) to be compatible with pytorch
         #   robot_head_rgb = ob['robot_head_rgb']
          #  robot_arm_rgb = ob['robot_arm_rgb']
            robot_head_depth = ob['robot_head_depth']
           # robot_arm_depth = ob['robot_arm_depth']

           # print("robot_head_rgb shape == " , robot_head_rgb.shape)
           # print("robot_arm_rgb shape == " , robot_arm_rgb.shape)
           # print("robot_head_depth shape == " , robot_head_depth.shape)
           # print("robot_arm_depth shape == " , robot_arm_depth.shape)

            ## receptacle number is important to calculate the loss of the auxilary head 
            ## in the planner (high-level) transformer
            
            receptacle_number= env.current_episode.target_receptacles[0][1]


          #  print("receptacle_number == " , receptacle_number)



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

            robot_head_depth_features=Feat_ext( torch.from_numpy(robot_head_depth).unsqueeze(0).permute(0,3,1,2).to(device))
            visual_obs_buffer.append(robot_head_depth_features.squeeze(0).cpu().detach().numpy())


          #  visual_obs_buffer.append(robot_head_depth)
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


            if number_of_steps==0 or number_of_steps % 10 ==0:
                with torch.no_grad():
                    actions=sample_actions_flow_matching(flow_matching_policy , torch.from_numpy(visual_obs_buffer_np).to(device).to(torch.float32) , torch.from_numpy(non_visual_obs_buffer_np).to(device).to(torch.float32) ,10, device)
                    actions[:,:,0:2]*=3.0

                    similarity_vector , best_traj_index = cosine_similarity_matrix_torch(actions)
                    if relative_pick_pos_base_polar[0]<1 or relative_place_pos_base_polar[0]<1:
                        best_traj_index,gripped=imagine_trajectories(env , actions,gripper_is_grasped,similarity_vector, render=True, viewer=viewer)
                    best_traj_index_q_values=best_traj_q_value(q_value_network, visual_obs_buffer_np, non_visual_obs_buffer_np, actions)
                    estimated_action_trajs=actions[best_traj_index]

            action=estimated_action_trajs[number_of_steps%10 ].detach().cpu().numpy()
            action[0:2]=np.clip(action[0:2],-3,3)  # Clip base actions
            action[2:9]=np.clip(action[2:9],-1,1)  # Clip arm actions
            action[9]=np.clip(action[9],-1,1)  # Clip gripper action 

            base_action=np.clip(action[0:2],-3,3)
          #  print("base action == " , base_action)
            arm_action=np.clip(action[2:9],-1,1)
           # print("arm action == " , arm_action)
            gripper_action=np.clip(action[9],-1,1)


            step_action={'action': 'BaseArmGripperAction2', 'action_args': {'base_action': (base_action) , 'arm_action':(arm_action) , 'gripper_action':gripper_action }, 'value': 2.9779255390167236}
            
            ob, reward, done, info = env.step(step_action)
            episode_reward += reward

           # print("step number == " , number_of_steps)
            number_of_steps+=1

            metrics = extract_scalars_from_info(info)
            success = metrics.get(config.RL.SUCCESS_MEASURE, -1)

            
            if number_of_steps%10==0:
                print("reset episode ?")
                x=input()
                if x=='y':
                    print("reseting epsidoe")
                    done=True
            



            if args.viewer and key == "r":
                done = True
            if number_of_steps>2000 or success:
                if success:
                    number_of_successful_episodes+=1
                    print("successful episode =")
                else:
                    print("Maximum steps reached, failed episode")
                #print("Reached max steps")
                done=True
            if done:
                print("total number of episodes == " , i_ep+1 , " total number of successful episodes == " , number_of_successful_episodes)
                print("sucess percentage == " , (number_of_successful_episodes/(i_ep+1))*100 )
                if not success:
                    print("Failed episode")
                break

        gc.collect()
        torch.cuda.empty_cache()
  

        # -------------------------------------------------------------------------- #
        # Update stats
        # -------------------------------------------------------------------------- #
        metrics = extract_scalars_from_info(info)
        episode_stats = metrics.copy()
        episode_stats["return"] = episode_reward
        all_episode_stats.append(episode_stats)

        logger.info(
            "Episode {} ({}/{}): {}".format(
                episode_id, i_ep, num_episodes, episode_stats
            )
        )

        success = metrics.get(config.RL.SUCCESS_MEASURE, -1)
        is_failure = success == False
        print("success == " , success)

       #     torch.save(current_episode_dict ,  f"/home/shokry/hab-mobile-manipulation/collected_dataset_transformer/successful_episode_{episode_id}_scene_{scene_id}_traj_num_{number_of_episodes}.pt" )
        #    input()
        if args.save_video == "all" or (
            args.save_video == "failure" and is_failure
        ):
            generate_video(
                video_option=["disk"],
                video_dir=config.VIDEO_DIR,
                images=rgb_frames,
                episode_id=episode_id,
                checkpoint_idx=-1,
                metrics={"success": success},
                fps=30,
                tb_writer=None,
            )

        if is_failure:
            failure_episodes.append(episode_id)

        if eval_episode_ids is not None:
            if len(all_episode_stats) >= len(eval_episode_ids):
                print("Completed")
                break

        number_of_episodes+=1

    env.close()

    # logging metrics
    aggregated_stats = {
        k: np.mean([ep_info[k] for ep_info in all_episode_stats])
        for k in all_episode_stats[0].keys()
    }
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.4f}")

    failure_episodes = sorted(failure_episodes)
    failure_episodes_str = ",".join(map(str, failure_episodes))
    logger.info("Failure episodes:\n{}".format(failure_episodes_str))

    if args.save_log:
        json_path = config.LOG_FILE.replace("log.txt", "result.json")
        with open(json_path, "w") as f:
            json.dump(all_episode_stats, f, indent=2)


if __name__ == "__main__":
    main()
