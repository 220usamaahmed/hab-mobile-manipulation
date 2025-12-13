import argparse
import json
import os
import os.path as osp
import re

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



Transformer_policy=True

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
    '/home/shokry/hab-mobile-manipulation/rl_trained_encoder/visual_encoder_nav_17_sept.pth',
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
        self.encoder_position_embedding=SinusoidalPositionalEncoding(hidden_dim, max_len=11)  # Sensor position embedding


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
        visual_obs=visual_obs=visual_obs.repeat(shape[0],1,1,1,1)  # Repeat condition for batch size
        non_visual_obs=non_visual_obs.repeat(shape[0],1,1) 


        for t in reversed(range(self.timesteps)):
            # Create timestep tensor
            t_tensor = torch.tensor([t]).to(device).to(torch.float32)#torch.full((shape[0],), t, device=device, dtype=torch.long)
            noisy_action_decoder = self.p_sample(model,  noisy_action_decoder, t_tensor, visual_obs , non_visual_obs)
        return noisy_action_decoder










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

    diffusion_policy=ConditionalDiffusionModel()
    diffusion_policy.load_state_dict(torch.load("/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/weights/concatenated_non_visual_obs_more_data/accurate_data/model_5_prev_obs_pretrained_enc_small_data_2400.pt",
                                            map_location=device))
    diffusion_policy.to(device)
    diffusion_policy.eval()
    for p in diffusion_policy.parameters():
        p.requires_grad_(False)
    scheduler = NoiseScheduler()

    print("Loaded Diffusion Policy")


    # -------------------------------------------------------------------------- #
    # Main
    # -------------------------------------------------------------------------- #
    num_episodes = env.number_of_episodes
    # num_episodes = len(env.habitat_env.episode_iterator.episodes)
    if args.num_episodes is not None:
        num_episodes = args.num_episodes

    done, info = True, {}
    all_episode_stats = []
    episode_reward = 0
    failure_episodes = []

    if args.save_video is not None:
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
    rgb_frames = []

    if args.viewer:
        viewer = OpenCVViewer(config.TASK_CONFIG.TASK.TYPE)

    number_of_episodes=0

    for i_ep in range(num_episodes):
        ob = env.reset()
        initial_robot_pos = env.env._env._sim.robot.base_pos
        #policy.reset(ob)

        done = False
        rnn_hidden_states = None   # reset context at episode start
        prev_actions = None

        # On the first step, mask = 0 (episode just reset)
        masks = torch.tensor([1.0], device=device)

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


        obs_ep_transformer=[]
        actions_ep_transformer=[]
        rewards_ep_transformer=[]
        masks_ep_transformer=[]
        infos_ep_transformer=[]


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




            visual_obs_buffer.append(robot_head_depth)
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
                    shape = (10,20,10)
                    actions=scheduler.sample(diffusion_policy, shape, torch.from_numpy(visual_obs_buffer_np).to(device).to(torch.float32) , torch.from_numpy(non_visual_obs_buffer_np).to(device).to(torch.float32) , device, num_random_samples=20)
                    actions[:,:,0:2]*=3.0
                    similarity_vector , best_traj_index = cosine_similarity_matrix_torch(actions)
                    if relative_pick_pos_base_polar[0]<1 or relative_place_pos_base_polar[0]<1:
                        best_traj_index,gripped=imagine_trajectories(env , actions,gripper_is_grasped,similarity_vector, render=True, viewer=viewer)
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
            

            if args.viewer and key == "r":
                done = True
            if number_of_steps>1000 or success:
                print("success =", success)
                #print("Reached max steps")
                done=True
            if done:
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
