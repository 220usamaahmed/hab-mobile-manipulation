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
from typing import Optional

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

from torch.utils.data import TensorDataset, DataLoader, random_split

from datetime import datetime



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()
torch.cuda.empty_cache()







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
        self.head_depth_projection= nn.Linear(depth_features_dim, hidden_dim)
        self.arm_depth_projection= nn.Linear(depth_features_dim, hidden_dim)
        self.non_visual_obs_projection= nn.Linear(sensor_dim, hidden_dim)

        self.head_depth_encoder = SimpleCNN(1, (128, 128), depth_features_dim)
        self.arm_depth_encoder = SimpleCNN(1, (128, 128), depth_features_dim)


        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),

        )
      #  self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear( hidden_dim,action_dim )
        
        self.decoder_position_embedding=SinusoidalPositionalEncoding(hidden_dim, max_len=21)  # Action position embedding
        self.encoder_position_embedding=SinusoidalPositionalEncoding(hidden_dim, max_len=16)  # Sensor position embedding


    def forward(self, head_depth_features, arm_depth_features , non_visual_obs, noisy_action, t):

        batch_size=non_visual_obs.shape[0]
        context_length=non_visual_obs.shape[1]
        


        noisy_action=self.action_input_proj(noisy_action.to(torch.float32))  



        head_depth_features=self.head_depth_projection(head_depth_features.to(torch.float32))
        arm_depth_features=self.arm_depth_projection(arm_depth_features.to(torch.float32))



       # print("shape after reshaping back to batch and context length == " , visual_obs.shape )

       # print("initial non visual observations shape == " , non_visual_obs.shape)
        non_visual_obs=self.non_visual_obs_projection(non_visual_obs.to(torch.float32))
      #  print("shape after reshaping back to batch and context length == " , non_visual_obs.shape )

        t=self.time_mlp(t.to(torch.float32))  # Time embedding
        t = t.unsqueeze(1)
        t=t.repeat(batch_size, 1, 1)  # Repeat to match action sequence length
     #   print("t shape after embedding == " , t.shape)

        encoder_input=torch.zeros((batch_size, (context_length*3)+1, non_visual_obs.shape[-1])).to(torch.float32).to(non_visual_obs.device)

        encoder_input[:,0:-2:3,:]=head_depth_features
        encoder_input[:,1::3,:]=arm_depth_features
        encoder_input[:,2::3,:]=non_visual_obs
        encoder_input[:, -1, :]=t.squeeze(1)           # print("encoder input shape == " , encoder_input.shape)
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








@torch.no_grad()
def sample_actions_flow_matching(
    model,
    
    head_depth_features: torch.Tensor,
    arm_depth_features: torch.Tensor,
    non_visual_obs: torch.Tensor,
    num_predicted_actions: int = 10,
    device: Optional[torch.device] = None,
    action_len: int = 20,
    action_dim: int = 10,
    num_steps: int = 50,
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
        device = head_depth_features.device
    batch_size=head_depth_features.shape[0]
    print("head depth features shape in flow matching sampler == " , head_depth_features.shape)
    head_depth_features = head_depth_features.to(device=device, dtype=torch.float32)
    arm_depth_features = arm_depth_features.to(device=device, dtype=torch.float32)
    head_depth_features=head_depth_features.unsqueeze(1).repeat(1,num_predicted_actions,1,1)  # Repeat condition for batch size

    arm_depth_features=arm_depth_features.unsqueeze(1).repeat(1,num_predicted_actions,1,1)
    non_visual_obs = non_visual_obs.to(device=device, dtype=torch.float32)
    non_visual_obs = non_visual_obs.unsqueeze(1).repeat(1,num_predicted_actions, 1,1)

    head_depth_features=head_depth_features.reshape(batch_size* num_predicted_actions , head_depth_features.shape[2], head_depth_features.shape[3]) 
    arm_depth_features=arm_depth_features.reshape(batch_size* num_predicted_actions , arm_depth_features.shape[2], arm_depth_features.shape[3])
    non_visual_obs=non_visual_obs.reshape(batch_size* num_predicted_actions , non_visual_obs.shape[2], non_visual_obs.shape[3])

    x = torch.randn((batch_size, num_predicted_actions, action_len, action_dim), device=device, dtype=torch.float32)
    x = x.reshape(batch_size* num_predicted_actions , action_len, action_dim)

    print("head depth features shape after reshaping in flow matching sampler == " , head_depth_features.shape)
    print("arm depth features shape after reshaping in flow matching sampler == " , arm_depth_features.shape)
    print("non visual obs shape after reshaping in flow matching sampler == " , non_visual_obs.shape)
    print("initial noise actions shape in flow matching sampler == " , x.shape)

    dt = 1.0 / num_steps
    for k in range(num_steps):
        t_k = k * dt
        t_k_tensor = torch.tensor(t_k, device=device, dtype=torch.float32)
        t_k_tensor = t_k_tensor.unsqueeze(0)


        #t_k_tensor = torch.full((num_predicted_actions,), t_k, device=device, dtype=torch.float32)
        #print("t k tensor == " , t_k_tensor)
       # t_k_tensor=t_k_tensor.unsqueeze(-1)

        #print("t_k tensor shape in flow matching sampler == " , t_k_tensor.shape)
        v_k = model(head_depth_features,arm_depth_features , non_visual_obs, x, t_k_tensor)

        if method.lower() == "euler":
            x = x + dt * v_k
        elif method.lower() == "heun":
            x_pred = x + dt * v_k
            t_k1 = (k + 1) * dt
            #t_k1_tensor = torch.full((num_predicted_actions,), t_k1, device=device, dtype=torch.float32)
            t_k1_tensor = torch.tensor( t_k1, device=device, dtype=torch.float32)
            t_k1_tensor = t_k1_tensor.unsqueeze(0)
            v_k1 = model(head_depth_features,arm_depth_features, non_visual_obs, x_pred, t_k1_tensor)
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

    def get_loss(self, model, x1, t, head_depth_batch,arm_depth_batch, non_visual_obs_batch):
        """
        Model now predicts velocity v at x_t (NOT noise epsilon).
        Signature matches the original diffusion scheduler for minimal code changes.
        """

        x_t, v, _ = self.sample_xt_and_v(x1, t)
        v_pred = model(
            head_depth_batch.to(torch.float32),
            arm_depth_batch.to(torch.float32),
            non_visual_obs_batch.to(torch.float32),
            x_t.to(torch.float32),
            t.to(torch.float32),
        )
        return F.mse_loss(v_pred, v)





def predict_flow_matching_actions(data_directory,diffusion_model_directory,skill='',num_random_trajectroies=20, num_predicted_actions=20,action_dim=10):  

    flow_matching_policy=ConditionalDiffusionModel()
    checkpoint=torch.load(diffusion_model_directory,
                                                map_location=device)
    flow_matching_policy.load_state_dict(checkpoint['model'])
    flow_matching_policy.to(device)
    flow_matching_policy.eval()
    for p in flow_matching_policy.parameters():
        p.requires_grad_(False)
    scheduler = FlowMatchingScheduler()
    #shape = (10,20,10)
    predicted_actions=None
    batch_size=2000
    
    for i in range(2,8):
       # if skill == 'place' and i==9:
        #    break
        #next_visual_obs,next_non_visual_obs=upload_next_observation_data(data_directory,skill)
        next_head_depth_obs=torch.load(os.path.join(data_directory, 'next_head_depth_obs_data_{}_task_p_{}.pt'.format(skill,i))).to(device)
        next_arm_depth_obs=torch.load(os.path.join(data_directory, 'next_arm_depth_obs_data_{}_task_p_{}.pt'.format(skill,i))).to(device)
        next_non_visual_obs=torch.load(os.path.join(data_directory, 'next_non_visual_obs_data_{}_task_p_{}.pt'.format(skill,i))).to(device)
        print("uploaded next head depth obs shape for {} task part {} == ".format(skill,i) , next_head_depth_obs.shape)
        print("uploaded next arm depth obs shape for {} task part {} == ".format(skill,i) , next_arm_depth_obs.shape)
        print("uploaded next non visual obs shape for {} task part {} == ".format(skill,i) , next_non_visual_obs.shape)
        number_of_samples=next_head_depth_obs.shape[0]
        number_of_iterations=math.ceil(number_of_samples/batch_size)
        for s in range(number_of_iterations):
            print("starting step {} , time == {}".format(s,datetime.now().strftime("%H:%M:%S")))
            end_index=min((s+1)*batch_size,number_of_samples)
            print("start index == {} , end index == {}".format(s*batch_size,end_index))
            next_head_depth_obs_step=next_head_depth_obs[s*batch_size:end_index]
            next_arm_depth_obs_step=next_arm_depth_obs[s*batch_size:end_index]
            next_non_visual_obs_step=next_non_visual_obs[s*batch_size:end_index]

            #predicted_actions_step= scheduler.sample( diffusion_policy, shape, next_visual_obs_step , next_non_visual_obs_step , device, num_random_samples=20)
            predicted_actions_step=  sample_actions_flow_matching(flow_matching_policy , next_head_depth_obs_step.to(device).to(torch.float32),  next_arm_depth_obs_step.to(device).to(torch.float32), next_non_visual_obs_step.to(device).to(torch.float32),num_random_trajectroies, device)
            predicted_actions_step=predicted_actions_step.reshape(-1,num_random_trajectroies, num_predicted_actions, action_dim)
            print("final predicted actions shape afte the sampling fn == " , predicted_actions_step.shape)
          #  input()

         #   print("predicted actions step shape == " , predicted_actions_step.shape)
            if predicted_actions is None:
                predicted_actions=predicted_actions_step#.unsqueeze(0)
            else:
                predicted_actions=torch.cat((predicted_actions,predicted_actions_step),dim=0)

            print("finished step {} out of {} for {} task part {} at time {}".format(s+1,number_of_iterations,skill,i,datetime.now().strftime("%H:%M:%S")))
        print("final predicted actions shape == " , predicted_actions.shape)
        #input()
        if predicted_actions.shape[0] == next_non_visual_obs.shape[0]:
            print("Number of actions and observations match!")
        else:
            print("Number of actions and observations do not match!")
            input()
        
        torch.save(predicted_actions, os.path.join(data_directory, 'predicted_actions_flow_matching_{}_task_p_{}.pt'.format(skill,i)))
        print("saved action predictions file")

        predicted_actions=None  # Reset for the next part

        
        







if __name__ == "__main__":
    data_directory='/lustre/mlnvme/data/s47ashok_hpc-data/final_dataset_27_jan/tidy_house/seed_100/diffusion_data/collected_dataset/processed_data_flow_matching_CNN_encoder_scratch_epoch_860'
   # data_directory='/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data'
    diffusion_model_directory='/lustre/mlnvme/data/s47ashok_hpc-data/final_dataset_27_jan/tidy_house/seed_100/diffusion_data/weights_flow_matching_CNN_encoder_scratch_head_and_arm_depth_without_action_extension/flow_matching_Cnn_encoder_scratch_without_act_ext_enhanced_sampling_860.pt'
    #diffusion_model_directory='/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/marvin_weights/pretrained_visual_encoder_2190.pt'


    required_skill='place_task'
    if required_skill!='nav_to_pick_pos' and required_skill!='nav_to_place_pos'and required_skill!='pick_task' and required_skill!='place_task' :
        print("please set a valid skill among nav, pick, place")
        input()
    #for required_skill in ['nav_to_pick_pos','nav_to_place_pos']:
    predict_flow_matching_actions(data_directory,diffusion_model_directory,skill=required_skill)
   





