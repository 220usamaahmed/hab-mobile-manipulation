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

from torch.utils.data import TensorDataset, DataLoader, random_split

from datetime import datetime



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()
torch.cuda.empty_cache()







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
    def sample(self, model,  visual_obs , non_visual_obs , device, num_random_trajectroies=None, num_predicted_actions=None,action_dim=10):
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

        if num_random_trajectroies is None or num_predicted_actions is None:
            print("specify the number of generated trajectories and predicted actions")
            input()

        # Start from pure noise
        batch_size=visual_obs.shape[0]
        visual_obs=visual_obs.unsqueeze(1).repeat(1, num_random_trajectroies , 1, 1)
        visual_obs=visual_obs.reshape(batch_size* num_random_trajectroies , visual_obs.shape[2], visual_obs.shape[3])  # Reshape to (batch_size * num_random_trajectroies, depth_features_dim)
        non_visual_obs=non_visual_obs.unsqueeze(1).repeat(1, num_random_trajectroies , 1, 1)
        non_visual_obs=non_visual_obs.reshape(batch_size* num_random_trajectroies , non_visual_obs.shape[2], non_visual_obs.shape[3])  # Reshape to (batch_size * num_random_trajectroies, sensor_dim)
        
        noisy_action_decoder = torch.randn((batch_size,num_random_trajectroies,num_predicted_actions,action_dim), device=device)
        noisy_action_decoder=noisy_action_decoder.reshape(batch_size* num_random_trajectroies , num_predicted_actions , action_dim)  # Reshape to (batch_size * num_random_trajectroies, num_predicted_actions, action_dim)
        print("visual observation shape in the sampling fucntion == " , visual_obs.shape)
        print("non visual observation shape in the sampling fucntion == " , non_visual_obs.shape)
        print("noisy action decoder shape in the sampling fucntion == " , noisy_action_decoder.shape)
     #   print("non visual observation [0] == " , non_visual_obs[0])  
     #   print("non visual observation [{}] == ".format(num_random_trajectroies-1) , non_visual_obs[num_random_trajectroies-1]) 
     #   print("non visual observation [{}] == ".format(2*num_random_trajectroies-1) , non_visual_obs[2*num_random_trajectroies-1])
     #   print("non visual observation [{}] == ".format(3*num_random_trajectroies-1) , non_visual_obs[3*num_random_trajectroies-1])


        for t in reversed(range(self.timesteps)):
            # Create timestep tensor
            t_tensor = torch.tensor([t]).to(device).to(torch.float32)#torch.full((shape[0],), t, device=device, dtype=torch.long)
            noisy_action_decoder = self.p_sample(model,  noisy_action_decoder, t_tensor, visual_obs , non_visual_obs)
        return noisy_action_decoder
    

    '''
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
    '''







def predict_diffusion_actions(data_directory,diffusion_model_directory,skill='',num_random_trajectroies=20, num_predicted_actions=20,action_dim=10):  

    diffusion_policy=ConditionalDiffusionModel()
    diffusion_policy.load_state_dict(torch.load(diffusion_model_directory,
                                                map_location=device))
    diffusion_policy.to(device)
    diffusion_policy.eval()
    for p in diffusion_policy.parameters():
        p.requires_grad_(False)
    scheduler = NoiseScheduler()
    #shape = (10,20,10)
    predicted_actions=None
    batch_size=2000
    
    for i in [9]:
       # if skill == 'place' and i==9:
        #    break
        #next_visual_obs,next_non_visual_obs=upload_next_observation_data(data_directory,skill)
        next_visual_obs=torch.load(os.path.join(data_directory, 'next_visual_obs_data_{}_task_p_{}.pt'.format(skill,i))).to(device)
        next_non_visual_obs=torch.load(os.path.join(data_directory, 'next_non_visual_obs_data_{}_task_p_{}.pt'.format(skill,i))).to(device)
        print("uploaded next visual obs shape for {} task part {} == ".format(skill,i) , next_visual_obs.shape)
        print("uploaded next non visual obs shape for {} task part {} == ".format(skill,i) , next_non_visual_obs.shape)
        number_of_samples=next_visual_obs.shape[0]
        number_of_iterations=math.ceil(number_of_samples/batch_size)
        for s in range(number_of_iterations):
            print("starting step {} , time == {}".format(s,datetime.now().strftime("%H:%M:%S")))
            end_index=min((s+1)*batch_size,number_of_samples)
            print("start index == {} , end index == {}".format(s*batch_size,end_index))
            next_visual_obs_step=next_visual_obs[s*batch_size:end_index]
            next_non_visual_obs_step=next_non_visual_obs[s*batch_size:end_index]

            #predicted_actions_step= scheduler.sample( diffusion_policy, shape, next_visual_obs_step , next_non_visual_obs_step , device, num_random_samples=20)
            predicted_actions_step= scheduler.sample( diffusion_policy,  next_visual_obs_step , next_non_visual_obs_step , device, num_random_trajectroies=num_random_trajectroies, num_predicted_actions=num_predicted_actions,action_dim=action_dim)
            predicted_actions_step=predicted_actions_step.reshape(-1,num_random_trajectroies, num_predicted_actions, action_dim)

            print("predicted actions step shape == " , predicted_actions_step.shape)
            if predicted_actions is None:
                predicted_actions=predicted_actions_step#.unsqueeze(0)
            else:
                predicted_actions=torch.cat((predicted_actions,predicted_actions_step),dim=0)

            print("finished step {} out of {} for {} task part {} at time {}".format(s+1,number_of_iterations,skill,i,datetime.now().strftime("%H:%M:%S")))
        print("final predicted actions shape == " , predicted_actions.shape)
        #input()
        torch.save(predicted_actions, os.path.join(data_directory, 'predicted_actions_diffusion_{}_task_p_{}.pt'.format(skill,i)))
        if predicted_actions.shape[0] == next_non_visual_obs.shape[0]:
            print("Number of actions and observations match!")
        else:
            print("Number of actions and observations do not match!")
            input()
        predicted_actions=None  # Reset for the next part

        
        







if __name__ == "__main__":
    data_directory='/lustre/mlnvme/data/s47ashok_hpc-data/new_accurate_data_individual_tasks_4_jan_2026/processed_data'
   # data_directory='/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data'
    diffusion_model_directory='/lustre/mlnvme/data/s47ashok_hpc-data/new_accurate_data_individual_tasks_4_jan_2026/weights_pretrained_visual_encoder/pretrained_visual_encoder_2200.pt'
    #diffusion_model_directory='/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/marvin_weights/pretrained_visual_encoder_2190.pt'
    predict_diffusion_actions(data_directory,diffusion_model_directory,skill='place')
   





