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




Feat_ext = SimpleCNN(1, (128, 128), 256).to(device).to(torch.float32)



#directory='/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data'
directory='/lustre/mlnvme/data/s47ashok_hpc-data/complete_trajs_datset_22_nov_tidy_house/more_data_with_rel_nav_pos_and_arm_depth'

Batch_size=64

def get_filenames_in_directory(directory):
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filenames.append(os.path.join(directory, filename))
    return filenames







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
    def __init__(self, action_dim=10, output_dim=10,sensor_dim=21,pos_dim=3, gripper_dim=1,qpos_dim=7, depth_features_dim=256, hidden_dim=256, num_layers=2):
        super().__init__()

        self.visual_feature_extractor = Feat_ext
        self.action_input_proj = nn.Linear(action_dim , hidden_dim)
        self.visual_obs_projection= nn.Linear(depth_features_dim, hidden_dim)
        self.rel_nav_pos_projection= nn.Linear(pos_dim, hidden_dim)
        self.qpos_projection= nn.Linear(qpos_dim, hidden_dim)
        self.rel_resting_pos_projection= nn.Linear(pos_dim, hidden_dim)
        self.gripper_state_projection= nn.Linear(gripper_dim, hidden_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),

        )
      #  self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear( hidden_dim,action_dim )
        
        self.decoder_position_embedding=SinusoidalPositionalEncoding(hidden_dim, max_len=21)  # Action position embedding
        self.encoder_position_embedding=SinusoidalPositionalEncoding(hidden_dim, max_len=26)  # Sensor position embedding


    def forward(self, visual_obs, rel_nav_pos, qpos ,rel_resting_pos ,gripper_state, noisy_action, t):

        batch_size=rel_nav_pos.shape[0]
        context_length=rel_nav_pos.shape[1]

        noisy_action=self.action_input_proj(noisy_action.to(torch.float32))  
     #   print("noisy action shape after projection == " , noisy_action.shape)
     #   print("initial visual observation shape == " , visual_obs.shape)
        if len(visual_obs.shape)==5:
            visual_obs=visual_obs.permute(0,1,4,2,3) 
            visual_obs=Feat_ext(rearrange(visual_obs, 'b s c h w -> (b s) c h w'))
        elif len(visual_obs.shape)==4:
            visual_obs=visual_obs.permute(0,3,1,2)
            visual_obs=Feat_ext(visual_obs)

      #  print("visual featurs shape after feature extraction == " , visual_obs.shape)
        visual_obs=visual_obs.reshape(batch_size*context_length, -1)  # Reshape to (batch_size * context_length, feature_dim)
        visual_obs=self.visual_obs_projection(visual_obs.to(torch.float32))  
       # print("shape after visual feature projection == " , visual_obs.shape )
        visual_obs=visual_obs.reshape(batch_size, context_length, -1)  
       # print("shape after reshaping back to batch and context length == " , visual_obs.shape )

        #print("initial non visual observations shape == " , non_visual_obs.shape)
        rel_nav_pos=rel_nav_pos.reshape(batch_size*context_length, -1)
        rel_nav_pos=self.rel_nav_pos_projection(rel_nav_pos.to(torch.float32))
        rel_nav_pos=rel_nav_pos.reshape(batch_size, context_length, -1)
      #  print("shape after rel nav pos projection == " , rel_nav_pos.shape )


        qpos=qpos.reshape(batch_size*context_length, -1)
        qpos=self.qpos_projection(qpos.to(torch.float32))
        qpos=qpos.reshape(batch_size, context_length, -1)
       # print("shape after qpos projection == " , qpos.shape )

        rel_resting_pos=rel_resting_pos.reshape(batch_size*context_length, -1)
        rel_resting_pos=self.rel_resting_pos_projection(rel_resting_pos.to(torch.float32))
        rel_resting_pos=rel_resting_pos.reshape(batch_size, context_length, -1)
        #print("shape after rel resting pos projection == " , rel_resting_pos.shape )

        #print("initial gripper state shape == " , gripper_state.shape)
        gripper_state=gripper_state.reshape(batch_size*context_length, -1)
        #print("shape after reshaping gripper state == " , gripper_state.shape )
        gripper_state=self.gripper_state_projection(gripper_state.to(torch.float32))
       # print("shape after gripper state projection == " , gripper_state.shape )
        gripper_state=gripper_state.reshape(batch_size, context_length, -1)
        #print("shape after gripper state projection == " , gripper_state.shape )



        t=self.time_mlp(t.to(torch.float32))  # Time embedding
        t = t.unsqueeze(1)
       # print("t shape after embedding == " , t.shape)

        encoder_input=torch.cat((visual_obs,rel_nav_pos,qpos,rel_resting_pos,gripper_state,t),dim=1)
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
    def __init__(self, timesteps=500, beta_start=1e-4, beta_end=0.02): #timesteps=500
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t=torch.reshape(t, (x_start.shape[0],-1))  # Ensure t is a 1D tensor
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().unsqueeze(1)
        sqrt_one_minus_alpha_cumprod = (1. - self.alpha_cumprod[t]).sqrt().unsqueeze(1)
        x_new=sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    def get_loss(self, model, x_start, t, visual_obs_batch , rel_nav_pos_batch, qpos_batch ,rel_resting_pos_batch ,gripper_state_batch): 
        noise = torch.randn_like(x_start).to(torch.float32)
        noisy_action = self.q_sample(x_start, t, noise)
        predicted_noise = model(visual_obs_batch.to(torch.float32), rel_nav_pos_batch.to(torch.float32), qpos_batch.to(torch.float32), 
                                rel_resting_pos_batch.to(torch.float32),gripper_state_batch.to(torch.float32), noisy_action.to(torch.float32), t.to(torch.float32))
        return F.mse_loss(predicted_noise, noise)



def process_episode_data(episode_data,num_prev_obs,num_predicted_actions):
    visual_obs=[]
    rel_nav_pos=[]
    qpos=[]
    rel_resting_pos=[]
    gripper_state=[]
    actions=[]

    number_of_steps=len(episode_data['action_to_save'])
    last_action=episode_data['action_to_save'][-1]
  #  print("last_action == " , last_action.shape)
    last_action_repeated=np.array([last_action for _ in range(num_predicted_actions)])
   # print("last_action_repeated shape == " , last_action_repeated.shape)
    episode_data['action_to_save']=np.concatenate( (episode_data['action_to_save'], last_action_repeated), axis=0)
    #print("New action_to_save shape == " , episode_data['action_to_save'].shape)
    for step in range(number_of_steps-num_prev_obs):
        vis_obs_step=[]
        for obs_idx in range(step,step+num_prev_obs):
            vis_obs_step.append(episode_data['robot_head_depth'][obs_idx])
        visual_obs.append(np.array(vis_obs_step))


        rel_nav_pos_step=[]
        for obs_idx in range(step,step+num_prev_obs):
            if episode_data['is_holding'][obs_idx]:
                rel_nav_pos_step.append(episode_data['rel_place_pos_base'][obs_idx])
            else:
                rel_nav_pos_step.append(episode_data['rel_pick_pos_base'][obs_idx])
        rel_nav_pos.append(np.array(rel_nav_pos_step))


        qpos_step=[]
        for obs_idx in range(step,step+num_prev_obs):
            qpos_step.append(episode_data['rob_qpos'][obs_idx])
        qpos.append(np.array(qpos_step))

        rel_resting_pos_step=[]
        for obs_idx in range(step,step+num_prev_obs):
            rel_resting_pos_step.append(episode_data['rel_resting_pos'][obs_idx])
        rel_resting_pos.append(np.array(rel_resting_pos_step))

        gripper_state_step=[]
        for obs_idx in range(step,step+num_prev_obs):
            gripper_state_step.append(episode_data['is_holding'][obs_idx])
        gripper_state.append(np.array(gripper_state_step))

        action_step=[]
        for act_idx in range(step+num_prev_obs,step+num_prev_obs+num_predicted_actions):
            action_step.append(episode_data['action_to_save'][act_idx])
        actions.append(np.array(action_step))

    return np.array(visual_obs), np.array(rel_nav_pos), np.array(qpos),np.array(rel_resting_pos),np.array(gripper_state) ,np.array(actions)





def train_diffusion_model(upload_directory,save_directory,num_prev_obs=5, num_predicted_actions=20):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    number_of_epochs=1000000
    
    print("device == " , device)

    file_names= get_filenames_in_directory(upload_directory)

    model = ConditionalDiffusionModel().to(device)
  #  model.load_state_dict(torch.load(os.path.join(save_directory, 'model_90.pt'))  )
    model.to(device)
    scheduler = NoiseScheduler()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ----------------------------
    # 5. TensorBoard Setup
    # ----------------------------
    log_dir = os.path.join(save_directory,"runs_diffusion_complete_trajs_cnn_encoder_scratch", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),"trained_from_scratch_cnn_encoder")
    writer = SummaryWriter(log_dir=log_dir)
    epoch_loss =0.0
    epoch_samples=0
    for epoch in range(number_of_epochs):
        current_file_idx=0
        for file_name in file_names:
            print("Processing file: ", file_name )
            with open(file_name, 'rb') as f:
                data = pickle.load(f)

            total_loss =0.0
            total_samples=0
            number_of_episodes_in_file=len(data)
        #    print("Processing file: ", file_name , " with number of episodes == " , number_of_episodes_in_file)
            for episode_key in data.keys():

                episode_data=data[episode_key]
                visual_obs_data , rel_nav_pos_data , qpos_data, rel_resting_pos_data , gripper_state_data ,action_data=process_episode_data(episode_data,num_prev_obs,num_predicted_actions)
                number_of_samples=action_data.shape[0]
                number_of_batches=math.ceil(number_of_samples/Batch_size)
                
                for batch in range(number_of_batches):
                    optimizer.zero_grad()
                    start_idx=batch*Batch_size
                    end_idx=start_idx+Batch_size
                    if end_idx>number_of_samples:
                        end_idx=number_of_samples
                    visual_obs_batch=torch.from_numpy(visual_obs_data[start_idx:end_idx]).to(device)
                    rel_nav_pos_batch=torch.from_numpy(rel_nav_pos_data[start_idx:end_idx]).to(device)
                    qpos_batch=torch.from_numpy(qpos_data[start_idx:end_idx]).to(device)
                    rel_resting_pos_batch=torch.from_numpy(rel_resting_pos_data[start_idx:end_idx]).to(device)
                    gripper_state_batch=torch.from_numpy(gripper_state_data[start_idx:end_idx]).to(device)
                    action_batch=torch.from_numpy(action_data[start_idx:end_idx]).to(device)
                    t = torch.randint(0, scheduler.timesteps, (visual_obs_batch.size(0),), device=device)
                    loss = scheduler.get_loss(model, action_batch, t, visual_obs_batch , rel_nav_pos_batch, qpos_batch ,rel_resting_pos_batch ,gripper_state_batch )*10
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.detach().item()
                    total_samples += number_of_samples
                    epoch_loss +=loss.detach().item()
                    epoch_samples+= number_of_samples
                
            current_file_idx+=1
            del visual_obs_data, rel_nav_pos_data , qpos_data, rel_resting_pos_data , gripper_state_data ,action_data
            gc.collect()
            torch.cuda.empty_cache()
            
            print("Epoch {} file [{}/{}], File: {}, Loss: {:.4f}".format(epoch+1 , current_file_idx, len(file_names), file_name, total_loss/total_samples))

        del data
        gc.collect()
        torch.cuda.empty_cache()
        train_loss = epoch_loss / epoch_samples
        print("Completed Epoch {}: Train Loss: {:.4f}".format(epoch+1, train_loss))
        writer.add_scalar('Loss/Train', train_loss, epoch)

        if epoch%10 ==0 :
         #   for name, param in model.named_parameters():
           #     if param.grad is not None:
            #        print(f"{name}: grad norm = {param.grad.norm().item()}")
          #  input()
            torch.save(model.state_dict(), os.path.join(save_directory, 'model_{}.pt'.format(epoch)) )



# ----------------------------
# 6. Train/Test Functions
# ----------------------------
def train(model, loader, optimizer, device,scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for label, target in loader:
        label, target = label.to(device).to(torch.float32), target.to(device).to(torch.float32)
        optimizer.zero_grad()
        t = torch.randint(0, scheduler.timesteps, (label.size(0),), device=device)
        loss = scheduler.get_loss(model, target, t, label)*10
        loss.backward()
        optimizer.step()

        total_loss += loss
        total += target.size(0)

    return (total_loss/total)/10 , correct  

def test(model, loader, device,scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for label, target in loader:
            label, target = label.to(device).to(torch.float32), target.to(device).to(torch.float32)
            t = torch.randint(0, scheduler.timesteps, (label.size(0),), device=device)
            loss = scheduler.get_loss(model, target, t, label)*10
            total_loss += loss

            total += target.size(0)

    return (total_loss/total)/10 , correct 




#upload_directory='/home/shokry/hab-mobile-manipulation/diffusion_dataset_new/accurate_data_5_aug/all_tasks_corrected_grasped_obs_26_aug'
upload_directory=directory

save_directory=directory+'/weights_diff_transformer_complete_trajs_cnn_encoder_scratch'
#save_directory='/home/shokry/hab-mobile-manipulation/diffusion_dataset_new/accurate_data_5_aug/all_tasks_corrected_grasped_obs_26_aug/weights_diff_transformer'

train_diffusion_model(upload_directory,save_directory)
