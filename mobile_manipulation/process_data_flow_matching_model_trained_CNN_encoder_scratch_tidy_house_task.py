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


    def forward(self, head_depth_images, arm_depth_images , non_visual_obs, noisy_action, t):

        batch_size=non_visual_obs.shape[0]
        context_length=non_visual_obs.shape[1]
        


        noisy_action=self.action_input_proj(noisy_action.to(torch.float32))  

       # print("initial visual observation shape == " , visual_obs.shape)
        if len(head_depth_images.shape)==5:
            head_depth_images=head_depth_images.permute(0,1,4,2,3) 
            arm_depth_images=arm_depth_images.permute(0,1,4,2,3)
       #     visual_obs=Feat_ext(rearrange(visual_obs, 'b s c h w -> (b s) c h w'))
        elif len(head_depth_images.shape)==4:
            head_depth_images=head_depth_images.permute(0,3,1,2)
            arm_depth_images=arm_depth_images.permute(0,3,1,2)
         #   visual_obs=Feat_ext(visual_obs)

        head_depth_images=head_depth_images.reshape(batch_size*context_length, head_depth_images.shape[-3], head_depth_images.shape[-2], head_depth_images.shape[-1])
        arm_depth_images=arm_depth_images.reshape(batch_size*context_length, arm_depth_images.shape[-3], arm_depth_images.shape[-2], arm_depth_images.shape[-1])
        head_depth_features=self.head_depth_encoder(head_depth_images.to(torch.float32))
        arm_depth_features=self.arm_depth_encoder(arm_depth_images.to(torch.float32))
        head_depth_features=head_depth_features.reshape(batch_size, context_length, -1)
        arm_depth_features=arm_depth_features.reshape(batch_size, context_length, -1)


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







skill='pick_task'


#directory='/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/complete_rearrange_trajs'
#directory='/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/seed_100'
directory = '/lustre/mlnvme/data/s47ashok_hpc-data/final_dataset_27_jan/tidy_house/seed_100/diffusion_data/collected_dataset'

flow_matching_model_directory='/lustre/mlnvme/data/s47ashok_hpc-data/final_dataset_27_jan/tidy_house/seed_100/diffusion_data/weights_flow_matching_CNN_encoder_scratch_head_and_arm_depth_without_action_extension'
checkpoint= torch.load( os.path.join(flow_matching_model_directory, 'flow_matching_Cnn_encoder_scratch_without_act_ext_enhanced_sampling_860.pt'),map_location=device )
flow_matching_model=ConditionalDiffusionModel()
flow_matching_model.load_state_dict(checkpoint['model'])
flow_matching_model.to(device)
flow_matching_model.eval()

head_depth_encoder=flow_matching_model.head_depth_encoder
arm_depth_encoder=flow_matching_model.arm_depth_encoder




def get_filenames_in_directory(directory):
    filenames = []
    for filename in os.listdir(directory) :
       # if filename.endswith('.pkl') and '{}_only_eval_dataset_ep_0'.format(skill) in filename:
        filenames.append(os.path.join(directory, filename))
    return filenames







def process_episode_data(episode_data,num_prev_obs,num_predicted_actions):

    sparse_reward=100
    head_depth_obs=[]
    arm_depth_obs=[]
    non_visual_obs=[]
    actions=[]
    rewards=[]
    next_head_depth_obs=[]
    next_arm_depth_obs=[]
    next_non_visual_obs=[]
    done=[]

    number_of_steps=len(episode_data['action_to_save'])
    head_depth_features=torch.zeros( (number_of_steps, 512), dtype=torch.float32).to(device)
    arm_depth_features=torch.zeros( (number_of_steps, 512), dtype=torch.float32).to(device)
    for i in range(number_of_steps):
        head_depth_features[i]=head_depth_encoder( torch.from_numpy( episode_data['robot_head_depth'][i].astype(np.float32) ).unsqueeze(0).permute(0,3,1,2).to(device))
        arm_depth_features[i]=arm_depth_encoder( torch.from_numpy( episode_data['robot_arm_depth'][i].astype(np.float32) ).unsqueeze(0).permute(0,3,1,2).to(device))
    head_depth_features=head_depth_features.cpu().detach().numpy()
    arm_depth_features=arm_depth_features.cpu().detach().numpy()



    last_action=episode_data['action_to_save'][-1]
  #  print("last_action == " , last_action.shape)
    last_action_repeated=np.array([last_action for _ in range(num_predicted_actions)])
   # print("last_action_repeated shape == " , last_action_repeated.shape)
    episode_data['action_to_save']=np.concatenate( (episode_data['action_to_save'], last_action_repeated), axis=0)
    #print("New action_to_save shape == " , episode_data['action_to_save'].shape)
    for step in range(number_of_steps-num_prev_obs-num_predicted_actions+2):

        head_depth_obs_step=[]
        arm_depth_obs_step=[]
        next_arm_depth_obs_step=[]
        next_head_depth_obs_step=[]
        if step==0:
            for obs_idx in range(step,step+num_prev_obs):
                head_depth_obs_step.append(head_depth_features[0])
                arm_depth_obs_step.append(arm_depth_features[0])
                next_head_depth_obs_step.append(head_depth_features[obs_idx+15])
                next_arm_depth_obs_step.append(arm_depth_features[obs_idx+15])
           # print("vis_obs_step shape at step 0 == " , np.array(vis_obs_step).shape)
            head_depth_obs.append(np.array(head_depth_obs_step))
            arm_depth_obs.append(np.array(arm_depth_obs_step))
            next_head_depth_obs.append(np.array(next_head_depth_obs_step))
            next_arm_depth_obs.append(np.array(next_arm_depth_obs_step))
            head_depth_obs_step=[]
            arm_depth_obs_step=[]
            next_head_depth_obs_step=[]
            next_arm_depth_obs_step=[]

        for obs_idx in range(step,step+num_prev_obs):
            head_depth_obs_step.append(head_depth_features[obs_idx])
            arm_depth_obs_step.append(arm_depth_features[obs_idx])
            next_head_depth_obs_step.append(head_depth_features[obs_idx+20-1])
            next_arm_depth_obs_step.append(arm_depth_features[obs_idx+20-1])
        head_depth_obs.append(np.array(head_depth_obs_step))
        arm_depth_obs.append(np.array(arm_depth_obs_step))
        next_head_depth_obs.append(np.array(next_head_depth_obs_step))
        next_arm_depth_obs.append(np.array(next_arm_depth_obs_step))
        
        non_vis_obs_step=[]
        next_non_vis_obs_step=[]
        if step==0:
            for obs_idx in range(step,step+num_prev_obs):

                non_vis_obs_step.append( np.concatenate((
                    episode_data['rel_resting_pos'][0],
                    episode_data['rel_pick_pos_ee'][0],
                    episode_data['rel_place_pos_ee'][0],
                    episode_data['rel_pick_pos_base_polar'][0],
                    episode_data['rel_place_pos_base_polar'][0],
                    episode_data['rob_qpos'][0],
                    np.array([int(episode_data['is_holding'][0])]),
                ),axis=-1) )
                next_non_vis_obs_step.append( np.concatenate((
                    episode_data['rel_resting_pos'][obs_idx+15],
                    episode_data['rel_pick_pos_ee'][obs_idx+15],
                    episode_data['rel_place_pos_ee'][obs_idx+15],
                    episode_data['rel_pick_pos_base_polar'][obs_idx+15],
                    episode_data['rel_place_pos_base_polar'][obs_idx+15],
                    episode_data['rob_qpos'][obs_idx+15],
                    np.array([int(episode_data['is_holding'][obs_idx+15])]),
                ),axis=-1) )


          #  print("non_vis_obs_step shape at step 0 == " , np.array(non_vis_obs_step).shape)
            non_visual_obs.append(np.array(non_vis_obs_step))
            next_non_visual_obs.append(np.array(next_non_vis_obs_step))
         #   rewards.append(np.array([-1]))
            non_vis_obs_step=[]
            next_non_vis_obs_step=[]

        for obs_idx in range(step,step+num_prev_obs):

            non_vis_obs_step.append( np.concatenate((
                episode_data['rel_resting_pos'][obs_idx],
                episode_data['rel_pick_pos_ee'][obs_idx],
                episode_data['rel_place_pos_ee'][obs_idx],
                episode_data['rel_pick_pos_base_polar'][obs_idx],
                episode_data['rel_place_pos_base_polar'][obs_idx],
                episode_data['rob_qpos'][obs_idx],
                np.array([int(episode_data['is_holding'][obs_idx])]),
            ),axis=-1) )
            next_non_vis_obs_step.append( np.concatenate((
                episode_data['rel_resting_pos'][obs_idx+20-1],
                episode_data['rel_pick_pos_ee'][obs_idx+20-1],
                episode_data['rel_place_pos_ee'][obs_idx+20-1],
                episode_data['rel_pick_pos_base_polar'][obs_idx+20-1],
                episode_data['rel_place_pos_base_polar'][obs_idx+20-1],
                episode_data['rob_qpos'][obs_idx+20-1],
                np.array([int(episode_data['is_holding'][obs_idx+20-1])]),
            ),axis=-1) )
        non_visual_obs.append(np.array(non_vis_obs_step))
        next_non_visual_obs.append(np.array(next_non_vis_obs_step))




        if step==0:
            rewards.append(np.array([0]))
            done.append(np.array([0]))
        if step == number_of_steps - num_prev_obs -num_predicted_actions + 1:
            if skill=='place_task':
                rewards.append(np.array([sparse_reward]))
            else:
                rewards.append(np.array([0]))
            done.append(np.array([1]))
        else:
            rewards.append(np.array([0]))
            done.append(np.array([0]))


        action_step=[]

        if step==0:
            for act_idx in range(step,step+num_predicted_actions):
                action_step.append(episode_data['action_to_save'][act_idx])
          #  print("action_step shape at step 0 == " , np.array(action_step).shape)
            actions.append(np.array(action_step))
            action_step=[]

        for act_idx in range(step+num_prev_obs-1,step+num_prev_obs+num_predicted_actions-1):
    #        print("action == " , episode_data['action_to_save'][act_idx])
            action_step.append(episode_data['action_to_save'][act_idx])
        actions.append(np.array(action_step))




        ### for the placing task
     #   print("episode_data['rel_place_pos_ee'][step+num_prev_obs] == ",episode_data['rel_place_pos_ee'][step+num_prev_obs])
      #  input()




        '''
        if episode_data['is_holding'][step+num_prev_obs] ==0 and skill=='place':
            print("placed the object, the added reward is 100 and breaking the loop")
            rewards[-1]=np.array([sparse_reward])
            done[-1]=np.array([1])
            break
        '''




  #  print("actions shape == " , np.array(actions).shape)
   # input()
    return np.array(head_depth_obs), np.array(arm_depth_obs) , np.array(non_visual_obs), np.array(actions) , np.array(rewards), np.array(next_head_depth_obs), np.array(next_arm_depth_obs), np.array(next_non_visual_obs), np.array(done)


def process_data(upload_directory,save_directory,num_prev_obs=5, num_predicted_actions=20):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print("device == " , device)

    required_skill=skill
    skill_files=[]
    if required_skill!='nav_to_pick_pos' and required_skill!='nav_to_place_pos'and required_skill!='pick_task' and required_skill!='place_task' :
        print("please set a valid skill among nav, pick, place")
        input()

    print("Collecting files for skill: ", required_skill )

    file_names= get_filenames_in_directory(upload_directory)
    print("Total number of files in the directory: ", len(file_names) )
    for file_name in file_names:
        if required_skill in file_name:
            skill_files.append(file_name)
    print("Total number of files for skill {} : {}".format(required_skill, len(skill_files)) )
    number_of_files=len(skill_files)
    file_head_depth_obs=[]
    file_arm_depth_obs=[]
    file_non_visual_obs=[]
    file_action_obs=[]
    file_rewards=[]
    file_next_head_depth_obs=[]
    file_next_arm_depth_obs=[]
    file_next_non_visual_obs=[]
    file_done=[]
    number_of_processed_files=0
    number_of_saved_parts=0
    for file_name in skill_files:

        print("Processing file: ", file_name )
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        number_of_episodes_in_file=len(data.keys())

        for episode_key in data.keys():
            episode_data=data[episode_key]
            head_depth_obs_data,arm_depth_obs_data,non_visual_obs_data,action_data,rewards,next_head_depth_obs_data,next_arm_depth_obs_data,next_non_visual_obs_data,done=process_episode_data(episode_data,num_prev_obs,num_predicted_actions)
            if head_depth_obs_data.shape[0]==0:
                print("Skipping episode ", episode_key , " due to zero valid steps after processing.")
                continue
            file_head_depth_obs.append(head_depth_obs_data)
            file_arm_depth_obs.append(arm_depth_obs_data)
            file_non_visual_obs.append(non_visual_obs_data)
            file_action_obs.append(action_data)
            file_rewards.append(rewards)
            file_next_head_depth_obs.append(next_head_depth_obs_data)
            file_next_arm_depth_obs.append(next_arm_depth_obs_data)
            file_next_non_visual_obs.append(next_non_visual_obs_data)
            file_done.append(done)
       # print("Completed processing file: ", file_name )
        #print("Total episodes processed so far: ", len(file_visual_obs) )
        number_of_processed_files+=1

        if number_of_processed_files % 20 ==0 or number_of_processed_files==number_of_files:
            head_depth_obs_data_all_episodes=np.concatenate(file_head_depth_obs,axis=0)
            arm_depth_obs_data_all_episodes=np.concatenate(file_arm_depth_obs,axis=0)
            non_visual_obs_data_all_episodes=np.concatenate(file_non_visual_obs,axis=0)
            action_data_all_episodes=np.concatenate(file_action_obs,axis=0)
            rewards_data_all_episodes=np.concatenate(file_rewards,axis=0)
            next_head_depth_obs_data_all_episodes=np.concatenate(file_next_head_depth_obs,axis=0)
            next_arm_depth_obs_data_all_episodes=np.concatenate(file_next_arm_depth_obs,axis=0)
            next_non_visual_obs_data_all_episodes=np.concatenate(file_next_non_visual_obs,axis=0)
            done_data_all_episodes=np.concatenate(file_done,axis=0)

            head_depth_obs_data_all_episodes=torch.from_numpy(head_depth_obs_data_all_episodes)
            arm_depth_obs_data_all_episodes=torch.from_numpy(arm_depth_obs_data_all_episodes)
            non_visual_obs_data_all_episodes=torch.from_numpy(non_visual_obs_data_all_episodes)
            action_data_all_episodes=torch.from_numpy(action_data_all_episodes)
            rewards_data_all_episodes=torch.from_numpy(rewards_data_all_episodes)
            next_head_depth_obs_data_all_episodes=torch.from_numpy(next_head_depth_obs_data_all_episodes)
            next_arm_depth_obs_data_all_episodes=torch.from_numpy(next_arm_depth_obs_data_all_episodes)
            next_non_visual_obs_data_all_episodes=torch.from_numpy(next_non_visual_obs_data_all_episodes)
            done_data_all_episodes=torch.from_numpy(done_data_all_episodes)
            number_of_saved_parts+=1
            print("Saving processed data part number {} to disk...".format(number_of_saved_parts) )
            print("head_depth_obs_data_all_episodes shape == " , head_depth_obs_data_all_episodes.shape)
            print("arm_depth_obs_data_all_episodes shape == " , arm_depth_obs_data_all_episodes.shape)
            print("non_visual_obs_data_all_episodes shape == " , non_visual_obs_data_all_episodes.shape)
            print("action_data_all_episodes shape == " , action_data_all_episodes.shape)
            print("rewards_data_all_episodes shape == " , rewards_data_all_episodes.shape)
            print("next_head_depth_obs_data_all_episodes shape == " , next_head_depth_obs_data_all_episodes.shape)
            print("next_arm_depth_obs_data_all_episodes shape == " , next_arm_depth_obs_data_all_episodes.shape)
            print("next_non_visual_obs_data_all_episodes shape == " , next_non_visual_obs_data_all_episodes.shape)
            print("done_data_all_episodes shape == " , done_data_all_episodes.shape)




            if head_depth_obs_data_all_episodes.shape[0] ==arm_depth_obs_data_all_episodes.shape[0] == non_visual_obs_data_all_episodes.shape[0] == action_data_all_episodes.shape[0] == rewards_data_all_episodes.shape[0] == next_head_depth_obs_data_all_episodes.shape[0] == next_arm_depth_obs_data_all_episodes.shape[0] == next_non_visual_obs_data_all_episodes.shape[0] == done_data_all_episodes.shape[0] :
                print("All saved tensors have the same number of samples")
            else:
                print("Saved data has incorrect dimension")
                input()



            torch.save(head_depth_obs_data_all_episodes, os.path.join(save_directory, 'head_depth_obs_data_{}_task_p_{}.pt'.format(skill,number_of_saved_parts)) )
            torch.save(arm_depth_obs_data_all_episodes, os.path.join(save_directory, 'arm_depth_obs_data_{}_task_p_{}.pt'.format(skill,number_of_saved_parts)) )
            torch.save(non_visual_obs_data_all_episodes, os.path.join(save_directory, 'non_visual_obs_data_{}_task_p_{}.pt'.format(skill,number_of_saved_parts)) )
            torch.save(action_data_all_episodes, os.path.join(save_directory, 'action_data_{}_task_p_{}.pt'.format(skill,number_of_saved_parts)) )
            torch.save(rewards_data_all_episodes, os.path.join(save_directory, 'rewards_data_{}_task_p_{}.pt'.format(skill,number_of_saved_parts)) )
            torch.save(next_head_depth_obs_data_all_episodes, os.path.join(save_directory, 'next_head_depth_obs_data_{}_task_p_{}.pt'.format(skill,number_of_saved_parts)) )
            torch.save(next_arm_depth_obs_data_all_episodes, os.path.join(save_directory, 'next_arm_depth_obs_data_{}_task_p_{}.pt'.format(skill,number_of_saved_parts)) )
            torch.save(next_non_visual_obs_data_all_episodes, os.path.join(save_directory, 'next_non_visual_obs_data_{}_task_p_{}.pt'.format(skill,number_of_saved_parts)) )
            torch.save(done_data_all_episodes, os.path.join(save_directory, 'done_data_{}_task_p_{}.pt'.format(skill,number_of_saved_parts)) )
            print("finished saving processed data part number {} to disk.".format(number_of_saved_parts) )

            file_head_depth_obs=[]
            file_arm_depth_obs=[]
            file_non_visual_obs=[]
            file_action_obs=[]
            file_rewards=[]
            file_next_head_depth_obs=[]
            file_next_arm_depth_obs=[]
            file_next_non_visual_obs=[]
            file_done=[]
            head_depth_obs_data_all_episodes=None
            arm_depth_obs_data_all_episodes=None
            non_visual_obs_data_all_episodes=None
            action_data_all_episodes=None
            rewards_data_all_episodes=None
            next_head_depth_obs_data_all_episodes=None
            next_arm_depth_obs_data_all_episodes=None
            next_non_visual_obs_data_all_episodes=None
            done_data_all_episodes=None
            print("Cleared accumulated data from memory after saving to disk.")
            gc.collect()
            torch.cuda.empty_cache()

    return None
   # return visual_obs_data_all_episodes, non_visual_obs_data_all_episodes, action_data_all_episodes, rewards_data_all_episodes, next_visual_obs_data_all_episodes, next_non_visual_obs_data_all_episodes, done_data_all_episodes

    







# ----------------------------
# 6. Train/Test Functions
# ----------------------------



#upload_directory='/home/shokry/hab-mobile-manipulation/diffusion_dataset_new/accurate_data_5_aug/all_tasks_corrected_grasped_obs_26_aug'
upload_directory=directory

save_directory=directory+'/processed_data_flow_matching_CNN_encoder_scratch_epoch_860'
#save_directory='/home/shokry/hab-mobile-manipulation/diffusion_dataset_new/accurate_data_5_aug/all_tasks_corrected_grasped_obs_26_aug/weights_diff_transformer'

process_data(upload_directory,save_directory)

'''
visual_obs_data_all_episodes, non_visual_obs_data_all_episodes, action_data_all_episodes,rewards_data_all_episodes,next_visual_obs_data_all_episodes, next_non_visual_obs_data_all_episodes, done_data_all_episodes=process_data(upload_directory,save_directory)
print("visual_obs_data_all_episodes shape == " , visual_obs_data_all_episodes.shape)
print("non_visual_obs_data_all_episodes shape == " , non_visual_obs_data_all_episodes.shape)
print("action_data_all_episodes shape == " , action_data_all_episodes.shape)
print("rewards_data_all_episodes shape == " , rewards_data_all_episodes.shape)
print("next_visual_obs_data_all_episodes shape == " , next_visual_obs_data_all_episodes.shape)
print("next_non_visual_obs_data_all_episodes shape == " , next_non_visual_obs_data_all_episodes.shape)
print("done_data_all_episodes shape == " , done_data_all_episodes.shape)
torch.save(visual_obs_data_all_episodes, os.path.join(save_directory, 'visual_obs_data_{}_task_with_rewards_eval_dataset_ep_0_5_prev_ob_20_acts.pt'.format(skill)) )
torch.save(non_visual_obs_data_all_episodes, os.path.join(save_directory, 'non_visual_obs_data_{}_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt'.format(skill)) )
torch.save(action_data_all_episodes, os.path.join(save_directory, 'action_data_{}_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt'.format(skill)) )
torch.save(rewards_data_all_episodes, os.path.join(save_directory, 'rewards_data_{}_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt'.format(skill)) )
torch.save(next_visual_obs_data_all_episodes, os.path.join(save_directory, 'next_visual_obs_data_{}_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt'.format(skill)) )
torch.save(next_non_visual_obs_data_all_episodes, os.path.join(save_directory, 'next_non_visual_obs_data_{}_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt'.format(skill)) )
torch.save(done_data_all_episodes, os.path.join(save_directory, 'done_data_{}_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt'.format(skill)) )
'''
