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



skill='pick'


#directory='/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/complete_rearrange_trajs'
directory='/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data'

Feat_ext = SimpleCNN(1, (128, 128), 512).to(device).to(torch.float32)
Feat_ext.load_state_dict(torch.load( os.path.join(directory, 'visual_encoder.pth'),
    map_location='cpu',
    weights_only=True
))
Feat_ext.to(device)
Feat_ext.eval()


Batch_size=64

def get_filenames_in_directory(directory):
    filenames = []
    for filename in os.listdir(directory) :
       # if filename.endswith('.pkl') and '{}_only_eval_dataset_ep_0'.format(skill) in filename:
        filenames.append(os.path.join(directory, filename))
    return filenames







def process_episode_data(episode_data,num_prev_obs,num_predicted_actions):

    sparse_reward=100
    visual_obs=[]
    non_visual_obs=[]
    actions=[]
    rewards=[]
    next_visual_obs=[]
    next_non_visual_obs=[]
    done=[]

    number_of_steps=len(episode_data['action_to_save'])

    visual_features=torch.zeros( (number_of_steps, 512), dtype=torch.float32).to(device)
    for i in range(number_of_steps):
        visual_features[i]=Feat_ext( torch.from_numpy( episode_data['robot_head_depth'][i].astype(np.float32) ).unsqueeze(0).permute(0,3,1,2).to(device))
    visual_features=visual_features.cpu().detach().numpy()



    last_action=episode_data['action_to_save'][-1]

    last_action[0:-1]=np.zeros(9)

    last_action_repeated=np.array([last_action for _ in range(num_predicted_actions)])
   # print("last_action_repeated shape == " , last_action_repeated.shape)
    episode_data['action_to_save']=np.concatenate( (episode_data['action_to_save'], last_action_repeated), axis=0)
    #print("New action_to_save shape == " , episode_data['action_to_save'].shape)
    for step in range(number_of_steps-num_prev_obs+1 -10):

        vis_obs_step=[]
        next_vis_obs_step=[]
        if step==0:
            for obs_idx in range(step,step+num_prev_obs):
                vis_obs_step.append(visual_features[0])
                if obs_idx+15 > number_of_steps-1:
                    next_vis_obs_step.append(visual_features[-1])
                else:
                    next_vis_obs_step.append(visual_features[obs_idx+15])
           # print("vis_obs_step shape at step 0 == " , np.array(vis_obs_step).shape)
            visual_obs.append(np.array(vis_obs_step))
            next_visual_obs.append(np.array(next_vis_obs_step))
            vis_obs_step=[]
            next_vis_obs_step=[]

        for obs_idx in range(step,step+num_prev_obs):
            vis_obs_step.append(visual_features[obs_idx])
            if obs_idx+20-1 > number_of_steps-1:
                next_vis_obs_step.append(visual_features[-1])
            else:
                next_vis_obs_step.append(visual_features[obs_idx+20-1])
        visual_obs.append(np.array(vis_obs_step))
        next_visual_obs.append(np.array(next_vis_obs_step))

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
                
                if obs_idx+15 > number_of_steps-1:
                    next_non_vis_obs_step.append( np.concatenate((
                    episode_data['rel_resting_pos'][-1],
                    episode_data['rel_pick_pos_ee'][-1],
                    episode_data['rel_place_pos_ee'][-1],
                    episode_data['rel_pick_pos_base_polar'][-1],
                    episode_data['rel_place_pos_base_polar'][-1],
                    episode_data['rob_qpos'][-1],
                    np.array([int(episode_data['is_holding'][-1])]),
                ),axis=-1) )
                
                else:
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
            
            if obs_idx+20-1 > number_of_steps-1:
                next_non_vis_obs_step.append( np.concatenate((
                episode_data['rel_resting_pos'][-1],
                episode_data['rel_pick_pos_ee'][-1],
                episode_data['rel_place_pos_ee'][-1],
                episode_data['rel_pick_pos_base_polar'][-1],
                episode_data['rel_place_pos_base_polar'][-1],
                episode_data['rob_qpos'][-1],
                np.array([int(episode_data['is_holding'][-1])]),
            ),axis=-1) )
            else:  
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




#### modify the action part

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


        if step==0:
            rewards.append(np.array([0]))
            done.append(np.array([0]))
        if step == number_of_steps - num_prev_obs - 1:
            if skill=='place':
                rewards.append(np.array([sparse_reward]))
            else:
                rewards.append(np.array([0]))
            done.append(np.array([1]))
        else:
            rewards.append(np.array([0]))
            done.append(np.array([0]))


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

    return np.array(visual_obs), np.array(non_visual_obs), np.array(actions) , np.array(rewards), np.array(next_visual_obs), np.array(next_non_visual_obs), np.array(done)


def process_data(upload_directory,save_directory,num_prev_obs=5, num_predicted_actions=20):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print("device == " , device)

    required_skill=skill
    skill_files=[]
    if required_skill!='nav' and required_skill!='pick' and required_skill!='place':
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
    file_visual_obs=[]
    file_non_visual_obs=[]
    file_action_obs=[]
    file_rewards=[]
    file_next_visual_obs=[]
    file_next_non_visual_obs=[]
    file_done=[]
    number_of_processed_files=0
    number_of_saved_parts=0
    for file_name in skill_files:

        print("Processing file: ", file_name )
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        number_of_episodes_in_file=len(data)

        for episode_key in data.keys():
            episode_data=data[episode_key]
            visual_obs_data,non_visual_obs_data,action_data,rewards,next_visual_obs_data,next_non_visual_obs_data,done=process_episode_data(episode_data,num_prev_obs,num_predicted_actions)
            if visual_obs_data.shape[0] == non_visual_obs_data.shape[0] == action_data.shape[0] == rewards.shape[0] == next_visual_obs_data.shape[0] == next_non_visual_obs_data.shape[0] == done.shape[0] :
                #print("Processed data have equal number of samples !!! ")
                pass
                
            else:
                print("error in processing episode {} in file {}".format(episode_key,file_name))
                input()
            if visual_obs_data.shape[0]==0:
                print("Skipping episode ", episode_key , " due to zero valid steps after processing.")
                continue
            file_visual_obs.append(visual_obs_data)
            file_non_visual_obs.append(non_visual_obs_data)
            file_action_obs.append(action_data)
            file_rewards.append(rewards)
            file_next_visual_obs.append(next_visual_obs_data)
            file_next_non_visual_obs.append(next_non_visual_obs_data)
            file_done.append(done)
       # print("Completed processing file: ", file_name )
        #print("Total episodes processed so far: ", len(file_visual_obs) )
        number_of_processed_files+=1

        if number_of_processed_files % 20 ==0 or number_of_processed_files==number_of_files:
            visual_obs_data_all_episodes=np.concatenate(file_visual_obs,axis=0)
            non_visual_obs_data_all_episodes=np.concatenate(file_non_visual_obs,axis=0)
            action_data_all_episodes=np.concatenate(file_action_obs,axis=0)
            rewards_data_all_episodes=np.concatenate(file_rewards,axis=0)
            next_visual_obs_data_all_episodes=np.concatenate(file_next_visual_obs,axis=0)
            next_non_visual_obs_data_all_episodes=np.concatenate(file_next_non_visual_obs,axis=0)
            done_data_all_episodes=np.concatenate(file_done,axis=0)

            visual_obs_data_all_episodes=torch.from_numpy(visual_obs_data_all_episodes)
            non_visual_obs_data_all_episodes=torch.from_numpy(non_visual_obs_data_all_episodes)
            action_data_all_episodes=torch.from_numpy(action_data_all_episodes)
            rewards_data_all_episodes=torch.from_numpy(rewards_data_all_episodes)
            next_visual_obs_data_all_episodes=torch.from_numpy(next_visual_obs_data_all_episodes)
            next_non_visual_obs_data_all_episodes=torch.from_numpy(next_non_visual_obs_data_all_episodes)
            done_data_all_episodes=torch.from_numpy(done_data_all_episodes)
            number_of_saved_parts+=1
            print("Saving processed data part number {} to disk...".format(number_of_saved_parts) )
            print("visual_obs_data_all_episodes shape == " , visual_obs_data_all_episodes.shape)
            print("non_visual_obs_data_all_episodes shape == " , non_visual_obs_data_all_episodes.shape)
            print("action_data_all_episodes shape == " , action_data_all_episodes.shape)
            print("rewards_data_all_episodes shape == " , rewards_data_all_episodes.shape)
            print("next_visual_obs_data_all_episodes shape == " , next_visual_obs_data_all_episodes.shape)
            print("next_non_visual_obs_data_all_episodes shape == " , next_non_visual_obs_data_all_episodes.shape)
            print("done_data_all_episodes shape == " , done_data_all_episodes.shape)
            
            if visual_obs_data_all_episodes.shape[0] == non_visual_obs_data_all_episodes.shape[0] == action_data_all_episodes.shape[0] == rewards_data_all_episodes.shape[0] == next_visual_obs_data_all_episodes.shape[0] == next_non_visual_obs_data_all_episodes.shape[0] == done_data_all_episodes.shape[0] :
                print("All saved tensors have the same number of samples")
            else:
                print("Saved data has incorrect dimension")
                input()

            torch.save(visual_obs_data_all_episodes, os.path.join(save_directory, 'visual_obs_data_{}_task_p_{}_with_half_action_extension.pt'.format(skill,number_of_saved_parts)) )
            torch.save(non_visual_obs_data_all_episodes, os.path.join(save_directory, 'non_visual_obs_data_{}_task_p_{}_with_half_action_extension.pt'.format(skill,number_of_saved_parts)) )
            torch.save(action_data_all_episodes, os.path.join(save_directory, 'action_data_{}_task_p_{}_with_half_action_extension.pt'.format(skill,number_of_saved_parts)) )
            torch.save(rewards_data_all_episodes, os.path.join(save_directory, 'rewards_data_{}_task_p_{}_with_half_action_extension.pt'.format(skill,number_of_saved_parts)) )
            torch.save(next_visual_obs_data_all_episodes, os.path.join(save_directory, 'next_visual_obs_data_{}_task_p_{}_with_half_action_extension.pt'.format(skill,number_of_saved_parts)) )
            torch.save(next_non_visual_obs_data_all_episodes, os.path.join(save_directory, 'next_non_visual_obs_data_{}_task_p_{}_with_half_action_extension.pt'.format(skill,number_of_saved_parts)) )
            torch.save(done_data_all_episodes, os.path.join(save_directory, 'done_data_{}_task_p_{}_with_half_action_extension.pt'.format(skill,number_of_saved_parts)) )
            print("finished saving processed data part number {} to disk.".format(number_of_saved_parts) )

            file_visual_obs=[]
            file_non_visual_obs=[]
            file_action_obs=[]
            file_rewards=[]
            file_next_visual_obs=[]
            file_next_non_visual_obs=[]
            file_done=[]
            visual_obs_data_all_episodes=None
            non_visual_obs_data_all_episodes=None
            action_data_all_episodes=None
            rewards_data_all_episodes=None
            next_visual_obs_data_all_episodes=None
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

save_directory=directory+'/processed_data_with_action_extension'
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
