# TODO:
# Create dataset object (equalent to d4rl) - Done
# Load behaviour model - Done
# Generate fake action sequences
# Load Q Model
# Train loop

import os
from os import path
import torch
from typing import Optional
import copy

# import gym
# import d4rl
import numpy as np
# import functools
# import copy
# import os
import torch.nn.functional as F
import tqdm
from scipy.special import softmax
from torch.optim import Adam

#from mobile_manipulation.ppo.trainers.ppo_trainer_v0 import ConditionalDiffusionModel, NoiseScheduler, TrainConfig, QTransformer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataclasses import dataclass
from torch import nn
import math

import gc
from einops import rearrange







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
    dropout: float = 0.1

    hist_len: int = 5
    horizon: int = 20

    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_epochs: int = 50

    gamma: float = 0.99
    num_action_samples: int = 20
    target_ema_tau: float = 0.005

    log_every: int = 50
    ckpt_every_steps: int = 2
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
        self.num_tokens = 3 * hist_len + horizon

        self.head_depth_proj = nn.Linear(d_vis, d_model)
        self.arm_depth_proj = nn.Linear(d_vis, d_model)
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

    def forward(self, head_depth_hist: torch.Tensor, arm_depth_hist: torch.Tensor , nonvis_hist: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        B = head_depth_hist.shape[0]
    #    print("Visual history shape in Q transformer == ", vis_hist.shape)
     #   print("Non visual history shape in Q transformer == ", nonvis_hist.shape)
        head_depth_tok = self.head_depth_proj(head_depth_hist)
        arm_depth_tok = self.arm_depth_proj(arm_depth_hist)
        nonvis_tok = self.nonvis_proj(nonvis_hist)

        state_tokens = torch.stack([head_depth_tok, arm_depth_tok, nonvis_tok], dim=2)
        state_tokens = state_tokens.view(B, 3 * self.hist_len, -1)



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







def _mem_usage(tensor):
    bytes_used = tensor.numel() * tensor.element_size()
    return f"{bytes_used / 1024**3:.2f} GB"





device = torch.device("cuda" if torch.cuda.is_available() else torch.device('cpu'))

ROOT = "/lustre/mlnvme/data/s47ashok_hpc-data/final_dataset_27_jan/tidy_house/seed_100/diffusion_data/collected_dataset/processed_data_flow_matching_CNN_encoder_scratch_epoch_860"
Diffusion_model_path = "/lustre/mlnvme/data/s47ashok_hpc-data/final_dataset_27_jan/tidy_house/seed_100/diffusion_data/weights_flow_matching_CNN_encoder_scratch_head_and_arm_depth_without_action_extension"
Critic_model_path = "/lustre/mlnvme/data/s47ashok_hpc-data/final_dataset_27_jan/tidy_house/seed_100/diffusion_data/collected_dataset/processed_data_flow_matching_CNN_encoder_scratch_epoch_860/critic_weights"


#ROOT = "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/sample_data"
#Diffusion_model_path = "/lustre/mlnvme/data/s47ashok_hpc-data/new_accurate_data_individual_tasks_4_jan_2026/weights_pretrained_visual_encoder"
#Critic_model_path = "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/sample_data/critic_weights"



MAX_BZ_SIZE = 1024
soft_Q_update = True

Discount_factor=0.98

Batch_size=2048

Total_num_of_training_epcohs=1000000

number_of_epochs_to_update_returns=100
number_of_MC_samples=10

class update_return_dataset(Dataset):

    def __init__(self,head_depth_states,arm_depth_states, non_visual_states, fake_actions):
        self.head_depth_states = head_depth_states
        self.arm_depth_states = arm_depth_states
        self.non_visual_states = non_visual_states
        self.fake_actions = fake_actions
        self.len = self.head_depth_states.shape[0]
    def __getitem__(self, index):
        i = index % self.len
        head_depth_states = self.head_depth_states[i]
        arm_depth_states = self.arm_depth_states[i]
        non_visual_states = self.non_visual_states[i]
        fake_actions = self.fake_actions[i]
        return head_depth_states,arm_depth_states, non_visual_states, fake_actions

    def __len__(self):
        return self.len



class Diffusion_buffer(Dataset):

    def __init__(self):
        self.normalise_return = True
        self.MC_dropout=False


        data = self._load_data()
        self.actions = data["actions"]
        self.head_depth_states = data["head_depth_states"]
        self.arm_depth_states = data["arm_depth_states"]
        self.non_visual_states = data["non_visual_states"]
        self.rewards = data["rewards"]
        self.done = data["done"]
        self.next_head_depth_states = data["next_head_depth_states"]
        self.next_arm_depth_states = data["next_arm_depth_states"]
        self.next_non_visual_states = data["next_non_visual_states"]
        self.fake_actions = data["fake_actions"]
        
        self.returns = data["returns"]
        data=[]
        #self.raw_returns = [self.returns.copy()]
        self.raw_returns = self.returns.clone().detach()
        self.raw_values = []
        self.returns_mean = torch.mean(self.returns)
        self.returns_std = max(torch.std(self.returns), torch.tensor(0.1))
       # print("returns mean {}  std {}".format(self.returns_mean, self.returns_std))
        if self.normalise_return:
            self.returns = (self.returns - self.returns_mean) / self.returns_std
            #data["returns"] = returns
            print("returns normalised at mean {}, std {}".format(self.returns_mean, self.returns_std))
        else:
            print("no normal")

        self.len = self.head_depth_states.shape[0]
        # make sure same number of data points exist in all tasks
        # self.fake_len = int(np.maximum(np.round(10000 / self.len), 1)) * self.len
        # print(self.len, "data loaded", self.fake_len, "data faked")

    
        
    def __getitem__(self, index):
        i = index % self.len
        actions = self.actions[i]
        rewards = self.rewards[i]
        head_depth_states = self.head_depth_states[i]
        arm_depth_states = self.arm_depth_states[i]
        non_visual_states = self.non_visual_states[i]
        done= self.done[i]
        next_head_depth_states = self.next_head_depth_states[i]
        next_arm_depth_states = self.next_arm_depth_states[i]
        next_non_visual_states = self.next_non_visual_states[i]
        fake_actions = self.fake_actions[i]
        returns= self.returns[i]
        return actions, rewards, head_depth_states, arm_depth_states , non_visual_states, done, next_head_depth_states, next_arm_depth_states , next_non_visual_states, fake_actions,returns

    def __len__(self):
        return self.len

    def _load_data(self):
        data = {}
        # Define task types and their partition counts
        tasks = [
            ("nav_to_pick_pos", 7),
            ("nav_to_place_pos", 7),
            ("pick_task", 7),
            ("place_task", 6),
            # ("place_task", 1),
        ]

        # Define data types and their file prefixes
        data_types = {
            "actions": "action_data",
            "done": "done_data",
            "non_visual_states": "non_visual_obs_data",
            "head_depth_states": "head_depth_obs_data",
            "arm_depth_states": "arm_depth_obs_data",
            "rewards": "rewards_data",
            "next_head_depth_states": "next_head_depth_obs_data",
            "next_arm_depth_states": "next_arm_depth_obs_data",
            "next_non_visual_states": "next_non_visual_obs_data",
            "fake_actions": "predicted_actions_flow_matching",
        }

        # Load all data using loops
        for data_key, file_prefix in data_types.items():
            tensors = []
            for task_name, num_partitions in tasks:
                for p in range(1, num_partitions + 1):
                    print("Loading partition", p, task_name)
                    file_name = f"{file_prefix}_{task_name}_task_p_{p}.pt"
                    # file_name = f"{file_prefix}_{task_name}_p_1.pt"
                    tensors.append(
                        torch.load(
                            path.join(ROOT, file_name),
                            map_location=torch.device("cpu"),
                        )
                    )
            data[data_key] = torch.cat(tensors, dim=0)

        if not data["done"][-1]:
            data["done"][-1] = True

        print("Loaded data:")
        print(
            "Actions",
            data["actions"].shape,
            data["actions"].device,
            _mem_usage(data["actions"]),
        )
        print(
            "Done",
            data["done"].shape,
            data["done"].device,
            _mem_usage(data["done"]),
        )
        print(
            "Non-Visual Obs",
            data["non_visual_states"].shape,
            data["non_visual_states"].device,
            _mem_usage(data["non_visual_states"]),
        )
        print(
            "Head_Depth Obs",
            data["head_depth_states"].shape,
            data["head_depth_states"].device,
            _mem_usage(data["head_depth_states"]),
        )
        print(
            "Arm_Depth Obs",
            data["arm_depth_states"].shape,
            data["arm_depth_states"].device,
            _mem_usage(data["arm_depth_states"]),
        )
        print(
            "Rewards",
            data["rewards"].shape,
            data["rewards"].device,
            _mem_usage(data["rewards"]),
        )
        print(
            "Next Non-Visual Obs",
            data["next_non_visual_states"].shape,
            data["next_non_visual_states"].device,
            _mem_usage(data["next_non_visual_states"]),
        )
        print(
            "Next Head Depth Obs",
            data["next_head_depth_states"].shape,
            data["next_head_depth_states"].device,
            _mem_usage(data["next_head_depth_states"]),
        )
        print(
            "Next Arm Depth Obs",
            data["next_arm_depth_states"].shape,
            data["next_arm_depth_states"].device,
            _mem_usage(data["next_arm_depth_states"]),
        )
        print(
            "Fake Actions",
            data["fake_actions"].shape,
            data["fake_actions"].device,
            _mem_usage(data["fake_actions"]),
        )

        print(torch.cuda.memory_allocated() / 1024**2, "MB")

        data["rewards"] = data["rewards"].squeeze()
        data["done"] = data["done"].squeeze()

        assert data["done"][-1]
        data["returns"] = torch.zeros(data["head_depth_states"].shape[0])

        last = 0

        # NOTE: We set the returns based on the the MC returns: discount the
        # last return until the first one.
        # This gives us the initial value for all returns in each state

        for i in range(data["returns"].shape[0] - 1, -1, -1):

            last = data["rewards"][i] + Discount_factor * last * (
                1.0 - data["done"][i]
            )
            data["returns"][i] = last

        return data
    


    def update_returns(self, score_model):
        # NOTE: We calculate the Q value at each state with all 16 fake actions
        # Then we do some processing to update the return values ???c
        # - Soft Q-update (weighted average favoring high values)
        # - Percentile (85th percentile)

        assert self.fake_actions is not None

        assert self.head_depth_states.shape[0] == self.fake_actions.shape[0]
        qs = None
        q = None

        update_return_dataset_instance = update_return_dataset(self.next_head_depth_states,self.next_arm_depth_states, self.next_non_visual_states, self.fake_actions)
        update_return_dataloader = DataLoader(update_return_dataset_instance, batch_size=2048*2, shuffle=False)

        for head_depth_states,arm_depth_states, non_visual_states, fake_actions in tqdm.tqdm(update_return_dataloader):
            head_depth_states = head_depth_states.to("cuda")
            arm_depth_states = arm_depth_states.to("cuda")
            non_visual_states = non_visual_states.to("cuda")
            fake_actions = fake_actions.to("cuda")
            
            head_depth_states = torch.repeat_interleave(head_depth_states, fake_actions.shape[1], dim=0)
            arm_depth_states = torch.repeat_interleave(arm_depth_states, fake_actions.shape[1], dim=0)
            non_visual_states = torch.repeat_interleave(non_visual_states, fake_actions.shape[1], dim=0)
            reshaped_fake_actions=fake_actions.reshape((head_depth_states.shape[0], fake_actions.shape[-2], fake_actions.shape[-1]))


            if self.MC_dropout:
                with torch.no_grad():
                    q = None
                    for T in range(number_of_MC_samples):
                        q_single = score_model.calculateQ(head_depth_states,arm_depth_states, non_visual_states, reshaped_fake_actions)
                        q_single= q_single.unsqueeze(1)

                        if q is None:
                            q = q_single
                        else:
                            q = torch.cat([q, q_single], dim=1)
                    print("sample q values in update returns ", q[0:2,0:5])
                    q_mean=torch.mean(q, dim=1, keepdim=True)
                    q_var=torch.var(q, dim=1, keepdim=True)
                    print("q mean sample in update returns ", q_mean[0:2])
                    print("q var sample in update returns ", q_var[0:2])

                    q_mean=q_mean.reshape((fake_actions.shape[0], fake_actions.shape[1]))

                    
                    q_var=q_var.reshape((fake_actions.shape[0], fake_actions.shape[1]))
                    q_var = q_var.clamp(min=1e-4)
                    eps=1e-8
                    print("original q var sample in update returns ", q_var[0:2])
                  #  q_var_max=torch.max(q_var, dim=1, keepdim=True).values
                  #  q_var_min=torch.min(q_var, dim=1, keepdim=True).values
                  #  q_var_normalized=(q_var - q_var_min) / (q_var_max - q_var_min + 1e-6)
                  #  print("q_var normalized == ", q_var_normalized[0:2])

                   # unnormalized_weights = 1.0 - q_var_normalized
                   # sum_weights = torch.sum(unnormalized_weights, dim=1, keepdim=True) + eps
                   # normalized_weights = unnormalized_weights / sum_weights
                    weights = 1.0 / (q_var + eps)                 # inverse-variance weights
                    weights = weights / weights.sum(dim=1, keepdim=True)
                   # weighted_q_values_normalized = q_mean_normalized * (1.0 - q_var_normalized)
                    updated_q = q_mean * weights
                    print("updated q values sample in update returns ", updated_q[0:2])
                 #   input()

                    if qs is None:
                        qs = updated_q
                    else:
                        qs=torch.cat([qs, updated_q], dim=0)

            else:
                with torch.no_grad():
                    q = score_model.calculateQ(head_depth_states,arm_depth_states, non_visual_states, reshaped_fake_actions)
                    q = q.reshape((fake_actions.shape[0], fake_actions.shape[1]))
                    q= q/ fake_actions.shape[1]
                    if qs is None:
                        qs = q
                    else:
                        qs=torch.cat([qs, q], dim=0)

 
                    






        print("total q values collected in update returns ", qs.shape)
        print("max q values collected in update returns ", torch.max(qs))
        print("min q values collected in update returns ", torch.min(qs))
        print("mean q values collected in update returns ", torch.mean(qs))
        print("std q values collected in update returns ", torch.std(qs))
       # input()
        values = np.array(qs.detach().cpu())
        #print("final value shape in update returns ", values.shape)


        #self.raw_values.append(values)
        self.raw_values=values.copy()
        if soft_Q_update:
            #values = np.sum(softmax(20 * values, axis=-1) * values, axis=-1, keepdims=1)
           # values = np.mean(values, axis=-1, keepdims=1)
            values = np.sum(values, axis=-1, keepdims=1)

        else:
            values = np.percentile(values, 85, axis=-1, keepdims=1)
        values = torch.FloatTensor(values)
        print("processed value shape in update returns ", values.shape)
     #   print("processed values sample ", values[0:5,0:5])
        if self.normalise_return:
            values = values * self.returns_std + self.returns_mean
      #  print("values after denormalisation sample ", values[0:5,0:5])
        #assert values.ndim == 2
        assert values.shape[0] == self.non_visual_states.shape[0]
        returns = torch.zeros(values.shape[0])
        last = 0
        num_truncated_traj = 0

       # print("total number of data in update returns ", returns.shape[0])
        for i in range(returns.shape[0] - 1, -1, -1):
            
            
            bootstrap = self.rewards[i] + Discount_factor * last * (1.0 - self.done[i])
           # imagainary = values[i, 0]
            imagainary =  self.rewards[i] + Discount_factor * values[i, 0]  * (1.0 - self.done[i])    
            if i % 10000 ==0 or i % 10000 ==1 or i % 10000 ==2:
                print("updating return for index ", i)
                print("reward ", self.rewards[i], " done ", self.done[i], " last ", last)
                print("bootstrap ", bootstrap, " imagainary ", imagainary)
           
            if bootstrap > imagainary:
                returns[i] = bootstrap
            else:
                returns[i] = imagainary
                num_truncated_traj += 1
            last = returns[i]
        print("num_truncated_traj perc", num_truncated_traj / returns.shape[0])
        #self.raw_returns.append(returns)
      #  returns=returns.squeeze()
        print("updated returns shape ", returns.shape)
      #  input()
        self.raw_returns=returns.clone().detach()
        self.returns_mean = torch.mean(returns)
        self.returns_std = torch.tensor(max(torch.std(returns), 0.1))
      #  print("returns mean {}  std {}".format(self.returns_mean, self.returns_std))
        if self.normalise_return:
            returns = (returns - self.returns_mean) / self.returns_std
            print("returns normalised at mean {}, std {}".format(self.returns_mean, self.returns_std))
        else:
            print("no normal")



        self.returns = returns.clone().detach()
      #  print("updated self.returns shape ", self.returns.shape)
     #   input()

        returns = None


        #self.ys = np.concatenate([returns, self.actions], axis=-1)
        #self.ys = self.ys.astype(np.float32)


       # self.rewards = torch.FloatTensor(returns.squeeze())

        print("update returns finished")


class ModelWrapper():
    def __init__(self):        
       # self.diffusion_policy = ConditionalDiffusionModel()
    #    self.diffusion_policy.load_state_dict(torch.load(path.join(Diffusion_model_path,"pretrained_visual_encoder_2200.pt"),
     #                                                       map_location=device))

        #self.diffusion_policy.to(device)
        #self.diffusion_policy.eval()
        #for p in self.diffusion_policy.parameters():
         #   p.requires_grad_(False)
        #self.scheduler = NoiseScheduler()

        cfg = TrainConfig()
        self.q_value_network = QTransformer(
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
        # ckpt = torch.load('/home/user/siddiquieu1/HRL-Usama/mobile-manipulation/ahmed_checkpoints/ckpt_step_4500.pt', map_location="cpu")
        # self.q_value_network.load_state_dict(ckpt["q_state_dict"]) 
        self.q_value_network = self.q_value_network.to(device)
      #  self.q_value_network.eval()
    '''
    def sample(self, visual_obs, non_visual_obs):
        # print(visual_obs.shape)
        # print(non_visual_obs.shape)
        # exit()

        shape = (20, 20, 10)
        with torch.no_grad():
            actions = self.scheduler.sample(self.diffusion_policy, shape, visual_obs.to(device).to(torch.float32), non_visual_obs.to(device).to(torch.float32), device, num_random_samples=20)

        return actions
    '''
    def calculateQ(self, head_depth_states, arm_depth_states, non_visual_states, actions):

        q_values = self.q_value_network(head_depth_states.to(device).to(torch.float32),arm_depth_states.to(device).to(torch.float32), non_visual_states.to(device).to(torch.float32), actions.to(device).to(torch.float32))


        return q_values


def train_critic(score_model, data_loader):
   # data_loader.dataset.update_returns(score_model)

    optimizer = Adam(score_model.q_value_network.parameters(), lr=3e-4)

    bk_model_sd = copy.deepcopy(score_model.q_value_network.state_dict())

    for epoch in range(Total_num_of_training_epcohs):
        gc.collect()
        torch.cuda.empty_cache()

        avg_loss = 0.
        num_items = 0
        for batch in tqdm.tqdm(data_loader):
            actions, rewards, head_depth_states, arm_depth_states , non_visual_states,done,next_head_depth_states, next_arm_depth_states,next_non_visual_states,fake_actions,returns = batch
            returns = returns.to(device)


            qs = score_model.calculateQ(head_depth_states, arm_depth_states, non_visual_states, actions)
       #     print("shape of returns in critic training epoch {} == {}".format(epoch, returns.shape))
       #     print("shape of Qs in critic training epoch {} == {}".format(epoch, qs.shape))
       #     print("maximum returns in critic training epoch {} == {}".format(epoch, torch.max(returns)))
        #    print("minimum returns in critic training epoch {} == {}".format(epoch, torch.min(returns)))
        #    print("mean returns in critic training epoch {} == {}".format(epoch, torch.mean(returns)))
        #    print("std returns in critic training epoch {} == {}".format(epoch, torch.std(returns)))
        #    print("maximum Qs in critic training epoch {} == {}".format(epoch, torch.max(qs)))
        #    print("minimum Qs in critic training epoch {} == {}".format(epoch, torch.min(qs)))
        #    print("mean Qs in critic training epoch {} == {}".format(epoch, torch.mean(qs)))
        #    print("std Qs in critic training epoch {} == {}".format(epoch, torch.std(qs)))
            loss = torch.mean((qs - returns)**2)
         #   print("Loss for critic training epoch {} == {}".format(epoch, loss.item()))
         #   input()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #score_model.condition = None
            avg_loss += loss.item() * actions.shape[0]
            num_items += actions.shape[0]

        print("shape of returns in critic training epoch {} == {}".format(epoch, returns.shape))
        print("shape of Qs in critic training epoch {} == {}".format(epoch, qs.shape))
        print("maximum returns in critic training epoch {} == {}".format(epoch, torch.max(returns)))
        print("minimum returns in critic training epoch {} == {}".format(epoch, torch.min(returns)))
        print("mean returns in critic training epoch {} == {}".format(epoch, torch.mean(returns)))
        print("std returns in critic training epoch {} == {}".format(epoch, torch.std(returns)))
        print("maximum Qs in critic training epoch {} == {}".format(epoch, torch.max(qs)))
        print("minimum Qs in critic training epoch {} == {}".format(epoch, torch.min(qs)))
        print("mean Qs in critic training epoch {} == {}".format(epoch, torch.mean(qs)))
        print("std Qs in critic training epoch {} == {}".format(epoch, torch.std(qs)))

        print("Average Loss for epoch {} == {}".format(epoch, avg_loss/num_items))


        if epoch != 0 and epoch % number_of_epochs_to_update_returns == 0:
            #print("Average Loss for epoch {} == {}".format(epoch, avg_loss/num_items))
            for name, param in score_model.q_value_network.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad norm = {param.grad.norm().item()}")
            data_loader.dataset.update_returns(score_model)
            
            ## save model
            torch.save({
                'q_state_dict': score_model.q_value_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, path.join(Critic_model_path, f'ckpt_with_normalization_without_uncertainty_head_arm_depth_no_input_dropout_tidy_house_ep_{epoch}.pt'))
    
            score_model.q_value_network.load_state_dict(bk_model_sd)
            optimizer = Adam(score_model.q_value_network.parameters(), lr=3e-4)


def critic():
    score_model = ModelWrapper()
    dataset = Diffusion_buffer()

    data_loader = DataLoader(dataset, batch_size=Batch_size, shuffle=True)

    # Generate fake actions
    # all_actions = []
    # for i in tqdm.tqdm(range(0, len(dataset), 128)):
    #     visual_states = dataset.visual_states[i:128]
    #     non_visual_states = dataset.non_visual_states[i:128]

    #     fake_actions = score_model.sample(visual_states, non_visual_states)
    #     all_actions.append(fake_actions.cpu().numpy())


    train_critic(score_model, data_loader)

if __name__ == "__main__":


        
    critic()

    '''
    uploaded_predicted_nav_actions=torch.load(path.join(ROOT, "predicted_actions_diffusion_pick_task_p_1.pt"))
    done_nav=torch.load(path.join(ROOT, "done_data_pick_task_p_1.pt"))
    for idx in range(uploaded_predicted_nav_actions.shape[0]):
        if done_nav[idx]:
            for j in range(uploaded_predicted_nav_actions.shape[1]):
                print("action step == {}".format(uploaded_predicted_nav_actions[idx][j][0]))
            input()
    '''

# ----- ROUGH TESTS -----


# buffer = Diffusion_buffer()
# print(len(buffer))
# a, r, v, nv = buffer[0]
# print("Action", a.shape)
# print("Reward", r.shape)
# print("Visual", v.shape)
# print("Non-Visual", nv.shape)

# all_actions = torch.load(path.join(ROOT, "action_data_nav_task_p_2.pt"))
# all_dones = torch.load(path.join(ROOT, "done_data_nav_task_p_2.pt"))
# all_non_visual_obs = torch.load(path.join(ROOT, "non_visual_obs_data_nav_task_p_2.pt"))
# all_visual_obs = torch.load(path.join(ROOT, "visual_obs_data_nav_task_p_2.pt"))
# all_rewards = torch.load(path.join(ROOT, "rewards_data_nav_task_p_2.pt"))

# print("Actions", all_actions.shape)
# print("Done", all_dones.shape)
# print("Non-Visual Obs", all_non_visual_obs.shape)
# print("Visual Obs", all_visual_obs.shape)
# print("Rewards", all_rewards.shape)

# actions = all_actions[0, :, :]
# non_visual_obs = all_non_visual_obs[0, :, :]
# visual_obs = all_visual_obs[0, :, :]
# rewards = all_rewards[0, :]

# print("Single Traj Actions", actions.shape)
# print("Single Traj Non-Visual Obs", non_visual_obs.shape)
# print("Single Traj Visual Obs", visual_obs.shape)
# print("Single Traj Rewards", rewards.shape)

# diffusion_policy = ConditionalDiffusionModel()
# diffusion_policy.load_state_dict(torch.load("/home/user/siddiquieu1/HRL-Usama/mobile-manipulation/ahmed_checkpoints/model_all_tasks_eval_dataset_5_prev_obs_20_act_4400.pt",
#                                                     map_location=device))

# diffusion_policy.to(device)
# diffusion_policy.eval()
# for p in diffusion_policy.parameters():
#     p.requires_grad_(False)
# scheduler = NoiseScheduler()

# with torch.no_grad():
#     shape = (10, 20, 10)
#     actions = scheduler.sample(diffusion_policy, shape, visual_obs.to(device).to(torch.float32), non_visual_obs.to(device).to(torch.float32), device, num_random_samples=20)

# cfg = TrainConfig()
# q_value_network=  QTransformer(
#     d_vis=cfg.d_vis,
#     d_nonvis=cfg.d_nonvis,
#     d_act=cfg.d_act,
#     d_model=cfg.d_model,
#     n_heads=cfg.n_heads,
#     n_layers=cfg.n_layers,
#     dropout=cfg.dropout,
#     hist_len=cfg.hist_len,
#     horizon=cfg.horizon,
# ).to(cfg.device)
# ckpt = torch.load('/home/user/siddiquieu1/HRL-Usama/mobile-manipulation/ahmed_checkpoints/ckpt_step_4500.pt', map_location="cpu")
# q_value_network.load_state_dict(ckpt["q_state_dict"])   
# q_value_network = q_value_network.to(device)
# q_value_network.eval()

# print("Visual Obs: ", visual_obs.shape)
# print("Non-Visual Obs: ", non_visual_obs.shape)
# print("Sampled Actions: ", actions.shape)

# with torch.no_grad():
#     num_trajectories=actions.shape[0]

#     vo_input = visual_obs.repeat(num_trajectories,1,1).to(device).to(torch.float32)
#     nvo_input = non_visual_obs.repeat(num_trajectories,1,1).to(device).to(torch.float32)
#     a_input = actions.to(device).to(torch.float32)

#     print(vo_input.shape, nvo_input.shape, a_input.shape)

#     # vo_input = vo_input[0].reshape(1, vo_input.shape[1], vo_input.shape[2])
#     # nvo_input = nvo_input[0].reshape(1, nvo_input.shape[1], nvo_input.shape[2])
#     # a_input = a_input[0].reshape(1, a_input.shape[1], a_input.shape[2])
#     # print(vo_input.shape, nvo_input.shape, a_input.shape)

#     q_values = q_value_network(vo_input, nvo_input, a_input)
#     print("Q Values: ", q_values)
