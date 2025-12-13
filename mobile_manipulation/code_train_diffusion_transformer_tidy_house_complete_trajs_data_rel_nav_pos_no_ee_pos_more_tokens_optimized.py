import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from einops import rearrange

import math

from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import datetime
import random

from typing import Dict, Tuple, Optional
import pickle

# ----------------------------
# 0. Configuration / Globals
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device == ", device)

#directory='/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data'
directory='/lustre/mlnvme/data/s47ashok_hpc-data/complete_trajs_diffusion/new_dataset_22_nov_tidy_house/more_data_with_rel_nav_pos_and_arm_depth'

Batch_size=64

def get_filenames_in_directory(directory):
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filenames.append(os.path.join(directory, filename))
    return filenames


class Flatten(nn.Module):
    r"""Copied from 'https://github.com/tuomaso/romi-cnn-
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


class SimpleCNN(nn.Sequential):
    """
    Copied from 'https://github.com/tuomaso/romi-cnn-
    """
    def __init__(
        self, in_channels, input_shape, out_channels
    ):
        self.input_shape = input_shape

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


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 ==0 :
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension for broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is shape (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class TimeEncoding(nn.Module):
    def __init__(self, time_dim):
        super(TimeEncoding, self).__init__()
        self.linear1 = nn.Linear(1, time_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(time_dim, time_dim)

    def forward(self, t):
        t = t.unsqueeze(-1)  # Make it (batch_size, 1) to pass through MLP
        t = self.linear1(t)
        t = self.relu(t)
        t = self.linear2(t)
      #  print("time_embedding shape == ", t.shape)
        return t

# ----------------------------
# 1. Transformer-based Model
# ----------------------------
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, 
                 visual_dim=256, 
                 rel_nav_pos_dim=5, 
                 qpos_dim=7, 
                 rel_resting_pos_dim=3,
                 gripper_state_dim=1,
                 action_dim=8,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=512,
                 dropout=0.1):
        super(ConditionalDiffusionModel, self).__init__()
        
        self.visual_dim = visual_dim
        self.rel_pick_pos_dim = rel_nav_pos_dim
        self.rel_place_pos_dim = rel_nav_pos_dim
        self.qpos_dim = qpos_dim
        self.rel_resting_pos_dim = rel_resting_pos_dim
        self.gripper_state_dim = gripper_state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Input projections to d_model
       # self.visual_proj = nn.Linear(visual_dim, d_model)
        self.rel_pick_pos_proj = nn.Linear(rel_nav_pos_dim, d_model)
        self.rel_place_pos_proj = nn.Linear(rel_nav_pos_dim, d_model)
        self.qpos_proj = nn.Linear(qpos_dim, d_model)
        self.rel_resting_pos_proj = nn.Linear(rel_resting_pos_dim, d_model)
        self.gripper_state_proj = nn.Linear(gripper_state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        
        # Positional Encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model)
        self.time_mlp = TimeEncoding(d_model)
      #  self.time_mlp = nn.Sequential(
       #     nn.Linear(d_model, dim_feedforward),
        #    nn.ReLU(),
         #   nn.Linear(dim_feedforward, d_model),
        #)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Final prediction layer for noise on action
        self.output_layer = nn.Linear(d_model, action_dim)

    def encode_visual_features(self, visual_obs):
        original_shape = visual_obs.shape
        #print("Original visual_obs shape:", original_shape)

        if len(original_shape) == 5:  # (batch, time, H, W, C) or (batch, time, C, H, W)
            if original_shape[-1] == 1:  # Assuming last dim is channel
                visual_obs = visual_obs.squeeze(-1)
            visual_obs = visual_obs.reshape(-1, original_shape[2], original_shape[3])
        elif len(original_shape) == 4:  # (batch, H, W, C)
            if original_shape[-1] == 1:
                visual_obs = visual_obs.squeeze(-1)
        elif len(original_shape) == 3:  # Already (batch, H, W)
            pass
        else:
            raise ValueError(f"Unexpected visual_obs shape: {original_shape}")

        #print("Visual obs shape before permute: ",visual_obs.shape)
        if visual_obs.dim() == 3:
            visual_obs = visual_obs.unsqueeze(1)  # Add channel dimension
      #  print("visual_obs.shape == " , visual_obs.shape)
        #visual_obs = rearrange(visual_obs, "T B H W -> B T H W") 
      #  print("visual_obs.shape == " , visual_obs.shape)
        visual_obs = visual_obs.to(torch.float32).to(device)
        conv_input_shape = visual_obs.shape[-2:]
        temp= SimpleCNN(1, conv_input_shape, 256).to(device).to(torch.float32)
        x = Feat_ext(visual_obs)
     #   print("feature map shape == ", x.shape)
        return x

    def forward(self, visual_obs, rel_pick_pos, rel_place_pos, qpos, rel_resting_pos, gripper_state, noisy_action, t):
        batch_size, seq_len, _ = noisy_action.shape

        # Encode inputs to d_model
        visual_features = self.encode_visual_features(visual_obs)
        visual_features = visual_features.unsqueeze(1).expand(-1, seq_len, -1)

        rel_pick_pos = self.rel_pick_pos_proj(rel_pick_pos)
        rel_place_pos = self.rel_place_pos_proj(rel_place_pos)
        qpos = self.qpos_proj(qpos)
        rel_resting_pos = self.rel_resting_pos_proj(rel_resting_pos)
      #  for i in range(gripper_state.shape[0]):
       #     for j in range(gripper_state.shape[1]):
        #        if gripper_state[i,j]:
         #           gripper_state[i,j]=0
          #      else:
           #         gripper_state[i,j]=-1
        gripper_state = gripper_state.unsqueeze(-1)  # Expand from (B, seq_len) to (B, seq_len, 1)
        gripper_state = self.gripper_state_proj(gripper_state)
        noisy_action = self.action_proj(noisy_action)

        # Stack context along feature dimension
        context = visual_features + rel_pick_pos + rel_place_pos + qpos + rel_resting_pos + gripper_state + noisy_action
       # context = torch.cat((rel_nav_pos , qpos , rel_resting_pos), dim=2)
        # Time embedding
        if t.dim() == 1:
            t = t.view(-1)
        t_emb = self.time_mlp(t.to(torch.float32))
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Add time embedding to context
        context = context + t_emb

        # Add positional encoding
        context = self.positional_encoding(context)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(context)

        # Predict noise for actions
        predicted_noise = self.output_layer(transformer_output)
        return predicted_noise

# Noise Scheduler (like DDPM)
class NoiseScheduler:
    """
    Simple DDPM-like noise scheduler with precomputed square roots for speed.
    """
    def __init__(self, timesteps: int = 500, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        # Precompute sqrt terms to avoid repeated sqrt calls
        self.alpha_cumprod = alpha_cumprod
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Diffusion forward process q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        # t is (B,), x_start is (B, T, D)
        t = t.long()
        if t.dim() != 1:
            t = t.view(-1)

        sqrt_ac = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)

        return sqrt_ac * x_start + sqrt_om * noise

    def get_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        visual_obs_batch: torch.Tensor,
        rel_pick_pos_batch: torch.Tensor,
        rel_place_pos_batch: torch.Tensor,
        qpos_batch: torch.Tensor,
        rel_resting_pos_batch: torch.Tensor,
        gripper_state_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE between predicted and true noise."""
        noise = torch.randn_like(x_start)
        noisy_action = self.q_sample(x_start, t, noise)

        predicted_noise = model(
            visual_obs_batch,
            rel_pick_pos_batch,
            rel_place_pos_batch,
            qpos_batch,
            rel_resting_pos_batch,
            gripper_state_batch,
            noisy_action,
            t,
        )
        return F.mse_loss(predicted_noise, noise)


def process_episode_data(episode_data, num_prev_obs: int, num_predicted_actions: int):
    """
    Vectorized preprocessing for a single episode.
    Builds sliding windows of observations and future actions.

    Returns:
        visual_obs:      (N, K, ...)      depth observations
        rel_nav_pos:     (N, K, 3)
        qpos:            (N, K, 7)
        rel_resting_pos: (N, K, 3)
        gripper_state:   (N, K)
        actions:         (N, M, A)
    where:
        N = number of training samples per episode
        K = num_prev_obs
        M = num_predicted_actions
    """
    try:
        # Required fields
        depth = np.asarray(episode_data['robot_head_depth'])
        is_holding = np.asarray(episode_data['is_holding'])
        rel_place = np.asarray(np.concatenate((episode_data['rel_place_pos_base_polar'], episode_data['rel_place_pos_ee']), axis=-1))
        rel_pick = np.asarray(np.concatenate((episode_data['rel_pick_pos_base_polar'], episode_data['rel_pick_pos_ee']), axis=-1))

        qpos = np.asarray(episode_data['rob_qpos'])
        rel_rest = np.asarray(episode_data['rel_resting_pos'])
        actions = np.asarray(episode_data['action_to_save'])
    except KeyError as e:
        # If any key is missing, skip this episode
        print(f"[WARN] Missing key {e} in episode, skipping episode.")
        return (
            np.empty((0,) + depth.shape[1:], dtype=depth.dtype if 'depth' in locals() else np.float32),
            np.empty((0, num_prev_obs, 3), dtype=np.float32),
            np.empty((0, num_prev_obs, 7), dtype=np.float32),
            np.empty((0, num_prev_obs, 3), dtype=np.float32),
            np.empty((0, num_prev_obs), dtype=np.float32),
            np.empty((0, num_predicted_actions, actions.shape[-1] if 'actions' in locals() else 1), dtype=np.float32),
        )

    T = actions.shape[0]
    K = num_prev_obs
    M = num_predicted_actions

    if T <= K:
        # Not enough steps to form one window
        return (
            np.empty((0,) + depth.shape[1:], dtype=depth.dtype),
            np.empty((0, K, rel_place.shape[-1]), dtype=rel_place.dtype),
            np.empty((0, K, qpos.shape[-1]), dtype=qpos.dtype),
            np.empty((0, K, rel_rest.shape[-1]), dtype=rel_rest.dtype),
            np.empty((0, K), dtype=is_holding.dtype),
            np.empty((0, M, actions.shape[-1]), dtype=actions.dtype),
        )

    # Extend actions with last action repeated M times for prediction horizon
    last_action = actions[-1]
    last_repeated = np.repeat(last_action[None, :], M, axis=0)
    actions_ext = np.concatenate([actions, last_repeated], axis=0)  # (T+M, A)

    # Compute navigation position depending on is_holding
   # nav_pos = np.where(is_holding[:, None], rel_place, rel_pick)  # (T, 3)

    num_samples = T - K

    # Sliding windows over time dimension (axis=0)
    try:
        vis_windows = sliding_window_view(depth, window_shape=K, axis=0)   # (T-K+1, K, ...)
        pick_windows = sliding_window_view(rel_pick, window_shape=K, axis=0) # (T-K+1, K, 3)
        place_windows = sliding_window_view(rel_place, window_shape=K, axis=0) # (T-K+1, K, 3)
        qpos_windows = sliding_window_view(qpos, window_shape=K, axis=0)   # (T-K+1, K, 7)
        rest_windows = sliding_window_view(rel_rest, window_shape=K, axis=0)  # (T-K+1, K, 3)
        grip_windows = sliding_window_view(is_holding, window_shape=K, axis=0)  # (T-K+1, K)
    except ValueError as e:
        # Any shape mismatch or bad windowing -> skip episode
        print(f"[WARN] Sliding window error in episode: {e}. Skipping episode.")
        return (
            np.empty((0,) + depth.shape[1:], dtype=depth.dtype),
            np.empty((0, K, rel_pick.shape[-1]), dtype=nav_pos.dtype),
            np.empty((0, K, qpos.shape[-1]), dtype=qpos.dtype),
            np.empty((0, K, rel_rest.shape[-1]), dtype=rel_rest.dtype),
            np.empty((0, K), dtype=is_holding.dtype),
            np.empty((0, M, actions.shape[-1]), dtype=actions.dtype),
        )

    vis_windows = vis_windows[:num_samples]
    pick_windows = pick_windows[:num_samples]
    place_windows = place_windows[:num_samples]
    qpos_windows = qpos_windows[:num_samples]
    rest_windows = rest_windows[:num_samples]
    grip_windows = grip_windows[:num_samples]

    # Action windows: shape (T+1, M, A) then select aligned with samples
    act_windows = sliding_window_view(actions_ext, window_shape=M, axis=0)  # (T+1, M, A)
    act_windows = act_windows[K:K + num_samples]  # (num_samples, M, A)

    return vis_windows, pick_windows, place_windows, qpos_windows, rest_windows, grip_windows, act_windows

def train_diffusion_model(upload_directory, save_directory, num_prev_obs: int = 5, num_predicted_actions: int = 20):
    """Main training loop for the conditional diffusion model.

    This version:
      * Uses vectorized episode preprocessing (process_episode_data).
      * Concatenates all episodes in a file into single arrays.
      * Wraps everything in a TensorDataset + DataLoader.
      * Adds basic error handling so a bad episode/file does not crash training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    number_of_epochs = 1000000

    print("device ==", device)

    file_names = get_filenames_in_directory(upload_directory)
    if not file_names:
        raise RuntimeError(f"No .pkl files found in directory: {upload_directory}")

    model = ConditionalDiffusionModel().to(device)
    model.to(device)
    scheduler = NoiseScheduler()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ----------------------------
    # 7. TensorBoard Logging Setup
    # ----------------------------
    log_dir = os.path.join(
        save_directory,
        "runs_diffusion_complete_trajs_cnn_encoder_scratch",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "trained_from_scratch_cnn_encoder",
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    os.makedirs(save_directory, exist_ok=True)

    for epoch in range(number_of_epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        current_file_idx = 0

        for file_name in file_names:
            print(f"[INFO] Epoch {epoch + 1}: processing file {file_name}")
            try:
                with open(file_name, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"[ERROR] Could not load file {file_name}: {e}. Skipping file.")
                continue

            vis_list = []
            pick_list = []
            place_list = []
            qpos_list = []
            rest_list = []
            grip_list = []
            act_list = []

            # --- collect all episodes into big arrays ---
            for episode_key, episode_data in data.items():
                try:
                    vis, pick, place , qpos, rest, grip, act = process_episode_data(
                        episode_data, num_prev_obs, num_predicted_actions
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to process episode {episode_key} in {file_name}: {e}. Skipping episode.")
                    continue

                if vis.shape[0] == 0:
                    continue

                vis_list.append(vis)
                pick_list.append(pick)
                place_list.append(place)
                qpos_list.append(qpos)
                rest_list.append(rest)
                grip_list.append(grip)
                act_list.append(act)

            if not vis_list:
                print(f"[WARN] No valid episodes in file {file_name}, skipping file.")
                continue

            try:
                visual_obs_data = np.concatenate(vis_list, axis=0)
                rel_pick_pos_data = np.concatenate(pick_list, axis=0)
                rel_place_pos_data = np.concatenate(place_list, axis=0)
                qpos_data = np.concatenate(qpos_list, axis=0)
                rel_rest_data = np.concatenate(rest_list, axis=0)
                grip_data = np.concatenate(grip_list, axis=0)
                action_data = np.concatenate(act_list, axis=0)
            except ValueError as e:
                print(f"[ERROR] Failed to concatenate episodes in file {file_name}: {e}. Skipping file.")
                continue

            # --- numpy -> torch once ---
            visual_obs_t = torch.from_numpy(visual_obs_data).float()
            rel_pick_pos_t = torch.from_numpy(rel_pick_pos_data).float()
            rel_place_pos_t = torch.from_numpy(rel_place_pos_data).float()
            qpos_t = torch.from_numpy(qpos_data).float()
            rel_rest_t = torch.from_numpy(rel_rest_data).float()
            grip_t = torch.from_numpy(grip_data.astype(np.float32))
            action_t = torch.from_numpy(action_data).float()

            dataset = TensorDataset(
                visual_obs_t,
                rel_pick_pos_t,
                rel_place_pos_t,
                qpos_t,
                rel_rest_t,
                grip_t,
                action_t,
            )

            loader = DataLoader(
                dataset,
                batch_size=Batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=4,   # adjust based on your system
                drop_last=False,
            )

            file_loss = 0.0
            file_samples = 0

            model.train()
            for batch in loader:
                (
                    visual_obs_batch,
                    rel_pick_pos_batch,
                    rel_place_pos_batch,
                    qpos_batch,
                    rel_resting_pos_batch,
                    gripper_state_batch,
                    action_batch,
                ) = batch

                # Move to GPU/CPU
                visual_obs_batch = visual_obs_batch.to(device, non_blocking=True)
                rel_pick_pos_batch = rel_pick_pos_batch.to(device, non_blocking=True)
                rel_place_pos_batch = rel_place_pos_batch.to(device, non_blocking=True)
                qpos_batch = qpos_batch.to(device, non_blocking=True)
                rel_resting_pos_batch = rel_resting_pos_batch.to(device, non_blocking=True)
                gripper_state_batch = gripper_state_batch.to(device, non_blocking=True)
                action_batch = action_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                t = torch.randint(0, scheduler.timesteps, (visual_obs_batch.size(0),), device=device)

                try:
                    loss = scheduler.get_loss(
                        model,
                        action_batch,
                        t,
                        visual_obs_batch,
                        rel_pick_pos_batch,
                        rel_place_pos_batch,
                        qpos_batch,
                        rel_resting_pos_batch,
                        gripper_state_batch,
                    ) * 10.0
                except RuntimeError as e:
                    # Catch CUDA or shape-related errors for this batch only
                    print(f"[ERROR] Training batch failed in file {file_name}: {e}. Skipping batch.")
                    continue

                loss.backward()
                optimizer.step()

                bs = action_batch.size(0)
                file_loss += loss.detach().item() * bs
                file_samples += bs

                epoch_loss += loss.detach().item() * bs
                epoch_samples += bs

            current_file_idx += 1
            if file_samples > 0:
                avg_file_loss = file_loss / file_samples
            else:
                avg_file_loss = float('nan')

            print(
                f"Epoch {epoch + 1} file [{current_file_idx}/{len(file_names)}], "
                f"File: {os.path.basename(file_name)}, Loss: {avg_file_loss:.4f}"
            )

            # Free some memory
            del data, vis_list, pick_list, place_list, qpos_list, rest_list, grip_list, act_list
            del visual_obs_data, rel_pick_pos_data, rel_place_pos_data, qpos_data, rel_rest_data, grip_data, action_data
            del visual_obs_t, rel_pick_pos_t, rel_place_pos_t, qpos_t, rel_rest_t, grip_t, action_t, dataset, loader

        if epoch_samples == 0:
            print(f"[WARN] Epoch {epoch + 1}: no samples processed, skipping logging and checkpoint.")
            continue

        train_loss = epoch_loss / epoch_samples
        print(f"Completed Epoch {epoch + 1}: Train Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/Train', train_loss, epoch)

        # Save every 10 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(save_directory, f'model_{epoch}.pt')
            try:
                torch.save(model.state_dict(), ckpt_path)
                print(f"[INFO] Saved checkpoint to {ckpt_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save checkpoint {ckpt_path}: {e}")


# ----------------------------
# 6. Train/Test Functions (legacy)
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
            total_loss += loss.item()
            total += target.size(0)

    return (total_loss/total)/10 , correct 


#upload_directory='/home/shokry/hab-mobile-manipulation/diffusion_dataset_new/accurate_data_5_aug/all_tasks_corrected_grasped_obs_26_aug'
upload_directory=directory

save_directory=directory+'/weights_diff_transformer_complete_trajs_cnn_encoder_scratch'
#save_directory='/home/shokry/hab-mobile-manipulation/diffusion_dataset_new/accurate_data_5_aug/all_tasks_corrected_grasped_obs_26_aug/weights_diff_transformer'

train_diffusion_model(upload_directory,save_directory)

