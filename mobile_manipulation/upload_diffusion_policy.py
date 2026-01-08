# skill_transformer_loader.py

import os
import copy
import torch
import yaml
from typing import Optional, Dict, Any, Tuple

from gym.spaces import Dict as gymDict

# --- Import your repo's classes (adjust paths if your package layout differs) ---
from habitat.config import Config
from mobile_manipulation.transformer_policy.transformer_policy import TransformerResNetPolicy
from mobile_manipulation.transformer_policy.action_distribution import MixedDistributionNet
import numpy as np
from gym.spaces import Box

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
    def __init__(self, action_dim=10, output_dim=10,sensor_dim=21,depth_features_dim=256, hidden_dim=256, num_layers=2):
        super().__init__()

        self.visual_feature_extractor = Feat_ext
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
        non_visual_obs=self.non_visual_obs_projection(non_visual_obs.to(torch.float32))
        #print("shape after reshaping back to batch and context length == " , non_visual_obs.shape )

        t=self.time_mlp(t.to(torch.float32))  # Time embedding
        t = t.unsqueeze(1)
       # print("t shape after embedding == " , t.shape)

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

    def get_loss(self, model, x_start, t, visual_obs_batch , non_visual_obs_batch): 
        noise = torch.randn_like(x_start).to(torch.float32)
        noisy_action = self.q_sample(x_start, t, noise)
        predicted_noise = model(visual_obs_batch.to(torch.float32), non_visual_obs_batch.to(torch.float32), noisy_action.to(torch.float32), t.to(torch.float32))
        return F.mse_loss(predicted_noise, noise)


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
        visual_obs=visual_obs.repeat(shape[0],1)  # Repeat condition for batch size
        non_visual_obs=non_visual_obs.repeat(shape[0],1) 
        for t in reversed(range(self.timesteps)):
            # Create timestep tensor
            t_tensor = torch.tensor([t]).to(device).to(torch.float32)#torch.full((shape[0],), t, device=device, dtype=torch.long)
            noisy_action_decoder = self.p_sample(model,  noisy_action_decoder, t_tensor, visual_obs , non_visual_obs)
        return noisy_action_decoder

























class DiffusionPolicyLoader:

    def __init__(self):
        self.diffusion_policy = ConditionalDiffusionModel()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
       
        self.diffusion_policy.eval()  # default safe mode
        self.noise_scheduler = NoiseScheduler()

    # ---------- High-level constructors ----------

    @classmethod
    def from_checkpoint(cls,
                        ckpt_path: str):
        """
        Build policy from the checkpoint file.
        - If the checkpoint contains 'config', that will be used unless you pass override_config.
        - If you pass override_config, it will take precedence (useful to tweak batch size, paths, etc.).
        """
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        ckpt = torch.load(ckpt_path)

        policy = TransformerResNetPolicy.from_config(policy_cfg,observation_space,action_space)
        policy.action_distribution = MixedDistributionNet(
                policy.net.output_size,
                policy_cfg.RL.POLICY.ACTION_DIST,
                action_space,
            )
        # 3) Prepare state dict (handle DDP prefixes, CPU↔GPU etc.)
        if "state_dict" not in ckpt:
            raise ValueError("Checkpoint missing 'state_dict'.")

        state_dict = ckpt["state_dict"]
        # handle nested containers (some trainers save under 'model'/'policy')
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if strip_prefix:
            state_dict = _strip_prefix_in_state_dict(state_dict, strip_prefix)

        # 4) Load weights (with strict control)
        missing, unexpected = policy.load_state_dict(state_dict, strict=strict)
        if not strict:
            # Helpful logging for partial compatibility
            if missing:
                print(f"[SkillTransformerPolicyLoader] Missing keys ({len(missing)}):")
                for k in missing[:20]:
                    print("  -", k)
                if len(missing) > 20:
                    print("  ...")

            if unexpected:
                print(f"[SkillTransformerPolicyLoader] Unexpected keys ({len(unexpected)}):")
                for k in unexpected[:20]:
                    print("  -", k)
                if len(unexpected) > 20:
                    print("  ...")

        loader = cls(policy=policy, policy_config=policy_cfg, device=device)
        loader._ckpt_blob = ckpt  # keep a handle if user wants optimizer/scheduler

        return loader

    @classmethod
    def from_config_and_weights(cls,
                                config_like: Any,
                                state_dict: Dict[str, torch.Tensor],
                                device: Optional[str] = None,
                                strict: bool = True,
                                strip_prefix: Optional[str] = "module.") -> "SkillTransformerPolicyLoader":
        """
        Build the architecture from a provided config (yaml/dict/Config) and load a given state_dict.
        """
        cfg = _to_config(config_like)

        policy = TransformerResNetPolicy.from_config(cfg)

        if strip_prefix:
            state_dict = _strip_prefix_in_state_dict(state_dict, strip_prefix)

        missing, unexpected = policy.load_state_dict(state_dict, strict=strict)
        if not strict:
            if missing:
                print(f"[SkillTransformerPolicyLoader] Missing keys ({len(missing)}):")
                for k in missing[:20]:
                    print("  -", k)
                if len(missing) > 20:
                    print("  ...")
            if unexpected:
                print(f"[SkillTransformerPolicyLoader] Unexpected keys ({len(unexpected)}):")
                for k in unexpected[:20]:
                    print("  -", k)
                if len(unexpected) > 20:
                    print("  ...")

        return cls(policy=policy, policy_config=cfg, device=device)

    # ---------- Convenience API ----------

    def to(self, device: str) -> "SkillTransformerPolicyLoader":
        self.device = torch.device(device)
        self.policy = _maybe_move(self.policy, self.device)
        return self

    def eval(self):
        self.policy.eval()

    def train(self):
        self.policy.train()
    '''
    @torch.no_grad()
    def act(self, observations: Dict[str, torch.Tensor], prev_actions: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Convenience wrapper for inference. You can extend this to match your repo's acting API.
        Example: return actions, skill logits, aux logits.
        """
        self.policy.eval()
        # The real acting path in your repo likely has more arguments (rnn state, masks, etc.)
        # Here we only show a minimal example.
        out = self.policy(
            observations=observations,
            rnn_hidden_states=kwargs.get("rnn_hidden_states"),
            prev_actions=prev_actions,
            masks=kwargs.get("masks"),
            rtgs=kwargs.get("rtgs"),
            offline_training=False,
        )
        return out
    '''

    # ---------- (Optional) Optimizer/Scheduler restore ----------

    def build_optimizer(self, opt_ctor, **kwargs) -> torch.optim.Optimizer:
        """
        Build a new optimizer bound to the current policy parameters.
        opt_ctor: e.g., torch.optim.AdamW
        kwargs: lr, weight_decay, betas, etc.
        """
        return opt_ctor(self.policy.parameters(), **kwargs)

    def maybe_load_opt_sched(self,
                             optimizer: Optional[torch.optim.Optimizer] = None,
                             scheduler: Optional[Any] = None) -> Tuple[Optional[torch.optim.Optimizer], Optional[Any]]:
        """
        If the checkpoint contained optimizer/scheduler states, load them.
        """
        ckpt = getattr(self, "_ckpt_blob", None)
        if ckpt is None:
            return optimizer, scheduler

        if optimizer is not None and "optim_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optim_state"])
                print("[SkillTransformerPolicyLoader] Optimizer state loaded.")
            except Exception as e:
                print("[SkillTransformerPolicyLoader] Optimizer state load failed:", e)

        if scheduler is not None and "sched_state" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["sched_state"])
                print("[SkillTransformerPolicyLoader] Scheduler state loaded.")
            except Exception as e:
                print("[SkillTransformerPolicyLoader] Scheduler state load failed:", e)

        return optimizer, scheduler

