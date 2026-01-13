# TODO:
# Create dataset object (equalent to d4rl) - Done
# Load behaviour model - Done
# Generate fake action sequences
# Load Q Model
# Train loop

import os
from os import path
import torch
from typing import Optional, List
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

# from mobile_manipulation.ppo.trainers.ppo_trainer_v0 import ConditionalDiffusionModel, NoiseScheduler, TrainConfig, QTransformer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataclasses import dataclass
from torch import nn
import math

import gc


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    d_vis: int = 512
    d_nonvis: int = 21
    d_act: int = 10

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.1

    hist_len: int = 5
    horizon: int = 20

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_epochs: int = 50

    gamma: float = 0.99
    num_action_samples: int = 8
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

        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.q_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        vis_hist: torch.Tensor,
        nonvis_hist: torch.Tensor,
        act_seq: torch.Tensor,
    ) -> torch.Tensor:
        B = vis_hist.shape[0]
        vis_tok = self.vis_proj(vis_hist)
        nonvis_tok = self.nonvis_proj(nonvis_hist)

        state_tokens = torch.stack([vis_tok, nonvis_tok], dim=2)
        state_tokens = state_tokens.view(B, 2 * self.hist_len, -1)

        act_tokens = self.act_proj(act_seq)

        x = torch.cat([state_tokens, act_tokens], dim=1)
        x = x + self.pos_emb
        x = self.dropout(x)

        h = self.encoder(x)
        pooled = h.mean(dim=1)
        q = self.q_head(pooled).squeeze(-1)
        return q


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2)

        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # Apply cos to odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer(
            "pe", pe
        )  # Not a parameter, but saved with the model

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encoding added
        """
        return x + self.pe[:, : x.size(1), :].to(x.device)


# Sinusoidal Timestep Embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device)
            * -(torch.log(torch.tensor(10000.0)) / half_dim)
        )
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# Cross-Attention Block
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

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

        q = rearrange(q, "b n_decoder (h d) -> b h n_decoder d", h=h)
        k = rearrange(k, "b n_context (h d) -> b h n_context d", h=h)
        v = rearrange(v, "b n_context (h d) -> b h n_context d", h=h)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn_scores.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n_decoder d -> b n_decoder (h d)")
        return self.to_out(out)


# Transformer Block with Cross Attention
class DiffusionTransformerBlock(nn.Module):
    def __init__(self, dim, cond_dim, heads=8, dim_head=128):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, batch_first=True, dim_feedforward=256
        )
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
    def __init__(
        self,
        action_dim=10,
        output_dim=10,
        sensor_dim=21,
        depth_features_dim=512,
        hidden_dim=256,
        num_layers=2,
    ):
        super().__init__()

        #    self.visual_feature_extractor = Feat_ext
        self.action_input_proj = nn.Linear(action_dim, hidden_dim)
        self.visual_obs_projection = nn.Linear(depth_features_dim, hidden_dim)
        self.non_visual_obs_projection = nn.Linear(sensor_dim, hidden_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
        )
        #  self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(hidden_dim, action_dim)

        self.decoder_position_embedding = SinusoidalPositionalEncoding(
            hidden_dim, max_len=21
        )  # Action position embedding
        self.encoder_position_embedding = SinusoidalPositionalEncoding(
            hidden_dim, max_len=21
        )  # Sensor position embedding

    def forward(self, visual_obs, non_visual_obs, noisy_action, t):

        batch_size = non_visual_obs.shape[0]
        context_length = non_visual_obs.shape[1]

        noisy_action = self.action_input_proj(noisy_action.to(torch.float32))

        visual_obs = self.visual_obs_projection(visual_obs.to(torch.float32))
        # print("shape after visual feature projection == " , visual_obs.shape )
        visual_obs = visual_obs.reshape(batch_size, context_length, -1)

        # print("shape after reshaping back to batch and context length == " , visual_obs.shape )

        # print("initial non visual observations shape == " , non_visual_obs.shape)
        non_visual_obs = self.non_visual_obs_projection(
            non_visual_obs.to(torch.float32)
        )
        #  print("shape after reshaping back to batch and context length == " , non_visual_obs.shape )

        t = self.time_mlp(t.to(torch.float32))  # Time embedding
        t = t.unsqueeze(1)
        t = t.repeat(
            batch_size, 1, 1
        )  # Repeat to match action sequence length
        #   print("t shape after embedding == " , t.shape)

        encoder_input = torch.cat((visual_obs, non_visual_obs, t), dim=1)
        # print("encoder input shape == " , encoder_input.shape)
        encoder_input = self.encoder_position_embedding(
            encoder_input
        )  # Apply sensor position embedding

        decoder_input = torch.cat((t, noisy_action), dim=1)
        # print("decoder input shape == " , decoder_input.shape)
        decoder_input = self.decoder_position_embedding(
            decoder_input
        )  # Apply action position embedding

        for block in self.transformer_blocks:
            decoder_input = block(decoder_input, encoder_input)
        out = self.output_proj(decoder_input)
        out = out[:, 1:, :]
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
        t = torch.reshape(t, (x_start.shape[0], -1))  # Ensure t is a 1D tensor
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().unsqueeze(1)
        sqrt_one_minus_alpha_cumprod = (
            (1.0 - self.alpha_cumprod[t]).sqrt().unsqueeze(1)
        )
        x_new = (
            sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
        )
        return (
            sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
        )

    def get_loss(
        self, model, x_start, t, visual_obs_batch, non_visual_obs_batch
    ):
        noise = torch.randn_like(x_start).to(torch.float32)
        noisy_action = self.q_sample(x_start, t, noise)
        predicted_noise = model(
            visual_obs_batch.to(torch.float32),
            non_visual_obs_batch.to(torch.float32),
            noisy_action.to(torch.float32),
            t.to(torch.float32),
        )
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, noisy_action, t, visual_obs, non_visual_obs):

        t.to(device)
        # Predict noise using the model

        predicted_noise = model(visual_obs, non_visual_obs, noisy_action, t)

        #  print("noisy action shape == " , noisy_action_decoder.shape)
        # print("predicted noise shape == " , predicted_noise.shape)
        # Extract coefficients for denoising
        alpha_t = self.alphas[t.to(device).to(torch.long)]  # .view(-1,1, 1)
        # alpha_t= alpha_t.repeat(10,20,1)
        # print("alpha t shape == " , alpha_t.shape)
        alpha_cumprod_t = self.alpha_cumprod[
            t.to(device).to(torch.long)
        ]  # .view(-1, 1)
        beta_t = self.betas[t.to(device).to(torch.long)]  # .view(-1, 1)

        # Compute mean of reverse process
        sqrt_alpha_t = torch.sqrt(alpha_t).to(device)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

        # Compute mean
        pred_mean = (
            noisy_action
            - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
        ) / sqrt_alpha_t
        #  print("predicted mean shape == " , pred_mean.shape)
        # Add noise for all timesteps except t=0
        if t.min() > 0:
            noise = torch.randn_like(noisy_action)
            pred_mean = pred_mean + torch.sqrt(beta_t) * noise

        return pred_mean

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        visual_obs,
        non_visual_obs,
        device,
        num_random_samples=20,
    ):
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
        visual_obs = visual_obs.repeat(
            shape[0], 1, 1
        )  # Repeat condition for batch size
        non_visual_obs = non_visual_obs.repeat(shape[0], 1, 1)

        for t in reversed(range(self.timesteps)):
            # Create timestep tensor
            t_tensor = (
                torch.tensor([t]).to(device).to(torch.float32)
            )  # torch.full((shape[0],), t, device=device, dtype=torch.long)
            noisy_action_decoder = self.p_sample(
                model,
                noisy_action_decoder,
                t_tensor,
                visual_obs,
                non_visual_obs,
            )
        return noisy_action_decoder


device = torch.device(
    "cuda" if torch.cuda.is_available() else torch.device("cpu")
)

ROOT = "/lustre/mlnvme/data/s47ashok_hpc-data/new_accurate_data_individual_tasks_4_jan_2026/processed_data"
Diffusion_model_path = "/lustre/mlnvme/data/s47ashok_hpc-data/new_accurate_data_individual_tasks_4_jan_2026/weights_pretrained_visual_encoder"
Critic_model_path = "/lustre/mlnvme/data/s47ashok_hpc-data/new_accurate_data_individual_tasks_4_jan_2026/critic_weights"


MAX_BZ_SIZE = 1024
soft_Q_update = True

Discount_factor = 0.98

Batch_size = 128

Total_num_of_training_epcohs = 10000

number_of_epochs_to_update_returns = 100


def _mem_usage(tensor):
    bytes_used = tensor.numel() * tensor.element_size()
    return f"{bytes_used / 1024**3:.2f} GB"


def _load_and_concat_prealloc(
    file_paths: List[str],
    *,
    map_location: torch.device,
) -> torch.Tensor:
    """Load many tensors from disk and concatenate along dim=0 with low peak RAM.

    This avoids holding all partitions in memory at once by:
    1) First pass: load one-by-one to compute total length and validate shapes.
    2) Allocate the final output tensor.
    3) Second pass: load one-by-one and copy into the preallocated output.
    """

    if len(file_paths) == 0:
        raise ValueError("No file paths provided")

    # Load first tensor to establish dtype/shape.
    first = torch.load(file_paths[0], map_location=map_location)
    if not torch.is_tensor(first):
        raise TypeError(
            f"Expected a Tensor in {file_paths[0]}, got {type(first)!r}"
        )

    rest_shape = tuple(first.shape[1:])
    total = int(first.shape[0])

    # First pass: determine total length.
    for fp in file_paths[1:]:
        t = torch.load(fp, map_location=map_location)
        if not torch.is_tensor(t):
            raise TypeError(f"Expected a Tensor in {fp}, got {type(t)!r}")
        if tuple(t.shape[1:]) != rest_shape:
            raise ValueError(
                f"Shape mismatch for {fp}: got {tuple(t.shape)}, expected (*, {rest_shape})"
            )
        total += int(t.shape[0])
        del t
        gc.collect()

    out = torch.empty((total, *rest_shape), dtype=first.dtype, device=first.device)
    offset = 0
    out[offset : offset + first.shape[0]].copy_(first)
    offset += int(first.shape[0])
    del first
    gc.collect()

    # Second pass: fill the output.
    for fp in file_paths[1:]:
        t = torch.load(fp, map_location=map_location)
        out[offset : offset + t.shape[0]].copy_(t)
        offset += int(t.shape[0])
        del t
        gc.collect()

    return out


class Diffusion_buffer(Dataset):

    def __init__(self):
        self.normalise_return = True

        data = self._load_data()
        self.actions = data["actions"]
        self.visual_states = data["visual_states"]
        self.non_visual_states = data["non_visual_states"]
        self.rewards = data["rewards"]
        self.done = data["done"]
        self.next_visual_states = data["next_visual_states"]
        self.next_non_visual_states = data["next_non_visual_states"]
        self.fake_actions = data["fake_actions"]

        self.returns = data["returns"]
        data = []
        self.raw_returns = [self.returns.copy()]
        self.raw_values = []
        self.returns_mean = np.mean(self.returns)
        self.returns_std = np.maximum(np.std(self.returns), 0.1)
        print(
            "returns mean {}  std {}".format(
                self.returns_mean, self.returns_std
            )
        )
        if self.normalise_return:
            self.returns = (
                self.returns - self.returns_mean
            ) / self.returns_std
            # data["returns"] = returns
            print(
                "returns normalised at mean {}, std {}".format(
                    self.returns_mean, self.returns_std
                )
            )
        else:
            print("no normal")

        self.len = self.visual_states.shape[0]
        # make sure same number of data points exist in all tasks
        # self.fake_len = int(np.maximum(np.round(10000 / self.len), 1)) * self.len
        # print(self.len, "data loaded", self.fake_len, "data faked")

    def __getitem__(self, index):
        i = index % self.len
        actions = self.actions[i]
        rewards = self.rewards[i]
        visual_states = self.visual_states[i]
        non_visual_states = self.non_visual_states[i]
        done = self.done[i]
        next_visual_states = self.next_visual_states[i]
        next_non_visual_states = self.next_non_visual_states[i]
        fake_actions = self.fake_actions[i]
        returns = self.returns[i]
        return (
            actions,
            rewards,
            visual_states,
            non_visual_states,
            done,
            next_visual_states,
            next_non_visual_states,
            fake_actions,
            returns,
        )

    def __len__(self):
        return self.len

    def _load_data(self):
        data = {}
        # Define task types and their partition counts
        tasks = [
            ("nav_task", 10),
            ("pick_task", 10),
            ("place_task", 9),
            # ("nav_task", 1),
            # ("pick_task", 1),
            # ("place_task", 1),
        ]

        # Define data types and their file prefixes
        data_types = {
            "actions": "action_data",
            "done": "done_data",
            "non_visual_states": "non_visual_obs_data",
            "visual_states": "visual_obs_data",
            "rewards": "rewards_data",
            "next_visual_states": "next_visual_obs_data",
            "next_non_visual_states": "next_non_visual_obs_data",
            "fake_actions": "predicted_actions_diffusion",
        }

        # Load all data without keeping every partition tensor in memory at once.
        map_location = torch.device("cpu")
        for data_key, file_prefix in data_types.items():
            file_paths: List[str] = []
            for task_name, num_partitions in tasks:
                for p in range(1, num_partitions + 1):
                    print("Queueing partition", p, task_name)
                    file_name = f"{file_prefix}_{task_name}_p_{p}.pt"
                    # file_name = f"{file_prefix}_{task_name}_p_1.pt"
                    file_paths.append(path.join(ROOT, file_name))

            print(f"Loading+concatenating {data_key} from {len(file_paths)} partitions...")
            data[data_key] = _load_and_concat_prealloc(
                file_paths, map_location=map_location
            )

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
            "Visual Obs",
            data["visual_states"].shape,
            data["visual_states"].device,
            _mem_usage(data["visual_states"]),
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
            "Next Visual Obs",
            data["next_visual_states"].shape,
            data["next_visual_states"].device,
            _mem_usage(data["next_visual_states"]),
        )
        print(
            "Fake Actions",
            data["fake_actions"].shape,
            data["fake_actions"].device,
            _mem_usage(data["fake_actions"]),
        )

        print(torch.cuda.memory_allocated() / 1024**2, "MB")

        input()
        # exit()

        data["rewards"] = data["rewards"].squeeze()
        data["done"] = data["done"].squeeze()

        assert data["done"][-1]
        data["returns"] = np.zeros((data["visual_states"].shape[0], 1))

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

        assert self.visual_states.shape[0] == self.fake_actions.shape[0]
        qs = []

        for i in range(0, self.len):
            # visual_states.repeat(num_trajectories,1,1).to(device).to(torch.float32)

            fake_actions = self.fake_actions[i]

            num_trajectories = fake_actions.shape[0]

            next_non_visual_states = self.next_non_visual_states[i].repeat(
                num_trajectories, 1, 1
            )
            next_visual_states = self.next_visual_states[i].repeat(
                num_trajectories, 1, 1
            )

            # non_visual_states = torch.repeat_interleave(non_visual_states, fake_actions.shape[1], dim=0)
            # visual_states = torch.repeat_interleave(visual_states, fake_actions.shape[1], dim=0)

            with torch.no_grad():
                q = score_model.calculateQ(
                    next_visual_states,
                    next_non_visual_states,
                    torch.FloatTensor(fake_actions),
                )
                print(
                    "Raw Q values of the generated actions shape == ", q.shape
                )
                qs.append(q.cpu().numpy())

        # for states, actions in tqdm.tqdm(zip(np.array_split(self.visual_states, self.visual_states.shape[0] // 128 + 1), np.array_split(self.fake_actions, self.visual_states.shape[0] // 128 + 1))):
        #     with torch.no_grad():
        #         states = torch.FloatTensor(states).to("cuda")
        #         actions = torch.FloatTensor(actions).to("cuda")
        #         states = torch.repeat_interleave(states, actions.shape[1], dim=0)
        #         # TODO: We need to use our Q model here which needs both visual and non visual states
        #         q = score_model.calculateQ(states, actions.reshape((states.shape[0], actions.shape[-1])))
        #         q = q.reshape((actions.shape[0], actions.shape[1]))
        #         qs.append(q.cpu().numpy())

        values = np.array(qs)

        self.raw_values.append(values)
        if soft_Q_update:
            values = np.sum(
                softmax(20 * values, axis=-1) * values, axis=-1, keepdims=1
            )
        else:
            values = np.percentile(values, 85, axis=-1, keepdims=1)
        if self.normalise_return:
            values = values * self.returns_std + self.returns_mean
        assert values.ndim == 2
        assert values.shape[0] == self.non_visual_states.shape[0]
        returns = torch.zeros_like(values)
        last = 0
        num_truncated_traj = 0
        for i in range(returns.shape[0] - 1, -1, -1):
            bootstrap = self.rewards[i] + Discount_factor * last * (
                1.0 - self.done[i]
            )
            imagainary = values[i, 0]
            if bootstrap > imagainary:
                returns[i, 0] = bootstrap
            else:
                returns[i, 0] = imagainary
                num_truncated_traj += 1
            last = returns[i, 0]
        print("num_truncated_traj perc", num_truncated_traj / returns.shape[0])
        # self.raw_returns.append(returns)
        self.raw_returns = returns.copy()
        self.returns_mean = np.mean(returns)
        self.returns_std = np.maximum(np.std(returns), 0.1)
        print(
            "returns mean {}  std {}".format(
                self.returns_mean, self.returns_std
            )
        )
        if self.normalise_return:
            returns = (returns - self.returns_mean) / self.returns_std
            print(
                "returns normalised at mean {}, std {}".format(
                    self.returns_mean, self.returns_std
                )
            )
        else:
            print("no normal")

        self.returns = returns.copy()

        returns = None

        # self.ys = np.concatenate([returns, self.actions], axis=-1)
        # self.ys = self.ys.astype(np.float32)

        # self.rewards = torch.FloatTensor(returns.squeeze())

        print("update returns finished")


class ModelWrapper:
    def __init__(self):
        self.diffusion_policy = ConditionalDiffusionModel()
        self.diffusion_policy.load_state_dict(torch.load(path.join(Diffusion_model_path,"pretrained_visual_encoder_2200.pt"),
                                                            map_location=device))

        self.diffusion_policy.to(device)
        self.diffusion_policy.eval()
        for p in self.diffusion_policy.parameters():
            p.requires_grad_(False)
        self.scheduler = NoiseScheduler()

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
        self.q_value_network.eval()

    def sample(self, visual_obs, non_visual_obs):
        # print(visual_obs.shape)
        # print(non_visual_obs.shape)
        # exit()

        shape = (20, 20, 10)
        with torch.no_grad():
            actions = self.scheduler.sample(
                self.diffusion_policy,
                shape,
                visual_obs.to(device).to(torch.float32),
                non_visual_obs.to(device).to(torch.float32),
                device,
                num_random_samples=20,
            )

        return actions

    def calculateQ(self, visual_states, non_visual_states, actions):
        # num_trajectories = actions.shape[0]

        # print(visual_states.shape, non_visual_states.shape, actions.shape)

        # vo_input = visual_states.repeat(num_trajectories,1,1).to(device).to(torch.float32)
        # nvo_input = non_visual_states.repeat(num_trajectories,1,1).to(device).to(torch.float32)
        # a_input = actions.to(device).to(torch.float32)

        # print(vo_input.shape, nvo_input.shape, a_input.shape)
        # exit()

        print(
            "Non visual states in claculate Q function == ",
            non_visual_states.shape,
        )
        print("Visual states in claculate Q function == ", visual_states.shape)
        print(
            "Generated actions by diffusion in claculate Q function",
            actions.shape,
        )

        """
        16 set of actions of length 20

        Non visual states torch.Size([16, 5, 21])
        Visual states torch.Size([16, 5, 512])
        Actions torch.Size([16, 20, 10])
        Q Values torch.Size([16])
        """

        q_values = self.q_value_network(
            visual_states.to(device).to(torch.float32),
            non_visual_states.to(device).to(torch.float32),
            actions.to(device).to(torch.float32),
        )

        print("Q Values", q_values.shape)

        # exit()

        return q_values


def train_critic(score_model, data_loader):
    # data_loader.dataset.update_returns(score_model)

    optimizer = Adam(score_model.q_value_network.parameters(), lr=1e-3)

    bk_model_sd = copy.deepcopy(score_model.q_value_network.state_dict())

    for epoch in range(Total_num_of_training_epcohs):
        gc.collect()
        torch.cuda.empty_cache()

        avg_loss = 0.0
        num_items = 0
        for batch in tqdm.tqdm(data_loader):
            (
                actions,
                rewards,
                visual_states,
                non_visual_states,
                done,
                next_visual_states,
                next_non_visual_states,
                fake_actions,
                returns,
            ) = batch
            returns = returns.to(device)

            qs = score_model.calculateQ(
                visual_states, non_visual_states, actions
            )
            loss = torch.mean((qs - returns) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # score_model.condition = None
            avg_loss += loss.item() * actions.shape[0]
            num_items += actions.shape[0]
        if epoch != 0 and epoch % number_of_epochs_to_update_returns == 0:
            print("Average Loss for epoch {} == {}".format(epoch, avg_loss))

            data_loader.dataset.update_returns(score_model)

            ## save model
            torch.save(
                {
                    "q_state_dict": score_model.q_value_network.state_dict(),
                },
                path.join(Critic_model_path, f"ckpt_step_{epoch}.pt"),
            )

            score_model.q_value_network.load_state_dict(bk_model_sd)
            optimizer = Adam(score_model.q_value_network.parameters(), lr=1e-3)


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
