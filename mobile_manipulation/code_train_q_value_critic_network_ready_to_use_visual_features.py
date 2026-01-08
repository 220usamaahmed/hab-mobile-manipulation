import os
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# NEW: for prioritized sampling
from torch.utils.data import WeightedRandomSampler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from einops import rearrange


# ============================================================
# NEW: Checkpoint helpers (auto-save "latest" + auto-resume)
# ============================================================

def _atomic_torch_save(obj, path: str):
    """
    NEW: Atomic checkpoint write to avoid corrupted files if job preempts mid-save.
    """
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def _list_ckpts(out_dir: str) -> List[str]:
    """
    NEW: List candidate ckpt files, ignoring temp files.
    """
    if not os.path.isdir(out_dir):
        return []
    files = []
    for fn in os.listdir(out_dir):
        if fn.endswith(".pt") and not fn.endswith(".tmp"):
            files.append(os.path.join(out_dir, fn))
    return files

def _parse_step_from_name(path: str) -> Optional[int]:
    """
    NEW: parse step from ckpt_step_12345.pt
    """
    base = os.path.basename(path)
    m = None
    if base.startswith("ckpt_step_"):
        m = __import__("re").match(r"ckpt_step_(\d+)\.pt$", base)
    if m:
        return int(m.group(1))
    return None

def _parse_epoch_from_name(path: str) -> Optional[int]:
    """
    NEW: parse epoch from ckpt_epoch_12.pt
    """
    base = os.path.basename(path)
    m = None
    if base.startswith("ckpt_epoch_"):
        m = __import__("re").match(r"ckpt_epoch_(\d+)\.pt$", base)
    if m:
        return int(m.group(1))
    return None

def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    """
    NEW: Choose the best checkpoint to resume from.
      Priority:
        1) latest.pt if exists
        2) highest step ckpt_step_*.pt
        3) highest epoch ckpt_epoch_*.pt
    """
    latest = os.path.join(out_dir, "latest.pt")
    if os.path.isfile(latest):
        return latest

    ckpts = _list_ckpts(out_dir)
    if not ckpts:
        return None

    step_ckpts = [(p, _parse_step_from_name(p)) for p in ckpts]
    step_ckpts = [(p, s) for (p, s) in step_ckpts if s is not None]
    if step_ckpts:
        step_ckpts.sort(key=lambda x: x[1])
        return step_ckpts[-1][0]

    epoch_ckpts = [(p, _parse_epoch_from_name(p)) for p in ckpts]
    epoch_ckpts = [(p, e) for (p, e) in epoch_ckpts if e is not None]
    if epoch_ckpts:
        epoch_ckpts.sort(key=lambda x: x[1])
        return epoch_ckpts[-1][0]

    return None

def save_checkpoint(
    out_dir: str,
    *,
    tag: str,
    step: int,
    epoch: int,
    q: nn.Module,
    q_targ: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg_dict: dict,
):
    """
    NEW: Unified checkpoint saver.
      - Writes tagged checkpoint (e.g. ckpt_step_1000.pt)
      - Also updates out_dir/latest.pt every time
      - Saves RNG states for exact resume
    """
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "step": step,
        "epoch": epoch,
        "q_state_dict": q.state_dict(),
        "q_target_state_dict": q_targ.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "cfg": cfg_dict,
        # RNG states (best-effort)
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    tagged_path = os.path.join(out_dir, f"{tag}.pt")
    _atomic_torch_save(payload, tagged_path)

    latest_path = os.path.join(out_dir, "latest.pt")
    _atomic_torch_save(payload, latest_path)

def try_resume(
    out_dir: str,
    q: nn.Module,
    q_targ: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: str,
) -> Tuple[int, int]:
    """
    NEW: Auto-resume from latest checkpoint if it exists.
    Returns: (global_step, start_epoch)
    """
    ckpt_path = find_latest_checkpoint(out_dir)
    if ckpt_path is None:
        print("[Resume] No checkpoint found. Starting fresh.")
        return 0, 0

    print(f"[Resume] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    q.load_state_dict(ckpt["q_state_dict"])
    q_targ.load_state_dict(ckpt["q_target_state_dict"])
    opt.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    # Restore RNG (best-effort; safe even if missing)
    if "torch_rng_state" in ckpt and ckpt["torch_rng_state"] is not None:
        torch.set_rng_state(ckpt["torch_rng_state"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])

    step = int(ckpt.get("step", 0))
    epoch = int(ckpt.get("epoch", 0))
    start_epoch = epoch

    q.to(device)
    q_targ.to(device)

    print(f"[Resume] Resumed at step={step}, epoch={start_epoch}")
    return step, start_epoch


# ============================================================
# NEW: Prioritized sampling helper
# ============================================================

class PrioritizedSampler:
    """
    Computes a WeightedRandomSampler that prioritizes:
      1) samples with high reward (reward > 0)
      2) then samples with done flag (done == 1)
    Everything else gets weight 1.0.

    You can tune reward_boost and done_boost. Keep them moderate to avoid
    training on a tiny subset forever.
    """
    def __init__(
        self,
        reward: torch.Tensor,
        done: torch.Tensor,
        reward_boost: float = 50.0,
        done_boost: float = 5.0,
        reward_threshold: float = 0.0,
        replacement: bool = True,
    ):
        # reward, done come as [N,1] or [N]; normalize to [N]
        r = reward.view(-1).float().cpu()
        d = done.view(-1).float().cpu()

        self.N = r.numel()

        high_reward = r > reward_threshold
        is_done = d > 0.5

        w = torch.ones(self.N, dtype=torch.float32)
        # Higher priority for high reward
        w[high_reward] *= reward_boost
        # Next priority for done
        w[is_done] *= done_boost

        self.weights = w
        self.replacement = replacement

        # Debug prints (kept, since you asked to keep printing stuff)
        print(f"[PrioritizedSampler] N = {self.N}")
        print(f"[PrioritizedSampler] high_reward count = {int(high_reward.sum().item())}")
        print(f"[PrioritizedSampler] done count = {int(is_done.sum().item())}")
        print(f"[PrioritizedSampler] weights min/mean/max = {w.min().item()} {w.mean().item()} {w.max().item()}")

    def make(self) -> WeightedRandomSampler:
        return WeightedRandomSampler(
            weights=self.weights,
            num_samples=self.N,     # one "epoch" draws N samples (with replacement if enabled)
            replacement=self.replacement
        )


# ============================================================
# Your original code begins (UNCHANGED except additions below)
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

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

class DiffusionTransformerBlock(nn.Module):
    def __init__(self, dim, cond_dim, heads=8, dim_head=128):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True,dim_feedforward=256)
        self.cross_attn = CrossAttention(dim, cond_dim, heads, dim_head)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, cond):
        x = self.attn(x)
        x = self.norm(x + self.cross_attn(x, cond))
        return x

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, action_dim=10, output_dim=10,sensor_dim=21,depth_features_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()
        self.action_input_proj = nn.Linear(action_dim , hidden_dim)
        self.visual_obs_projection= nn.Linear(depth_features_dim, hidden_dim)
        self.non_visual_obs_projection= nn.Linear(sensor_dim, hidden_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
        )

        self.transformer_blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear( hidden_dim,action_dim )

        self.decoder_position_embedding=SinusoidalPositionalEncoding(hidden_dim, max_len=21)
        self.encoder_position_embedding=SinusoidalPositionalEncoding(hidden_dim, max_len=21)

    def forward(self, visual_obs, non_visual_obs, noisy_action, t):
        batch_size=non_visual_obs.shape[0]
        context_length=non_visual_obs.shape[1]

        noisy_action=self.action_input_proj(noisy_action.to(torch.float32))
        visual_obs=self.visual_obs_projection(visual_obs.to(torch.float32))
        visual_obs=visual_obs.reshape(batch_size, context_length, -1)
        non_visual_obs=self.non_visual_obs_projection(non_visual_obs.to(torch.float32))

        t=self.time_mlp(t.to(torch.float32))
        t = t.unsqueeze(1)
        t=t.repeat(batch_size, 1, 1)

        encoder_input=torch.cat((visual_obs,non_visual_obs,t),dim=1)
        encoder_input=self.encoder_position_embedding(encoder_input)

        decoder_input=torch.cat((t,noisy_action),dim=1)
        decoder_input = self.decoder_position_embedding(decoder_input)

        for block in self.transformer_blocks:
            decoder_input = block(decoder_input, encoder_input)
        out=self.output_proj(decoder_input)
        out=out[:,1:,:]
        return out

class NoiseScheduler:
    def __init__(self, timesteps=500, beta_start=1e-4, beta_end=0.02):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t=torch.reshape(t, (x_start.shape[0],-1))
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().unsqueeze(1)
        sqrt_one_minus_alpha_cumprod = (1. - self.alpha_cumprod[t]).sqrt().unsqueeze(1)
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    def get_loss(self, model, x_start, t, visual_obs_batch , non_visual_obs_batch):
        noise = torch.randn_like(x_start).to(torch.float32)
        noisy_action = self.q_sample(x_start, t, noise)
        predicted_noise = model(visual_obs_batch.to(torch.float32), non_visual_obs_batch.to(torch.float32), noisy_action.to(torch.float32), t.to(torch.float32))
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, noisy_action, t, visual_obs , non_visual_obs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t.to(device)
        predicted_noise = model(visual_obs, non_visual_obs, noisy_action, t )
        alpha_t = self.alphas[t.to(device).to(torch.long)]
        alpha_cumprod_t = self.alpha_cumprod[t.to(device).to(torch.long)]
        beta_t = self.betas[t.to(device).to(torch.long)]
        sqrt_alpha_t = torch.sqrt(alpha_t).to(device)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        pred_mean = (noisy_action - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t) / sqrt_alpha_t
        if t.min() > 0:
            noise = torch.randn_like(noisy_action)
            pred_mean = pred_mean + torch.sqrt(beta_t) * noise
        return pred_mean

    @torch.no_grad()
    def sample(self, model, shape, visual_obs , non_visual_obs , device, num_random_samples=20):
        batch_size=shape[0]
        number_diffusion_trajectories=shape[1]
        noisy_action_decoder = torch.randn(shape, device=device)
        noisy_action_decoder=noisy_action_decoder.reshape(batch_size*number_diffusion_trajectories,shape[2],shape[3])

        visual_obs=visual_obs.repeat(1,number_diffusion_trajectories,1,1)
        visual_obs=visual_obs.reshape(batch_size*number_diffusion_trajectories,visual_obs.shape[2],visual_obs.shape[3])

        non_visual_obs=non_visual_obs.repeat(1,number_diffusion_trajectories,1,1)
        non_visual_obs=non_visual_obs.reshape(batch_size*number_diffusion_trajectories,non_visual_obs.shape[2],non_visual_obs.shape[3])

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t]).to(device).to(torch.float32)
            noisy_action_decoder = self.p_sample(model,  noisy_action_decoder, t_tensor, visual_obs , non_visual_obs)
        return noisy_action_decoder


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


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_concat(filepaths: List[str], small_dataset=False) -> torch.Tensor:
    tensors = []
    for fp in filepaths:
        t = torch.load(fp, map_location="cpu")
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"Expected Tensor in {fp}, got {type(t)}")
        if small_dataset:
            t=t[0:5000]
        tensors.append(t)
    return torch.cat(tensors, dim=0)


class SkillDataset(Dataset):
    def __init__(
        self,
        vis_hist: torch.Tensor,
        nonvis_hist: torch.Tensor,
        act_seq: torch.Tensor,
        vis_next: torch.Tensor,
        nonvis_next: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        self.vis_hist = vis_hist
        self.nonvis_hist = nonvis_hist
        self.act_seq = act_seq
        self.vis_next = vis_next
        self.nonvis_next = nonvis_next

        self.reward = reward.view(-1).float()
        self.done = done.view(-1).float()

        n = self.vis_hist.shape[0]
        assert self.nonvis_hist.shape[0] == n
        assert self.act_seq.shape[0] == n
        assert self.vis_next.shape[0] == n
        assert self.nonvis_next.shape[0] == n
        assert self.reward.shape[0] == n
        assert self.done.shape[0] == n

    def __len__(self):
        return self.vis_hist.shape[0]

    def __getitem__(self, idx):
        return (
            self.vis_hist[idx],
            self.nonvis_hist[idx],
            self.act_seq[idx],
            self.vis_next[idx],
            self.nonvis_next[idx],
            self.reward[idx],
            self.done[idx],
        )


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

    def forward(self, vis_hist: torch.Tensor, nonvis_hist: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
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


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float):
    for p_t, p in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    TORCH_FILES: Dict[str, List[str]] = {
        "vis_hist": [
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/visual_obs_data_nav_task_with_rewards_eval_dataset_ep_0_5_prev_ob_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/visual_obs_data_pick_task_with_rewards_eval_dataset_ep_0_5_prev_ob_20_acts.pt",
            "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/visual_obs_data_place_task_with_rewards_eval_dataset_ep_0_5_prev_ob_20_acts.pt",
       ],
        "nonvis_hist": [
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/non_visual_obs_data_nav_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/non_visual_obs_data_pick_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/non_visual_obs_data_place_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
        ],
        "act_seq": [
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/action_data_nav_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/action_data_pick_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/action_data_place_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
        ],
        "vis_next": [
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/next_visual_obs_data_nav_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/next_visual_obs_data_pick_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/next_visual_obs_data_place_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
        ],
        "nonvis_next": [
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/next_non_visual_obs_data_nav_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/next_non_visual_obs_data_pick_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/next_non_visual_obs_data_place_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
        ],
        "reward": [
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/rewards_data_nav_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/rewards_data_pick_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/rewards_data_place_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
        ],
        "done": [
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/done_data_nav_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/done_data_pick_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
             "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/done_data_place_task_with_rewards_eval_dataset_ep_0_5_prev_obs_20_acts.pt",
        ],
    }

    for k, v in TORCH_FILES.items():
        if len(v) == 0:
            raise ValueError(f"Please provide at least one filepath for '{k}' in TORCH_FILES.")

    small_dataset=True  
    vis_hist = load_and_concat(TORCH_FILES["vis_hist"],small_dataset=small_dataset)
    nonvis_hist = load_and_concat(TORCH_FILES["nonvis_hist"],small_dataset=small_dataset)
    act_seq = load_and_concat(TORCH_FILES["act_seq"],small_dataset=small_dataset)
    vis_next = load_and_concat(TORCH_FILES["vis_next"],small_dataset=small_dataset)
    nonvis_next = load_and_concat(TORCH_FILES["nonvis_next"],small_dataset=small_dataset)
    reward = load_and_concat(TORCH_FILES["reward"],small_dataset=small_dataset)
    done = load_and_concat(TORCH_FILES["done"],small_dataset=small_dataset)

    assert vis_hist.ndim == 3 and vis_hist.shape[1] == cfg.hist_len and vis_hist.shape[2] == cfg.d_vis, vis_hist.shape
    assert act_seq.ndim == 3 and act_seq.shape[1] == cfg.horizon and act_seq.shape[2] == cfg.d_act, act_seq.shape
    assert vis_next.shape == vis_hist.shape, (vis_next.shape, vis_hist.shape)
    assert nonvis_next.shape == nonvis_hist.shape, (nonvis_next.shape, nonvis_hist.shape)

    print("vis_hist shape:", vis_hist.shape)
    print("nonvis_hist shape:", nonvis_hist.shape)
    print("act_seq shape:", act_seq.shape)
    print("vis_next shape:", vis_next.shape)
    print("nonvis_next shape:", nonvis_next.shape)
    print("reward shape:", reward.shape)
    print("done shape:", done.shape)

    if cfg.d_nonvis != nonvis_hist.shape[-1]:
        print(f"[Info] Overriding cfg.d_nonvis {cfg.d_nonvis} -> {nonvis_hist.shape[-1]}")
        cfg.d_nonvis = nonvis_hist.shape[-1]

    dataset = SkillDataset(
        vis_hist=vis_hist,
        nonvis_hist=nonvis_hist,
        act_seq=act_seq,
        vis_next=vis_next,
        nonvis_next=nonvis_next,
        reward=reward,
        done=done,
    )

    # ============================================================
    # NEW: Prioritized sampling (runs by default)
    #   - high reward samples get large weights
    #   - done samples get moderate weights
    # ============================================================
    prio = PrioritizedSampler(
        reward=reward,
        done=done,
        reward_boost=1.0,   # you can tune
        done_boost=20.0,      # you can tune
        reward_threshold=0.0,  # >0 means "success" samples
        replacement=True,
    )
    sampler = prio.make()

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,          # IMPORTANT when sampler is provided
        sampler=sampler,        # NEW
        pin_memory=True,
        drop_last=True,
    )

    diffusion_policy = ConditionalDiffusionModel()
    diffusion_policy.load_state_dict(torch.load(
        "/home/shokry/hab-mobile-manipulation/collected_data_diffusion/tidy_house/more_data/processed_data/marvin_weights/model_all_tasks_eval_dataset_5_prev_obs_20_act_4400.pt",
        map_location=cfg.device
    ))
    diffusion_policy.to(cfg.device)
    diffusion_policy.eval()
    for p in diffusion_policy.parameters():
        p.requires_grad_(False)

    scheduler = NoiseScheduler()

    q = QTransformer(
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

    q_targ = QTransformer(
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
    q_targ.load_state_dict(q.state_dict())
    q_targ.eval()

    opt = torch.optim.AdamW(q.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device.startswith("cuda")))

    # ============================================================
    # NEW: Auto-resume (runs by default)
    # ============================================================
    global_step, start_epoch = try_resume(
        cfg.out_dir, q, q_targ, opt, scaler, cfg.device
    )

    t0 = time.time()
    running_loss = 0.0
    number_diffusion_trajectories = 10

    for epoch in range(start_epoch, cfg.num_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", dynamic_ncols=True)
        for batch in pbar:
            (b_vis, b_nonvis, b_act, b_vis_next, b_nonvis_next, b_r, b_done) = batch

            b_vis = b_vis.to(cfg.device, non_blocking=True).float()
            b_nonvis = b_nonvis.to(cfg.device, non_blocking=True).float()
            b_act = b_act.to(cfg.device, non_blocking=True).float()
            b_vis_next = b_vis_next.to(cfg.device, non_blocking=True).float()
            b_nonvis_next = b_nonvis_next.to(cfg.device, non_blocking=True).float()
            b_r = b_r.to(cfg.device, non_blocking=True).float()
            b_done = b_done.to(cfg.device, non_blocking=True).float()

            with torch.no_grad():
                shape = (cfg.batch_size, number_diffusion_trajectories, cfg.horizon, cfg.d_act)
                a_next = scheduler.sample(
                    diffusion_policy, shape, b_vis_next, b_nonvis_next, cfg.device, num_random_samples=20
                )
                a_next = a_next.reshape(cfg.batch_size, number_diffusion_trajectories, cfg.horizon, cfg.d_act)

                B, K, H, Da = a_next.shape
                flat_vis_next = b_vis_next.unsqueeze(1).expand(B, K, cfg.hist_len, cfg.d_vis).reshape(B*K, cfg.hist_len, cfg.d_vis)
                flat_nonvis_next = b_nonvis_next.unsqueeze(1).expand(B, K, cfg.hist_len, cfg.d_nonvis).reshape(B*K, cfg.hist_len, cfg.d_nonvis)
                flat_a_next = a_next.reshape(B*K, cfg.horizon, cfg.d_act)

                q_next_all = q_targ(flat_vis_next, flat_nonvis_next, flat_a_next)
                print("q_next_all shape before view:", q_next_all.shape)
                q_next_all = q_next_all.view(B, K)
                print("q_next_all shape after view:", q_next_all.shape)

                q_next = q_next_all.mean(dim=1)
                print("q_next shape:", q_next.shape)
                target = b_r + cfg.gamma * (1.0 - b_done) * q_next

                print("q_next_all == ", q_next_all)
                print("q_next == ", q_next)
                print("done == ", b_done)
                print("reward == ", b_r)
                print("target == ", target)

            q_pred = q(b_vis, b_nonvis, b_act)
            loss = F.mse_loss(q_pred, target)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip_norm)

            scaler.step(opt)
            scaler.update()

            ema_update(q_targ, q, cfg.target_ema_tau)

            global_step += 1

            running_loss += loss.item()
            print("Global Step:", global_step)
            print("Loss:", running_loss)
            if global_step % cfg.log_every == 0:
                avg_loss = running_loss / cfg.log_every
                running_loss = 0.0
                elapsed = time.time() - t0
                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{avg_loss:.4f}",
                    "t(s)": f"{elapsed:.0f}",
                    "q_pred": f"{q_pred.mean().item():.2f}",
                    "targ": f"{target.mean().item():.2f}",
                })

            if global_step % cfg.ckpt_every_steps == 0:
                save_checkpoint(
                    cfg.out_dir,
                    tag=f"ckpt_step_{global_step}",
                    step=global_step,
                    epoch=epoch,
                    q=q,
                    q_targ=q_targ,
                    opt=opt,
                    scaler=scaler,
                    cfg_dict=cfg.__dict__,
                )

            if global_step %10 == 0:
                for name, param in q.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: grad norm = {param.grad.norm().item()}")

        save_checkpoint(
            cfg.out_dir,
            tag=f"ckpt_epoch_{epoch+1}",
            step=global_step,
            epoch=epoch,
            q=q,
            q_targ=q_targ,
            opt=opt,
            scaler=scaler,
            cfg_dict=cfg.__dict__,
        )

    print("Training complete.")
    print(f"Checkpoints saved in: {cfg.out_dir}")
    print(f"Latest checkpoint: {os.path.join(cfg.out_dir, 'latest.pt')}")


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
