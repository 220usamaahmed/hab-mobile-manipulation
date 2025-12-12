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

def _to_config(cfg_like: Any) -> Config:
    """
    Normalize any 'config-like' object to a habitat Config.
    Accepts:
      - Config
      - dict
      - path to YAML file (str/path)
    """
    if isinstance(cfg_like, Config):
        return cfg_like

    if isinstance(cfg_like, (str, os.PathLike)):
        with open(cfg_like, "r") as f:
            d = yaml.safe_load(f)
        return Config(d)

    if isinstance(cfg_like, dict):
        return Config(cfg_like)

    raise TypeError(f"Unsupported config type: {type(cfg_like)}")


def _strip_prefix_in_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """
    Remove a leading prefix (like 'module.' from DDP) in all keys if present.
    """
    if not state_dict:
        return state_dict
    needs_strip = all(k.startswith(prefix) for k in state_dict.keys())
    if not needs_strip:
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def _maybe_move(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return model.to(device)


class SkillTransformerPolicyLoader:
    """
    Rebuilds the transformer policy architecture and loads weights from a checkpoint.

    Two main entry points:
      - from_checkpoint(ckpt_path, ...)
      - from_config_and_weights(config, state_dict, ...)

    The checkpoint format is expected to be something like:
      {
        "state_dict": <policy.state_dict()>,
        "optim_state": <optional optimizer state>,
        "sched_state": <optional scheduler state>,
        "config": <optional policy.config.to_dict()>,
        ...
      }
    """

    def __init__(self,
                 policy: TransformerResNetPolicy,
                 policy_config: Config,
                 device: Optional[torch.device] = None):
        self.policy = policy
        self.config = policy_config
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _ = _maybe_move(self.policy, self.device)
        self.policy.eval()  # default safe mode

    # ---------- High-level constructors ----------

    @classmethod
    def from_checkpoint(cls,
                        ckpt_path: str,
                        device: Optional[str] = None,
                        override_config: Optional[Any] = None,
                        strict: bool = True,
                        map_location: Optional[str] = "cpu",
                        strip_prefix: Optional[str] = "module.") -> "SkillTransformerPolicyLoader":
        """
        Build policy from the checkpoint file.
        - If the checkpoint contains 'config', that will be used unless you pass override_config.
        - If you pass override_config, it will take precedence (useful to tweak batch size, paths, etc.).
        """
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=map_location)
      #  print("checkpoint keys:", ckpt['state_dict'].keys())
       # input()




        # 1) Resolve config
        if override_config is not None:
            policy_cfg = _to_config(override_config)
        else:
            if "config" in ckpt:
                policy_cfg = _to_config(ckpt["config"])

            else:
                raise ValueError(
                    "Checkpoint does not contain 'config'. "
                    "Pass override_config=<yaml path or dict or Config>."
                )

        # 2) Rebuild policy architecture exactly as training
        low = np.array([ -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,-1. ])
        high = np.array([  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1. , 1.])

        action_space = Box(low=low, high=high, shape=(12,))
       # action_space = Box([ -1. , -1.,  -1. , -1. , -1. , -1. , -1. , -1. ,-20., -20.  ,-1.], [ 1. , 1. , 1. , 1. , 1.,  1. , 1. , 1., 20., 20. , 1.], (11,))
        observation_space = gymDict(
            is_holding = Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            joint = Box(-3.4028235e+38, 3.4028235e+38, shape=(7,), dtype=np.float32),
            obj_goal_gps_compass = Box(-3.4028235e+38, 3.4028235e+38, shape=(2,), dtype=np.float32),
            obj_goal_sensor = Box(-3.4028235e+38, 3.4028235e+38, shape=(3,), dtype=np.float32),
            obj_start_gps_compass = Box(-3.4028235e+38, 3.4028235e+38, shape=(2,), dtype=np.float32),
            obj_start_sensor = Box(-3.4028235e+38, 3.4028235e+38, shape=(3,), dtype=np.float32),
            relative_resting_position = Box(-3.4028235e+38, 3.4028235e+38, shape=(3,), dtype=np.float32),
            robot_head_depth = Box(0.0, 1.0, shape=(128, 128, 1), dtype=np.float32)
        )
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

