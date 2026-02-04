#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
from gym import spaces
from torch import device, nn as nn
import torch.nn.functional as F
import numpy as np
from habitat_baselines.config.default import get_config
from habitat.config import Config
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ppo.policy import Policy
from mobile_manipulation.transformer_policy.transformer_model import (
    GPTConfig,
    ActionInference,
    SkillInference,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import get_num_actions
from mobile_manipulation.transformer_policy.action_distribution import (
    ActionDistribution,
)
from habitat_baselines.rl.ppo import NetPolicy
from habitat_baselines.common.tensor_dict import TensorDict

from habitat.core.spaces import ActionSpace, EmptySpace

from .focal_loss import FocalLoss



@baseline_registry.register_policy
class TransformerResNetPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        context_length: int = 30,
        max_episode_step: int = 200,
        n_layer: int = 6,
        n_head: int = 8,
        reg_flags=None,
        model_type: str = "reward_conditioned",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        force_blind_policy: bool = False,
        policy_config: Config = None,
        fuse_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        include_visual_keys = policy_config.include_visual_keys
        self.offline_training = policy_config.offline
        self.train_planner = policy_config.train_planner
        self.train_control = policy_config.train_control

        self.context_length=context_length
        self.action_dim=12

        super().__init__(
            TransformerResnetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                context_length=context_length,
                max_episode_step=max_episode_step,
                model_type=model_type,
                n_head=n_head,
                n_layer=n_layer,
                reg_flags=reg_flags,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                force_blind_policy=force_blind_policy,
                discrete_actions=False,
                fuse_keys=[
                    k for k in fuse_keys if k not in include_visual_keys
                ],
                include_visual_keys=include_visual_keys,
                use_rgb=False,#policy_config.use_rgb,
            ),
            self.action_dim,   ### action dimension
            policy_config=policy_config,
        )
        self.boundaries_mean = torch.linspace(-1, 1, 21).cuda()
        self.boundaries = torch.linspace(-1.025, 1.025, 22).cuda()
        if self.offline_training:
            self.loss_vars = nn.parameter.Parameter(torch.zeros((3,)))
            self.focal_loss_planner = FocalLoss(gamma=5).cuda()
            self.focal_loss_arm = FocalLoss(gamma=5).cuda()
            self.focal_loss_loc = FocalLoss(gamma=5).cuda()
            self.focal_loss_pick = FocalLoss(
                alpha=(1 - torch.tensor([0.8, 0.1, 0.1])), gamma=5
            ).cuda()
            self.aux_head = nn.Linear(512, 5).cuda()  # DTHACK

        self.action_config = policy_config.ACTION_DIST

        if self.action_distribution_type == "categorical":
            self.len_logit = [21 * 7, 21 * 2, 3]
        elif self.action_distribution_type == "gaussian":
            self.len_logit = [7, 2, 1]
        elif self.action_distribution_type == "mixed":
            self.len_logit = [
                21 * 7 if self.action_config.discrete_arm else 7,
                
                21 * 2 if self.action_config.discrete_base else 2,
                3,
            ]
        else:
            raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space: spaces.Dict,
        action_space,
        orig_action_space=None,
        **kwargs,
    ):
        orig_action_space = ActionSpace(
            {
                "ARM_ACTION": spaces.Dict(
                    {
                        "arm_action": spaces.Box(
                            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
                        ),
                        "grip_action": spaces.Box(
                            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                        ),
                    }
                ),
                "BASE_VELOCITY": spaces.Dict(
                    {
                        "base_vel": spaces.Box(
                            low=-20.0, high=20.0, shape=(2,), dtype=np.float32
                        )
                    }
                ),
                "REARRANGE_STOP": EmptySpace(),
            }
        )
        return cls(
            observation_space=observation_space,
            action_space=orig_action_space,
            hidden_size=config.RL.TRANSFORMER.hidden_size,
            context_length=config.RL.TRANSFORMER.context_length,
            max_episode_step=config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS,
            model_type=config.RL.TRANSFORMER.model_type,
            n_head=config.RL.TRANSFORMER.n_head,
            n_layer=config.RL.TRANSFORMER.n_layer,
            reg_flags=config.RL.TRANSFORMER.reg_flags,
            backbone=config.RL.TRANSFORMER.backbone,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            policy_config=config.RL.POLICY,
            fuse_keys=[
                "robot_head_depth",
                "relative_resting_position",
                "obj_start_sensor",
                "obj_goal_sensor",
                "obj_start_gps_compass",
                "obj_goal_gps_compass",
                "joint",
                "is_holding",
            ]
            # fixed input keys.
        )

    def act_original(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        envs_to_pause=None,
        rtgs=None,
    ):
        (
            value,
            action,
            action_log_probs,
            rnn_hidden_states,
        ) = super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=deterministic,
            envs_to_pause=envs_to_pause,
            rtgs=rtgs,
        )
        action = action.float()
        if self.action_distribution_type == "mixed":
            if self.action_config.discrete_base:
                action[:, 7:9] = self.boundaries_mean[
                    action[:, 7:9].to(torch.long)
                ]
            if self.action_config.discrete_arm:
                action[:, :7] = self.boundaries_mean[
                    action[:, :7].to(torch.long)
                ]
            # gripper actions are "changes" in gripper states. 
            # Convert back to gripper states as required by the environment.
         #   action[:, 9] = (
          #      torch.argmax(action[:, 9:12]) == 1).int()
            #    + 2 * (torch.argmax(action[:, 9:12]) == 0).int()
           #     + 3 * (torch.argmax(action[:,  9:12]) == 2).int()
             #   - 2
            #)
        #    action[:,10:12]=np.array([0,0])
          #  mask = action[:, 7:8] == -1
           # action = torch.cat(
            #    [action, torch.zeros_like(mask.float())], dim=-1
            #)
        # #============= advance hidden state ===============
        mask = ~torch.any((rnn_hidden_states.sum(-1) == 0), -1)
        rnn_hidden_states[mask] = rnn_hidden_states[mask].roll(-1, 1)
        rnn_hidden_states[mask, -1, :] = 0   ###this part performs rollout of the rnn_hidden_state, if the buffer of an env is full the buffer will be shifted left by one step and the most recent data will be inserted at the end of the buffer

        # #============= reset arm ===============
        B = rnn_hidden_states.shape[0]
        if not hasattr(self, "reset_mask"):
            self.reset_mask = torch.zeros(
                B, dtype=torch.bool, device=rnn_hidden_states.device
            )

        if not hasattr(self, "holding_mask"):
            self.holding_mask = torch.zeros(
                B, dtype=torch.bool, device=rnn_hidden_states.device
            )

        if not hasattr(self, "_initial_delta"):
            self._initial_delta = torch.zeros(
                (B, 7), dtype=torch.float, device=rnn_hidden_states.device
            )

        state_index = list(range(self.reset_mask.shape[0]))
        if envs_to_pause is not None:
            envs_to_pause.sort()
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)

        holding_mask = self.holding_mask[state_index] != observations[
            "is_holding"
        ].reshape(B)
        self.holding_mask[state_index] = (
            observations["is_holding"].reshape(B).bool()
        )

        self.reset_mask[state_index] = (
            self.reset_mask[state_index]
            | holding_mask
            | self.net.reset_mask[state_index]
        )
        self._reset_arm(
            observations,
            action,
            rnn_hidden_states,
            holding_mask | self.net.reset_mask[state_index],
            state_index,
        )

       # mask = action[:, 7] == -1   ## this represents the release action which represents the end of the episode, this is because the action space is 7D arm, 1D gripper, 2D base, 1 empty stop action
                                    ## the total size is 11D
        #action[:, -1] = mask.float()

        return (
            value,
            action,
            action_log_probs,
            rnn_hidden_states,
        )

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info=None,
        evaluate_aux_losses=True,
    ):
        if self.action_distribution_type == "mixed":
            action = action[:, :12]
            if self.action_config.discrete_base:
                action[:, 8:10] = (
                    torch.bucketize(action[:, 8:10], self.boundaries) - 1
                )
            if self.action_config.discrete_arm:
                action[:, :7] = (
                    torch.bucketize(action[:, :7], self.boundaries) - 1
                )
            action[:, 7] = (
                (action[:, 7] == 0).int()
                + 2 * (action[:, 7] == -1).int()
                + 3 * (action[:, 7] == 1).int()
                - 1
            )
        return super().evaluate_actions(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            action,
            rnn_build_seq_info=rnn_build_seq_info,
        )

    def forward(
        self,
        states,
        actions,
        targets,
        rtgs,
        timesteps,
    ):
        if not self.offline_training:
            raise ValueError
        features, planner_logits = self.net(
            states, None, actions, None, rtgs=rtgs, offline_training=True
        )    ### feautures are the logits produced by the action inference module , and planner logits are the output of the planner transformer
             ###input of the action inference module has 3 tokens per time step and the output (refered to as feautures) is the output of or decoded state token
        
        # if we are given some desired targets also calculate the loss
        loss = 0
        loss_dict = dict()
        
        if self.train_planner:
            B = actions.shape[0]

            aux_logits = planner_logits[1]
            planner_logits = planner_logits[0]


            # ========================== planner aux loss ============================
            if "all_predicates" in states.keys():
                # only calculate when hidden object is present
                mask_open_skill = (
                    (states["skill"].reshape(B, -1) == 3)
                    | (states["skill"].reshape(B, -1) == 1)
                    | (states["skill"].reshape(B, -1) == 5)
                    | (states["skill"].reshape(B, -1) == 6)
                )
                mask_predicate = mask_open_skill & torch.any(
                    states["all_predicates"].reshape(
                        B, -1, states["all_predicates"].shape[-1]
                    )[..., :5],
                    dim=-1,
                )
                temp_target = states["all_predicates"].reshape(
                    B, -1, states["all_predicates"].shape[-1]
                )[..., :5]
                temp_target = torch.cat(
                    [
                        ~torch.any(temp_target, dim=-1, keepdim=True),
                        temp_target,
                    ],
                    dim=-1,
                )
                temp_target = torch.argmax(temp_target.long(), dim=-1)
                loss_aux = F.cross_entropy(
                    aux_logits[..., :6].permute(0, 2, 1),  
                    temp_target.long(),
                    label_smoothing=0.05,
                    reduction="none",
                )
                loss_aux = loss_aux.reshape(B, -1)
                loss_aux = (
                    torch.mean(loss_aux[mask_open_skill])
                    if loss_aux[mask_open_skill].shape[0] != 0
                    else torch.tensor(0)
                )
                loss = loss + loss_aux
                accuracy_aux = torch.sum(
                    torch.argmax(
                    #    aux_logits[mask_open_skill][..., 4:10], dim=-1   ####### This line is very weired, I guess the index should be 1:10
                    aux_logits[mask_open_skill][..., 1:10], dim=-1 
                    )
                    == temp_target[mask_open_skill].long()
                ) / np.prod(temp_target[mask_open_skill].shape)
                loss_dict.update(
                    {
                        "planner_aux_loss": loss_aux.detach().item(),
                        "planner_predicate_accuracy": accuracy_aux.detach().item(),
                    }
                )

            # ========================== planner action ============================
            temp_target = states["skill"].reshape(B, -1, 1)
            loss_p = self.focal_loss_planner(
                planner_logits.permute(0, 2, 1),
                temp_target.reshape(B, -1).long(),
            )
            accuracy_p = torch.sum(
                torch.argmax(planner_logits[:, :, :], dim=-1)
                == temp_target.reshape(B, -1).long()
            ) / np.prod(temp_target.shape)

            loss_dict.update(
                {
                    "planner_skill": loss_p.detach().item(),
                    "accuracy_planner_skill": accuracy_p.detach().item(),
                    "aux_loss": loss_aux.detach().item(),
                }
            )
            loss = loss + loss_p

        if self.train_control:
            if self.action_distribution_type == "categorical":
                distribution = self.action_distribution(features)
                logits = distribution.probs
            elif self.action_distribution_type == "gaussian":
                distribution = self.action_distribution(features)
                logits = distribution.mean
            elif self.action_distribution_type == "mixed":
                logits = self.action_distribution(features, return_logits=True)   ## this is just a linear projection for dimensionality
                
            else:
                raise NotImplementedError

            # ======================== separate logits ==========================
            logits_arm, logits_loc, logits_pick = torch.split(
                logits, self.len_logit, -1
            )

            # =========================== locomotion ============================
            if self.action_config.discrete_base:
                temp_target = (
                    torch.bucketize(targets[:, :, 7:9], self.boundaries) - 1
                )
                logits_loc = logits_loc.view(*logits_loc.shape[:2], 2, 21)
                loss1 = self.focal_loss_loc(
                    logits_loc[:, :, :, :].permute(0, 3, 1, 2),
                    temp_target[:, :, :],
                )
                accuracy1 = torch.sum(
                    torch.argmax(logits_loc[:, :, :, :], dim=-1)
                    == temp_target[:, :, :]
                ) / np.prod(temp_target[:, :, :].shape)
            else:
                temp_target = targets[:, :, 7:9]
                loss1 = F.mse_loss(logits_loc, temp_target)

            # =========================== arm action ============================
            if self.action_config.discrete_arm:
                temp_target = (
                    torch.bucketize(targets[:, :, :7], self.boundaries) - 1
                )
                logits_arm = logits_arm.view(*logits_arm.shape[:2], 7, 21)
                loss2 = self.focal_loss_arm(
                    logits_arm[:, :, :, :].permute(0, 3, 1, 2),
                    temp_target[:, :, :7],
                )
                accuracy2 = torch.sum(
                    torch.argmax(logits_arm[:, :, :, :], dim=-1)
                    == temp_target[:, :, :7]
                ) / np.prod(temp_target[:, :, :7].shape)
            else:
                temp_target = targets[:, :, :7]
                loss2 = F.mse_loss(logits_arm, temp_target)


            # ========================= gripper action ==========================
            loss3 = self.focal_loss_pick(
                logits_pick.permute(0, 2, 1), targets[:, :, 9].long()
            )
            accuracy3 = torch.sum(
                torch.argmax(logits_pick[:, :, :], dim=-1)
                == targets[:, :, 9].long()
            ) / np.prod(targets[:, :, 9].shape)

            # ========================== aux loss ============================

            loss_dict.update(
                {
                    "locomotion": loss1.detach().item(),
                    "arm": loss2.detach().item(),
                    "pick": loss3.detach().item(),
                    "accuracy_pick": accuracy3.detach().item(),
                }
            )

            if self.action_config.discrete_base:
                loss_dict.update(
                    {
                        "accuracy_nav": accuracy1.detach().item(),
                    }
                )
            else:
                loss_dict.update(
                    {
                        "mse_base": loss1.detach().item(),
                    }
                )
            if self.action_config.discrete_arm:
                loss_dict.update(
                    {
                        "accuracy_arm": accuracy2.detach().item(),
                    }
                )
            else:
                loss_dict.update(
                    {
                        "mse_arm": loss2.detach().item(),
                    }
                )


            loss1 = torch.exp(-self.loss_vars[0]) * loss1 + self.loss_vars[0]
            loss2 = torch.exp(-self.loss_vars[1]) * loss2 + self.loss_vars[1]
            loss3 = torch.exp(-self.loss_vars[2]) * loss3 + self.loss_vars[2]
            loss = loss + loss1 + loss2 + loss3   ## the first loss is for the planner and the three other losses are for the action inference and they have a trainable weights on the loss
        return loss, loss_dict

    def get_policy_info(self, infos, dones):
        policy_infos = []
        for i, info in enumerate(infos):
            policy_info = {
                "cur_skill": self.net.cur_skill[i],
            }
            policy_infos.append(policy_info)

        return policy_infos

    @property
    def hidden_state_hxs_dim(self):
        return self.net.hidden_state_hxs_dim

    # hard-coded reset action brought from the expert policy
    def _reset_arm(
        self,
        observations,
        prev_actions,
        rnn_hidden_states,
        reset_mask,
        state_index,
    ):
        self._target = torch.tensor(
            [
                -4.5003259e-01,
                -1.0799699e00,
                9.9526465e-02,
                9.3869519e-01,
                -7.8854430e-04,
                1.5702540e00,
                4.6168058e-03,
            ],
            device=rnn_hidden_states.device,
        )
        self._initial_delta[state_index] = (
            self._target - observations["joint"]
        ) * reset_mask.reshape(-1, 1) + self._initial_delta[state_index] * (
            ~reset_mask
        ).reshape(
            -1, 1
        )

        current_joint_pos = observations["joint"]
        delta = self._target - current_joint_pos

        # Dividing by max initial delta means that the action will
        # always in [-1,1] and has the benefit of reducing the delta
        # amount was we converge to the target.
        delta = delta / torch.maximum(
            self._initial_delta[state_index].max(-1, keepdims=True)[0],
            torch.tensor(1e-5, device=rnn_hidden_states.device),
        )

        prev_actions[self.reset_mask[state_index], :7] = delta[
            self.reset_mask[state_index]
        ]

        self.reset_mask[state_index] = self.reset_mask[state_index] & ~(
            torch.abs(current_joint_pos - self._target).max(-1)[0] < 5e-2
        )

    def _back_up(
        self,
        observations,
        prev_actions,
        rnn_hidden_states,
        reset_mask,
        state_index,
    ):
        if self.net.timeout[0] > 100 and (
            self.net.cur_skill[0] == 0 or self.net.cur_skill[0] == 4
        ):
            prev_actions[0, 8:10] = torch.tensor([-1, 0]).cuda()
            self.net.timeout[0] -= 20












    @torch.no_grad()
    def act(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = True,
        rtgs: torch.Tensor = None,
        envs_to_pause=None,
        
    ) -> Dict[str, Any]:
        """
        Produce the action for the *next single time step* for each environment.

        Arguments
        ---------
        observations:
            Dict mapping obs names -> tensors.
            Shapes are (B, ...) for 1D features, (B, C, H, W) for images.
            This should already be on the same device as the policy.

        rnn_hidden_states:
            Rolling transformer context buffer of shape:
                (B, context_length, slot_dim)
            Each "slot" along dimension 1 stores information for one time step
            (state embedding, skill field, RTG, prev_action, etc.), as defined
            in TransformerResnetNet.forward.
            This is passed back and forth between calls.

        prev_actions:
            Tensor of shape (B, action_dim).
            Action actually executed at the *previous* step for each env.
            For the very first step of an episode, this can be zeros.

        masks:
            Tensor of shape (B,).
            - 1.0  => episode is continuing for that env.
            - 0.0  => episode just reset for that env.
            Used inside the backbone to zero out the context for reset envs.

        deterministic:
            If True, the action head will return the mode/mean instead of sampling.

        rtgs:
            Optional "return-to-go" tensor of shape (B, 1) or (B, T, 1),
            if you trained with RTG conditioning. Otherwise None.

        envs_to_pause:
            Optional list of env indices that are paused (not used here
            unless you integrate multi-env pausing logic).

        Returns
        -------
        A dict with:
            "actions":           (B, action_dim)
            "rnn_hidden_states": updated context buffer
            "prev_actions":      same as "actions" (for feeding next call)
            "extra":             any extra info the backbone returned
        """

        # ---------------------------------------------------------
        # 0. Setup: infer batch size, device, and initialize memory
        # ---------------------------------------------------------
        # Infer batch size from masks (all other batch dims must match)
        B = 1   #prev_actions.shape[0]

        # Pick device from one of the inputs (they should all match)
        device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

        # Initialize the rolling context buffer the first time we call act.
        # We allocate a fixed window length = context_length. The "slot_dim"
        # (size of the last dimension) is determined by the backbone.
        
        
        
        
        if rnn_hidden_states is None or rnn_hidden_states.numel() == 0:
            # The net knows how big each time-step slot is; we can query it.
            # If you don't have such an attribute, you can hardcode or compute it.
            slot_dim = 256+21+1+12#self.net._hxs_dim #self.net.rnn_state_dim  # e.g., defined in TransformerResnetNet.__init__
            rnn_hidden_states = torch.zeros(
                B,
                self.context_length,
                slot_dim,
                device=device,
                dtype=torch.float32,
            ).float()










        # Make sure shapes are consistent with the current batch size
        # (e.g. if you change number of envs between runs).
        if rnn_hidden_states.shape[0] != B:
            raise ValueError(
                f"rnn_hidden_states batch dim ({rnn_hidden_states.shape[0]}) "
                f"does not match masks batch dim ({B})."
            )
        # If prev_actions has wrong batch size, re-init it
        if prev_actions is None or prev_actions.shape[0] != B:
            prev_actions = torch.zeros(
                B, self.action_dim, device=device, dtype=torch.float32
            )


        # Ensure contiguity to avoid subtle CUDA kernel issues
        rnn_hidden_states = rnn_hidden_states.contiguous()
        prev_actions = prev_actions.contiguous()

        # ---------------------------------------------------------
        # 1. Maintain the rolling window: keep only recent context
        # ---------------------------------------------------------
        # We want rnn_hidden_states[b] (for each env b) to always hold *at most*
        # self.context_length time steps. This buffer is filled step-by-step
        # by the backbone. When it becomes full (no zero rows left), we
        # "roll" it to the left (drop the oldest step) and zero the last slot.

        # For each env b, rnn_hidden_states[b] has shape (Tctx, slot_dim).
        # If any row is all zeros, that indicates a *free* slot (not yet used).
        # We detect envs whose buffer is already *full* (no zeros left).
        # Sum over the slot_dim, compare to zero ⇒ True for zero-rows.
        zero_rows = (rnn_hidden_states.sum(-1) == 0)  # (B, Tctx) bool

        # torch.any(zero_rows, -1) is True if there is at least one zero-row
        # in that env's buffer. Negating (~) gives True if the env is *full*.
        mask_full = ~torch.any(zero_rows, dim=-1)  # (B,) bool

        if mask_full.any():
            # For envs with a full buffer, roll their context along the time axis:
            # time t becomes t-1, etc. The last slot wraps around but we will zero it.
            rnn_hidden_states[mask_full] = rnn_hidden_states[mask_full].roll(
                shifts=-1, dims=1
            )
            # Now explicitly zero the last slot; this is where the *new* time step
            # will be written by the backbone.
            rnn_hidden_states[mask_full, -1, :] = 0.0

        # Important: we do *not* choose the current index here. The backbone
        # (TransformerResnetNet.forward) will:
        #   - zero out envs whose masks == 0,
        #   - find the first all-zero row per env,
        #   - write the new step there,
        #   - run planner/controller transformer,
        #   - and return features for that "current step".

        # ---------------------------------------------------------
        # 2. Call the transformer backbone to update context + get features
        # ---------------------------------------------------------
        # The backbone does:
        #   - build state embeddings from observations (images + 1D features),
        #   - maintain its internal skill/timeout/reset_mask state,
        #   - write the current step into rnn_hidden_states at the first empty slot,
        #   - run the planner transformer (skill + aux),
        #   - run the controller transformer (low-level actions),
        #   - return JUST the features for the current time step for each env.
        ctrl_features_t, rnn_hidden_states, extra = self.net.forward(
            observations=observations,
            rnn_hidden_states=rnn_hidden_states,
            prev_actions=prev_actions,
            masks=masks,
            rnn_build_seq_info=None,
            rtgs=rtgs,
            offline_training=False,  # online / acting mode
            envs_to_pause=envs_to_pause,
        )
        # ctrl_features_t: shape (B, hidden_dim_for_action_head)

        # ---------------------------------------------------------
        # 3. Turn controller features into an action (next step only)
        # ---------------------------------------------------------
        # Build the action distribution from the controller features at the
        # *current* timestep only. This head knows how to parameterize the
        # arm/base/gripper distributions (Gaussian / categorical / mixed).
        
    #    print("ctrl_features_t == " , ctrl_features_t.shape)
        action = self.action_distribution(ctrl_features_t,return_logits=True)

       # if deterministic:
            # Use the mode of the distribution (e.g., mean for Gaussians,
            # argmax for categoricals) for deterministic evaluation.
           
           
           # action = dist.mode()
         #  action = dist.sample()
        #   
        #else:
            # Sample a stochastic action (and, optionally, log-prob).
         #   action = dist.sample()

        # Ensure shape is (B, action_dim) and on the correct device

        action = action.view(B, self.action_dim).to(device)

        # ---------------------------------------------------------
        # 4. Prepare outputs for the caller
        # ---------------------------------------------------------
        # We treat the produced action as the "prev_actions" for the next call.
        next_prev_actions = action.detach()

        return {
            "actions": action,                        # (B, action_dim)
            "rnn_hidden_states": rnn_hidden_states,   # updated context window
            "prev_actions": next_prev_actions,        # for next step
            "extra": extra,                           # optional dict with planner/aux info
        }


















class TransformerResnetNet(nn.Module):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        context_length: int,
        max_episode_step: int,
        model_type: str,
        n_head: int,
        n_layer: int,
        reg_flags,
        backbone,
        resnet_baseplanes,
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        fuse_keys: Optional[List[str]] = None,
        include_visual_keys: Optional[List[str]] = None,
        use_rgb=False,
        num_skills=10,
    ):
        super().__init__()
        self.context_length = context_length

        self.discrete_actions = discrete_actions
        if discrete_actions:
            num_actions = action_space.n + 1
        else:
            num_actions = get_num_actions(action_space)

        self.num_actions=12  ### added by me to control the dimensionality of the actions 
        num_actions=12
        self.observation_dimension=21
        
        rnn_input_size = 0
        self.include_visual_keys = include_visual_keys

        self._fuse_keys = fuse_keys
        if self._fuse_keys is not None:
            rnn_input_size += sum(
                [observation_space.spaces[k].shape[0] for k in self._fuse_keys]
            )
        rnn_input_size=self.observation_dimension

        self._hidden_size = hidden_size

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        elif (
            self.include_visual_keys is not None
            and len(self.include_visual_keys) != 0
        ):
            use_obs_space = spaces.Dict(
                {
                    k: v
                    for k, v in observation_space.spaces.items()
                    if k in ["robot_head_rgb"]
                }
            )
        else:
            use_obs_space = observation_space

        if use_rgb:
            self.visual_encoder_rgb = ResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                use_input_norm=False,
            )
        else:
            self.visual_encoder_rgb = None

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        elif (
            self.include_visual_keys is not None
            and len(self.include_visual_keys) != 0
        ):
            use_obs_space = spaces.Dict(
                {
                    k: v
                    for k, v in observation_space.spaces.items()
                    if k in ["robot_head_depth"]
                }
            )
        else:
            use_obs_space = observation_space

        self.visual_encoder = ResNetEncoder(
            use_obs_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
          #  use_input_norm=False,
        )

        if not self.visual_encoder.is_blind:
            if use_rgb:
                self.visual_fc_rgb = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder_rgb.output_shape),
                        hidden_size // 2,
                    ),
                    nn.ReLU(True),
                )
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )
            else:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape),
                        hidden_size // 2,
                    ),
                    nn.ReLU(True),
                )
        self._hxs_dim = (self._hidden_size // 2) + rnn_input_size + num_actions

        self._hxs_dim += 1 if num_skills != 0 else 0
        self._num_actions = num_actions
        self.action_dim = self._num_actions

        self.obs_dim = (
            self.observation_dimension+ 1 if num_skills != 0 else self.observation_dimension
        )
        mconf = GPTConfig(
            num_actions,
            context_length,
            num_states=[
                (0 if self.is_blind else self._hidden_size // 2),
                rnn_input_size,
            ],
            n_layer=n_layer,
            n_head=n_head,
            n_embd=self._hidden_size,
            model_type=model_type,
            max_timestep=max_episode_step,
            num_skills=num_skills,
            use_rgb=use_rgb,
            reg_flags=reg_flags,
        )  # 6,8
        self.state_encoder = ActionInference(mconf)

        mconf = GPTConfig(
            num_actions,
            context_length,
            num_states=[
                (0 if self.is_blind else self._hidden_size // 2),
                rnn_input_size,
            ],
            n_layer=2,
            n_head=8,
            n_embd=self._hidden_size,
            model_type=model_type,
            max_timestep=max_episode_step,
            num_skills=num_skills,
            use_rgb=use_rgb,
            reg_flags=reg_flags,
        )  # 6,8
        self.planner_encoder = SkillInference(mconf)

        self.train()

    @property
    def hidden_state_hxs_dim(self):
        return self._hxs_dim

    @property
    def num_recurrent_layers(self):
        return self.context_length

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info=None,
        rtgs=None,
        offline_training=False,
        envs_to_pause=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []
        B = prev_actions.shape[0]


        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)
                visual_feats = self.visual_fc(visual_feats)

                x.append(visual_feats)
                if self.visual_encoder_rgb is not None:
                    visual_feats = self.visual_encoder_rgb(observations)
                    visual_feats = self.visual_fc_rgb(visual_feats)
                    x.append(visual_feats)


        if self._fuse_keys is not None:

            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys], dim=-1
            )

            fuse_states=torch.unsqueeze(fuse_states,0)
            x.append(fuse_states)


        x = torch.cat(x, dim=1)
        x = x.reshape(B, -1, *x.shape[1:])

        if offline_training:
            # Move valid state-action-reward pair to the left

            out2 = self.planner_encoder(x)
            ## observation skills is represented by an integer 
            if "skill" in observations.keys():
                x = torch.cat(
                    [x, observations["skill"].reshape(B, -1, 1).float()],
                    dim=-1,
                )
            ### this part is added by me and it should be modified
            actions_for_token =prev_actions 

            out = self.state_encoder(
                x,
                actions_for_token,
                rtgs=rtgs,
            )
            return out, out2

        #rnn_hidden_states *= masks.view(-1, 1, 1).long()  ##set the states of the reseted (done)envs to zeros
        

        current_context = torch.argmax(
            (rnn_hidden_states.sum(-1) == 0).float(), -1
        )
         
        ### find the current context (time step in the window size) index
  #      print("rnn  hiden state shape == " , rnn_hidden_states.shape)
   #     print("rnn_hidden_states.sum(-1) == " , rnn_hidden_states.sum(-1).shape)
        
        # Write obs to context
        # print(
        #     f"Rnn shape {rnn_hidden_states.shape}, batch {B}, ctx {current_context}, ac dim {action_dim} x shape {x.shape}, actions shape {prev_actions.shape}"
        # )

        obs_dim = self.obs_dim

        action_dim = self.action_dim



  #      print("rnn_hidden_states before adding == " , rnn_hidden_states[0,current_context,-(action_dim+1):])
      #  rnn_hidden_states[
       #     torch.arange(B), current_context, :-obs_dim   ### only state, without actions or skill
      #  ] = x.view(B, -1)
        rnn_hidden_states[
            torch.arange(B), current_context, :- (action_dim+1)  ### only state, without actions or skill
        ] = x.view(B, -1).float()
        
        
        
        # Write actions to context
        
        if current_context>0:
            rnn_hidden_states[
                torch.arange(B), current_context-1, -action_dim:
            ] = prev_actions.view(B, -1).float()

            

        
        
      #  print("prev acts == " , prev_actions.view(B, -1).float())
        out, predicted_dist = self.planner_encoder(
            rnn_hidden_states[..., :-(action_dim+1)],
        )
        self.predicted_dist = predicted_dist[torch.arange(B), current_context]
        rnn_hidden_states[
            torch.arange(B), current_context, -(action_dim+1)
        ] = torch.argmax(out[torch.arange(B), current_context], dim=-1).float()   ### add the estimated skill by the planner to the rnn state before feeding it to the action inference
        
        current_skill_index=torch.argmax(out[torch.arange(B), current_context], dim=-1).float()
        print("skill logits == ", out[torch.arange(B), current_context])
        if current_skill_index == 0:
            print("current skill == {} nav to pick pos".format(current_skill_index))
        elif current_skill_index == 1:
            print("current skill == {} pick".format(current_skill_index))
        elif current_skill_index == 2:
            print("current skill == {} place".format(current_skill_index))
        elif current_skill_index == 3:
            print("current skill == {} pick offset".format(current_skill_index))
        elif current_skill_index == 4:
            print("current skill == {} nav to place pos".format(current_skill_index))
        elif current_skill_index == 5:
            print("curent skill == {} open".format(current_skill_index))
        elif current_skill_index == 6:
            print("curent skill == {} reset arm".format(current_skill_index))
        else:
            pass
      #  input()
        #input_skill=input("enter the required skill ")
        
      #  rnn_hidden_states[
       #     torch.arange(B), current_context, -(action_dim+1)
       # ] = float(input_skill) #torch.argmax(out[torch.arange(B), current_context], dim=-1).float()   ### add the estimated skill by the planner to the rnn state before feeding it to the action inference
        
        



       # print("rnn_hidden_states after adding == " , rnn_hidden_states[0,current_context,-(action_dim+1):])
        '''
        if not hasattr(self, "cur_skill"):
            self.cur_skill = torch.zeros(B, device=rnn_hidden_states.device).float()

        if not hasattr(self, "timeout"):
            self.timeout = torch.zeros(B, device=rnn_hidden_states.device).float()

        if not hasattr(self, "reset_mask"):
            self.reset_mask = torch.zeros(
                B, dtype=torch.bool, device=rnn_hidden_states.device
            ).float()

        state_index = list(range(self.timeout.shape[0]))
        if envs_to_pause is not None:
            envs_to_pause.sort()
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)

        self.timeout[state_index] *= masks.view(-1).float()

        mask = (
            rnn_hidden_states[torch.arange(B), current_context, -(action_dim+1)]
            == self.cur_skill[state_index]     ### check if the current skill is the same as the previous skill (at the previous time step)
        )
        self.timeout[state_index] += mask.float()
        self.timeout[state_index] *= mask.float()  ### accumlates the total time spent in the current skill

        # give a timeout reset to the skills, consistent with the expert behavior
        mask = self.timeout[state_index] > 200 & (
            (self.cur_skill[state_index] == 1)
            | (self.cur_skill[state_index] == 2)
        )
        self.reset_mask[mask] = True

        # give a reset when skill changes, consistent with the expert behavior
        self.reset_mask[state_index] = (
            rnn_hidden_states[torch.arange(B), current_context, -(action_dim+1)]
            != self.cur_skill[state_index]
        ).float()

        self.cur_skill[state_index] = rnn_hidden_states[
            torch.arange(B), current_context, -(action_dim+1)
        ]

        # add return if desired
        if rtgs is not None:
            rnn_hidden_states[
                torch.arange(B), current_context, -obs_dim : -obs_dim + 1
            ] = rtgs.cuda()
            rtgs = rnn_hidden_states[..., -obs_dim : -obs_dim + 1]
        '''
        rnn_hidden_states = rnn_hidden_states.contiguous()
        



        out = self.state_encoder(
            rnn_hidden_states[..., :-action_dim],
            rnn_hidden_states[..., -action_dim:],
            rtgs=rtgs,
        )
     #   print("current_context == " ,current_context)
      #  input()
        return out[torch.arange(B), current_context], rnn_hidden_states, {}
