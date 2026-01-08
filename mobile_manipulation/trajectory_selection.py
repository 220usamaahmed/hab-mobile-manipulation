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

from mobile_manipulation.ppo.trainers.ppo_trainer_v0 import ConditionalDiffusionModel, NoiseScheduler, TrainConfig, QTransformer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else torch.device('cpu'))

ROOT = "/home/user/siddiquieu1/HRL-Usama/mobile-manipulation/ahmed_checkpoints/example_traj_data/"

MAX_BZ_SIZE = 1024
soft_Q_update = True

class Diffusion_buffer(Dataset):

    def __init__(self):
        self.normalise_return = True

        L = 10

        data = self._load_data()
        self.actions = data["actions"][:L]
        self.visual_states = data["visual_states"][:L]
        self.non_visual_states = data["non_visual_states"][:L]
        self.rewards = data["rewards"][:L]
        self.done = data["done"][:L]
        
        returns = data["returns"]
        self.raw_returns = [returns]
        self.raw_values = []
        self.returns_mean = np.mean(returns)
        self.returns_std = np.maximum(np.std(returns), 0.1)
        print("returns mean {}  std {}".format(self.returns_mean, self.returns_std))
        if self.normalise_return:
            returns = (returns - self.returns_mean) / self.returns_std
            print("returns normalised at mean {}, std {}".format(self.returns_mean, self.returns_std))
        else:
            print("no normal")

        self.len = self.visual_states.shape[0]
        # make sure same number of data points exist in all tasks
        # self.fake_len = int(np.maximum(np.round(10000 / self.len), 1)) * self.len
        # print(self.len, "data loaded", self.fake_len, "data faked")

        self.fake_actions: Optional[np.ndarray] = None
    
        
    def __getitem__(self, index):
        i = index % self.len
        actions = self.actions[i]
        rewards = self.rewards[i]
        visual_states = self.visual_states[i]
        non_visual_states = self.non_visual_states[i]
        return actions, rewards, visual_states, non_visual_states

    def __len__(self):
        return self.len
    
    def _load_data(self):
        all_actions = torch.load(path.join(ROOT, "action_data_nav_task_p_2.pt"))
        all_dones = torch.load(path.join(ROOT, "done_data_nav_task_p_2.pt"))
        all_non_visual_obs = torch.load(path.join(ROOT, "non_visual_obs_data_nav_task_p_2.pt"))
        all_visual_obs = torch.load(path.join(ROOT, "visual_obs_data_nav_task_p_2.pt"))
        all_rewards = torch.load(path.join(ROOT, "rewards_data_nav_task_p_2.pt"))

        if not all_dones[-1]:
            all_dones[-1] = True

        data = {}        
        data["visual_states"] = all_visual_obs
        data["non_visual_states"] = all_non_visual_obs
        data["actions"] = all_actions
        data["rewards"] = all_rewards.squeeze()
        data["done"] = all_dones.squeeze()

        assert data["done"][-1]
        data["returns"] = np.zeros((data["visual_states"].shape[0], 1))

        last = 0

        # NOTE: We set the returns based on the the MC returns: discount the
        # last return until the first one.
        # This gives us the initial value for all returns in each state

        for i in range(data["returns"].shape[0] - 1, -1, -1):
            last = data["rewards"][i] + 0.99 * last * (1. - data["done"][i])
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

            non_visual_states = self.non_visual_states[i].repeat(num_trajectories,1,1)
            visual_states = self.visual_states[i].repeat(num_trajectories,1,1)

            # non_visual_states = torch.repeat_interleave(non_visual_states, fake_actions.shape[1], dim=0)
            # visual_states = torch.repeat_interleave(visual_states, fake_actions.shape[1], dim=0)

            with torch.no_grad():
                q = score_model.calculateQ(visual_states, non_visual_states, torch.FloatTensor(fake_actions))
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
            values = np.sum(softmax(20 * values, axis=-1) * values, axis=-1, keepdims=1)
        else:
            values = np.percentile(values, 85, axis=-1, keepdims=1)
        if self.normalise_return:
            values = values * self.returns_std + self.returns_mean
        assert values.ndim == 2
        assert values.shape[0] == self.non_visual_states.shape[0]
        returns = np.zeros_like(values)
        last = 0
        num_truncated_traj = 0
        for i in range(returns.shape[0] - 1, -1, -1):
            bootstrap = self.rewards[i] + 0.99 * last * (1.0 - self.done[i])
            imagainary = values[i, 0]
            if bootstrap > imagainary:
                returns[i, 0] = bootstrap
            else:
                returns[i, 0] = imagainary
                num_truncated_traj += 1
            last = returns[i, 0]
        print("num_truncated_traj perc", num_truncated_traj / returns.shape[0])
        self.raw_returns.append(returns)
        self.returns_mean = np.mean(returns)
        self.returns_std = np.maximum(np.std(returns), 0.1)
        print("returns mean {}  std {}".format(self.returns_mean, self.returns_std))
        if self.normalise_return:
            returns = (returns - self.returns_mean) / self.returns_std
            print("returns normalised at mean {}, std {}".format(self.returns_mean, self.returns_std))
        else:
            print("no normal")

        # self.ys = np.concatenate([returns, self.actions], axis=-1)
        # self.ys = self.ys.astype(np.float32)

        self.rewards = torch.FloatTensor(returns.squeeze())

        print("update returns finished")


class ModelWrapper():
    def __init__(self):        
        self.diffusion_policy = ConditionalDiffusionModel()
        self.diffusion_policy.load_state_dict(torch.load("/home/user/siddiquieu1/HRL-Usama/mobile-manipulation/ahmed_checkpoints/model_all_tasks_eval_dataset_5_prev_obs_20_act_4400.pt",
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

        shape = (16, 20, 10)
        with torch.no_grad():
            actions = self.scheduler.sample(self.diffusion_policy, shape, visual_obs.to(device).to(torch.float32), non_visual_obs.to(device).to(torch.float32), device, num_random_samples=20)

        return actions

    def calculateQ(self, visual_states, non_visual_states, actions):
        # num_trajectories = actions.shape[0]

        # print(visual_states.shape, non_visual_states.shape, actions.shape)

        # vo_input = visual_states.repeat(num_trajectories,1,1).to(device).to(torch.float32)
        # nvo_input = non_visual_states.repeat(num_trajectories,1,1).to(device).to(torch.float32)
        # a_input = actions.to(device).to(torch.float32)

        # print(vo_input.shape, nvo_input.shape, a_input.shape)
        # exit()

        print("Non visual states", non_visual_states.shape)
        print("Visual states", visual_states.shape)
        print("Actions", actions.shape)

        """
        16 set of actions of length 20

        Non visual states torch.Size([16, 5, 21])
        Visual states torch.Size([16, 5, 512])
        Actions torch.Size([16, 20, 10])
        Q Values torch.Size([16])
        """
        

        q_values = self.q_value_network(visual_states.to(device).to(torch.float32), non_visual_states.to(device).to(torch.float32), actions.to(device).to(torch.float32))

        print("Q Values", q_values.shape)

        exit()

        return q_values


def train_critic(score_model, data_loader):
    data_loader.dataset.update_returns(score_model)

    optimizer = Adam(score_model.q_value_network.parameters(), lr=1e-3)

    bk_model_sd = copy.deepcopy(score_model.q_value_network.state_dict())

    for epoch in range(100):
        avg_loss = 0.
        num_items = 0
        for batch in tqdm.tqdm(data_loader):
            actions, rewards, visual_states, non_visual_states = batch
            rewards = rewards.to(device)

            qs = score_model.calculateQ(visual_states, non_visual_states, actions)
            loss = torch.mean((qs - rewards)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            score_model.condition = None
            avg_loss += loss.item() * actions.shape[0]
            num_items += actions.shape[0]
        if epoch != 0 and epoch % 10 == 0:
            print("Average Loss", avg_loss)

            data_loader.dataset.update_returns(score_model)

            score_model.q_value_network.load_state_dict(bk_model_sd)
            optimizer = Adam(score_model.q_value_network.parameters(), lr=1e-3)


def critic():
    score_model = ModelWrapper()
    dataset = Diffusion_buffer()

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Generate fake actions
    # all_actions = []
    # for i in tqdm.tqdm(range(0, len(dataset), 128)):
    #     visual_states = dataset.visual_states[i:128]
    #     non_visual_states = dataset.non_visual_states[i:128]

    #     fake_actions = score_model.sample(visual_states, non_visual_states)
    #     all_actions.append(fake_actions.cpu().numpy())

    fake_actions_path = path.join(ROOT, "fake_actions.npy")
    if path.exists(fake_actions_path):
        cached = np.load(fake_actions_path)
        if cached.shape[0] == len(dataset):
            dataset.fake_actions = cached
            print(f"Loaded cached fake actions from: {fake_actions_path}")
        else:
            print(
                f"Found cached fake actions but length mismatched (cache={cached.shape[0]}, dataset={len(dataset)}). Regenerating..."
            )

    if dataset.fake_actions is None:
        all_fake_actions = []
        for i in tqdm.tqdm(range(len(dataset))):
            _, _, visual_obs, non_visual_obs = dataset[i]
            fake_actions = score_model.sample(visual_obs, non_visual_obs)
            all_fake_actions.append(fake_actions.cpu().numpy())

        print("Generated fake actions")
        dataset.fake_actions = np.array(all_fake_actions)

        tmp_path = fake_actions_path + ".tmp.npy"
        np.save(tmp_path, dataset.fake_actions)
        os.replace(tmp_path, fake_actions_path)
        print(f"Saved fake actions to: {fake_actions_path}")

    print("Fake actions shape:", dataset.fake_actions.shape, dataset.rewards.shape)

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