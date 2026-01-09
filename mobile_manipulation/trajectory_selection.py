import os
import copy

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

from mobile_manipulation.ppo.trainers.ppo_trainer_v0 import TrainConfig, QTransformer

device = torch.device("cuda" if torch.cuda.is_available() else torch.device('cpu'))

# Path to the folder containing the sample data files.
SAMPLE_DATA_ROOT = "/home/user/shokry/sample_data"
SKILLS = ["nav", "place", "pick"]

RETURN_UPDATE_TYPE = "SOFT-Q"  # or "PERCENTILE"
NORMALIZE_RETURN = True

EPOCHS = 100
UPDATE_RETURNS_EVERY = 10
SAVE_MODEL_EVERY = 20
RETURN_UPDATE_BATCH_SIZE = 16
TRAIN_BATCH_SIZE = 128
MODEL_SAVE_FOLDER = "/home/user/siddiquieu1/HRL-Usama/mobile-manipulation/ahmed_checkpoints"

class Buffer(Dataset):

    def __init__(self):
        self._load_data()
        self._set_mc_returns()
        if NORMALIZE_RETURN:
            self._normalize_returns()
        

    def _load_data(self):
        super().__init__()

        visual_obs_list = []
        non_visual_obs_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        predicted_actions_list = []

        for skill in SKILLS:
            visual_obs = torch.load(
                os.path.join(SAMPLE_DATA_ROOT, f"visual_obs_data_{skill}_task_p_1.pt")
            )
            non_visual_obs = torch.load(
                os.path.join(SAMPLE_DATA_ROOT, f"non_visual_obs_data_{skill}_task_p_1.pt")
            )
            actions = torch.load(
                os.path.join(SAMPLE_DATA_ROOT, f"action_data_{skill}_task_p_1.pt")
            )
            rewards = torch.load(
                os.path.join(SAMPLE_DATA_ROOT, f"rewards_data_{skill}_task_p_1.pt")
            )
            dones = torch.load(
                os.path.join(SAMPLE_DATA_ROOT, f"done_data_{skill}_task_p_1.pt")
            )
            predicted_actions = torch.load(
                os.path.join(
                    SAMPLE_DATA_ROOT, f"predicted_actions_diffusion_{skill}_task_p_1.pt"
                )
            )

            print("Loaded data for skill:", skill)
            print("  Visual obs", visual_obs.shape)
            print("  Non-visual obs", non_visual_obs.shape)
            print("  Actions", actions.shape)
            print("  Rewards", rewards.shape)
            print("  Dones", dones.shape)
            print("  Predicted actions", predicted_actions.shape)

            visual_obs_list.append(visual_obs)
            non_visual_obs_list.append(non_visual_obs)
            actions_list.append(actions)
            rewards_list.append(rewards)
            dones_list.append(dones)
            predicted_actions_list.append(predicted_actions)

            # break # NOTE: For testing, only load one skill.

        # Concatenate along the first dimension (trajectory dimension)
        self.visual_obs = torch.cat(visual_obs_list, dim=0)
        self.non_visual_obs = torch.cat(non_visual_obs_list, dim=0)
        self.actions = torch.cat(actions_list, dim=0)
        self.returns = torch.cat(rewards_list, dim=0).squeeze()
        self.dones = torch.cat(dones_list, dim=0).squeeze()
        self.predicted_actions = torch.cat(predicted_actions_list, dim=0)

        print("\nCombined dataset shapes (all skills):")
        print("Visual obs", self.visual_obs.shape)
        print("Non-visual obs", self.non_visual_obs.shape)
        print("Actions", self.actions.shape)
        print("Rewards", self.returns.shape)
        print("Dones", self.dones.shape)
        print("Predicted actions", self.predicted_actions.shape)


    def _set_mc_returns(self):
        assert self.dones[-1] == 1, "Last done flag must be 1 for MC return calculation."

        updated_returns = torch.zeros_like(self.returns)

        last = 0
        for i in range(updated_returns.shape[0] - 1, -1, -1):
            last = self.returns[i] + 0.99 * last * (1. - self.dones[i])
            updated_returns[i] = last

        self.returns = updated_returns

    def _normalize_returns(self):
        returns = self.returns.cpu().numpy()

        self.returns_mean = np.mean(returns)
        self.returns_std = np.maximum(np.std(returns), 0.1)
        print("returns mean {}  std {}".format(self.returns_mean, self.returns_std))
        self.returns = torch.FloatTensor((returns - self.returns_mean) / self.returns_std)
        print("returns normalised at mean {}, std {}".format(self.returns_mean, self.returns_std))

    def __len__(self):
        return self.visual_obs.shape[0]
    
    def __getitem__(self, idx):
        idx = idx % self.visual_obs.shape[0]
        return self.visual_obs[idx], self.non_visual_obs[idx], self.actions[idx], self.predicted_actions[idx], self.returns[idx]

    def update_returns(self, q_model, batch_size = RETURN_UPDATE_BATCH_SIZE):
        # Num trajectories per state = 20
        # We process batch_size states at a time -> batch_size * 20 actions

        all_qs = []

        print("Updating returns")

        for i in tqdm(range(0, len(self), batch_size)):
            visual_obs = self.visual_obs[i:i+batch_size]
            non_visual_obs = self.non_visual_obs[i:i+batch_size]
            predicted_actions = self.predicted_actions[i:i+batch_size]

            num_trajectories = predicted_actions.shape[1]

            # Reshape/stack so we evaluate each (state, trajectory) pair as a batch item.
            # visual_obs:        [B, 5, d_vis]      -> [B*T, 5, d_vis]
            # non_visual_obs:    [B, 5, d_nonvis]   -> [B*T, 5, d_nonvis]
            # predicted_actions: [B, T, 20, 10]     -> [B*T, 20, 10]
            # where B=batch_size, T=num_trajectories
            B = visual_obs.shape[0]
            T = num_trajectories

            visual_obs = (
                visual_obs.unsqueeze(1)
                .repeat(1, T, 1, 1)
                .reshape(B * T, visual_obs.shape[1], visual_obs.shape[2])
            )
            non_visual_obs = (
                non_visual_obs.unsqueeze(1)
                .repeat(1, T, 1, 1)
                .reshape(B * T, non_visual_obs.shape[1], non_visual_obs.shape[2])
            )
            predicted_actions = predicted_actions.contiguous().reshape(
                B * T, predicted_actions.shape[2], predicted_actions.shape[3]
            )
            
            # print("Visual obs batch shape:", visual_obs.shape)
            # print("Non-visual obs batch shape:", non_visual_obs.shape)
            # print("Predicted actions batch shape:", predicted_actions.shape)

            with torch.no_grad():
                qs = q_model(
                    visual_obs.to(device).to(torch.float32), 
                    non_visual_obs.to(device).to(torch.float32), 
                    predicted_actions.to(device).to(torch.float32)
                )
                qs = qs.reshape((-1, predicted_actions.shape[1]))
                all_qs.append(qs.cpu().numpy())

        values = np.concatenate(all_qs)

        if RETURN_UPDATE_TYPE == "SOFT-Q":
            values = np.sum(softmax(20 * values, axis=-1) * values, axis=-1, keepdims=1)
        elif RETURN_UPDATE_TYPE == "PERCENTILE":
            values = np.percentile(values, 85, axis=-1, keepdims=1)
        else:
            raise NotImplementedError
        if NORMALIZE_RETURN:
            values = values * self.returns_std + self.returns_mean
        assert values.ndim == 2
        assert values.shape[0] == self.non_visual_obs.shape[0]
        returns = np.zeros_like(values)
        last = 0
        num_truncated_traj = 0
        for i in range(returns.shape[0] - 1, -1, -1):
            bootstrap = self.returns[i] + 0.99 * last * (1.0 - self.dones[i])
            imagainary = values[i, 0]
            if bootstrap > imagainary:
                returns[i, 0] = bootstrap
            else:
                returns[i, 0] = imagainary
                num_truncated_traj += 1
            last = returns[i, 0]
        print("num_truncated_traj perc", num_truncated_traj / returns.shape[0])
        self.returns_mean = np.mean(returns)
        self.returns_std = np.maximum(np.std(returns), 0.1)
        print("returns mean {}  std {}".format(self.returns_mean, self.returns_std))
        if NORMALIZE_RETURN:
            returns = (returns - self.returns_mean) / self.returns_std
            print("returns normalised at mean {}, std {}".format(self.returns_mean, self.returns_std))
        else:
            print("no normal")

        self.returns = torch.FloatTensor(returns.squeeze())

        print("update returns finished")


def train_critic(q_model, buffer):
    data_loader = DataLoader(buffer, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    optimizer = Adam(q_model.parameters(), lr=1e-3)
    bk_model = copy.deepcopy(q_model.state_dict())

    for epoch in range(EPOCHS):
        avg_loss = 0
        num_items = 0
        for batch in tqdm(data_loader):
            visual_obs, non_visual_obs, actions, _, returns = batch

            # print("Visual obs batch shape:", visual_obs.shape)
            # print("Non-visual obs batch shape:", non_visual_obs.shape)
            # print("Actions batch shape:", actions.shape)
            # print("Returns batch shape:", returns.shape)

            qs = q_model(
                visual_obs.to(device).to(torch.float32), 
                non_visual_obs.to(device).to(torch.float32), 
                actions.to(device).to(torch.float32)
            )

            loss = torch.mean((qs - returns.to(device).to(torch.float32)) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * actions.shape[0]
            num_items += actions.shape[0]

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss / num_items}")

        if (epoch + 1) % UPDATE_RETURNS_EVERY == 0:
            buffer.update_returns(q_model)
            q_model.load_state_dict(bk_model)
            optimizer = Adam(q_model.parameters(), lr=1e-3)
        if (epoch + 1) % SAVE_MODEL_EVERY == 0:
            model_save_path = os.path.join(MODEL_SAVE_FOLDER, f"q_model_epoch_{epoch+1}.pt")
            torch.save(q_model.state_dict(), model_save_path)
            print(f"Saved Q-model checkpoint at epoch {epoch+1} to {model_save_path}")


if __name__ == "__main__":
    buffer = Buffer()
    
    cfg = TrainConfig()
    q_model = QTransformer(
        d_vis=cfg.d_vis,
        d_nonvis=cfg.d_nonvis,
        d_act=cfg.d_act,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        hist_len=cfg.hist_len,
        horizon=cfg.horizon,
    )
    q_model = q_model.to(cfg.device)
    q_model.eval()
    
    train_critic(q_model, buffer)
            