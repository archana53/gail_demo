import numpy as np
import torch
from torch import nn, optim

import GAIL.pytorch_util as ptu


class Discriminator(nn.Module):
    def __init__(self, params, expert_pairs, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = params["ob_dim"]
        self.ac_dim = params["ac_dim"]
        self.size = params["size"]
        self.n_layers = params["n_layers"]
        self.gamma = params["gamma"]
        self.learning_rate = params["learning_rate"]
        self.net = ptu.build_mlp(
            input_size=self.ob_dim + self.ac_dim,
            output_size=1,
            n_layers=self.n_layers,
            size=self.size,
            activation="tanh",
            output_activation="sigmoid",
        )
        self.optimizer = optim.Adam(
            self.net.parameters(),
            self.learning_rate,
        )
        self.loss = nn.BCELoss()
        self.expert_pairs = ptu.from_numpy(expert_pairs)
        self.net.to(ptu.device)

    def forward(self, obs_ac_pairs):
        if not isinstance(obs_ac_pairs, torch.Tensor):
            obs_ac_pairs = ptu.from_numpy(obs_ac_pairs)
        pred = self.net(obs_ac_pairs)
        return pred

    def calculate_reward(self, pred, terminals):
        reward = -torch.log(pred).squeeze(1)
        reward = torch.cat((reward, torch.tensor([0]).to(ptu.device)))
        ret = torch.zeros_like(reward)
        for i in (np.arange(reward.shape[0] - 1))[::-1]:
            ret[i] = reward[i] + (ret[i + 1] * self.gamma) * (1 - terminals[i])
        return ret[:-1].detach()

    def update(self, obs_acs, rews, next_ob, terminals):
        obs_acs = ptu.from_numpy(obs_acs)
        terminals = ptu.from_numpy(terminals)
        pred_agent = self(obs_acs)
        pred_expert = self(self.expert_pairs)

        # Update the discriminator parameters
        self.optimizer.zero_grad()
        loss = self.loss(pred_agent, torch.zeros_like(pred_agent)) + self.loss(
            pred_expert, torch.ones_like(pred_expert)
        )
        loss.backward()
        self.optimizer.step()

        # Return new reward for TRPO
        advantages = self.calculate_reward(pred_agent, terminals)
        return ptu.to_numpy(advantages)
