import numpy as np
import torch
from torch import nn, optim
from demo.GAIL.pytorch_util import *


class Discriminator(nn.Module):
    def __init__(self, params, expert_pairs, kwargs):
        super().__init__(**kwargs)
        self.ob_dim = params["ob_dim"]
        self.ac_dim = params["ac_dim"]
        self.size = params["size"]
        self.n_layers = params["n_layers"]
        self.learning_rate = params["lr"]
        self.net = build_mlp(
            input_size=self.ob_dim * self.ac_dim,
            output_size=2,
            n_layers=self.n_layers,
            size=self.size,
            activation="tanh",
            output_activation="sigmoid",
        )
        self.optimizer = optim.Adam(
            self.mean_net.parameters(),
            self.learning_rate,
        )
        self.loss = nn.BCELoss()
        self.expert_pairs = expert_pairs

    def forward(self, obs_ac_pairs):
        obs_ac_pairs = from_numpy(obs_ac_pairs)
        pred = self.net(obs_ac_pairs)
        return pred

    def calculate_reward(self, pred, next_ob, rews, terminals):
        reward = torch.log(pred)
        torch.cat(reward, torch.tensor([0]))
        ret = np.zeros_like(reward)
        for i in np.reverse(np.arange(reward.shape[0])):
            ret[i] = reward[i] + torch.mul(
                torch.mul(self.gamma, ret[i + 1]), (1 - terminals)
            )
        return ret[:-1]

    def update(self, obs, acs, next_ob, rews, terminals):
        self.optimizer.zero_grad()
        pred_agent = self(torch.cat(obs, acs))
        pred_expert = self(self.expert_pairs)

        # Update the discriminator parameters
        loss = self.loss(pred_agent, np.zeros_like(pred_agent)) + self.loss(
            pred_expert, np.ones_like(pred_expert)
        )
        loss.backward()
        self.optimizer.step()

        # Return new reward for TRPO
        advantages = self.calculate_reward(
            pred_agent, next_ob, rews, terminals
        ).detach()
        return advantages
