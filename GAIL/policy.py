import abc
import itertools

import numpy as np
import torch
from torch import distributions, nn, optim
from torch.nn import functional as F

import GAIL.pytorch_util as ptu
from GAIL.utils import (
    conjugate_gradient,
    get_flat_grads,
    get_flat_params,
    normalize,
    rescale_and_linesearch,
    set_params,
)


class BasePolicy(object, metaclass=abc.ABCMeta):
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        discrete=False,
        learning_rate=1e-4,
        training=True,
        nn_baseline=True,
        entropy_coeff=0,
        clip_eps=0.2,
        max_grad_norm=100,
        gamma=0.99,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # PPO properties
        self.entropy_coeff = entropy_coeff
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm

        # TRPO properties
        self.max_kl = 0.001
        self.cg_damping = 1
        self.gamma = 0.99

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(), self.learning_rate)
            self.parameters = self.logits_na.parameters()
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate,
            )
            self.parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from HW1
        if not isinstance(obs, torch.Tensor):
            obs = ptu.from_numpy(obs.astype(np.float32))
        with torch.no_grad():
            action = ptu.to_numpy(self.forward(obs).sample())
        return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution


#####################################################
#####################################################


class MLPPolicyGAIL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def ppo_update(self, observations, actions, advantages, q_values=None):
        n_obs = observations.shape[0]

        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        q_values = ptu.from_numpy(q_values)
        q_values = normalize(q_values, (q_values).mean(), (q_values).std())

        logprobs_old = (
            self.forward(observations).log_prob(actions).reshape(advantages.shape)
        )

        for i in range(15):
            distribution = self.forward(observations)
            entropy = distribution.entropy().mean()
            logprobs = distribution.log_prob(actions).reshape(advantages.shape)
            ratios = (logprobs - logprobs_old.detach()).exp_()
            loss_actor1 = torch.mul(ratios, advantages)
            loss_actor2 = (
                torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                * advantages
            )
            loss_actor = -torch.min(loss_actor1, loss_actor2).mean()

            self.optimizer.zero_grad()
            (loss_actor - self.entropy_coeff * entropy).backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self.parameters, 40)
            # nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
            self.optimizer.step()

            if self.nn_baseline:
                self.baseline_optimizer.zero_grad()
                loss_b = self.baseline_loss(
                    self.baseline(observations).reshape(q_values.shape),
                    q_values,
                )
                loss_b.backward(retain_graph=False)
                # nn.utils.clip_grad_norm_(self.parameters, 40)
                self.baseline_optimizer.step()

            train_log = {
                "Training Loss": ptu.to_numpy(
                    ((loss_actor - self.entropy_coeff * entropy))
                ),
            }
        return train_log

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: update the policy using policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
        # is the expectation over collected trajectories of:
        # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
        # by the `forward` method

        logprobs = (
            self.forward(observations).log_prob(actions).reshape(advantages.shape)
        )

        loss = -torch.sum(
            torch.mul(
                logprobs,
                advantages,
            )
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # TODO

        if self.nn_baseline:
            ## TODO: update the neural network baseline using the q_values as
            ## targets. The q_values should first be normalized to have a mean
            ## of zero and a standard deviation of one.

            ## Note: You will need to convert the targets into a tensor using
            ## ptu.from_numpy before using it in the loss

            # TODO
            # Normalise the q_values
            q_values = normalize(q_values, np.mean(q_values), np.std(q_values))
            q_values = ptu.from_numpy(q_values)
            self.baseline_optimizer.zero_grad()
            loss_b = self.baseline_loss(
                self.baseline(observations).reshape(q_values.shape), q_values
            )
            loss_b.backward(retain_graph=False)
            # nn.utils.clip_grad_norm_(self.baseline.parameters(), self.max_grad_norm)
            self.baseline_optimizer.step()

        train_log = {
            "Training Loss": ptu.to_numpy(((loss))),
        }
        return train_log

    def run_baseline_prediction(self, observations):
        """
        Helper function that converts `observations` to a tensor,
        calls the forward method of the baseline MLP,
        and returns a np array

        Input: `observations`: np.ndarray of size [N, 1]
        Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())
