import numpy as np

from GAIL.policy import MLPPolicyGAIL
from GAIL.utils import normalize, unnormalize


class BaseAgent(object):
    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


class GAILAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(GAILAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = agent_params["gamma"]
        self.standardize_advantages = agent_params["standardize_advantages"]
        self.nn_baseline = agent_params["nn_baseline"]
        self.reward_to_go = agent_params["reward_to_go"]
        self.gae_lambda = agent_params["gae_lambda"]

        # actor/policy
        self.actor = MLPPolicyGAIL(
            self.agent_params["ac_dim"],
            self.agent_params["ob_dim"],
            self.agent_params["n_layers"],
            self.agent_params["size"],
            discrete=self.agent_params["discrete"],
            learning_rate=self.agent_params["learning_rate"],
            nn_baseline=self.agent_params["nn_baseline"],
        )

    def train(
        self, observations, actions, rewards, returns, next_observations, terminals
    ):
        """
        Training a PG agent refers to updating its actor using the given observations/actions
        and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO: update the PG actor/policy using the given batch of data
        # using helper functions to compute qvals and advantages, and
        # return the train_log obtained from updating the policy

        # Here rewards = q_values because we get the montecarlo return from the discriminator

        # Use terminals to find end of reward list
        advantages = self.estimate_advantage(observations, rewards, returns, terminals)
        train_log = self.actor.ppo_update(observations, actions, advantages, returns)

        return train_log

    def estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ):
        """
        Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            print(values_unnormalized.ndim, q_values.ndim)
            assert values_unnormalized.ndim == q_values.ndim
            values = unnormalize(
                values_unnormalized, np.mean(q_values), np.std(q_values)
            )
            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in np.arange(batch_size)[::-1]:
                    ## recursively compute advantage estimates starting from
                    ## timestep T.
                    ## is 1 if the state is the last in its trajectory, and
                    ## 0 otherwise.
                    if terminals[i] == 1:
                        advantages[i] = rewards[i] - values[i]
                    else:
                        advantages[i] = (
                            rewards[i]
                            + self.gamma * values[i + 1]
                            - values[i]
                            + self.gamma * self.gae_lambda * advantages[i + 1]
                        )

                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                ##compute advantage estimates using q_values, and values as baselines
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages to have a mean of zero
        # and a standard deviation of one
        if self.standardize_advantages:
            advantages = normalize(advantages, np.mean(advantages), np.std(advantages))

        return advantages
