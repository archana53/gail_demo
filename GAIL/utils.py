from typing import Dict, List, Tuple

import gym
import numpy as np
import torch

import GAIL.pytorch_util as ptu

MJ_ENV_NAMES = [
    "Ant-v4",
    "Walker2d-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "CartPole-v0",
    "Acrobot-v1",
]
MJ_ENV_KWARGS = {name: {"render_mode": "rgb_array"} for name in MJ_ENV_NAMES}
MJ_ENV_KWARGS["Ant-v4"]["use_contact_forces"] = True


def sample_trajectories(
    env, policy, min_timesteps_per_batch, max_path_length, render=False
):
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        current_path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(current_path)
        timesteps_this_batch += get_pathlength(current_path)
    return paths, timesteps_this_batch


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean


def sample_trajectory(env, policy, max_path_length, render=False):
    # initialize env for the beginning of a new rollout
    ob = env.reset()

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render image of the simulated env
        if render:
            if hasattr(env, "sim"):
                image_obs.append(
                    env.sim.render(camera_name="track", height=500, width=500)[::-1]
                )
            else:
                image_obs.append(env.render())

        # use the most recent ob to decide what to do
        obs.append(ob)

        # get action from the expert policy
        ac = policy.get_action(ob)
        if isinstance(ac, torch.Tensor):
            ac = ptu.to_numpy(ac)
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        rollout_done = done or max_path_length == steps
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def convert_path_to_obs_actions(paths, discrete=None, ac_dim=None):
    obs_action_pairs = []
    for path in paths:
        if discrete:
            path_len = get_pathlength(path)
            acs = np.zeros((path_len, ac_dim))
            acs[np.arange(path_len), np.squeeze((path["action"]).astype(int))] = 1
            obs_action_pairs.append(np.concatenate((path["observation"], acs), axis=1))
        else:
            obs_action_pairs.append(
                np.concatenate((path["observation"], path["action"]), axis=1)
            )
    return np.concatenate(obs_action_pairs)


def convert_listofrollouts(paths, concat_rew=True):
    """
    Take a list of rollout dictionaries
    and return separate arrays,
    where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals


############################################
############################################


def get_pathlength(path):
    return len(path["reward"])


class SampleTrajectoryVectorizedData:
    TRACKED_OBJECTS = [
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminals",
    ]

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs

        self.data: Dict[List[List[np.ndarray]]] = {
            name: self.init_vectorized_data_object() for name in self.TRACKED_OBJECTS
        }

    def init_vectorized_data_object(self) -> List[List]:
        num_envs = self.num_envs

        return [[] for _ in range(num_envs)]

    def update_object(self, key: str, updates: np.ndarray) -> None:
        num_envs = self.num_envs
        obj_to_update = self.data[key]
        terminals = self.data["terminals"]

        for i in range(num_envs):
            terminated = terminals[i] and terminals[i][-1]
            if terminated:
                continue

            obj_to_update[i].append(updates[i])

    def to_paths_list(self) -> List[Dict[str, np.ndarray]]:
        num_envs = self.num_envs
        data = self.data
        tracked_objects = self.TRACKED_OBJECTS

        def create_path(args: Tuple[np.ndarray]) -> Dict[str, np.ndarray]:
            return Path(args[0], [], *args[1:])

        paths: List[Dict[str, np.ndarray]] = []
        for i in range(num_envs):
            args = tuple(data[k][i] for k in tracked_objects)
            paths.append(create_path(args))

        return paths


def sample_trajectory_vectorized(
    env: gym.vector.VectorEnv, policy, max_path_length: int
) -> List[Dict[str, np.ndarray]]:
    """
    N_p -> number of parallel gym envs
    T_sample -> length of the path length of a particular sample
    D_o -> observation dim
    D_a -> action dim
    """

    # Initialize env for new rollouts
    # `observations` is of shape (N_p, D_o)
    observations: np.ndarray = env.reset()
    num_envs = observations.shape[0]

    # Initialize our data
    data = SampleTrajectoryVectorizedData(num_envs)

    steps = 0
    while True:
        # Record our observations
        data.update_object("observations", observations)

        # Get the policy actions of shape (N_p, D_a)
        actions = policy.get_action(observations)
        data.update_object("actions", actions)

        # Take the actions
        observations, rewards, terminals, _ = env.step(actions)

        # Record results of taking that action
        steps += 1
        data.update_object("next_observations", observations)
        data.update_object("rewards", rewards)

        # Process terminals
        max_path_length_reached = steps >= max_path_length
        if max_path_length_reached:
            terminals = np.full(num_envs, True, dtype=bool)

        data.update_object("terminals", terminals)

        # If all the envs are terminated, we exit
        if terminals.all():
            break

    paths = data.to_paths_list()
    return paths


def sample_trajectories_vectorized(
    env: gym.vector.VectorEnv, policy, min_timesteps_total: int, max_path_length: int
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_total:
        paths = sample_trajectory_vectorized(env, policy, max_path_length)

        for path in paths:
            timesteps_this_path = get_pathlength(path)
            paths.append(path)
            timesteps_this_batch += timesteps_this_path

            if timesteps_this_batch >= min_timesteps_total:
                break

    return paths, timesteps_this_batch


### TRPO Utils


def get_flat_grads(f, net):
    flat_grads = torch.cat(
        [
            grad.view(-1)
            for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
        ]
    )

    return flat_grads


def get_flat_params(net):
    return torch.cat([param.view(-1) for param in net.parameters()])


def set_params(net, new_flat_params):
    start_idx = 0
    for param in net.parameters():
        end_idx = start_idx + np.prod(list(param.shape))
        param.data = torch.reshape(new_flat_params[start_idx:end_idx], param.shape)

        start_idx = end_idx


def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def rescale_and_linesearch(
    g, s, Hs, max_kl, L, kld, old_params, pi, max_iter=10, success_ratio=0.1
):
    set_params(pi, old_params)
    L_old = L().detach()

    beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

    for _ in range(max_iter):
        new_params = old_params + beta * s

        set_params(pi, new_params)
        kld_new = kld().detach()

        L_new = L().detach()

        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
            return new_params

        beta *= 0.5

    print("The line search was failed!")
    return old_params
