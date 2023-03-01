import pickle
import time
from collections import OrderedDict

import gym
import numpy as np
import torch

import GAIL.pytorch_util as ptu
import GAIL.utils as utils
from GAIL.agents import GAILAgent
from GAIL.discriminator import Discriminator
from GAIL.logger import Logger


class GAIL:
    def __init__(self, params):
        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params["logdir"])

        # Set random seeds
        seed = self.params["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(use_gpu=not self.params["no_gpu"], gpu_id=self.params["which_gpu"])

        #############
        ## ENV
        #############

        # Video Logging Setup
        if self.params["video_log_freq"] == -1:
            self.log_video = 0
            self.params["env_kwargs"]["render_mode"] = None

        # Initialise vectorized gym environment
        self.num_envs = self.params["multiprocess_gym_envs"]
        if self.num_envs > 1:
            dummy_env = gym.make(self.params["env_name"], **self.params["env_kwargs"])
            env_for_properties = dummy_env
            self.env = gym.vector.make(
                self.params["env_name"],
                num_envs=self.num_envs,
                **self.params["env_kwargs"],
            )
        else:
            self.env = gym.make(self.params["env_name"], **self.params["env_kwargs"])
            env_for_properties = self.env

        self.env.reset(seed=seed)

        # Maximum length for episodes
        self.params["ep_len"] = (
            self.params["ep_len"] or env_for_properties.spec.max_episode_steps
        )
        MAX_VIDEO_LEN = self.params["ep_len"]

        # Is this env continuous, or self.discrete?
        discrete = isinstance(env_for_properties.action_space, gym.spaces.Discrete)
        self.discrete = discrete
        self.params["agent_params"]["discrete"] = discrete

        # Is the observation an image?
        img = len(env_for_properties.observation_space.shape) > 2

        # Observation and action sizes
        ob_dim = (
            env_for_properties.observation_space.shape
            if img
            else env_for_properties.observation_space.shape[0]
        )
        ac_dim = (
            env_for_properties.action_space.n
            if discrete
            else env_for_properties.action_space.shape[0]
        )

        # Set obs, ac dimensions
        self.params["agent_params"]["ac_dim"] = ac_dim
        self.params["agent_params"]["ob_dim"] = ob_dim

        # simulation timestep, will be used for video saving
        if "model" in dir(env_for_properties):
            self.fps = 1 / env_for_properties.model.opt.timestep
        elif "video.frames_per_second" in env_for_properties.env.metadata.keys():
            self.fps = env_for_properties.env.metadata["video.frames_per_second"]

        self.env.seed(seed)

        # Setup GAIL
        agent_params = self.params["agent_params"]
        self.expert_data = self.params["expert_data"]
        self.agent = GAILAgent(self.env, agent_params)
        with open(self.expert_data, "rb") as f:
            self.expert_paths = pickle.loads(f.read())
        obs, _, _, _, _ = utils.convert_listofrollouts(self.expert_paths)
        if params["states_only"]:
            self.expert_samples = obs
        else:
            self.expert_samples = utils.convert_path_to_obs_actions(
                self.expert_paths,
                discrete,
                ac_dim,
            )
        self.discriminator = Discriminator(
            params=agent_params,
            expert_pairs=self.expert_samples,
            states_only=params["states_only"],
        )
        self.gamma = agent_params["gamma"]

        self.IL_trajectories = {}
        self.REWARDS = []

    def run_training_loop(self, n_iters):
        self.total_envsteps = 0
        self.start_time = time.time()
        training_logs = []
        for itr in range(n_iters):
            if (
                itr % self.params["video_log_freq"] == 0
                and self.params["video_log_freq"] != -1
            ):
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params["scalar_log_freq"] == -1:
                self.logmetrics = False
            elif itr % self.params["scalar_log_freq"] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            print(f"*** Training Interation {itr} ****")
            paths, steps_this_batch = self.collect_training_trajectories(
                self.agent.actor,
                self.params["batch_size"],
                self.params["ep_len"],
            )
            self.total_envsteps += steps_this_batch
            ob, ac, rew, next_ob, terminal = utils.convert_listofrollouts(paths)

            rl_returns = np.zeros(ob.shape[0] + 1)
            for i in (np.arange(rew.shape[0]))[::-1]:
                rl_returns[i] = rew[i] + (rl_returns[i + 1] * self.gamma) * (
                    1 - terminal[i]
                )

            if self.params["states_only"]:
                novice_data = ob
            else:
                novice_data = utils.convert_path_to_obs_actions(
                    paths, self.discrete, self.params["agent_params"]["ac_dim"]
                )
            rewards, returns = self.discriminator.update(
                novice_data, rew, next_ob, terminal
            )
            self.REWARDS.append(
                {
                    "rl_reward": rew,
                    "rl_return": rl_returns[:-1],
                    "disc_reward": rewards,
                    "disc_returns": returns,
                }
            )

            train_log = self.agent.train(ob, ac, rewards, returns, next_ob, terminal)
            training_logs.append(train_log)

            self.perform_logging(itr, paths, self.agent.actor, None, training_logs)
        with open(f"reward_data_{self.params['env_name']}.pkl", "wb") as f:
            pickle.dump(self.REWARDS, f)

    def collect_training_trajectories(
        self, collect_policy, batch_size, max_path_length
    ):
        print("\nCollecting data to be used for training...")
        env = self.env
        max_path_length = self.params["ep_len"]
        num_envs = self.num_envs

        if num_envs > 1:
            paths, envsteps_this_batch = utils.sample_trajectories_vectorized(
                env, collect_policy, batch_size, max_path_length
            )
        else:
            paths, envsteps_this_batch = utils.sample_trajectories(
                env, collect_policy, batch_size, max_path_length, False
            )

        return paths, envsteps_this_batch

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):
        last_log = all_logs[-1]
        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        if self.num_envs > 1:
            eval_paths, _ = utils.sample_trajectories_vectorized(
                self.env,
                eval_policy,
                self.params["eval_batch_size"],
                self.params["ep_len"],
            )

            """
            eval_returns = np.mean(
                [eval_path["reward"].sum() for eval_path in eval_paths]
            )
            with open(
                f"trajectories_{self.params['env_name']}_{int(eval_returns)}", "wb"
            ) as f:
                pickle.dump(eval_paths, f)
            """
        else:
            eval_paths, _ = utils.sample_trajectories(
                self.env,
                eval_policy,
                self.params["eval_batch_size"],
                self.params["ep_len"],
            )

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print("\nCollecting video rollouts eval")
            eval_video_paths = utils.sample_n_trajectories(
                self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True
            )

            # save train/eval videos
            print("\nSaving train rollouts as videos...")
            self.logger.log_paths_as_videos(
                train_video_paths,
                itr,
                fps=self.fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="train_rollouts",
            )
            self.logger.log_paths_as_videos(
                eval_video_paths,
                itr,
                fps=self.fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            ##Log the discriminator rewards and true RL rewards

            ###Save Eval Trajectories at every iteration

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            self.logger.flush()
