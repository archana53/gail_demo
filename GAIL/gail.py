import pickle
import time
from collections import OrderedDict

import gym
import numpy as np
import torch

import pytorch_util as ptu
import utils
from discriminator import Discriminator
from logger import Logger
from policy import MLPPolicyGAIL


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

        # Make the gym environment
        if self.params["video_log_freq"] == -1:
            self.params["env_kwargs"]["render_mode"] = None
        self.env = gym.make(self.params["env_name"], **self.params["env_kwargs"])
        self.env.reset(seed=seed)

        # Maximum length for episodes
        self.params["ep_len"] = self.params["ep_len"] or self.env.spec.max_episode_steps
        MAX_VIDEO_LEN = self.params["ep_len"]

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params["agent_params"]["discrete"] = discrete

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params["agent_params"]["ac_dim"] = ac_dim
        self.params["agent_params"]["ob_dim"] = ob_dim

        # simulation timestep, will be used for video saving
        if "model" in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        else:
            self.fps = self.env.env.metadata["render_fps"]

        self.env.seed(seed)

        # Setup GAIL
        agent_params = self.params["agent_params"]
        self.expert_data = params["expert_data"]
        self.actor = MLPPolicyGAIL(
            ac_dim=ac_dim,
            ob_dim=ob_dim,
            n_layers=agent_params["n_layers"],
            size=agent_params["size"],
            discrete=agent_params["discrete"],
        )
        with open(self.expert_data, "rb") as f:
            self.expert_paths = pickle.loads(f.read())
        self.expert_oa_pairs = utils.convert_path_to_obs_actions(self.expert_paths)
        self.discriminator = Discriminator(
            params=agent_params, expert_pairs=self.expert_oa_pairs
        )

    def run_training_loop(self, n_iters, num_time_steps):
        for i in range(n_iters):
            print(f"*** Training Interation {i} ****")
            paths = utils.sample_trajectories(
                self.env,
                self.actor,
                self.params["batch_size"],
                self.params["ep_len"],
            )
            ob, ac, rew, next_ob, terminal = utils.convert_listofrollouts(paths)
            advantages = self.discriminator.update(ob, ac, rew, next_ob, terminal)
            training_logs = self.actor.update(
                paths["observations"], paths["actions"], advantages
            )

            self.perform_logging(i, paths, self.actor, None, training_logs)

    def perform_logging(
        self, itr, paths, eval_policy, train_video_paths, training_logs
    ):
        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
            self.env, eval_policy, self.params["eval_batch_size"], self.params["ep_len"]
        )

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
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

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

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
            last_log = training_logs[-1]  # Only use the last log for now
            logs.update(last_log)
            print(training_logs)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            self.logger.flush()