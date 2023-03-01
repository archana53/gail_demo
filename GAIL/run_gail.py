import os
import time

from GAIL.agents import GAILAgent
from GAIL.gail import GAIL
from GAIL.utils import MJ_ENV_KWARGS, MJ_ENV_NAMES


class GAIL_Trainer(object):
    def __init__(self, params):
        #######################
        ## AGENT PARAMS
        #######################

        computation_graph_args = {
            "n_layers": params["n_layers"],
            "size": params["size"],
            "learning_rate": params["learning_rate"],
        }

        estimate_advantage_args = {
            "gamma": params["discount"],
            "standardize_advantages": not (params["dont_standardize_advantages"]),
            "reward_to_go": params["reward_to_go"],
            "nn_baseline": params["nn_baseline"],
            "gae_lambda": params["gae_lambda"],
        }

        agent_params = {
            **computation_graph_args,
            **estimate_advantage_args,
        }

        self.params = params
        self.params["env_kwargs"] = MJ_ENV_KWARGS[self.params["env_name"]]
        self.params["agent_class"] = GAILAgent
        self.params["agent_params"] = agent_params
        ################
        ## RL TRAINER
        ################

        self.gail = GAIL(self.params)  ## HW1: you will modify this

    def run_training_loop(self):
        self.gail.run_training_loop(
            n_iters=self.params["n_iter"],
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()

    # Expert data parameters
    parser.add_argument(
        "--expert_policy_file", "-epf", type=str, required=False
    )  # relative to where you're running this script from
    parser.add_argument("--num_expert_trajectories", "-net", type=int, default=4)
    parser.add_argument(
        "--expert_data",
        "-ed",
        type=str,
        default="gail_demo/expert/expert_data_HalfCheetah-v4.pkl",
    )  # relative to where you're running this script from

    # Environment and Experiment
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--exp_name", type=str, default="todo")

    # Training parameters
    parser.add_argument("--n_iter", "-n", type=int, default=300)
    parser.add_argument("--reward_to_go", "-rtg", action="store_true", default=True)
    parser.add_argument("--nn_baseline", action="store_true", default=True)
    parser.add_argument("--gae_lambda", type=float, default=0.97)
    parser.add_argument(
        "--dont_standardize_advantages", "-dsa", action="store_true", default=False
    )
    parser.add_argument("--states_only", action="store_true", default=False)
    parser.add_argument(
        "--batch_size", "-b", type=int, default=5000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=5000
    )  # steps collected per eval iteration
    parser.add_argument(
        "--train_batch_size", "-tb", type=int, default=1000
    )  ##steps used per gradient step

    # Learning Parameters
    parser.add_argument("--discount", type=float, default=0.995)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--size", "-s", type=int, default=100)
    parser.add_argument("--multiprocess_gym_envs", type=int, default=1)
    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    parser.add_argument("--save_params", action="store_true")
    parser.add_argument("--action_noise_std", type=float, default=0)

    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data")
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = (
        args.exp_name + "_" + args.env_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    params["logdir"] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    ###################
    ### RUN TRAINING
    ###################

    trainer = GAIL_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
