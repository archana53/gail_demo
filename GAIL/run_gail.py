import os
import time

from gail import GAIL
from utils import MJ_ENV_KWARGS, MJ_ENV_NAMES


class GAIL_Trainer(object):
    def __init__(self, params):
        #######################
        ## AGENT PARAMS
        #######################

        agent_params = {
            "n_layers": params["n_layers"],
            "size": params["size"],
            "learning_rate": params["learning_rate"],
            "max_replay_buffer_size": params["max_replay_buffer_size"],
        }

        self.params = params
        self.params["agent_params"] = agent_params
        self.params["env_kwargs"] = MJ_ENV_KWARGS[self.params["env_name"]]

        ################
        ## RL TRAINER
        ################

        self.gail_trainer = GAIL(self.params)  ## HW1: you will modify this

    def run_training_loop(self):
        self.gail_trainer.run_training_loop(
            n_iter=self.params["n_iter"],
            initial_expertdata=self.params["expert_data"],
            policy=self.gail_trainer.agent,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert_policy_file", "-epf", type=str, required=False
    )  # relative to where you're running this script from
    parser.add_argument(
        "--expert_data", "-ed", type=str, required=True
    )  # relative to where you're running this script from
    parser.add_argument(
        "--env_name",
        "-env",
        type=str,
        help=f'choices: {", ".join(MJ_ENV_NAMES)}',
        required=True,
    )
    parser.add_argument(
        "--exp_name", "-exp", type=str, default="pick an experiment name", required=True
    )
    parser.add_argument("--ep_len", type=int)

    parser.add_argument("--n_iter", "-n", type=int, default=100)

    parser.add_argument(
        "--batch_size", type=int, default=5000
    )  # training data collected (in the env) during each iteration

    parser.add_argument(
        "--n_layers", type=int, default=2
    )  # depth, of policy to be learned
    parser.add_argument(
        "--size", type=int, default=100
    )  # width of each layer, of policy to be learned
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=5e-3
    )  # LR for supervised learning

    parser.add_argument("--video_log_freq", type=int, default=5)
    parser.add_argument("--scalar_log_freq", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", type=int, default=0)
    parser.add_argument("--max_replay_buffer_size", type=int, default=1000000)
    parser.add_argument("--save_params", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")
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
