import os
import numpy as np
import random
import torch
import shutil
import argparse

from safety_gridworld import *

# SARSA agents
from safe_sarsa_agent import SafeSarsaAgent
from lyp_sarsa_agent import LypSarsaAgent
from utils import log


def get_filename(args):
    """
    Filter what to print based on agent type, and return the filename for the models and logs

    :param args:
    :return: name (string)
    """
    # all the params that go in for the logging go over her
    toprint = ['agent', 'lr', 'batch_size', ]
    args_param = vars(args)

    if args.agent == "ppo":
        toprint += ['num_envs', 'ppo_updates', 'gae', 'clip', 'traj_len', 'beta', 'value_loss_coef', ]
    elif args.agent == "a2c":
        toprint += ['num_envs', 'traj_len', 'critic_lr', 'beta']
    elif args.agent == "sarsa":
        toprint += ['num_envs', 'traj_len', ]
    # bvf agents
    elif args.agent == "bvf-sarsa":
        toprint += ['num_envs', 'traj_len', 'cost_reverse_lr', 'cost_q_lr', ]
    # elif args.agent == "safe-ppo":
    #     toprint += ['num_envs', 'cost_reverse_lr', 'cost_q_lr', 'traj_len', 'beta',
    #                 'ppo_updates', 'gae', 'clip', 'value_loss_coef', ]
    elif args.agent == "bvf-ppo":
        toprint += ['num_envs', 'cost_reverse_lr', 'cost_q_lr', 'traj_len', 'beta', 'gae', 'clip',
                    'ppo_updates', 'd0', 'cost_sg_coeff', 'prob_alpha']
    elif args.agent == "safe-a2c":
        toprint += ['num_envs', 'cost_reverse_lr', 'cost_q_lr', 'traj_len', 'beta']
    # lyapunov agents
    elif args.agent == "lyp-a2c":
        toprint += ['num_envs', 'cost_q_lr', 'traj_len', 'beta', 'd0', 'cost_sg_coeff']
    elif args.agent == "lyp-sarsa":
        toprint += ['num_envs', 'traj_len', 'cost_q_lr', 'd0', 'cost_sg_coeff']
    elif args.agent == "lyp-ppo":
        toprint += ['num_envs', 'cost_q_lr', 'ppo_updates', 'traj_len', 'value_loss_coef', 'd0',
                    'cost_sg_coeff', 'prob_alpha']
    else:
        raise Exception("Not implemented yet!!")

    # for every safe agent
    if "safe" or "bvf" in args.agent:
        toprint += ['d0', 'cost_sg_coeff']

    # if early stopping for ppo
    if args.early_stop:
        toprint += ['early_stop']

    name = ''
    for arg in toprint:
        name += '_{}_{}'.format(arg, args_param[arg])


    return name


def create_env(args):
    """
    the main method which creates any environment
    """
    env = None


    if args.env_name == "Gridworld":
        # create the Gridworld with pits env
        env = PitWorld(size = 14,
                       max_step = 400,
                       per_step_penalty = -1.0,
                       goal_reward = 1000.0,
                       obstace_density = 0.3,
                       constraint_cost = 1.0,
                       random_action_prob = 0.005,
                       one_hot_features=True,
                       rand_goal=False, # for testing purposes
                       )
    else:
        raise Exception("Not implemented yet")

    return env


def get_args():
    """
    Utility for getting the arguments from the user for running the experiment

    :return: parsed arguments
    """

    # Env
    parser = argparse.ArgumentParser(description='collect arguments')

    parser.add_argument('--env-name', default='pg',
                        help="pg: point gather env\n"\
                             "cheetah: safe-cheetah env\n"\
                             "Gridworld: Gridworld world env\n"\
                            "pc: point circle env\n"\
                        )

    parser.add_argument('--agent', default='ppo',
                        help="the RL algo to use\n"\
                             "ppo: for ppo\n"\
                             "lyp-ppo: for Lyapnunov based ppo\n" \
                             "bvf-ppo: for Backward value function based ppo\n" \
                             "sarsa: for n-step sarsa\n" \
                             "lyp-sarsa: for Lyapnunov based sarsa\n"\
                             "bvf-sarsa: for Backward Value Function based sarsa"\
                        )
    parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")
    parser.add_argument('--d0', type=float, default=5.0, help="the threshold for safety")

    # Actor Critic arguments goes here
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                            help="learning rate")
    parser.add_argument('--target-update-steps', type=int, default=int(1e4),
                        help="number of steps after to train the agent")
    parser.add_argument('--beta', type=float, default=0.001, help='entropy regularization')
    parser.add_argument('--critic-lr', type=float, default=1e-3, help="critic learning rate")
    parser.add_argument('--updates-per-step', type=int, default=1, help='model updates per simulator step (default: 1)')
    parser.add_argument('--tau', type=float, default=0.001, help='soft update rule for target netwrok(default: 0.001)')

    # PPO arguments go here
    parser.add_argument('--num-envs', type=int, default=10, help='the num of envs to gather data in parallel')
    parser.add_argument('--ppo-updates', type=int, default=1, help='num of ppo updates to do')
    parser.add_argument('--gae', type=float, default=0.95, help='GAE coefficient')
    parser.add_argument('--clip', type=float, default=0.2, help='clipping param for PPO')
    parser.add_argument('--traj-len', type=int, default= 10, help="the maximum length of the trajectory for an update")
    parser.add_argument('--early-stop', action='store_true',
                        help="early stop pi training based on target KL ")

    # Optmization arguments
    parser.add_argument('--lr', type=float, default=1e-2,
                            help="learning rate")
    parser.add_argument('--adam-eps', type=float, default=0.95, help="momenturm for RMSProp")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='size of minibatch for ppo/ ddpg update')

    # Safety params
    parser.add_argument('--cost-reverse-lr', type=float, default=5e-4,
                            help="reverse learning rate for reviewer")
    parser.add_argument('--cost-q-lr', type=float, default=5e-4,
                            help="reverse learning rate for critic")
    parser.add_argument('--cost-sg-coeff', type=float, default=0.0,
                            help="the coeeficient for the safe guard policy, minimizes the cost")
    parser.add_argument('--prob-alpha', type=float, default=0.6,
                        help="the kappa parameter for the target networks")
    parser.add_argument('--target', action='store_true',
                        help="use the target network based implementation")

    # Training arguments
    parser.add_argument('--num-steps', type=int, default=int(1e4),
                        help="number of steps to train the agent")
    parser.add_argument('--num-episodes', type=int, default=int(1e4),
                        help="number of episodes to train the agetn")
    parser.add_argument('--max-ep-len', type=int, default=int(15),
                        help="number of steps in an episode")

    # Evaluation arguments
    parser.add_argument('--eval-every', type=float, default=1000,
                        help="eval after these many steps")
    parser.add_argument('--eval-n', type=int, default=1,
                        help="average eval results over these many episodes")

    # Experiment specific
    parser.add_argument('--gpu', action='store_true', help="use the gpu and CUDA")
    parser.add_argument('--log-mode-steps', action='store_true',
                            help="changes the mode of logging w.r.r num of steps instead of episodes")
    parser.add_argument('--log-every', type=int, default=100,
                        help="logging schedule for training")
    parser.add_argument('--checkpoint-interval', type=int, default=1e5,
                        help="when to save the models")
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=str, default='/tmp/safe/models/')
    parser.add_argument('--log-dir', type=str, default="/tmp/safe/logs/")
    parser.add_argument('--reset-dir', action='store_true',
                        help="give this argument to delete the existing logs for the current set of parameters")

    args = parser.parse_args()
    return args





########################################################################################################################
# get the args from argparse
args = get_args()
# dump the args
log(args)


# initialize a random seed for the experiment
seed = np.random.randint(1,1000)
args.seed = seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# pytorch multiprocessing flag
torch.set_num_threads(1)

# check the device here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get the filename
name = get_filename(args)
args.out = os.path.join(args.out, args.env_name, args.agent, name)
tb_log_dir = os.path.join(args.log_dir, args.env_name, args.agent, name, 'tb_logs')
if args.reset_dir:
    shutil.rmtree(args.out, ignore_errors=True) #delete the results dir
    shutil.rmtree(tb_log_dir, ignore_errors=True) #delete the tb dir

os.makedirs(args.out, exist_ok=True)
os.makedirs(tb_log_dir, exist_ok=True)

# don't use tb on cluster
tb_writer = None

# print the dir in the beginning
print("Log dir", tb_log_dir)
print("Out dir", args.out)


agent = None

# create the env here
env = create_env(args)
print(env.to_string())
# create the agent here
#  SARSA based agent
if args.agent == "bvf-sarsa":
    agent = SafeSarsaAgent(args, env, writer=tb_writer)
elif args.agent == "lyp-sarsa":
    agent = LypSarsaAgent(args, env, writer=tb_writer)
else:
    raise Exception("Not implemented yet")


# start the run process here
agent.run()

# notify when finished
print("finished!")
