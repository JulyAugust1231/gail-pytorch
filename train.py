import os
import json
import pickle
import argparse

import torch
import gym

from models.nets import Expert
from models.gail import GAIL
from models.safety_gridworld import PitWorld

def main(env_name):
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    if env_name not in ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3", "Gridworld"]:
        print("The environment name is wrong!")
        return

    expert_ckpt_path = "experts"
    expert_ckpt_path = os.path.join(expert_ckpt_path, env_name)

    with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
        expert_config = json.load(f)

    ckpt_path = os.path.join(ckpt_path, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[env_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    if env_name in ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3"]:
        env = gym.make(env_name)
        env.reset()
        state_dim = len(env.observation_space.high)
    else:
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
        state_dim = env.reset().shape
        state_dim = state_dim[0]


    if env_name in ["CartPole-v1","Gridworld"]:
        discrete = True
        action_dim = env.action_space.n
        print(env.to_string())
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    expert = Expert(
        state_dim, action_dim, discrete, **expert_config
    ).to(device)
    expert.pi.load_state_dict(
        torch.load(
            os.path.join(expert_ckpt_path, "policy.ckpt"), map_location=device
        )
    )

    model = GAIL(state_dim, action_dim, discrete, config).to(device)

    results = model.train(env, expert)

    env.close()

    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(
            model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt")
        )
    if hasattr(model, "v"):
        torch.save(
            model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt")
        )
    if hasattr(model, "d"):
        torch.save(
            model.d.state_dict(), os.path.join(ckpt_path, "discriminator.ckpt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3,Gridworld]"
    )
    args = parser.parse_args()

    main(**vars(args))
