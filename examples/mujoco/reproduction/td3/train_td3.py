"""A training script of TD3 on OpenAI Gym Mujoco environments.

This script follows the settings of http://arxiv.org/abs/1802.09477 as much
as possible.
"""

import argparse
import logging
import sys

import gym
import gym.wrappers
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl import experiments, explorers, replay_buffers, utils
from pfrl.distributions import Delta

class PolicyFunc(nn.Module):
    def __init__(self,action_size,obs_size,hidden_size = 256):
        super().__init__()
        self.action_size = action_size
        self.lx = nn.Linear(obs_size, hidden_size)
        
        self.l11 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l21 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l31 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l41 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l_out = nn.Linear(hidden_size, action_size)

        # self.l1 = nn.Linear(hidden_size+obs_size,hidden_size)
        # self.l2 = nn.Linear(hidden_size+obs_size,hidden_size)
        # self.l3 = nn.Linear(hidden_size+obs_size,hidden_size)
        # self.l_out = nn.Linear(hidden_size, action_size)

    def forward(self,x):
        h1 = torch.relu(self.lx(x))
        h11 = torch.tanh(self.l11(h1)) * torch.sigmoid(self.l11(h1))
        h2 = torch.relu(self.l2(h1 + h11))
        h21 = torch.tanh(self.l11(h2)) * torch.sigmoid(self.l11(h2))
        h3 = torch.relu(self.l2(h1 + h2 + h21))
        h_out = torch.tanh(self.l_out(h3))
        
        # h = torch.relu(self.lx(x))
        # h = torch.cat([x,h],1)
        # h = torch.relu(self.l1(h))
        # h = torch.cat([x,h],1)
        # h = torch.relu(self.l2(h))
        # h = torch.cat([x,h],1)
        # h = torch.relu(self.l3(h))
        # h_out = torch.tanh(self.l_out(h))

        return torch.distributions.Independent(Delta(loc=h_out), 1)

class QFunction(nn.Module):
    def __init__(self,action_size,obs_size,hidden_size = 256):
        super().__init__()
        self.lx = nn.Linear(obs_size+action_size, hidden_size)
        
        self.l11 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l21 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l31 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l41 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l_out = nn.Linear(hidden_size,1)

        # self.l1 = nn.Linear(hidden_size+obs_size+action_size,hidden_size)
        # self.l2 = nn.Linear(hidden_size+obs_size+action_size,hidden_size)
        # self.l3 = nn.Linear(hidden_size+obs_size+action_size,hidden_size)
        # self.l_out = nn.Linear(hidden_size, 1)

    def forward(self,x):
        x = torch.cat(x, dim=-1)

        h1 = torch.relu(self.lx(x))
        h11 = torch.tanh(self.l11(h1)) * torch.sigmoid(self.l11(h1))
        h2 = torch.relu(self.l2(h1 + h11))
        h21 = torch.tanh(self.l11(h2)) * torch.sigmoid(self.l11(h2))
        h3 = torch.relu(self.l2(h1 + h2 + h21))
        h_out = self.l_out(h3)

        # h = torch.relu(self.lx(x))
        # h = torch.cat([x,h],1)
        # h = torch.relu(self.l1(h))
        # h = torch.cat([x,h],1)
        # h = torch.relu(self.l2(h))
        # h = torch.cat([x,h],1)
        # h = torch.relu(self.l3(h))
        # h_out = self.l_out(h)

        return h_out

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default="HalfCheetah-v2",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=10000,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Minibatch size")
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    def make_env(test):
        env = gym.make(args.env)
        # Unwrap TimeLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if args.render and not test:
            env = pfrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    # policy = nn.Sequential(
    #     nn.Linear(obs_size, 400),
    #     nn.ReLU(),
    #     nn.Linear(400, 300),
    #     nn.ReLU(),
    #     nn.Linear(300, action_size),
    #     nn.Tanh(),
    #     pfrl.policies.DeterministicHead(),
    # )
    # policy_optimizer = torch.optim.Adam(policy.parameters())

    # def make_q_func_with_optimizer():
    #     q_func = nn.Sequential(
    #         pfrl.nn.ConcatObsAndAction(),
    #         nn.Linear(obs_size + action_size, 400),
    #         nn.ReLU(),
    #         nn.Linear(400, 300),
    #         nn.ReLU(),
    #         nn.Linear(300, 1),
    #     )
    #     q_func_optimizer = torch.optim.Adam(q_func.parameters())
    #     return q_func, q_func_optimizer

    # q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    # q_func2, q_func2_optimizer = make_q_func_with_optimizer()
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    policy = PolicyFunc(action_size,obs_size)
    policy.apply(init_weights)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    q_func1 = QFunction(action_size,obs_size)
    # q_func1.apply(init_weights)
    q_func1_optimizer = torch.optim.Adam(q_func1.parameters(), lr=3e-4)
    q_func2 = QFunction(action_size,obs_size)
    # q_func2.apply(init_weights)
    q_func2_optimizer = torch.optim.Adam(q_func2.parameters(), lr=3e-4)

    rbuf = replay_buffers.ReplayBuffer(10 ** 6)

    explorer = explorers.AdditiveGaussian(
        scale=0.1, low=action_space.low, high=action_space.high
    )

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = pfrl.agents.TD3(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        soft_update_tau=5e-3,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
    )

    if len(args.load) > 0 or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not len(args.load) > 0 or not args.load_pretrained
        if len(args.load) > 0:
            agent.load(args.load)
        else:
            agent.load(
                utils.download_model("TD3", args.env, model_type=args.pretrained_type)[
                    0
                ]
            )

    eval_env = make_env(test=True)
    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
        import json
        import os

        with open(os.path.join(args.outdir, "demo_scores.json"), "w") as f:
            json.dump(eval_stats, f)
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_env=eval_env,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            train_max_episode_len=timestep_limit,
            use_tensorboard=True,
        )


if __name__ == "__main__":
    main()
