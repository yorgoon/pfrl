"""A training script of Soft Actor-Critic on OpenAI Gym Mujoco environments.

This script follows the settings of https://arxiv.org/abs/1812.05905 as much
as possible.
"""
import argparse
import functools
import logging
import sys
from distutils.version import LooseVersion
import math
import gym
import gym.wrappers
import numpy as np
import torch
from torch import distributions, nn
from torch.nn import functional as F
import pfrl
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda
from importlib import reload

reload(pfrl)

def swish(x):
    return x * torch.sigmoid(x)

def glu(x):
    return torch.tanh(x) * torch.sigmoid(x)

class MaxPool1d(nn.Module):
    def __init__(self, kernel, stride):
        super().__init__()
        self.max_pool_layer= nn.MaxPool1d(kernel,stride)
    
    def forward(self, x):
        return torch.squeeze(self.max_pool_layer(x.unsqueeze(0)))

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class PolicyFunc(nn.Module):
    def __init__(self,action_size,obs_size,hidden_size = 256):
        super().__init__()
        self.action_size = action_size
        # self.lx = nn.Linear(obs_size, hidden_size*4)
        # self.l1 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l11 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l2 = nn.Linear(hidden_size*2, hidden_size*2)
        # self.l21 = nn.Linear(hidden_size*2, hidden_size*2)
        # self.l3 = nn.Linear(hidden_size, hidden_size)
        # self.l31 = nn.Linear(hidden_size, hidden_size)
        # self.l4 = nn.Linear(hidden_size*2, hidden_size*2)
        # self.l5 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l51 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l_out = nn.Linear(hidden_size*4, action_size*2)
        # self.max_pool = nn.MaxPool1d(2,2,return_indices=True)
        # self.max_unpool = nn.MaxUnpool1d(2,2)

        # self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        # self.beta = nn.Parameter(torch.randn(hidden_size*4, requires_grad=True))
        # self.beta2 = nn.Parameter(torch.randn(hidden_size*4, requires_grad=True))
        # self.bias = nn.Parameter(torch.zeros(hidden_size*4, requires_grad=True))

        # nn.init.xavier_uniform_(self.beta)

        # self.lx = nn.Linear(obs_size, hidden_size*4)
        # self.l1 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l2 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l3 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l_out = nn.Linear(hidden_size*4, action_size*2)

        self.lx = nn.Linear(obs_size, hidden_size*4)
        self.l1 = nn.Linear(hidden_size*4+obs_size, hidden_size*4)
        self.l11 = nn.Linear(hidden_size*4, hidden_size*4)
        self.l2 = nn.Linear(hidden_size*4+obs_size, hidden_size*4)
        self.l3 = nn.Linear(hidden_size*4+obs_size, hidden_size*4)
        self.l_out = nn.Linear(hidden_size*4, action_size*2)

        # self.lx = nn.Linear(obs_size, hidden_size*4)
        # self.l1 = NoisyLinear(hidden_size*4+obs_size, hidden_size*4)
        # self.l11 = NoisyLinear(hidden_size*4, hidden_size*4)
        # self.l2 = NoisyLinear(hidden_size*4+obs_size, hidden_size*4)
        # self.l3 = NoisyLinear(hidden_size*4+obs_size, hidden_size*4)
        # self.l_out = NoisyLinear(hidden_size*4, action_size*2)

    def forward(self,x):
    #     h1 = torch.relu(self.lx(x))
        
    #     h2 = torch.relu(self.l1(h1))
    #     h2 = h2 * torch.sigmoid(self.l11(h2))
        
    #     h3, indices1 = self.max_pool(h2.unsqueeze(0))
    #     h3 = torch.squeeze(h3)

    #     h4 = torch.relu(self.l2(h3))
        
    #     h5, indices2 = self.max_pool(h4.unsqueeze(0))
    #     h5 = torch.squeeze(h5)

    #     h6 = torch.relu(self.l3(h5))
        
    #     h7 = self.max_unpool(h6.unsqueeze(0), indices2)
    #     h7 = torch.squeeze(h7)
    #     h7 = h4+h7

    #     h8 = torch.relu(self.l4(h7))
    #     h8 = h3+h8

    #     h9 = self.max_unpool(h8.unsqueeze(0), indices1)
    #     h9 = torch.squeeze(h9)
    #     h9 = h2+h9

    #     h10 = torch.relu(self.l5(h9))
    #    # h10 = h10 * torch.sigmoid(self.l1(h10))
    #     h10 = h1+h10
        
    #     h_out = self.l_out(h10)

        h = torch.relu(self.lx(x))
        h = torch.cat([x,h],1)
        h = torch.relu(self.l1(h))
        # h = h * torch.sigmoid(self.l11(h))
        h = torch.cat([x,h],1)
        h = torch.relu(self.l2(h))
        h = torch.cat([x,h],1)
        h = torch.relu(self.l3(h))
        h_out = self.l_out(h)

        assert h_out.shape[-1] == self.action_size * 2
        mean, log_scale = torch.chunk(h_out, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1)
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )
    def reset_noise(self):
        """Reset all noisy layers."""
        self.l1.reset_noise()
        self.l11.reset_noise()
        self.l2.reset_noise()
        self.l3.reset_noise()
        self.l_out.reset_noise()

class QFunction(nn.Module):
    def __init__(self,action_size,obs_size,hidden_size = 256):
        super().__init__()
        # self.lx = nn.Linear(obs_size+action_size, hidden_size*4)
        # self.l1 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l11 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l2 = nn.Linear(hidden_size*2, hidden_size*2)
        # self.l21 = nn.Linear(hidden_size*2, hidden_size*2)
        # self.l3 = nn.Linear(hidden_size, hidden_size)
        # self.l31 = nn.Linear(hidden_size, hidden_size)
        # self.l4 = nn.Linear(hidden_size*2, hidden_size*2)
        # self.l5 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l51 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l_out = nn.Linear(hidden_size*4, 1)
        # self.max_pool = nn.MaxPool1d(2,2,return_indices=True)
        # self.max_unpool = nn.MaxUnpool1d(2,2)

        # self.beta = nn.Parameter(torch.randn(hidden_size*4, requires_grad=True))
        # self.beta2 = nn.Parameter(torch.randn(hidden_size*4, requires_grad=True))
        # self.bias = nn.Parameter(torch.zeros(hidden_size*4, requires_grad=True))

        # self.lx = nn.Linear(obs_size+action_size, hidden_size*4)
        # self.l1 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l2 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l3 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l_out = nn.Linear(hidden_size*4, 1)
        
        # self.lx = nn.Linear(obs_size+action_size, hidden_size*4)
        # self.l1 = nn.Linear(hidden_size*4+obs_size+action_size, hidden_size*4)
        # self.l11 = nn.Linear(hidden_size*4, hidden_size*4)
        # self.l2 = nn.Linear(hidden_size*4+obs_size+action_size, hidden_size*4)
        # self.l3 = nn.Linear(hidden_size*4+obs_size+action_size, hidden_size*4)
        # self.l_out = nn.Linear(hidden_size*4, 1)

        self.lx = nn.Linear(obs_size+action_size, hidden_size*4)
        self.l1 = NoisyLinear(hidden_size*4+obs_size+action_size, hidden_size*4)
        self.l11 = NoisyLinear(hidden_size*4, hidden_size*4)
        self.l2 = NoisyLinear(hidden_size*4+obs_size+action_size, hidden_size*4)
        self.l3 = NoisyLinear(hidden_size*4+obs_size+action_size, hidden_size*4)
        self.l_out = NoisyLinear(hidden_size*4, 1)

    def forward(self,x):
        x = torch.cat(x, dim=-1)
    #     h1 = torch.relu(self.lx(x))
        
    #     h2 = torch.relu(self.l1(h1))
    #     h2 = h2 * torch.sigmoid(self.l11(h2))
        
    #     h3, indices1 = self.max_pool(h2.unsqueeze(0))
    #     h3 = torch.squeeze(h3)

    #     h4 = torch.relu(self.l2(h3))
        
    #     h5, indices2 = self.max_pool(h4.unsqueeze(0))
    #     h5 = torch.squeeze(h5)

    #     h6 = torch.relu(self.l3(h5))
        
    #     h7 = self.max_unpool(h6.unsqueeze(0), indices2)
    #     h7 = torch.squeeze(h7)
    #     h7 = h4+h7

    #     h8 = torch.relu(self.l4(h7))
    #     h8 = h3+h8

    #     h9 = self.max_unpool(h8.unsqueeze(0), indices1)
    #     h9 = torch.squeeze(h9)
    #     h9 = h2+h9

    #     h10 = torch.relu(self.l5(h9))
    #    # h10 = h10 * torch.sigmoid(self.l1(h10))
    #     h10 = h1+h10
        
    #     h_out = self.l_out(h10)

        h = torch.relu(self.lx(x))
        h = torch.cat([x,h],1)
        h = torch.relu(self.l1(h))
        # h = h * torch.sigmoid(self.l11(h))
        h = torch.cat([x,h],1)
        h = torch.relu(self.l2(h))
        h = torch.cat([x,h],1)
        h = torch.relu(self.l3(h))
        h_out = self.l_out(h)

        return h_out
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.l1.reset_noise()
        self.l11.reset_noise()
        self.l2.reset_noise()
        self.l3.reset_noise()
        self.l_out.reset_noise()

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
    parser.add_argument(
        "--num-envs", type=int, default=6, help="Number of envs run in parallel."
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
        default=3*(10 ** 6),
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=100,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=20000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=10000,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
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
        "--log-interval",
        type=int,
        default=1000,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
        "--policy-output-scale",
        type=float,
        default=1.0,
        help="Weight initialization scale of policy output.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Unwrap TimiLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        # Normalize action space to [-1, 1]^n
        env = pfrl.wrappers.NormalizeActionSpace(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )

    sample_env = make_env(process_idx=0, test=False)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
        raise Exception("This script requires a PyTorch version >= 1.5.0")

    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )

    # policy = nn.Sequential(
    #     nn.Linear(obs_size, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, action_size * 2),
    #     Lambda(squashed_diagonal_gaussian_head),
    # )
    # torch.nn.init.xavier_uniform_(policy[0].weight)
    # torch.nn.init.xavier_uniform_(policy[2].weight)
    # torch.nn.init.xavier_uniform_(policy[4].weight, gain=args.policy_output_scale)
    # policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # def make_q_func_with_optimizer():
    #     q_func = nn.Sequential(
    #         pfrl.nn.ConcatObsAndAction(),
    #         nn.Linear(obs_size + action_size, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 1),
    #     )
    #     torch.nn.init.xavier_uniform_(q_func[1].weight)
    #     torch.nn.init.xavier_uniform_(q_func[3].weight)
    #     torch.nn.init.xavier_uniform_(q_func[5].weight)
    #     q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
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
    q_func1.apply(init_weights)
    q_func1_optimizer = torch.optim.Adam(q_func1.parameters(), lr=3e-4)
    q_func2 = QFunction(action_size,obs_size)
    q_func2.apply(init_weights)
    q_func2_optimizer = torch.optim.Adam(q_func2.parameters(), lr=3e-4)

    rbuf = replay_buffers.ReplayBuffer(10 ** 6)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = pfrl.agents.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size,
        temperature_optimizer_lr=3e-4,
    )
    if len(args.load) > 0 or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not len(args.load) > 0 or not args.load_pretrained
        if len(args.load) > 0:
            agent.load(args.load)
        else:
            agent.load(
                utils.download_model("SAC", args.env, model_type=args.pretrained_type)[
                    0
                ]
            )

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
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
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(test=False),
            eval_env=make_batch_env(test=True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            max_episode_len=timestep_limit,
            use_tensorboard=True,
        )


if __name__ == "__main__":
    main()
