"""Cooperation game environment"""

from collections import namedtuple
import itertools
import os
import random
import sys

import numpy as np
import dill as pickle

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__),
                                 "../.."))
)

from gym import spaces
from gym.utils import seeding
from utils import apply_concav
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from PIL import Image
from matplotlib import pyplot as plt


DEFAULT_CONFIG = {
    "seed": 0,
    "max_time": 10000,
    "env_width": 5,
    "perturb": 0.0,
    "noisy_perturb": False,  # Switch to random perturbation
    "episode_dr": False,  # Apply randomization upon each episode
    "action_perturb": -1,
    "dr": -1,
    "concav": -1,
    "planner_ref": 4,
    "planner_norm": 0.5,
    "num_players": 3,
}

BIG_NUMBER = 1e20
STAY = 0
UP = 1
DOWN = 2


class CoopBimatrixEnvWrapper3(MultiAgentEnv):

    def _limit_coordinates(self, coord):
        """Limit movements inside grid"""
        for i in range(self.num_players + 1):
            coord[i] = min(coord[i], self.shape[i] - 1)
            coord[i] = max(coord[i], 0)
        return coord

    def __init__(self, env_config):
        # Load config
        self.env_config_dict = DEFAULT_CONFIG.copy()
        self.env_config_dict.update(env_config["env_config_dict"])
        self.num_players = self.env_config_dict["num_players"]
        self.shape = [self.env_config_dict["env_width"] for _ in range(self.num_players + 1)]

        # Concav DR settings
        self.action_perturb = self.env_config_dict.get("action_perturb", -1)
        self.concav = self.env_config_dict.get("concav", -1)
        self.dr = self.env_config_dict.get("dr", -1)
        self.episode_dr = self.env_config_dict["episode_dr"]

        # Get environment id (for Ray workers)
        if hasattr(env_config, "worker_index"):
            self.env_id = (
                env_config["num_envs_per_worker"] * (env_config.worker_index - 1)
            ) + env_config.vector_index
        else:
            self.env_id = None

        # Seed
        self._seed = self.env_config_dict["seed"]
        np.random.seed(self._seed)

        # Gym specifications
        self.action_space = spaces.Discrete(3)
        self.action_space_pl = spaces.Discrete(3)
        self.observation_space = spaces.Dict({"oa": spaces.Box(low=0.0, high=self.env_config_dict["env_width"], shape=(self.num_players + 1,), dtype=np.float32)})
        self.observation_space_pl = spaces.Dict({"oa": spaces.Box(low=0.0, high=self.env_config_dict["env_width"], shape=(self.num_players + 1,), dtype=np.float32)})

        # Create payoff matrices
        self.p_payoff = np.random.normal(size=(self.shape))
        self.a_payoffs = [np.random.normal(size=(self.shape)) for _ in range(self.num_players)]
        self.a_perturbs = [0 for _ in range(self.num_players)]
        if self.env_config_dict["perturb"] > 0:
            for i, _ in enumerate(self.a_payoffs):
                self.a_payoffs[i] = -self.p_payoff * self.env_config_dict["perturb"] + self.a_payoffs[i] * (1 - self.env_config_dict["perturb"])

        # Calculate transition probabilities
        a_map = {STAY: 0, UP: 1, DOWN: -1}
        self.T = {}
        for s in range(np.prod(self.shape)):
            position = np.unravel_index(s, self.shape)
            self.T[tuple(position)] = {}
            for pas in itertools.product(*[[STAY, UP, DOWN] for _ in range(self.num_players + 1)]):
                new_position = np.array(position) + np.array([a_map[pas[i]] for i in range(self.num_players + 1)])
                new_position = self._limit_coordinates(new_position).astype(int)
                self.T[tuple(position)][tuple(pas)] = new_position

        # Placeholder for state
        self.disp = None
        self.time = 0
        self.state = None
        self.state_history = []

        if not self.env_id:
            print("[EnvWrapper] Spaces")
            print("[EnvWrapper] Obs (a)   ", self.observation_space)
            print("[EnvWrapper] Obs (p)   ", self.observation_space_pl)
            print("[EnvWrapper] Action (a)", self.action_space)
            print("[EnvWrapper] Action (p)", self.action_space_pl)

    @property
    def n_agents(self):
        # Always 2 agents + planner
        return self.num_players

    @property
    def pickle_file(self):
        if self.env_id is None:
            return "game_object.pkl"
        else:
            return "game_object_{:03d}.pkl".format(self.env_id)

    def save_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        data = {"state": self.state, "time": self.time, "a_perturbs": self.a_perturbs}
        with open(path, "wb") as F:
            pickle.dump(data, F)

    def load_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        try:
            path = os.path.join(save_dir, self.pickle_file)
            with open(path, "rb") as F:
                self.__dict__.update(pickle.load(F))
        except:
            return

    @property
    def summary(self):
        return {}

    def get_seed(self):
        return int(self._seed)

    def seed(self, seed):
        _, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31

        print(
            "[EnvWrapper {}] twisting seed {} -> {} -> {} (final)".format(
                self.env_id, seed, seed1, seed2
            )
        )

        seed = int(seed2)
        np.random.seed(seed2)
        random.seed(seed2)
        self._seed = seed2

    def reset(self):
        self.time = 0
        self.state = np.random.randint(self.shape)
        # self.state_history = [self.state]

        if self.env_config_dict["noisy_perturb"]:
            for i in range(self.num_players):
                self.a_perturbs[i] = np.random.uniform(-self.env_config_dict["perturb"], self.env_config_dict["perturb"])

        if self.dr > -1 and self.episode_dr:
            for i in range(self.num_players):
                self.a_perturbs[i] = np.random.uniform(-self.dr, self.dr)

        state = {"oa": self.state}
        obs = {"p": state, **{str(i): state for i in range(self.num_players)}}
        return obs

    @property
    def episode_log(self):
        return ([], self.p_payoff, self.a_payoffs)

    def step(self, actions):
        # Preprocess actions
        pp_a = actions["p"]
        pas = []
        for i in range(self.num_players):
            pas.append(actions[str(i)])
        pas.append(pp_a)

        # Apply action perturbations
        if self.action_perturb > -1:
            for i in range(self.num_players):
                pas[i] = np.random.randint(0, 3)

        # Transition
        state = self.T[tuple(self.state)][tuple(pas)]
        p_reward, a_rewards = (self.p_payoff[tuple(state)],
                               [self.a_payoffs[i][tuple(state)] for i in range(self.num_players)])
        self.state = state
        # self.state_history.append(self.state)

        # Apply episode or step perturbations
        if self.dr > -1 and not self.episode_dr:
            for i in range(self.num_players):
                self.a_perturbs[i] = np.random.uniform(-self.dr, self.dr)
        for i in range(self.num_players):
            self.a_payoffs[i] += self.a_perturbs[i]

        # Apply concav usage
        if self.concav > -1:
            for i in range(self.num_players):
                a_rewards[i] = apply_concav(a_rewards[i], self.concav)

        # Update state
        done = self.time >= self.env_config_dict["max_time"]
        self.time += 1

        state = {"oa": self.state}
        rewards = {"p": p_reward, **{str(i): a_rewards[i] for i in range(self.num_players)}}
        obs = {"p": state, **{str(i): state for i in range(self.num_players)}}

        return obs, rewards, {"__all__": done}, {}
