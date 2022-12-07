"""Mall cop allocation game environment"""

import numpy as np
import os
import random
import sys
import dill as pickle

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__),
                                 "../.."))
)

from collections import namedtuple
from gym import spaces
from gym.utils import seeding
from utils import apply_concav
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from PIL import Image
from matplotlib import pyplot as plt


DEFAULT_CONFIG = {
    "seed": 0,
    "max_time": 10000,
    "scale": 30,
    "env_width": 5,
    "max_officers": 12,
    "officer_weight": 1.5,
    "perturb": 0.0,
    "noisy_perturb": False,  # Switch to random perturbation
    "episode_dr": False,  # Apply randomization upon each episode
    "action_perturb": -1,
    "dr": -1,
    "concav": -1,
}

BIG_NUMBER = 1e20
STAY = 0
UP = 1
DOWN = 2


class BimatrixEnvWrapper(MultiAgentEnv):

    def _limit_coordinates(self, coord):
        """Limit movements inside grid"""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def __init__(self, env_config):
        # Load config
        self.env_config_dict = DEFAULT_CONFIG.copy()
        self.env_config_dict.update(env_config["env_config_dict"])
        self.shape = (self.env_config_dict["env_width"], self.env_config_dict["env_width"])

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
        self.action_space_pl = spaces.Discrete(5) # self.env_config_dict["max_officers"] + 1)
        self.observation_space = spaces.Dict({"oa": spaces.Discrete(self.shape[0] * self.shape[1])})
        self.observation_space_pl = spaces.Dict({"oa": spaces.Discrete(self.shape[0] * self.shape[1]),
                                              "op": spaces.Discrete(self.env_config_dict["max_officers"] + 1)})

        # Create payoff matrices
        self.p_payoff = np.random.uniform(size=(self.shape))
        self.p1_payoff = np.random.uniform(size=(self.shape))
        self.p2_payoff = np.random.uniform(size=(self.shape))
        self.p1_perturb = 0
        self.p2_perturb = 0
        if self.env_config_dict["perturb"] > 0:
            self.p1_payoff -= self.p_payoff * self.env_config_dict["perturb"]
            self.p2_payoff -= self.p_payoff * self.env_config_dict["perturb"]
        self.officers = 0

        # Calculate transition probabilities
        a_map = {STAY: 0, UP: 1, DOWN: -1}
        self.P = {}
        for s in range(self.shape[0] * self.shape[1]):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {}
            for p1_a in [STAY, UP, DOWN]:
                for p2_a in [STAY, UP, DOWN]:
                    new_position = np.array(position) + np.array([a_map[p1_a], a_map[p2_a]])
                    new_position = self._limit_coordinates(new_position).astype(int)
                    new_state = np.ravel_multi_index(tuple(new_position), self.shape)
                    reward = (self.p_payoff[tuple(new_position)],
                              self.p1_payoff[tuple(new_position)],
                              self.p2_payoff[tuple(new_position)])
                    self.P[s][(p1_a, p2_a)] = (new_state, reward)

        # Placeholder for state
        self.disp = None
        self.time = 0
        self.state = None
        self.state_history = []
        self.officer_history = []
        self.avg_officers = 0

        if not self.env_id:
            print("[EnvWrapper] Spaces")
            print("[EnvWrapper] Obs (a)   ", self.observation_space)
            print("[EnvWrapper] Obs (p)   ", self.observation_space_pl)
            print("[EnvWrapper] Action (a)", self.action_space)
            print("[EnvWrapper] Action (p)", self.action_space_pl)

    @property
    def n_agents(self):
        # Always 2 agents + planner
        return 2

    @property
    def pickle_file(self):
        if self.env_id is None:
            return "game_object.pkl"
        else:
            return "game_object_{:03d}.pkl".format(self.env_id)

    def save_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        data = {"state": self.state, "time": self.time, "officers": self.officers, "p1_perturb": self.p1_perturb, "p2_perturb": self.p2_perturb}
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
        return {"avg_officers": self.avg_officers}

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
        if self.officer_history:
            self.avg_officers = np.mean(self.officer_history)
        self.officer_history = []

        self.time = 0
        self.officers = 0
        self.state = np.random.randint(self.shape[0] * self.shape[1])
        self.state_history = [self.state]

        if self.env_config_dict["noisy_perturb"]:
            self.p1_perturb = np.random.uniform(-self.env_config_dict["perturb"], self.env_config_dict["perturb"])
            self.p2_perturb = np.random.uniform(-self.env_config_dict["perturb"], self.env_config_dict["perturb"])

        if self.dr > -1 and self.episode_dr:
            self.p1_perturb = np.random.uniform(-self.dr, self.dr)
            self.p2_perturb = np.random.uniform(-self.dr, self.dr)

        state = {"oa": self.state}
        p_state = {"oa": self.state, "op": self.officers}
        obs = {"p": p_state, "0": state, "1": state}
        return obs

    @property
    def episode_log(self):
        return (self.state_history, self.p_payoff, self.p1_payoff, self.p2_payoff, self.officer_history)

    def step(self, actions):
        pp_a = actions["p"]
        p1_a = actions["0"]
        p2_a = actions["1"]

        # Potentially perturb
        if self.action_perturb > -1:
            if np.random.random() < self.action_perturb:
                res = []
                for i in range(3):
                    p_reward = self.P[self.state][(i, p2_a)][1][0]
                    res.append((p_reward, i))
                p1_a = sorted(res, key=lambda x: x[0])[0][1]

        # Transition
        state, (p_reward, p1_reward, p2_reward) = self.P[self.state][(p1_a, p2_a)]
        self.state = state
        self.state_history.append(self.state)

        # Apply planner action
        if pp_a == 0:
            self.officers -= 2
        elif pp_a == 1:
            self.officers -= 1
        elif pp_a == 2:
            pass
        elif pp_a == 3:
            self.officers += 1
        elif pp_a == 4:
            self.officers += 2
        else:
            raise ValueError()
        self.officers = min(self.officers, self.env_config_dict["max_officers"])
        self.officers = max(self.officers, 0)
        self.officer_history.append(self.officers)

        # Compute planner reward
        p_reward *= (self.env_config_dict["officer_weight"] * self.officers / self.env_config_dict["max_officers"] + 1)
        p_reward -= np.square(self.officers / self.env_config_dict["max_officers"])

        # Compute agent rewards
        if self.dr > -1 and not self.episode_dr:
            self.p1_perturb = np.random.uniform(-self.dr, self.dr)
            self.p2_perturb = np.random.uniform(-self.dr, self.dr)
        p1_reward += self.p1_perturb
        p2_reward += self.p2_perturb

        rewards = {"p": p_reward,
                   "0": p1_reward, "1": p2_reward}

        done = self.time >= self.env_config_dict["max_time"]
        self.time += 1

        state = {"oa": self.state}
        p_state = {"oa": self.state, "op": self.officers}
        obs = {"p": p_state, "0": state, "1": state}

        # Apply concav usage
        if self.concav > -1:
            rewards["p"] = apply_concav(rewards["p"], self.concav)

        return obs, rewards, {"__all__": done}, {}
