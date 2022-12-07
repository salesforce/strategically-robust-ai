import numpy as np
import os
import random
import sys
import dill as pickle

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__),
                                 "../.."))
)
from ai_economist import foundation
from gym import spaces
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv

BIG_NUMBER = 1e20


def recursive_list_to_np_array(d):
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, list):
                new_d[k] = np.array(v)
            elif isinstance(v, dict):
                new_d[k] = recursive_list_to_np_array(v)
            elif isinstance(v, (float, int)):
                new_d[k] = np.array([v])
            elif isinstance(v, np.ndarray):
                new_d[k] = v
            else:
                raise AssertionError

        return new_d
    else:
        raise AssertionError


class AIEEnvWrapper(MultiAgentEnv):
    def __init__(self, env_config):
        self.env_config_dict = env_config["env_config_dict"]

        if hasattr(env_config, "worker_index"):
            self.env_id = (
                env_config["num_envs_per_worker"] * (env_config.worker_index - 1)
            ) + env_config.vector_index
        else:
            self.env_id = None

        self.env = foundation.make_env_instance(**self.env_config_dict)

        obs = self.env.reset()

        self.observation_space = self._dict_to_spaces_dict(obs["0"])
        self.observation_space_pl = self._dict_to_spaces_dict(obs["p"])

        if self.env.world.agents[0].multi_action_mode:
            self.action_space = spaces.MultiDiscrete(
                self.env.get_agent("0").action_spaces
            )
            self.action_space.dtype = np.int32
            self.action_space.nvec = self.action_space.nvec.astype(np.int32)

        else:
            self.action_space = spaces.Discrete(self.env.get_agent("0").action_spaces)
            self.action_space.dtype = np.int32

        if self.env.world.planner.multi_action_mode:
            self.action_space_pl = spaces.MultiDiscrete(
                self.env.get_agent("p").action_spaces
            )
            self.action_space_pl.dtype = np.int32
            self.action_space_pl.nvec = self.action_space_pl.nvec.astype(np.int32)

        else:
            self.action_space_pl = spaces.Discrete(
                self.env.get_agent("p").action_spaces
            )
            self.action_space_pl.dtype = np.int32

        self._seed = None

        if not self.env_id:
            print("[EnvWrapper] Spaces")
            print("[EnvWrapper] Obs (a)   ", self.observation_space)
            print("[EnvWrapper] Obs (p)   ", self.observation_space_pl)
            print("[EnvWrapper] Action (a)", self.action_space)
            print("[EnvWrapper] Action (p)", self.action_space_pl)

    def _dict_to_spaces_dict(self, obs):
        dict_of_spaces = {}
        for k, v in obs.items():

            # list of lists are listified np arrays
            _v = v
            if isinstance(v, list):
                _v = np.array(v)
            elif isinstance(v, (int, float)):
                _v = np.array([v])

            # assign Space
            if isinstance(_v, np.ndarray):
                x = float(BIG_NUMBER)
                box = spaces.Box(
                    low=-x, high=x, shape=_v.shape, dtype=_v.dtype
                )
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                # This loop avoids issues with overflow to make sure low/high are good.
                while not low_high_valid:
                    x = x // 2
                    box = spaces.Box(
                        low=-x, high=x, shape=_v.shape, dtype=_v.dtype
                    )
                    low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                dict_of_spaces[k] = box

            elif isinstance(_v, dict):
                dict_of_spaces[k] = self._dict_to_spaces_dict(_v)
            else:
                raise TypeError
        return spaces.Dict(dict_of_spaces)

    @property
    def pickle_file(self):
        if self.env_id is None:
            return "game_object.pkl"
        else:
            return "game_object_{:03d}.pkl".format(self.env_id)

    def save_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        with open(path, "wb") as F:
            pickle.dump(self.env, F)

    def load_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        with open(path, "rb") as F:
            self.env = pickle.load(F)

    @property
    def n_agents(self):
        return self.env.n_agents

    @property
    def summary(self):
        last_completion_metrics = self.env.previous_episode_metrics
        if last_completion_metrics is None:
            return {}
        last_completion_metrics["completions"] = int(self.env._completions)
        return last_completion_metrics

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

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return recursive_list_to_np_array(obs)

    def step(self, actions):
        obs, rew, done, info = self.env.step(actions)
        return recursive_list_to_np_array(obs), rew, done, info
