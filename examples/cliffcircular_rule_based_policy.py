from gymnasium.envs.toy_text.cliffcircular import CliffCircularEnv

from omnisafe.envs.core import CMDP, make
from omnisafe.envs.discrete_env import DiscreteEnv

import numpy as np
import torch
from enum import Enum


class Action(Enum):
    NOOP = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class RuleBasedPolicy:
    # render_mode: str = 'human'
    render_mode: str = 'rgb_array'
    width: int = 5
    height: int = 5

    def __init__(self):
        self.env = make(env_id='CliffCircular-v1', class_name=None, render_mode=self.render_mode)

        # Statistics
        self._num_episodes: int = 0
        self._ep_rew_sum: float = 0.
        self._ep_cost_sum: float = 0.

        self._last_action: Action = Action.UP  # First action is up by default

    def play(self, episode_num: int = 5):
        """
        Play CliffCircular using the rule-based policy for a specified number of episodes then print the statistics
        Args:
            episode_num:

        Returns:

        """
        obs, info = self.env.reset()

        ep_reward: float = 0.
        ep_cost: float = 0.
        while self._num_episodes < episode_num:
            # Get rule-based action
            action = self.rule(obs=obs.numpy().reshape((self.height, self.width)), action=self._last_action)

            # Update history actions
            self._last_action = action

            # Step environment
            obs, reward, cost, terminated, truncated, info = self.env.step(torch.tensor([action.value]))

            ep_reward += reward.item()
            ep_cost += cost.item()

            if terminated or truncated:
                self._ep_rew_sum += ep_reward
                self._ep_cost_sum += ep_cost
                self._num_episodes += 1
                ep_reward = 0.
                ep_cost = 0.
                obs, info = self.env.reset()

        print('*' * 40)
        print(f'Episode num: {self._num_episodes}')
        print(f'Mean_ep_rew: {self._ep_rew_sum / self._num_episodes:.1f}')
        print(f'Mean_ep_cost: {self._ep_cost_sum / self._num_episodes:.1f}')

    def rule(self, obs: np.ndarray, action: Action) -> Action:
        """
        A clockwise track following rule
        """
        original_action = action

        if action == Action.UP:
            if self._last_action != Action.RIGHT and self._up_cliff_far(obs) and not self._right_cliff_near(obs):
                action = Action.RIGHT
            elif self._up_cliff_near(obs):
                action = Action.LEFT
        elif action == Action.RIGHT:
            if self._last_action != Action.DOWN and self._right_cliff_far(obs) and not self._down_cliff_near(obs):
                action = Action.DOWN
            elif self._right_cliff_near(obs):
                action = Action.UP
        elif action == Action.DOWN:
            if self._last_action != Action.LEFT and self._down_cliff_far(obs) and not self._left_cliff_near(obs):
                action = Action.LEFT
            elif self._down_cliff_near(obs):
                action = Action.RIGHT
        elif action == Action.LEFT:
            if self._last_action != Action.UP and self._left_cliff_far(obs) and not self._up_cliff_near(obs):
                action = Action.UP
            elif self._left_cliff_near(obs):
                action = Action.DOWN

        if original_action == action:
            return action
        else:
            action = self.rule(obs, action)
            return action

    def _up_cliff_near(self, obs: np.ndarray) -> bool:
        # Near cliff
        return obs[1, 2] == 1

    def _up_cliff_far(self, obs: np.ndarray) -> bool:
        # Far cliff
        return np.all(obs[0, :] == 1)

    def _right_cliff_near(self, obs: np.ndarray) -> bool:
        # Near cliff
        return obs[2, 3] == 1

    def _right_cliff_far(self, obs: np.ndarray) -> bool:
        # Far cliff
        return np.all(obs[:, 4] == 1)

    def _down_cliff_near(self, obs: np.ndarray) -> bool:
        # Near cliff
        return obs[3, 2] == 1

    def _down_cliff_far(self, obs: np.ndarray) -> bool:
        # Far cliff
        return np.all(obs[4, :] == 1)

    def _left_cliff_near(self, obs: np.ndarray) -> bool:
        # Near cliff
        return obs[2, 1] == 1

    def _left_cliff_far(self, obs: np.ndarray) -> bool:
        # Far cliff
        return np.all(obs[:, 0] == 1)

    def _up_right_cliff(self, obs: np.ndarray) -> bool:
        assert obs.shape == (self.height, self.width)
        return obs[1, 3] == 1

    def _down_right_cliff(self, obs: np.ndarray) -> bool:
        assert obs.shape == (self.height, self.width)
        return obs[3, 3] == 1

    def _up_left_cliff(self, obs: np.ndarray) -> bool:
        assert obs.shape == (self.height, self.width)
        return obs[1, 1] == 1

    def _down_left_cliff(self, obs: np.ndarray) -> bool:
        assert obs.shape == (self.height, self.width)
        return obs[3, 1] == 1


if __name__ == '__main__':
    rbp = RuleBasedPolicy()
    rbp.play(episode_num=100)
