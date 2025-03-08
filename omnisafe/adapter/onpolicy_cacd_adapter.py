# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""OnPolicy CACD (Constrained Actor Critic Dynamics) Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from rich.progress import track
from gymnasium.spaces import Discrete, MultiBinary, MultiDiscrete

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer, VectorOnPolicyCACDBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic_dynamics import ConstraintActorCriticDynamics
from omnisafe.utils.config import Config


class OnPolicyCACDAdapter(OnPolicyAdapter):
    """OnPolicy Adapter for OmniSafe.

    :class:`OnPolicyAdapter` is used to adapt the environment to the on-policy training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._reset_log()

        self.gru_layer_num: int = cfgs.model_cfgs.num_gru_layers
        self.gru_latent_size: int = cfgs.model_cfgs.latent_size

        if isinstance(self.action_space, Discrete):
            self.nominal_action = torch.tensor([[0]])  # CliffCircular
        elif isinstance(self.action_space, MultiDiscrete):
            self.nominal_action = torch.tensor([[1] * self.action_space.nvec.shape[0]])  # SRE
        else:
            print(f'Nominal action for {type(self.action_space)} is not supported.')
            raise NotImplementedError
        print(f'Nominal (no_op) action in this env: {self.nominal_action}')

    def rollout_old(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCriticDynamics,
        buffer: VectorOnPolicyCACDBuffer,
        logger: Logger,
        enable_safety_layer: bool = False,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCriticDynamics): Constraint actor-critic dynamics, including actor , reward critic,
            cost critic and semantic dynamics model.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            enable_safety_layer (bool): Whether enable the sdm-based safety layer
        """
        self._reset_log()

        obs, _ = self.reset()
        latent = None

        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            # Step CACD
            # act, logp, reward_pred, cost_value_pred, latent = agent.step(obs, latent)  # cost value approximated
            act, logp, act_overlaid, reward_pred, cost_pred, latent = agent.step(
                obs,
                latent,
                deterministic=False,
                enable_safety_layer=enable_safety_layer,
            )  # immediate cost approximated

            if act_overlaid.item() is True:
                logger.store({'Train/ActionOverlaid': act_overlaid})

            # Step environment
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            # print(f'{next_obs.shape=}')
            # print(f'{step=} {act=} {reward=} {cost=} {terminated=} {truncated=}')

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                # logger.store({'Value/cost': cost_value_pred})
                logger.store({'Value/cost': cost_pred})
            logger.store({'Value/reward': reward_pred})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                done=terminated or truncated,
                reward_pred=reward_pred,
                cost_pred=cost_pred,
                # value_c=cost_value_pred,
                next_obs=next_obs,
                logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            logger.log(
                                f'Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.',
                            )
                            _, _, _, last_r, last_value_c, _ = agent.step(obs, latent)

                        if time_out:
                            _, _, _, last_r, last_value_c, _ = agent.step(obs, latent)

                    if done or time_out:
                        obs, _ = self.reset()

                        self._log_metrics(logger, idx)
                        # print(f'Episodic cost is {self._ep_cost[0]}.')
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0

                    buffer.finish_path(last_r, last_value_c, idx)

                    # Log mean and max episodic rewards from buffer
                    logger.store({'Metrics/EpRetMean': buffer.buffers[0].mean_ep_ret})
                    logger.store({'Metrics/EpRetMax': buffer.buffers[0].max_ep_ret})

                    # Reset latent for next episode
                    latent = None

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCriticDynamics,
        buffer: VectorOnPolicyCACDBuffer,
        logger: Logger,
        enable_safety_layer: bool = False,
        safety_layer_use_reward: bool = False,
    ) -> None:
        """Rollout the environment and store the data in the buffer.
        This version does not interrupt a trajectory that is not yet terminated or truncated.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCriticDynamics): Constraint actor-critic dynamics, including actor , reward critic,
            cost critic and semantic dynamics model.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            enable_safety_layer (bool): Whether enable the sdm-based safety layer.
            safety_layer_use_reward (bool): Whether use reward-cost tradeoff score to select safe action
        """
        self._reset_log()

        obs, _ = self.reset()
        latent = None  # Last latent
        last_action = self.nominal_action.clone()  # means no_op action
        step: int = 0  # Number of steps taken by the agent

        while True:
            # Step CACD
            # act, logp, reward_pred, cost_value_pred, latent = agent.step(obs, latent)  # cost value approximated
            act, logp, act_overlaid, reward_pred, cost_pred, latent = agent.step(
                obs=obs,
                last_act=last_action,
                lagrangian_multiplier=logger.get_stats('Metrics/LagrangeMultiplier')[0],
                latent=latent,
                deterministic=False,
                enable_safety_layer=enable_safety_layer,
                safety_layer_use_reward=safety_layer_use_reward,
            )  # immediate cost approximated

            if act_overlaid.item() is True:
                logger.store({'Train/ActionOverlaid': act_overlaid.type(torch.float32)})

            # Step environment
            next_obs, reward, cost, terminated, truncated, info = self.step(act)
            step += 1
            last_action.copy_(act)

            # print(f'{next_obs.shape=}')
            # print(f'{step=} {act=} {reward=} {cost=} {terminated=} {truncated=}')

            # Keep record of reward/cost statistics
            self._log_value(reward=reward, cost=cost, info=info)

            # Log estimated immediate rewards and costs
            if self._cfgs.algo_cfgs.use_cost:
                # logger.store({'Value/cost': cost_value_pred})
                logger.store({'Value/cost': cost_pred})
            logger.store({'Value/reward': reward_pred})

            # Store info of this step (state-action-state transition)
            buffer.store(
                obs=obs,
                act=act,
                act_overlaid=act_overlaid,
                reward=reward,
                cost=cost,
                done=terminated or truncated,
                reward_pred=reward_pred,
                cost_pred=cost_pred,
                # value_c=cost_value_pred,
                next_obs=next_obs,
                logp=logp,
            )

            # Let the last episode finish naturally before concluding this epoch
            epoch_end = False
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if done or time_out:
                    # print(f'Episode end.')
                    # Judge epoch end
                    if step >= steps_per_epoch:
                        # print(f'Epoch End.')
                        epoch_end = True

                    # Log epoch statistics
                    self._log_metrics(logger, idx)
                    # Log mean and max episodic rewards from buffer
                    logger.store({'Metrics/EpRetMean': buffer.buffers[0].mean_ep_ret})
                    logger.store({'Metrics/EpRetMax': buffer.buffers[0].max_ep_ret})

                    # Reset epoch statistics
                    self._reset_log(idx)

                    # Finish epoch by calculating advantages
                    buffer.finish_path(idx=idx)

                    # Log mean and max episodic rewards from buffer
                    logger.store({'Metrics/EpRetMean': buffer.buffers[0].mean_ep_ret})
                    logger.store({'Metrics/EpRetMax': buffer.buffers[0].max_ep_ret})

                    # Reset obs and latent for next episode
                    obs, _ = self.reset()
                    latent = None
                    last_action = self.nominal_action.clone()  # means no_op action
                else:
                    # While loop continues with the latest observation
                    obs = next_obs

            # Break out of the while loop
            if epoch_end:
                logger.log(f'Epoch is done with {step} steps.')
                break

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
