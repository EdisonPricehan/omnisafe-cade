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
"""Implementation of the FOCOPS CACD algorithm."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Distribution, Categorical
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.clip_grad import clip_grad_norm_

from rich.progress import track
from typing import Any, Union, List
from gymnasium.spaces import Discrete, MultiDiscrete

from omnisafe.adapter.onpolicy_cade_adapter import OnPolicyCADEAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.common.buffer import VectorOnPolicyCADEBuffer
from omnisafe.common.lagrange import Lagrange
from omnisafe.common.mpc_lagrange import MPCLagrange
from omnisafe.utils import distributed
from omnisafe.typing import AdvatageEstimator
from omnisafe.utils.episode_dataset import EpisodeDataset
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.models.actor_critic.constraint_actor_dynamics_estimator import ConstraintActorDynamicsEstimator
from omnisafe.models.actor.latent_multi_categorical_actor import LatentMultiCategoricalActor
from omnisafe.utils.math import (get_dist_mean_std,
                                 get_multi_dist_mean_std,
                                 kld_multi_categorical,
                                 kld_multi_categorical_flattened,
                                 logits_from_multi_categorical)


@registry.register
class FOCOPS_CADE(PolicyGradient):
    """
    Constrained Actor Dynamics Estimator structure, built on
    the First Order Constrained Optimization in Policy Space (FOCOPS) algorithm.

    References:
        - Title: First Order Constrained Optimization in Policy Space
        - Authors: Yiming Zhang, Quan Vuong, Keith W. Ross.
        - URL: `FOCOPS <https://arxiv.org/abs/2002.06506>`_
    """

    _p_dist: Union[Categorical, List[Categorical]]

    def _init_env(self) -> None:
        self._env: OnPolicyCADEAdapter = OnPolicyCADEAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init(self) -> None:
        """Initialize the FOCOPS specific model.

        The FOCOPS algorithm uses a Lagrange multiplier to balance the cost and reward.
        """
        # Expand buffer size by 1 episode length
        size: int = self._steps_per_epoch + self._cfgs.train_cfgs.max_episode_steps

        self._buf: VectorOnPolicyCADEBuffer = VectorOnPolicyCADEBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            # size=self._steps_per_epoch,
            size=size,
            gamma=self._cfgs.algo_cfgs.gamma,
            gamma_c=self._cfgs.algo_cfgs.cost_gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            lookahead_steps=self._cfgs.algo_cfgs.lookahead_steps,
            cost_limit=self._cfgs.lagrange_cfgs.cost_limit,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

        if self._cfgs.algo_cfgs.use_mpc_lagrangian:
            # MPC Lagrange
            self._lagrange: MPCLagrange = MPCLagrange(
                cacd=self._actor_critic,
                horizon=3,
                rollout_num=5,
                buffer=self._buf.buffers[0],
                alpha_init=0.99,
                **self._cfgs.lagrange_cfgs,
            )
        else:
            # Standard Lagrange
            self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

        self._advantage_estimation_method: AdvatageEstimator = self._cfgs.algo_cfgs.adv_estimation_method

        # Exponential moving average (EMA) of actor loss and reward estimator loss
        self._pi_loss_mean: float = 0
        self._reward_loss_mean: float = 0
        # self._beta: float = 0.1  # Smoothing factor
        self._beta: float = 0.9  # Smoothing factor
        self._last_loss_pi: float = 1.0
        self._last_loss_r: float = 1.0

    def _init_model(self) -> None:
        self._actor_critic: ConstraintActorDynamicsEstimator = ConstraintActorDynamicsEstimator(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
            is_value_critic=('subm' not in self._cfgs.algo_cfgs.adv_estimation_method),  # estimate state value for non-MonteCarlo methods
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        # if self._cfgs.model_cfgs.exploration_noise_anneal:
        #     self._actor_critic.set_annealing(
        #         epochs=[0, self._cfgs.train_cfgs.epochs],
        #         std=self._cfgs.model_cfgs.std_range,
        #     )

        self._disable_no_op: bool = self._cfgs.model_cfgs.disable_no_op

    def _init_log(self) -> None:
        """Log the FOCOPS specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()

        # Setup additional keys that need to be logged
        self._logger.register_key('Metrics/LagrangeMultiplier', window_length=1)  # keep the latest value
        self._logger.register_key('Metrics/EpRetMean')
        self._logger.register_key('Metrics/EpRetMax')
        self._logger.register_key('Train/PolicyStd')
        self._logger.register_key('Train/PolicyRatioClipped')
        self._logger.register_key('Loss/Loss_reward_estimator')
        self._logger.register_key('Loss/Loss_sdm')
        self._logger.register_key('Loss/Loss_reward_ratio')
        self._logger.register_key('Loss/Loss_policy_ratio')
        self._logger.register_key('Loss/Loss_cost_estimator')
        self._logger.register_key('Value/Adv_r')
        self._logger.register_key('Value/Adv_c')
        if self._cfgs.algo_cfgs.use_mpc_lagrangian:
            self._logger.register_key('Metrics/MPCLagrangeMultiplier')
            self._logger.register_key('Metrics/MPCAlpha')

        # For safety layer
        self._logger.register_key('Train/ActionOverlaid')

        # Setup models that need to be saved regularly
        what_to_save: dict[str, Any] = {
            'actor_critic': self._actor_critic,
        }
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

    def _loss_pi_safe(
        self,
        distribution: Union[Distribution, List[Distribution]],
        act: torch.Tensor,
        act_overlaid: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute pi/actor loss.

        In FOCOPS, the loss is defined as:

        .. math::
            :nowrap:

            \begin{eqnarray}
                L = \nabla_{\theta} D_{K L} \left( \pi_{\theta}^{'} \| \pi_{\theta} \right)[s]
                - \frac{1}{\eta} \underset{a \sim \pi_{\theta}}{\mathbb{E}} \left[
                    \frac{\nabla_{\theta} \pi_{\theta} (a \mid s)}{\pi_{\theta}(a \mid s)}
                    \left( A^{R}_{\pi_{\theta}} (s, a) - \lambda A^C_{\pi_{\theta}} (s, a) \right)
                \right]
            \end{eqnarray}

        where :math:`\eta` is a hyperparameter, :math:`\lambda` is the Lagrange multiplier,
        :math:`A_{\pi_{\theta_k}}(s, a)` is the advantage function,
        :math:`A^C_{\pi_{\theta_k}}(s, a)` is the cost advantage function,
        :math:`\pi^*` is the optimal policy, and :math:`\pi_{\theta}` is the current policy.

        Args:
            distribution (Distribution): The ``distribution`` of current actor with sampled obs from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        if self._disable_no_op:
            if isinstance(self._actor_critic.act_space, Discrete):
                act -= 1
            elif isinstance(self._actor_critic.act_space, MultiDiscrete):
                pass  # TODO
            else:
                raise NotImplementedError

        logp_ = self._actor_critic.actor.log_prob(act)
        # std = self._actor_critic.actor.std
        if isinstance(distribution, list):  # Multi-discrete
            _, stds = get_multi_dist_mean_std(distribution)
            std = torch.mean(torch.stack(stds), dim=0).item()
        else:
            _, std = get_dist_mean_std(distribution)
        ratio = torch.exp(logp_ - logp)
        ratio_clipped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )

        # Calculate KLD of current policy from the initial policy that interacted with the environment
        if isinstance(distribution, list):  # Multi-discrete
            if self._cfgs.model_cfgs.first_non_no_op_win:  # Use flattened 9-way KLD
                if self._cfgs.model_cfgs.block_backward_action:  # Multi-Discrete [3,3,2,3] with 8-way KLD
                    kl = kld_multi_categorical_flattened(distribution, self._p_dist)
                else:  # Multi-Discrete [3,3,3,3] with 9-way KLD
                    kl = kld_multi_categorical_flattened(distribution, self._p_dist,
                                                         non_noop_idx=([0,2],[0,2],[0,2],[0,2]))
            else:  # Use multi-categorical KLD
                kl = kld_multi_categorical(distribution, self._p_dist)
        else:
            kl = torch.distributions.kl_divergence(distribution, self._p_dist).sum(-1, keepdim=True)

        # Safe action used to update policy regardless of kld
        # kl_mask = kl.detach() <= self._cfgs.algo_cfgs.focops_eta
        # combined_mask = kl_mask | act_overlaid.type(torch.bool)
        # loss = (kl - (1 / self._cfgs.algo_cfgs.focops_lam) * ratio * adv) * combined_mask.type(torch.float32)

        # Use unclipped ratio (FOCOPS original version)
        loss = ((kl - (1 / self._cfgs.algo_cfgs.focops_lam) * ratio * adv) *
                (kl.detach() <= self._cfgs.algo_cfgs.focops_eta).type(torch.float32))

        # Use symmetrically clipped ratio
        # loss = (kl - (1 / self._cfgs.algo_cfgs.focops_lam) * ratio_clipped * adv) * (
        #     kl.detach() <= self._cfgs.algo_cfgs.focops_eta
        # ).type(torch.float32)

        # Use asymmetrically clipped ratio
        # loss = (kl - (1 / self._cfgs.algo_cfgs.focops_lam) * torch.min(ratio_clipped * adv, ratio * adv)) * (
        #     kl.detach() <= self._cfgs.algo_cfgs.focops_eta
        # ).type(torch.float32)

        # Use asymmetrically clipped ratio without kld loss (PPO)
        # loss = -torch.min(ratio_clipped * adv, ratio * adv)

        loss = loss.mean()

        # Add entropy to loss
        if isinstance(distribution, list):
            loss -= self._cfgs.algo_cfgs.entropy_coef * torch.sum(
                torch.stack([dist.entropy().mean() for dist in distribution]))
        else:
            loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

        if isinstance(distribution, list):
            entropy = torch.sum(torch.stack([dist.entropy().mean() for dist in distribution])).item()
        else:
            entropy = distribution.entropy().mean().item()

        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Train/PolicyRatioClipped': ratio_clipped,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss

    def _loss_reward(
        self,
        reward: torch.Tensor,
        reward_pred: torch.Tensor,
    ) -> torch.Tensor:
        loss = nn.functional.mse_loss(reward_pred.squeeze(), reward.squeeze())

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        return loss

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        FOCOPS uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        if self._cfgs.algo_cfgs.use_lagrangian:
            # return (adv_r - self._lagrange.lagrangian_multiplier * adv_c) / (1 + self._lagrange.lagrangian_multiplier)

            # TODO a simplified version of advantage without downscaling with the multiplier
            return adv_r - self._lagrange.lagrangian_multiplier * adv_c
        else:
            return super()._compute_adv_surrogate(adv_r=adv_r, adv_c=adv_c)

    def _update(self) -> None:
        r"""Update actor, reward/cost critic, Lagrange multiplier and dynamics model parameters.

        In FOCOPS, the Lagrange multiplier is updated as the naive lagrange multiplier update.

        Then in each iteration of the policy update, FOCOPS calculates current policy's
        distribution, which used to calculate the policy loss.
        """
        if not self._cfgs.algo_cfgs.use_mpc_lagrangian:
            # Only update lagrange multiplier when reached certain epochs
            # cur_epochs = self._logger.get_stats('Train/Epoch')[0]
            # # TODO temp update lagrange after certain epochs
            # if cur_epochs > self._cfgs.model_cfgs.dynamics.engage_after_epochs:
            #
            #     # note that logger already uses MPI statistics across all processes.
            #     Jc = self._logger.get_stats('Metrics/EpCost')[0]
            #
            #     # first update Lagrange multiplier parameter
            #     self._lagrange.update_lagrange_multiplier(Jc)

            # note that logger already uses MPI statistics across all processes.
            Jc = self._logger.get_stats('Metrics/EpCost')[0]

            # first update Lagrange multiplier parameter
            self._lagrange.update_lagrange_multiplier(Jc)

        # Get all needed data from buffer
        data = self._buf.get()
        obs, act, act_overlaid, logp, done, reward, reward_pred, next_obs, target_value_c, adv_r, adv_c, cost, cost_pred = (
            data['obs'],
            data['act'],
            data['act_overlaid'],
            data['logp'],
            data['done'],
            data['reward'],
            data['reward_pred'],
            data['next_obs'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
            data['cost'],
            data['cost_pred'],
        )
        if self._advantage_estimation_method != 'subm':
            target_value_r = data['target_value_r']
        else:
            target_value_r = torch.zeros_like(target_value_c)

        # Construct an episode dataset with only observation
        episode_dataset = EpisodeDataset(
            (
                obs,
                act,
            ),
            done_tensor=done,
            padding=self._cfgs.algo_cfgs.ep_padding,
            max_seq_len=self._cfgs.train_cfgs.max_episode_steps,
        )

        # Get the current actor's action distribution for KL divergence calculation
        ep_obs_list = episode_dataset.padded_episodes[0]  # List of episodes of obs: Episode (Batch) x Sequence x Feature
        ep_act_list = episode_dataset.padded_episodes[1]  # List of episodes of act
        with torch.no_grad():
            old_distribution: Union[Categorical, List[Categorical]] = self._actor_critic.forward_actor(ep_obs_list, ep_act_list)

        if isinstance(old_distribution, list):
            old_logits = logits_from_multi_categorical(old_distribution)
        else:
            old_logits = old_distribution.logits
        # print(f'{old_logits.shape=}')

        # Construct an episode dataloader for training
        episode_dataset = EpisodeDataset(
            (
                obs,
                act,
                act_overlaid,
                logp,
                old_logits,
                reward,
                reward_pred,
                target_value_r,
                next_obs,
                target_value_c,
                adv_r,
                adv_c,
                cost,
                cost_pred,
            ),
            done_tensor=done,
            padding=self._cfgs.algo_cfgs.ep_padding,
            max_seq_len=self._cfgs.train_cfgs.max_episode_steps,
        )
        episode_dataloader = DataLoader(
            dataset=episode_dataset,
            batch_size=self._cfgs.algo_cfgs.ep_batch_size if self._cfgs.algo_cfgs.ep_padding else 1,
            shuffle=True,
        )

        # Train CACD
        final_steps = self._cfgs.algo_cfgs.update_iters
        kl_early_stopped: bool = False
        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating ...'):
            for (
                obs,
                act,
                act_overlaid,
                logp,
                old_logits,
                reward,
                reward_pred,
                target_value_r,
                next_obs,
                target_value_c,
                adv_r,
                adv_c,
                cost,
                cost_pred,
            ) in episode_dataloader:
                # Step 1: Update semantic dynamics model
                if self._cfgs.algo_cfgs.use_sdm:
                    obs, act, next_obs = self._view2d(obs), self._view2d(act), self._view2d(next_obs)
                    self._update_sdm(obs=obs, act=act, next_obs=next_obs)

                # Step 2: Update cost (value) critic
                if self._cfgs.algo_cfgs.use_cost:
                    # target_value_c = self._view2d(target_value_c, squeeze=True)
                    # self._update_cost_critic(obs, target_value_c)

                    cost = self._view2d(cost, squeeze=True)
                    # self._update_cost_critic(obs, cost)
                    self._update_cost_critic(next_obs, cost)  # cost corresponds to the next obs

                # Step 3: Update Lagrangian multiplier
                if self._cfgs.algo_cfgs.use_mpc_lagrangian:
                    assert self._cfgs.algo_cfgs.use_sdm, f'Needs to enable use_sdm for mpc lagrangian update.'
                    assert self._cfgs.algo_cfgs.use_cost, f'Needs to enable use_cost for mpc lagrangian update.'

                    episodic_cost = torch.sum(cost)
                    self._lagrange.update_lagrange_multiplier(Jc=episodic_cost.item(), ep_states=obs)

                    self._logger.store(
                        {
                            'Metrics/MPCLagrangeMultiplier': self._lagrange.lagrangian_multiplier,
                            'Metrics/MPCAlpha': torch.sigmoid(self._lagrange.alpha_raw),
                        }
                    )

                # Step 4 and 5: Update actor and reward estimator
                if kl_early_stopped:  # Update reward estimator only
                    if self._advantage_estimation_method == 'subm':
                        reward = self._view2d(reward, squeeze=True)
                        act = self._view2d(act)
                        self._update_reward_estimator(obs, act, reward)
                    else:
                        target_value_r = self._view2d(target_value_r, squeeze=True)
                        self._update_reward_critic(obs, act, target_value_r)
                else:  # Update actor and (immediate) reward estimator
                    old_logits = self._view2d(old_logits)
                    if isinstance(self._actor_critic.actor, LatentMultiCategoricalActor):
                        self._p_dist = [Categorical(logits=split) for split in
                                        torch.split(old_logits, list(self._actor_critic.actor._act_dim_list), dim=-1)]
                    else:
                        self._p_dist = Categorical(logits=old_logits)

                    act, act_overlaid, logp, adv_r, adv_c, reward, target_value_r = (
                        self._view2d(act),
                        self._view2d(act_overlaid),
                        self._view2d(logp),
                        self._view2d(adv_r),
                        self._view2d(adv_c),
                        self._view2d(reward, squeeze=True),
                        self._view2d(target_value_r, squeeze=True),
                    )

                    if self._cfgs.algo_cfgs.use_sdm:
                        obs = obs.unsqueeze(0)  # Revert obs to 3 dim: [Batch, Sequence, Feature]

                    self._update_actor_and_reward(obs, act, act_overlaid, logp, adv_r, adv_c,
                                                  reward if self._advantage_estimation_method == 'subm' else target_value_r)

            if kl_early_stopped:
                continue

            new_distribution = self._actor_critic.forward_actor(ep_obs_list, ep_act_list)
            if isinstance(new_distribution, list):  # Multi-discrete
                if self._cfgs.model_cfgs.first_non_no_op_win:
                    # Note the order of new_distribution and old_distribution is different from the original FOCOPS.
                    # This is to align with the order of new and old distributions in policy loss calculation
                    kl = kld_multi_categorical_flattened(new_distribution, old_distribution)
                else:
                    kl = kld_multi_categorical(old_distribution, new_distribution)
            else:  # Discrete
                kl = (
                    torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                    .sum(-1, keepdim=True)
                    .mean()
                )
            kl = distributed.dist_avg(kl)

            self._logger.store({'Train/KL': kl.item()})
            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                final_steps = i + 1
                self._logger.log(f'Actor training is early stopped at iter {i + 1} '
                                 f'due to reaching max kl {self._cfgs.algo_cfgs.target_kl}: {kl.item()}')

                if not self._cfgs.algo_cfgs.kl_continue_other:
                    break
                else:
                    kl_early_stopped = True

        # Log advantage
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        self._logger.store(
            {
                'Train/StopIter': final_steps,
                'Value/Adv_r': adv_r.mean().item(),
                'Value/Adv_c': adv_c.mean().item(),
                'Value/Adv': adv.mean().item(),
            },
        )

        # Log lagrange multiplier
        if not self._cfgs.algo_cfgs.use_mpc_lagrangian:
            self._logger.store(
                {
                    'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
                }
            )

    def _view2d(self, tensor: torch.Tensor, squeeze: bool = False) -> torch.Tensor:
        """
        View a 3d tensor as 2d tensor.
        :param tensor: input tensor
        :param squeeze: whether to squeeze the last dimension if it is 1
        :return: 2d tensor
        """
        if tensor.dim() == 2:  # Batch x Feature
            tensor = tensor.unsqueeze(1)  # Batch x Sequence x Feature

        if tensor.dim() == 3:  # Batch x Sequence x Feature
            batch_size, sequence_length, feature_size = tensor.shape
            tensor_out = tensor.reshape(batch_size * sequence_length, feature_size)
            if squeeze and feature_size == 1:
                tensor_out = tensor_out.squeeze(-1)
                assert tensor_out.dim() == 1, f'Tensor reshape wrong: {tensor_out.dim()=}'
            else:
                assert tensor_out.dim() == 2, f'Tensor reshape wrong: {tensor_out.dim()=}'

            return tensor_out
        else:
            print(f'Only support viewing 3d tensor as 2d, given {tensor.dim()}d tensor.')
            raise NotImplementedError

    def _update_actor_and_reward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        act_overlaid: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        reward: torch.Tensor,
    ) -> None:
        """
        Update actor and reward networks. When using separate GRU networks,
        actor path and reward path are trained independently.
        """
        # Forward pass actor and reward estimator to get network outputs
        distribution, reward_pred = self._actor_critic.forward_actor_reward(obs, act)

        # Calculate losses
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        # loss_pi = self._loss_pi(distribution, act, logp, adv)  # original version
        loss_pi = self._loss_pi_safe(distribution, act, act_overlaid, logp, adv)  # safe version
        loss_r = self._loss_reward(reward, reward_pred[0])  # either for reward estimator or reward value critic

        # (Optional) Use EMA to normalize losses to balance the effects of actor and reward estimator
        # self._pi_loss_mean = (1 - self._beta) * self._pi_loss_mean + loss_pi.mean().item()
        # self._reward_loss_mean = (1 - self._beta) * self._reward_loss_mean + loss_r.mean().item()
        # loss_pi = loss_pi / (self._pi_loss_mean + 1e-6)
        # loss_r = loss_r / (self._reward_loss_mean + 1e-6) * 0  # scale down to emphasize actor loss

        # (Optional) Use DWA (Dynamic Weight Average) to balance 2 losses
        # T = 0.5  # Temperature term
        # K = 2.  # Sum of weights
        # weights_pi_r = torch.Tensor([loss_pi.mean().item() / (self._last_loss_pi + 1e-6),
        #                              loss_r.mean().item() / (self._last_loss_r + 1e-6)]) / T
        # weights_pi_r = torch.nn.functional.softmax(weights_pi_r) * K
        # loss = weights_pi_r[0] * loss_pi + weights_pi_r[1] * loss_r  # Dynamically weighted loss

        # Update the latest losses for logging
        self._last_loss_pi = loss_pi.mean().item()
        self._last_loss_r = loss_r.mean().item()

        if self._advantage_estimation_method == 'subm':
            self._logger.store({'Loss/Loss_reward_estimator': loss_r.mean().item()})
            # self._logger.store({'Loss/Loss_policy_ratio': weights_pi_r[0]})
            # self._logger.store({'Loss/Loss_reward_ratio': weights_pi_r[1]})
        else:
            self._logger.store({'Loss/Loss_reward_critic': loss_r.mean().item()})

        # Zero all relevant gradients
        self._actor_critic.gru_optimizer.zero_grad()
        self._actor_critic.actor_optimizer.zero_grad()
        self._actor_critic.reward_critic_optimizer.zero_grad()
        if self._actor_critic._separate_reward_gru:
            self._actor_critic.reward_gru_optimizer.zero_grad()

        # When using separate GRUs, optimize actor and reward paths independently
        if self._actor_critic._separate_reward_gru:
            # Optimize actor path (GRU + actor)
            loss_pi.backward()
            if self._cfgs.algo_cfgs.use_max_grad_norm:
                clip_grad_norm_(self._actor_critic.gru.parameters(), self._cfgs.algo_cfgs.max_grad_norm)
                clip_grad_norm_(self._actor_critic.actor.parameters(), self._cfgs.algo_cfgs.max_grad_norm)
            distributed.avg_grads(self._actor_critic.gru)
            distributed.avg_grads(self._actor_critic.actor)
            self._actor_critic.gru_optimizer.step()
            self._actor_critic.actor_optimizer.step()

            # Optimize reward path (reward GRU + reward critic)
            loss_r.backward()
            if self._cfgs.algo_cfgs.use_max_grad_norm:
                clip_grad_norm_(self._actor_critic.reward_gru.parameters(), self._cfgs.algo_cfgs.max_grad_norm)
                clip_grad_norm_(self._actor_critic.reward_critic.parameters(), self._cfgs.algo_cfgs.max_grad_norm)
            distributed.avg_grads(self._actor_critic.reward_gru)
            distributed.avg_grads(self._actor_critic.reward_critic)
            self._actor_critic.reward_gru_optimizer.step()
            self._actor_critic.reward_critic_optimizer.step()
        else:
            # When using shared GRU, optimize everything together
            loss = loss_pi + loss_r  # Update both actor and reward estimator
            # loss = loss_pi  # Only update actor

            # Backpropagate total loss
            loss.backward()

            # Clip gradients if needed
            if self._cfgs.algo_cfgs.use_max_grad_norm:
                clip_grad_norm_(self._actor_critic.gru.parameters(), self._cfgs.algo_cfgs.max_grad_norm)
                clip_grad_norm_(self._actor_critic.actor.parameters(), self._cfgs.algo_cfgs.max_grad_norm)
                clip_grad_norm_(self._actor_critic.reward_critic.parameters(), self._cfgs.algo_cfgs.max_grad_norm)

            # Average gradients across processes
            distributed.avg_grads(self._actor_critic.gru)
            distributed.avg_grads(self._actor_critic.actor)
            distributed.avg_grads(self._actor_critic.reward_critic)

            # Update parameters
            self._actor_critic.gru_optimizer.step()
            self._actor_critic.actor_optimizer.step()
            self._actor_critic.reward_critic_optimizer.step()

    def _update_reward_estimator(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rewards: torch.Tensor,
    ) -> None:
        """Update reward estimator network.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{r} - r)^2

        where :math:`\hat{V}` is the predicted reward and :math:`V` is the target reward.

        Args:
            obs (torch.Tensor): The observation.
            act (torch.Tensor): The actions.
            rewards (torch.Tensor): The true rewards.
        """
        # Zero all related gradients
        self._actor_critic.reward_critic_optimizer.zero_grad()
        if self._actor_critic._separate_reward_gru:
            self._actor_critic.reward_gru_optimizer.zero_grad()

        # Forward pass through reward network and calculate loss
        pred_rewards = self._actor_critic.forward_reward(obs, act)[0]
        loss = nn.functional.mse_loss(pred_rewards.squeeze(), rewards.squeeze())

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        # Backpropagate
        loss.backward()

        # Clip gradients if needed
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
            if self._actor_critic._separate_reward_gru:
                clip_grad_norm_(
                    self._actor_critic.reward_gru.parameters(),
                    self._cfgs.algo_cfgs.max_grad_norm,
                )

        # Average gradients across processes
        distributed.avg_grads(self._actor_critic.reward_critic)
        if self._actor_critic._separate_reward_gru:
            distributed.avg_grads(self._actor_critic.reward_gru)

        # Update parameters
        self._actor_critic.reward_critic_optimizer.step()
        if self._actor_critic._separate_reward_gru:
            self._actor_critic.reward_gru_optimizer.step()

        self._logger.store({'Loss/Loss_reward_estimator': loss.mean().item()})


    def _update_cost_critic(self, obs: torch.Tensor, target_value_c: torch.Tensor) -> None:
        r"""Update cost estimator network.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_c (torch.Tensor): The ``target_value_c`` sampled from buffer.
        """
        self._actor_critic.cost_critic_optimizer.zero_grad()

        # value_c = self._actor_critic.cost_critic(obs)[0]
        value_c = self._actor_critic.cost_critic(obs)[0]

        loss = nn.functional.mse_loss(value_c.squeeze(), target_value_c.squeeze())  # MSE loss
        # loss = nn.functional.l1_loss(value_c, target_value_c)  # L1 loss

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)

        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_estimator': loss.mean().item()})

    def _update_sdm(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """
        Update semantic dynamics model (SDM).

        Args:
            obs: Observation tensor with shape [Batch, Feature].
            act: Action tensor with shape [Batch, Action].
            next_obs: Next observation tensor with shape [Batch, Feature].

        Returns:
            None
        """
        # Concat obs and act
        obs_act = torch.cat((obs, act), dim=-1)

        # Forward pass
        delta = self._actor_critic.sdm(obs_act)

        # Calculate loss
        # loss = self._actor_critic.sdm.loss_l1(obs, act, delta, next_obs)
        loss = self._actor_critic.sdm.loss_iou(obs, act, delta, next_obs)

        # Backpropagate
        self._actor_critic.sdm.backprop(loss)

        # Log sdm loss
        self._logger.store({'Loss/Loss_sdm': loss.mean().item()})
