"""Implementation of LatentCategoricalActor."""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import MultiDiscrete
from typing import List

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution

from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


# pylint: disable-next=too-many-instance-attributes
class LatentMultiCategoricalActor(Actor):
    """Implementation of LatentCategoricalActor.

    LatentMultiCategoricalActor is an actor suitable for multi-discrete action. It is used in
    multi-discrete action space environment such as ``Riverine Environment`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        latent_size (int): Latent dimension of upstream recurrent network.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        first_non_no_op_win (bool, optional): Whether to use the first non-no-op action as the winning action.
            Defaults to ``True``.
    """

    _current_dist_list: List[Categorical]

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        latent_size: int = 128,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        first_non_no_op_win: bool = True,
    ) -> None:
        assert isinstance(act_space, MultiDiscrete), f'Only supports multi-categorical action space!'

        """Initialize an instance of :class:`MultiCategoricalActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self._latent_size: int = latent_size

        self._act_dim_list: List[int] = act_space.nvec
        self._act_dim_exec: int = len(act_space.nvec)
        # print(f'{self._act_dim_list=}')

        self._first_non_no_op_win: bool = first_non_no_op_win

        self.logits: nn.Module = build_mlp_network(
            sizes=[self._latent_size, *self._hidden_sizes, self._act_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def _distribution(self, latent: torch.Tensor) -> List[Categorical]:
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        Args:
            latent (torch.Tensor): Latent from upstream recurrent network.

        Returns:
            List of categorical distribution over actions based on the actor's logits.
        """
        logits = self.logits(latent)
        return [Categorical(logits=split) for split in torch.split(logits, list(self._act_dim_list), dim=-1)]

    def predict(self, latent: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict the action based on given observations.

        The predicted action depends on the ``deterministic`` flag.

        - If ``deterministic`` is ``True``, the predicted action is the action with highest probability.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.

        Args:
            latent (torch.Tensor): Latent from upstream recurrent network.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            The action with highest probability if deterministic is True,
            otherwise a sampled action from the distribution.
        """
        self._current_dist_list = self._distribution(latent=latent)
        self._after_inference = True
        if deterministic:
            action = torch.stack([torch.argmax(dist.logits, dim=-1, keepdim=True) for dist in self._current_dist_list], dim=-1)
        else:
            action = torch.stack([dist.sample() for dist in self._current_dist_list], dim=-1)
        return action.view(-1, self._act_dim_exec)

    def forward(self, latent: torch.Tensor) -> List[Categorical]:
        """Forward method.

        Args:
            latent (torch.Tensor): Latent from upstream recurrent network.

        Returns:
            The current distribution.
        """
        self._current_dist_list = self._distribution(latent)
        self._after_inference = True
        return self._current_dist_list

    def _log_prob_first_non_no_op(self, act: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of the action when the first non-no-op action is considered as the winning action.

        Args:
            act (torch.Tensor): Action tensor of shape [B, 4], where B is the batch size and 4 is the number of axes.

        Returns:
            torch.Tensor: Log probability of the action, shape [B].
        """
        no_op = 1
        B = act.size(0)
        device = act.device

        # 1) log π_j(no-op) for each axis j → [B,4]
        logp_noop = []
        for dist in self._current_dist_list:  # 4 Categorical distributions
            noop_actions = torch.full((B,), no_op, device=device, dtype=torch.long)
            logp_noop.append(dist.log_prob(noop_actions))
        logp_noop = torch.stack(logp_noop, dim=1)

        # 2) log π_j(act[:, j]) for each axis j → [B,4]
        logp_given = []
        for j, dist in enumerate(self._current_dist_list):
            logp_given.append(dist.log_prob(act[:, j]))
        logp_given = torch.stack(logp_given, dim=1)

        # 3) for each batch element, find executed axis i (first non–no-op), then assemble log-prob
        out = torch.zeros(B, device=device)
        for b in range(B):
            # find first axis where action != no-op
            exec_idx = -1
            for j in range(4):
                if int(act[b, j].item()) != no_op:
                    exec_idx = j
                    break

            if exec_idx == -1:
                # all no-ops: product over j of π_j(no-op) → sum of logs
                out[b] = logp_noop[b].sum()
            else:
                # sum_{j < i} log π_j(no-op)  +  log π_i(direction)
                out[b] = logp_noop[b, :exec_idx].sum() + logp_given[b, exec_idx]

        return out

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the action given the current distribution.

        .. warning::
            You must call :meth:`forward` or :meth:`predict` before calling this method.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward` .

        Returns:
            Log probability of the action.
        """
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False

        if self._first_non_no_op_win:
            return self._log_prob_first_non_no_op(act)
        else:
            # Original implementation that computes log probability for each axis indiscriminately
            return torch.stack(
                [dist.log_prob(action) for dist, action in zip(self._current_dist_list, torch.unbind(act, dim=-1))], dim=-1
            ).sum(dim=-1)


