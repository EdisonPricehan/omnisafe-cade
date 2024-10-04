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
    ) -> None:
        assert isinstance(act_space, MultiDiscrete), f'Only supports multi-categorical action space!'

        """Initialize an instance of :class:`MultiCategoricalActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self._latent_size: int = latent_size

        self._act_dim_list: List[int] = act_space.nvec
        self._act_dim_exec: int = len(act_space.nvec)
        print(f'{self._act_dim_list=}')

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

    def forward(self, latent: torch.Tensor) -> Distribution:
        """Forward method.

        Args:
            latent (torch.Tensor): Latent from upstream recurrent network.

        Returns:
            The current distribution.
        """
        self._current_dist_list = self._distribution(latent)
        self._after_inference = True
        return self._current_dist_list

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
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self._current_dist_list, torch.unbind(act, dim=-1))], dim=-1
        ).sum(dim=-1)


