"""Implementation of RCritic."""

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class RCritic(Critic):
    """Implementation of RCritic.

        A R-function approximator that uses a multi-layer perceptron (MLP) to map latent variables to immediate reward.
        This class is an inherit class of :class:`Critic`.

        Args:
            obs_dim (int): Observation dimension.
            act_dim (int): Action dimension.
            latent_size (int): Latent dimension of upstream recurrent network.
            hidden_sizes (list of int): List of hidden layer sizes.
            activation (Activation, optional): Activation function. Defaults to ``'relu'``.
            weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
                ``'kaiming_uniform'``.
            num_critics (int, optional): Number of critics. Defaults to 1.
        """
    def __init__(self,
                 obs_space: OmnisafeSpace,
                 act_space: OmnisafeSpace,
                 latent_size: int,
                 hidden_sizes: list[int],
                 activation: Activation = 'relu',
                 weight_initialization_mode: InitFunction = 'kaiming_uniform',
                 num_critics: int = 1,
                 ):
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder=False,
        )
        self._latent_size: int = latent_size
        self.net_lst: list[nn.Module]
        self.net_lst = []

        for idx in range(self._num_critics):
            # Maps latent to immediate reward
            net = build_mlp_network(
                sizes=[self._latent_size, *self._hidden_sizes, 1],
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)

    def forward(
        self,
        latent: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Forward function.

        Specifically, R function approximator maps latent neurons to immediate reward.

        Args:
            latent (torch.Tensor): 1d latent variables from any recurrent network.

        Returns:
            The R critic value (immediate reward) of latent.
        """
        res = []
        for critic in self.net_lst:
            res.append(torch.squeeze(critic(latent), -1))
        return res
