"""Implementation of RCritic."""

import torch
import torch.nn as nn
from typing import Optional

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network, get_act_dim


class RCritic(Critic):
    """Implementation of RCritic.

        A R-function approximator that uses a multi-layer perceptron (MLP) to map latent variables to immediate reward.
        This class is an inherit class of :class:`Critic`.

        Args:
            obs_dim (int): Observation dimension.
            act_dim (int): Action dimension.
            latent_size (int): Latent dimension of upstream recurrent network.
            hidden_sizes (list of int): List of hidden layer sizes.
            pred_value (bool): If true, input is latent state, output is (latent) state value.
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
                 pred_value: bool = False,
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
        self._act_exec_dim: int = get_act_dim(act_space, execution_dim=True)
        self.pred_value: bool = pred_value

        self.net_lst: list[nn.Module] = []

        for idx in range(self._num_critics):
            # Maps latent to immediate reward
            net = build_mlp_network(
                sizes=[self._latent_size, *self._hidden_sizes, 1]  # obs only dependence
                if pred_value else
                [self._latent_size + self._act_exec_dim, *self._hidden_sizes, 1],  # obs+act dependence
                activation=self._activation,
                # output_activation='sigmoid',
                weight_initialization_mode=self._weight_initialization_mode,
            )
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)

    def forward(
        self,
        latent: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """Forward function.

        Specifically, R function approximator maps latent neurons to immediate reward.

        Args:
            latent (torch.Tensor): 1d latent variables from any recurrent network.
            action (Optional[torch.Tensor]): 1d action tensor upon this (latent) observation, or None

        Returns:
            The R critic value (immediate reward) of latent.
        """
        res = []
        latent_act = latent if self.pred_value and action is None else torch.cat([latent, action], dim=-1)
        for critic in self.net_lst:
            res.append(torch.squeeze(critic(latent_act), -1))
        return res
