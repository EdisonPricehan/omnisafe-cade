import numpy as np
from typing import Union, List, Type, Tuple

import torch
from torch import nn, optim
import torch.nn.functional as F

from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.model import get_obs_dim, get_act_dim, build_mlp_network


class SemanticDynamicsModelMLP(nn.Module):
    def __init__(self,
                 obs_space: OmnisafeSpace,
                 act_space: OmnisafeSpace,
                 model_cfgs: ModelConfig,
                 weight_initialization_mode: InitFunction = 'kaiming_uniform',
                 ):
        """
        The Multi-layer Perception version of the Semantic Dynamics Model
        Args:
            obs_space:
            act_space:
            model_cfgs:
            weight_initialization_mode:
        """
        super().__init__()

        # Constants
        self._obs_space: OmnisafeSpace = obs_space
        self._act_space: OmnisafeSpace = act_space
        self._obs_dim: int = get_obs_dim(obs_space)
        self._act_dim: int = get_act_dim(act_space, execution_dim=True)
        self._hidden_sizes: List[int] = model_cfgs.dynamics.hidden_sizes
        self._activation: Activation = model_cfgs.dynamics.activation

        # Define semantic dynamics model and its optimizer
        self.model = build_mlp_network(sizes=[self._obs_dim + self._act_dim, *self._hidden_sizes, self._obs_dim],
                                       activation=self._activation,
                                       weight_initialization_mode=weight_initialization_mode)
        self.sigmoid = nn.Sigmoid()

        if model_cfgs.dynamics.lr is not None:
            self.model_optimizer: optim.Adam = optim.Adam(self.parameters(), lr=model_cfgs.dynamics.lr)

    def forward(self, obs_act: torch.Tensor) -> torch.Tensor:
        """
        Forward method.
        Args:
            obs_act:  tensor with shape (N, obs_feature + act_feature)

        Returns:

        """
        return self.sigmoid(self.model(obs_act))

    def predict(self, obs_act: torch.Tensor, round_to_int: bool = True) -> torch.Tensor:
        """
        Given current observation and action, predict the next observation.
        Args:
            obs_act:
            round_to_int:

        Returns:

        """
        with torch.no_grad():
            pred_obs_next = self.forward(obs_act)
            if round_to_int:
                return torch.round(pred_obs_next)
            else:
                return pred_obs_next

    def backprop(self, loss: torch.Tensor) -> None:
        """
        Update sdm parameters by loss
        Args:
            loss:

        Returns:

        """
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
