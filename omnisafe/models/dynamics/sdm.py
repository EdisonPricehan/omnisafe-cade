import numpy as np
from typing import Union, List, Type, Tuple

import torch
from torch import nn, optim
import torch.nn.functional as F

import kornia as K

from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.model import get_obs_dim, get_act_dim, build_mlp_network


class SemanticDynamicsModel(nn.Module):
    def __init__(self,
                 obs_space: OmnisafeSpace,
                 act_space: OmnisafeSpace,
                 model_cfgs: ModelConfig,
                 patch_rows: int = 5,
                 patch_cols: int = 5,
                 weight_initialization_mode: InitFunction = 'kaiming_uniform',
                 ):
        super().__init__()

        # Constants
        self._obs_space: OmnisafeSpace = obs_space
        self._act_space: OmnisafeSpace = act_space
        self._obs_dim: int = get_obs_dim(obs_space)
        self._act_dim: int = get_act_dim(act_space, execution_dim=True)
        self._hidden_sizes: List[int] = model_cfgs.dynamics.hidden_sizes
        self._activation: Activation = model_cfgs.dynamics.activation
        self._output_size: int = 8  # 4 corner coordinate offsets
        self._patch_rows: int = patch_rows
        self._patch_cols: int = patch_cols
        # 4 Corners coordinates for a patchified image
        self._corners_coord: torch.Tensor = torch.Tensor([
            [0, 0],
            [self._patch_rows - 1, 0],
            [self._patch_rows - 1, self._patch_cols - 1],
            [0, self._patch_cols - 1]
        ])

        # Define semantic dynamics model and its optimizer
        self.model = build_mlp_network(sizes=[self._obs_dim + self._act_dim, *self._hidden_sizes, self._output_size],
                                       activation=self._activation,
                                       weight_initialization_mode=weight_initialization_mode)
        if model_cfgs.dynamics.lr is not None:
            self.model_optimizer: optim.Adam = optim.Adam(self.model.parameters(),
                                                          lr=model_cfgs.dynamics.lr)

    def forward(self, obs_act: torch.Tensor) -> torch.Tensor:
        """
        Forward method.
        Args:
            obs_act:  tensor with shape (N, obs_feature + act_feature)

        Returns: x and y coordinates of 4 corners, tensor with shape (N, 8)

        """
        delta = self.model(obs_act)
        return delta

    def predict(self, obs_act: torch.Tensor, round_to_int: bool = True) -> torch.Tensor:
        """
        Given current observation and action, predict the next observation.
        Args:
            obs_act:
            round_to_int:

        Returns:

        """
        with torch.no_grad():
            delta = self.forward(obs_act)  # B x (H x W)
            pred_obs_next = self._predict_next_obs(obs_act, delta).view(-1, self._patch_rows * self._patch_cols)
            if round_to_int:
                return torch.round(pred_obs_next)
            else:
                return pred_obs_next

    def _predict_next_obs(self, obs_act: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Calculate next observation based on sdm-predicted corner coordinates offsets, which are used to estimate the
        homography matrix from current observation to the next predicted observation.
        Args:
            obs_act:
            delta:

        Returns:

        """
        batch_size = obs_act.shape[0]
        corners_cur = self._corners_coord.expand(batch_size, 4, 2)  # B x 4 x 2
        # print(f'{obs_act.shape=}')
        # print(f'{obs_act.requires_grad=}')
        # print(f'{batch_size=}')
        obs_cur = obs_act[:, :-self._act_dim].view(batch_size, 1, self._patch_rows, self._patch_cols)  # B x 1 x H x W

        # Get 4 offseted corner coordinates
        delta = delta.view(-1, 4, 2)
        corners_next = corners_cur + delta

        # Calculate homography matrix
        H = K.geometry.get_perspective_transform(corners_cur, corners_next)
        # print(f'{H[0]}')

        # Warp current observation by the above homography matrix
        # Fill in the unknown pixels by 0.5 for neural uncertainty (input observation is binary)
        pred_obs_next = K.geometry.warp_perspective(obs_cur, H, (self._patch_rows, self._patch_cols),
                                                    mode='nearest', padding_mode='fill', fill_value=torch.ones(3) * 0.5)

        # 3-channel warped image is given, but only need a single channel
        pred_obs_next = pred_obs_next[:, 0]  # B x H x W
        # print(f'{pred_obs_next.shape=}')
        return pred_obs_next

    def loss_l1(
        self,
        obs_cur: torch.Tensor,
        act: torch.Tensor,
        delta: torch.Tensor,
        obs_next: torch.Tensor) -> torch.Tensor:
        """
        Calculate the L1 loss between the true and the estimated next observation.
        Args:
            obs_cur:
            act:
            delta:
            obs_next:

        Returns:

        """
        batch_size = obs_cur.shape[0]
        obs_next = obs_next.view(batch_size, self._patch_rows, self._patch_cols)
        obs_act = torch.cat((obs_cur, act), dim=-1)
        pred_obs_next = self._predict_next_obs(obs_act, delta)
        loss = F.l1_loss(pred_obs_next, obs_next)
        # print(f'L1 {loss=}')
        return loss

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
