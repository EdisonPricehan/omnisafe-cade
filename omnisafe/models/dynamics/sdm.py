from typing import Union, List

import torch
from torch import nn, optim
import torch.nn.functional as F

import kornia as K

from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.model import get_obs_dim, get_act_dim, build_mlp_network
from omnisafe.utils.math import soft_iou_loss


class SemanticDynamicsModel(nn.Module):
    def __init__(self,
                 obs_space: OmnisafeSpace,
                 act_space: OmnisafeSpace,
                 model_cfgs: ModelConfig,
                 patch_rows: int = 5,
                 patch_cols: int = 5,
                 weight_initialization_mode: InitFunction = 'kaiming_uniform',
                 perturb_delta: bool = False,
                 device: torch.device = torch.device('cuda:0'),
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
        self.perturb_delta: bool = perturb_delta
        self.device: torch.device = device if torch.cuda.is_available() else torch.device('cpu')
        print(f'SDM using device: {self.device}')

        # 4 Corners coordinates for a patchified image
        self._corners_coord: torch.Tensor = torch.Tensor([
            [0, 0],
            [self._patch_rows - 1, 0],
            [self._patch_rows - 1, self._patch_cols - 1],
            [0, self._patch_cols - 1]
        ]).to(self.device)

        # Define semantic dynamics model and its optimizer
        self.model = build_mlp_network(
            sizes=[self._obs_dim + self._act_dim, *self._hidden_sizes, self._output_size],
            # sizes=[self._obs_dim + self._act_dim, *self._hidden_sizes, 2],
            activation=self._activation,
            weight_initialization_mode=weight_initialization_mode
        ).to(self.device)

        # Learnable scaling factors for different movements
        if self.perturb_delta:
            self.scaling_translation = nn.Parameter(torch.ones(1))  # For forward/backward and left/right movement
            self.scaling_vertical = nn.Parameter(torch.ones(1))  # For up/down movement
            self.scaling_rotation = nn.Parameter(torch.ones(1))  # For rotation

        if model_cfgs.dynamics.lr is not None:
            self.model_optimizer: optim.Adam = optim.Adam(self.parameters(), lr=model_cfgs.dynamics.lr)

    def forward(self, obs_act: torch.Tensor) -> torch.Tensor:
        """
        Forward method.
        Args:
            obs_act:  tensor with shape (N, obs_feature + act_feature)

        Returns: x and y coordinates of 4 corners, tensor with shape (N, 8)

        """
        delta = self.model(obs_act)

        # Apply action perturbations with scaling
        if self.perturb_delta:
            action = obs_act[:, -self._act_dim:]  # Extract action
            delta_adjusted = self.apply_action_perturbation(delta, action)
            print(f'{delta=} {delta_adjusted=}')
            return delta_adjusted
        else:
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

    def apply_action_perturbation(self, delta: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Apply action perturbations to the predicted corner deviations using learnable scaling factors.
        Args:
            delta (torch.Tensor): Predicted corner deviations from the MLP.
            action (torch.Tensor): Action input (4D vector) where only one component is active.

        Returns:
            torch.Tensor: Adjusted corner deviations with action perturbations applied.
        """
        # Extract action components
        action_up_down = action[:, 0]  # Up or down movement (vertical scaling)
        action_rotate = action[:, 1]  # Rotation (left or right)
        action_forward_back = action[:, 2]  # Forward or backward movement (vertical translation)
        action_left_right = action[:, 3]  # Leftward or rightward movement (horizontal translation)

        # Initialize delta_adjusted with the original prediction
        delta_adjusted = delta

        # Scaling for up/down movement (scaling effect)
        if torch.any(action_up_down != 1):  # Up/down action is active
            # 0 means upward (positive scaling), 2 means downward (negative scaling)
            scaling_factor = torch.where(action_up_down == 0, 1, -1)  # Up is positive, down is negative
            scaling_factor = 1 + self.scaling_vertical * scaling_factor  # Learnable scaling applied uniformly

            center = torch.tensor([self._patch_cols // 2, self._patch_rows // 2], dtype=delta.dtype).unsqueeze(0)

            # Apply scaling uniformly to all corners relative to the image center
            corners_relative_to_center = delta_adjusted.view(-1, 4, 2) - center
            scaling_factor = scaling_factor.view(-1, 1, 1)  # Ensure scaling_factor is broadcasted correctly
            delta_adjusted = corners_relative_to_center * scaling_factor + center

        # Translation for forward/backward movement
        elif torch.any(action_forward_back != 1):  # Forward/backward action is active
            # 0 means forward (positive), 2 means backward (negative)
            direction = torch.where(action_forward_back == 0, 1, -1)  # Map 0 to forward (+), 2 to backward (-)
            print(f'{action_forward_back=} {direction=} {self.scaling_translation.data=}')
            translation = self.scaling_translation * direction
            translation = translation.view(-1, 1).repeat(1, 8)  # Adjust translation to match delta size
            delta_adjusted += translation  # Apply translation uniformly to all corners

        # Translation for left/right movement
        elif torch.any(action_left_right != 1):  # Left/right action is active
            # 0 means left (positive), 2 means right (negative)
            direction = torch.where(action_left_right == 0, 1, -1)  # Map 0 to left (+), 2 to right (-)
            translation = self.scaling_translation * direction
            translation = translation.view(-1, 1).repeat(1, 8)  # Adjust translation to match delta size
            delta_adjusted += translation  # Apply translation uniformly to all corners

        # Apply rotation (if action_rotate is active)
        elif torch.any(action_rotate != 1):  # Rotation action is active
            # 0 means rotate left, 2 means rotate right
            scaled_rotation = self.scaling_rotation * torch.where(action_rotate == 0, -1, 1)  # Left (-), right (+)

            # Compute the center of the image
            center = torch.tensor([self._patch_cols // 2, self._patch_rows // 2], dtype=delta.dtype).unsqueeze(0)

            # Create a rotation matrix
            cos_theta = torch.cos(scaled_rotation)
            sin_theta = torch.sin(scaled_rotation)
            rotation_matrix = torch.stack([
                torch.stack([cos_theta, -sin_theta], dim=-1),
                torch.stack([sin_theta, cos_theta], dim=-1)
            ], dim=1)  # Shape (batch_size, 2, 2)

            # Apply the rotation around the center of the image
            corners_relative_to_center = delta_adjusted.view(-1, 4, 2) - center
            rotated_corners = torch.bmm(corners_relative_to_center, rotation_matrix)  # Batch matrix multiplication
            delta_adjusted = rotated_corners + center
        # print(f'{delta=}')
        # print(f'{delta_adjusted=}')
        return delta_adjusted.view(-1, 8)  # Reshape back to the original size (batch_size, 8)

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
        obs_cur = obs_act[:, :-self._act_dim].view(batch_size, 1, self._patch_rows, self._patch_cols)  # B x 1 x H x W

        # Get 4 offseted corner coordinates
        delta = delta.view(-1, 4, 2)
        corners_next = corners_cur + delta
        # corners_next = corners_cur + delta * 5  # scale the delta
        # print(f'{delta=}')

        # Only use the translations predicted by mlp to construct the homography matrix
        # delta = delta.view(-1, 2)
        # H = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        # H[:, 0, 2] = delta[:, 0]  # tx (translation in x direction)
        # H[:, 1, 2] = delta[:, 1]  # ty (translation in y direction)
        # print(f'{delta[0]=}')

        # Calculate homography matrix
        H = K.geometry.get_perspective_transform(corners_cur, corners_next)
        # print(f'{H[0]=}')

        # Warp current observation by the above homography matrix
        # Fill in the unknown pixels by 0.5 for neural uncertainty (input observation is binary)
        pred_obs_next = K.geometry.warp_perspective(
            obs_cur,
            H,
            (self._patch_rows, self._patch_cols),
            # mode='nearest',
            # padding_mode='fill',
            padding_mode='border',
            fill_value=torch.ones(3) * 0.5,
            # fill_value=torch.ones(3) * 1,
        )

        # 3-channel warped image is given, but only need a single channel
        pred_obs_next = pred_obs_next[:, 0]  # B x H x W

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

    def loss_iou(
        self,
        obs_cur: torch.Tensor,
        act: torch.Tensor,
        delta: torch.Tensor,
        obs_next: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = obs_cur.shape[0]
        obs_next = obs_next.view(batch_size, self._patch_rows, self._patch_cols)
        obs_act = torch.cat((obs_cur, act), dim=-1)
        pred_obs_next = self._predict_next_obs(obs_act, delta)
        loss = soft_iou_loss(pred_obs_next, obs_next)
        # print(f'{loss.data=}')

        return loss

    def loss_mse(
        self,
        obs_cur: torch.Tensor,
        act: torch.Tensor,
        delta: torch.Tensor,
        obs_next: torch.Tensor,
    ):
        batch_size = obs_cur.shape[0]
        obs_next = obs_next.view(batch_size, self._patch_rows, self._patch_cols)
        obs_act = torch.cat((obs_cur, act), dim=-1)
        pred_obs_next = self._predict_next_obs(obs_act, delta)

        # print(f'Pred: {pred_obs_next}')
        # print(f'True: {obs_next}')

        loss = F.mse_loss(pred_obs_next[:, 1: self._patch_rows - 1, 1: self._patch_cols - 1],
                          obs_next[:, 1: self._patch_rows - 1, 1: self._patch_cols - 1])

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
        # print(f'{self.scaling_translation.grad=} {self.scaling_translation.requires_grad=}')

