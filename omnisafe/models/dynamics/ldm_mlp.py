import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import List

from omnisafe.utils.model import build_mlp_network
from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.config import ModelConfig


class LatentDynamicsModelMLP(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        act_dim: int,
        model_cfgs: ModelConfig,
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ):
        """
        The Multi-layer Perception version of the Latent Dynamics Model
        Args:
            latent_dim:
            act_dim:
            model_cfgs:
            weight_initialization_mode:
        """
        super().__init__()

        # Store the dimensions
        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.hidden_sizes: List[int] = model_cfgs.dynamics.hidden_sizes
        self.activation: Activation = model_cfgs.dynamics.activation

        # Define the MLP network for latent dynamics
        self.model = build_mlp_network(
            sizes=[latent_dim + act_dim, *self.hidden_sizes, latent_dim],
            activation=self.activation,
            weight_initialization_mode=weight_initialization_mode
        )

        # Define optimizer
        if model_cfgs.dynamics.lr is not None:
            self.model_optimizer: optim.Adam = optim.Adam(self.parameters(), lr=model_cfgs.dynamics.lr)

    def forward(self, latent_act: torch.Tensor) -> torch.Tensor:
        """
        Forward method for predicting the next latent representation.
        Args:
            latent_act: Tensor with shape (batch_size, latent_dim + act_dim)

        Returns:
            Tensor with predicted next latent representation
        """
        return self.model(latent_act)

    def predict(self, latent_act: torch.Tensor) -> torch.Tensor:
        """
        Predict the next latent representation using the dynamics model.
        Args:
            latent_act: Input tensor (latent + action)

        Returns:
            Predicted next latent representation
        """
        with torch.no_grad():
            return self.forward(latent_act)

    def loss_l1(self, latent_next_pred: torch.Tensor, latent_next: torch.Tensor) -> torch.Tensor:
        """
        Calculate the L1 loss between the predicted and true next latent representation.
        Args:
            latent_next_pred: Predicted next latent representation
            latent_next: True next latent representation

        Returns:
            L1 loss
        """
        return F.l1_loss(latent_next_pred, latent_next)

    def loss_mse(self, latent_next_pred: torch.Tensor, latent_next: torch.Tensor) -> torch.Tensor:
        """
        Calculate the MSE loss between the predicted and true next latent representation.
        Args:
            latent_next_pred: Predicted next latent representation
            latent_next: True next latent representation

        Returns:
            MSE loss
        """
        return F.mse_loss(latent_next_pred, latent_next)

    def backprop(self, loss: torch.Tensor) -> None:
        """
        Perform a backpropagation step to update model parameters.
        Args:
            loss: The calculated loss tensor

        Returns:
            None
        """
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
