import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LatentDynamicsModel(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 act_dim: int,
                 gru_hidden_dim: int,
                 num_gru_layers: int = 1,
                 lr: float = 1e-3,
                 weight_initialization_mode: str = 'kaiming_uniform'):
        """
        Latent Dynamics Model that learn the dynamics of latent embeddings with memory
        Args:
            latent_dim:
            act_dim:
            gru_hidden_dim:
            num_gru_layers:
            lr:
            weight_initialization_mode:
        """
        super().__init__()

        self.latent_dim = latent_dim  # Latent representation dimension from VAE
        self.act_dim = act_dim  # Action space dimension
        self.gru_hidden_dim = gru_hidden_dim  # GRU hidden state dimension

        # GRU layer for recurrent dynamics
        self.gru = nn.GRU(input_size=latent_dim + act_dim,
                          hidden_size=gru_hidden_dim,
                          num_layers=num_gru_layers,
                          batch_first=False,
                          )

        # MLP to predict the next latent state z_{t+1} from the GRU hidden state
        self.fc = nn.Linear(gru_hidden_dim, latent_dim)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Initialize weights if needed
        self.init_weights(weight_initialization_mode)

    def init_weights(self, mode):
        """
        Initialize weights according to the specified mode
        Args:
            mode:

        Returns:

        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if mode == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif mode == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_prev: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict the next latent state z_{t+1} given current z_t, action a_t, and hidden state h_{t-1}.

        Args:
            z_t: Current latent representation (batch_size, latent_dim)
            a_t: Current action (batch_size, act_dim)
            h_prev: Previous hidden state of the GRU (num_gru_layers, batch_size, gru_hidden_dim)

        Returns:
            z_next: Predicted next latent representation (batch_size, latent_dim)
            h_next: Updated GRU hidden state (num_gru_layers, batch_size, gru_hidden_dim)
        """
        # Concatenate latent state and action as the input to GRU
        gru_input = torch.cat([z_t, a_t], dim=-1).unsqueeze(1)  # (seq_size, batch_size=1, latent_dim + act_dim)

        # Pass through GRU
        # gru_output: (seq_len, batch_size=1, gru_hidden_dim)
        # h_next: (num_gru_layers, batch_size=1, gru_hidden_dim)
        gru_output, h_next = self.gru(gru_input, h_prev)

        # Predict the next latent state z_{t+1} from the top layer hidden state
        z_next = self.fc(gru_output)  # (seq_len, batch_size=1, latent_dim)

        return z_next, h_next

    def predict(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_prev: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the next latent state given the current latent state and action.

        Args:
            z_t: Current latent representation, shape (batch_size, latent_dim) or (1, latent_dim) if unbatched
            a_t: Current action, shape (batch_size, act_dim) or (1, act_dim) if unbatched
            h_prev: Previous hidden state of the GRU (num_gru_layers, batch_size, gru_hidden_dim) or (num_gru_layers, 1, gru_hidden_dim)

        Returns:
            z_pred: Predicted next latent state, shape (batch_size, latent_dim) or (1, latent_dim) if unbatched
            h_next: Updated GRU hidden state (num_gru_layers, batch_size, gru_hidden_dim) or (num_gru_layers, 1, gru_hidden_dim)
        """
        # Ensure that the input tensors have the correct dimensions
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)  # Add batch dimension if unbatched
        if a_t.dim() == 1:
            a_t = a_t.unsqueeze(0)  # Add batch dimension if unbatched

        # Concatenate the latent state and action
        gru_input = torch.cat([z_t, a_t], dim=-1).unsqueeze(
            0)  # Add sequence length dimension (1, batch_size, latent_dim + act_dim)

        # Forward pass through the GRU
        gru_output, h_next = self.gru(gru_input, h_prev)  # gru_output: (1, batch_size, gru_hidden_dim)

        # Predict the next latent state
        z_pred = self.fc(gru_output.squeeze(0))  # Remove the sequence length dimension, shape (batch_size, latent_dim)

        return z_pred, h_next

    def loss_mse(self, pred_latent: torch.Tensor, true_latent: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Error loss between predicted and true latent states.

        Args:
            pred_latent: Predicted next latent state
            true_latent: True next latent state

        Returns:
            MSE loss
        """
        return F.mse_loss(pred_latent, true_latent)

    def backprop(self, loss: torch.Tensor) -> None:
        """
        Perform a backpropagation step to update model parameters.

        Args:
            loss: The calculated loss tensor

        Returns:
            None
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
