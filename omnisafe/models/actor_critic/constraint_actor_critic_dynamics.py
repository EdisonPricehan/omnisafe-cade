import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR, LinearLR
from torch.distributions import Distribution

from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.model import get_obs_dim, get_act_dim
from omnisafe.models.actor.actor_builder import ActorBuilder
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.models.base import Actor, Critic
from omnisafe.models.dynamics.sdm import SemanticDynamicsModel

from typing import Optional, Union, List, Tuple


class ConstraintActorCriticDynamics(nn.Module):
    """
    This module contains 4 networks:
    2 Non-Markovian parts:
    Actor-critic network with shared GRU layers that account for the non-Markovian decision making and reward gaining
    2 Markovian parts:
    Cost critic network (MLP) that estimates the cost of an observation
    Semantic dynamics network (MLP) that estimates the next observation given current observation and action
    """
    def __init__(self,
                 obs_space: OmnisafeSpace,
                 act_space: OmnisafeSpace,
                 model_cfgs: ModelConfig,
                 epochs: int,
                 learn_reward_value: bool = False,
                 ):
        super().__init__()

        # Define common constants
        self.obs_dim = get_obs_dim(obs_space)
        self.act_dim = get_act_dim(act_space, execution_dim=True)
        self.learn_reward_value = learn_reward_value

        # Define shared GRU layer for actor and reward critic
        self.gru = nn.GRU(input_size=self.obs_dim,
                          hidden_size=model_cfgs.latent_size,
                          num_layers=model_cfgs.num_gru_layers,
                          batch_first=True)
        self.gru_optimizer: optim.Optimizer = optim.Adam(self.gru.parameters(), lr=model_cfgs.gru_lr)
        self.gru_latent: Optional[torch.Tensor] = None
        self.add_module('gru', self.gru)

        # Define actor head
        self.actor: Actor = ActorBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.actor.hidden_sizes,
            latent_size=model_cfgs.latent_size,
            activation=model_cfgs.actor.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
        ).build_actor(
            actor_type=model_cfgs.actor_type,
        )
        self.add_module('actor', self.actor)
        if model_cfgs.actor.lr is not None:
            self.actor_optimizer: optim.Optimizer = optim.Adam(self.actor.parameters(), lr=model_cfgs.actor.lr)
        if model_cfgs.actor.lr is not None:
            self.actor_scheduler: LinearLR | ConstantLR
            if model_cfgs.linear_lr_decay:
                self.actor_scheduler = LinearLR(
                    self.actor_optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=epochs,
                    verbose=True,
                )
            else:
                self.actor_scheduler = ConstantLR(
                    self.actor_optimizer,
                    factor=1.0,
                    total_iters=epochs,
                    verbose=True,
                )

        # Define reward critic head (immediate reward estimator)
        self.reward_critic: Critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            latent_size=model_cfgs.latent_size,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic(critic_type='r')  # critic that accepts latent vector as input
        self.add_module('reward_critic', self.reward_critic)
        if model_cfgs.critic.lr is not None:
            self.reward_critic_optimizer: optim.Optimizer = optim.Adam(
                self.reward_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

        # Define cost critic
        # Since cost is Markovian, a simple MLP would suffice (no upstream recurrent layers)
        self.cost_critic: Critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic(critic_type='v')
        self.add_module('cost_critic', self.cost_critic)
        if model_cfgs.critic.lr is not None:
            self.cost_critic_optimizer: optim.Optimizer
            self.cost_critic_optimizer = optim.Adam(
                self.cost_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

        # Define semantic dynamics model
        self.sdm = SemanticDynamicsModel(obs_space=obs_space,
                                         act_space=act_space,
                                         model_cfgs=model_cfgs,
                                         patch_rows=5,  # for CliffCircular
                                         patch_cols=5,  # for CliffCircular
                                         # TODO read patch number from environment cuz they are obs shape
                                         weight_initialization_mode='kaiming_uniform')
        self.add_module('sdm', self.sdm)

    def step(self,
             obs: torch.Tensor,
             latent: Optional[torch.Tensor] = None,
             deterministic: bool = False,
             ) -> tuple[torch.Tensor, ...]:
        """
        Get necessary network outputs (log_prob, reward_pred, cost_value_pred) to calculate policy loss
        Args:
            obs:
            latent:
            deterministic:

        Returns:

        """
        with torch.no_grad():
            # Step shared layers of actor and reward estimator
            gru_output, latent_output = self.gru(obs, latent)
            gru_output = gru_output.squeeze(1)  # batch x latent_dim

            # Step actor
            action = self.actor.predict(gru_output, deterministic=deterministic)
            log_prob = self.actor.log_prob(action)

            # Step reward (value) estimator
            reward = self.reward_critic(gru_output)[0]  # assume single estimator

            # Step cost critic
            value_cost = self.cost_critic(obs)[0]  # assume single cost critic

            # Step sdm
            # TODO if sdm helps with any of the above network outputs, call sdm's forward function here

        return action, log_prob, reward, value_cost, latent_output

    def forward_gru(
        self,
        obs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        if isinstance(obs, List):
            # List of tensors (each tensor of shape (sequence, feature))
            gru_outputs = []
            for seq in obs:
                seq = seq.unsqueeze(0)  # Add batch dimension (1, sequence, feature)

                gru_output, _ = self.gru(seq)
                # gru_output = gru_output.view(-1, gru_output.size(-1))  # Flatten the output
                gru_output = gru_output.reshape(-1, gru_output.size(-1))  # Flatten the output
                gru_outputs.append(gru_output)

            gru_output = torch.cat(gru_outputs, dim=0)  # Concatenate along the sequence dimension

        elif isinstance(obs, torch.Tensor):
            # Original tensor input
            assert obs.dim() == 3, f'Observation shape should be (Batch, Sequence, Feature), current: {obs.shape}'

            gru_output, _ = self.gru(obs)
            # print(f'{gru_output.shape=}')
            # gru_output = gru_output.view(-1, gru_output.size(-1))  # Flatten the output
            gru_output = gru_output.reshape(-1, gru_output.size(-1))  # Flatten the output
        else:
            raise TypeError(f"Expected input type torch.Tensor or list, got {type(obs)}")

        return gru_output

    def forward_actor(
        self,
        obs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Distribution:
        # Pass obs to shared GRU layers, but no grad for separate pass
        with torch.no_grad():
            gru_output = self.forward_gru(obs)

        # Pass the concatenated GRU outputs to the actor MLP
        distribution: Distribution = self.actor(gru_output)

        return distribution

    def predict_actor(
        self,
        obs: Union[torch.Tensor, List[torch.Tensor]],
        deterministic: bool = False,
    ) -> torch.Tensor:
        gru_output, self.gru_latent = self.gru(obs, self.gru_latent)
        action = self.actor.predict(gru_output, deterministic=deterministic)
        return action

    def forward_reward(
        self,
        obs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        # Pass obs to shared GRU layers, but no grad for separate pass
        with torch.no_grad():
            gru_output = self.forward_gru(obs)

        # Pass the concatenated GRU outputs to the reward estimator MLP
        reward_pred = self.reward_critic(gru_output)

        return reward_pred

    def forward_actor_reward(
        self,
        obs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Tuple[Distribution, List[torch.Tensor]]:
        # Pass obs to shared GRU layers, grad is required for combined pass
        gru_output = self.forward_gru(obs)

        # Pass the concatenated GRU outputs to the actor MLP
        distribution: Distribution = self.actor(gru_output)

        # Pass the concatenated GRU outputs to the reward estimator MLP
        reward_pred = self.reward_critic(gru_output)

        return distribution, reward_pred

    def forward_sdm(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, act], dim=-1)
        # obs_act.requires_grad_(True)
        delta = self.sdm(obs_act)
        return delta

    def forward(self,
                obs: torch.Tensor,
                latent: torch.Tensor,
                deterministic: bool = False,
                ) -> tuple[torch.Tensor, ...]:
        return self.step(obs, latent, deterministic)

