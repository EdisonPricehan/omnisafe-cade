import numpy as np
from io import StringIO
from contextlib import closing

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR, LinearLR
from torch.distributions import Distribution

from omnisafe.typing import Activation, InitFunction, OmnisafeSpace, ActorType
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.model import get_obs_dim, get_act_dim
from omnisafe.models.actor.actor_builder import ActorBuilder
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.models.base import Actor, Critic
from omnisafe.models.dynamics.sdm import SemanticDynamicsModel

from typing import Optional, Union, List, Tuple

from gymnasium.spaces import Discrete, MultiDiscrete


class ConstraintActorDynamicsEstimator(nn.Module):
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
                 ):
        super().__init__()

        # Define common constants
        self.obs_space = obs_space
        self.act_space = act_space
        self.obs_dim = get_obs_dim(obs_space)
        self.act_dim = get_act_dim(act_space, execution_dim=True)
        self.model_cfgs: ModelConfig = model_cfgs

        # Define shared GRU layer for actor and reward critic
        self.gru = nn.GRU(
            # input_size=self.obs_dim,
            input_size=self.obs_dim + self.act_dim,  # [obs_{t}, act_{t-1}]
            hidden_size=model_cfgs.latent_size,
            num_layers=model_cfgs.num_gru_layers,
            batch_first=True,
        )
        self.gru_optimizer: optim.Optimizer = optim.Adam(self.gru.parameters(), lr=model_cfgs.gru_lr)
        self.gru_latent: Optional[torch.Tensor] = None
        self.add_module('gru', self.gru)

        # Define actor head
        self._disable_no_op: bool = model_cfgs.disable_no_op
        self.actor: Actor = ActorBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.actor.hidden_sizes,
            latent_size=model_cfgs.latent_size,
            disable_no_op=self._disable_no_op,
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
        patch_dim = int(np.sqrt(self.obs_dim))  # assuming square first-person-view (patchified) observation
        self.sdm = SemanticDynamicsModel(
            obs_space=obs_space,
            act_space=act_space,
            model_cfgs=model_cfgs,
            patch_rows=patch_dim,
            patch_cols=patch_dim,
            weight_initialization_mode='kaiming_uniform',
            perturb_delta=False,
        )
        self.add_module('sdm', self.sdm)

    def sample_trajectory_cost(
        self,
        initial_obs: torch.Tensor,
        initial_latent: torch.Tensor,
        trajectory_length: int,
        lam_c: float,
        initial_action: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, float):
        """
        Samples a single trajectory and calculates its cumulative discounted cost.
        Starts with initial_action if provided; otherwise samples action from policy.

        Args:
            initial_obs (torch.Tensor): Initial observation tensor.
            initial_latent (torch.Tensor): Latent tensor for the initial observation.
            trajectory_length (int): Length of the trajectory to sample.
            lam_c (float): Discount factor for future costs.
            initial_action (torch.Tensor, optional): Given action to start the trajectory; if None, samples from policy.

        Returns:
            torch.Tensor: The first action in the trajectory.
            float: The cumulative discounted cost for the trajectory.
        """
        assert trajectory_length > 0, f'Trajectory len should be positive, given {trajectory_length}.'

        obs = initial_obs.clone()
        latent_state = initial_latent.clone()
        cumulative_cost = 0.0
        discount = lam_c

        # Choose initial action
        if initial_action is not None:
            action = initial_action
        else:
            with torch.no_grad():
                action = self.actor.predict(latent_state, deterministic=False)

        first_action = action.clone()  # Store the first action

        # Sample the trajectory
        with torch.no_grad():
            for t in range(trajectory_length):
                # Predict next observation using SDM based on the current action
                obs_act = torch.cat([obs, action], dim=-1)
                predicted_obs = self.sdm.predict(obs_act, round_to_int=True)

                # Predict immediate cost of this predicted future observation
                immediate_cost = self.cost_critic(predicted_obs)[0].item()

                # Discounted cumulative cost
                cumulative_cost += discount * immediate_cost
                discount *= lam_c

                # Update observation
                obs = predicted_obs

                # print(f'Predicted next obs \n')
                # print(self._print_obs(obs))
                # print(f'{immediate_cost=} \n')

                # For subsequent steps, sample actions from the policy
                if t < trajectory_length - 1:
                    obs_last_act = torch.cat([obs, action], dim=-1)
                    gru_output, latent_state = self.gru(obs_last_act, latent_state)
                    action = self.actor.predict(gru_output, deterministic=False)

        return first_action, cumulative_cost

    def sample_trajectory_reward_cost(
        self,
        initial_obs: torch.Tensor,
        initial_latent: torch.Tensor,
        trajectory_length: int,
        lam_r: float,
        lam_c: float,
        initial_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Samples a single trajectory and calculates its cumulative discounted reward and cost.
        Starts with initial_action if provided; otherwise samples action from policy.

        Args:
            initial_obs:
            initial_latent:
            trajectory_length:
            lam_r:
            lam_c:
            initial_action:

        Returns:

        """
        # Init params
        obs = initial_obs.clone()
        latent_state = initial_latent.clone()
        cumulative_reward = 0.0
        cumulative_cost = 0.0
        discount_r = lam_r
        discount_c = lam_c

        # Choose initial action
        if initial_action is not None:
            action = initial_action
        else:
            with torch.no_grad():
                action = self.actor.predict(latent_state, deterministic=False)

        first_action = action.clone()  # Store the first action

        # Sample the trajectory
        with torch.no_grad():
            for t in range(trajectory_length):
                # Predict next immediate reward
                immediate_reward = self.reward_critic(latent_state, action)[0]  # Predicted

                # Discounted cumulative reward
                cumulative_reward += discount_r * immediate_reward
                discount_r *= lam_r

                # Predict next observation using SDM based on the current action
                obs_act = torch.cat([obs, action], dim=-1)
                predicted_obs = self.sdm.predict(obs_act, round_to_int=True)
                immediate_cost = self.cost_critic(predicted_obs)[0].item()

                # Discounted cumulative cost
                cumulative_cost += discount_c * immediate_cost
                discount_c *= lam_c

                # Update observation
                obs = predicted_obs

                # print(f'Predicted next obs \n')
                # print(self._print_obs(obs))
                # print(f'{immediate_cost=} \n')

                # For subsequent steps, sample actions from the policy
                if t < trajectory_length - 1:
                    obs_last_act = torch.cat([obs, action], dim=-1)
                    gru_output, latent_state = self.gru(obs_last_act, latent_state)
                    action = self.actor.predict(gru_output, deterministic=False)

        return first_action, cumulative_reward, cumulative_cost

    def select_safe_action_by_imagined_costs(
        self,
        start_obs: torch.Tensor,
        start_latent: torch.Tensor,
        start_action: torch.Tensor,
        num_samples: int,
        trajectory_length: int,
        horizon_cost_threshold: float,
        lam_c: float,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Conditionally applies safe action selection by sampling multiple trajectories, starting with
        the given action if available, and choosing the best action if any trajectory meets the threshold.
        A second round of sampling (without start_action) is performed if none meet the threshold.

        Args:
            start_obs (torch.Tensor): Initial observation tensor.
            start_latent (torch.Tensor): Latent tensor for initial observation.
            start_action (torch.Tensor): Policy-chosen action on the start_obs.
            num_samples (int): Number of action trajectories to sample.
            trajectory_length (int): Length of each sampled trajectory.
            horizon_cost_threshold (float): Maximum allowed cumulative costs over the prediction horizon.
            lam_c (float): Discount factor for future costs.

        Returns:
            torch.Tensor: The first action in the trajectory with the lowest cumulative cost within threshold,
                          or the best action found if none meet the threshold.
            bool: whether the policy action is overlaid by the safety layer
        """
        safe_action = start_action.clone()
        lowest_cost = float('inf')
        action_overlay = True  # Flag to check if any trajectory is within the cost threshold

        # First round: try trajectories starting with start_action
        for _ in range(num_samples):
            action, cumulative_cost = self.sample_trajectory_cost(
                start_obs, start_latent, trajectory_length, lam_c, start_action
            )

            if cumulative_cost <= horizon_cost_threshold:
                action_overlay = False
                break

        # Second round: if no trajectories met the threshold, sample trajectories without start_action
        if action_overlay:
            for _ in range(num_samples):
                action, cumulative_cost = self.sample_trajectory_cost(
                    start_obs, start_latent, trajectory_length, lam_c
                )
                if cumulative_cost < lowest_cost:
                    safe_action, lowest_cost = action, cumulative_cost
        else:
            pass
            # print(f'Action no overlay, act: {safe_action.item()}.')

        # Force negate action overlay if the safe action is no op
        if (isinstance(self.act_space, Discrete) and safe_action.item() == 0 or
            isinstance(self.act_space, MultiDiscrete) and torch.all(safe_action == 1)):
            safe_action = start_action
            action_overlay = False

        # Print overlay
        if action_overlay:
            print(self._print_obs(start_obs))
            print(f'Action overlaid, orig act: {start_action}, safe act: {safe_action}, horizon lowest cost: {lowest_cost}.')
            print('-' * 60)

        return safe_action, action_overlay

    def select_safe_action_by_imagined_rewards_and_costs(
        self,
        start_obs: torch.Tensor,
        start_latent: torch.Tensor,
        start_action: torch.Tensor,
        num_samples: int,
        trajectory_length: int,
        horizon_cost_threshold: float,
        lam_r: float,
        lam_c: float,
        lagrangian_multiplier: float,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Conditionally applies safe action selection by sampling multiple trajectories, starting with
        the given action if available, and choosing the best action if any trajectory meets the threshold.
        A second round of sampling (without start_action) is performed if none meet the threshold.

        Args:
            start_obs (torch.Tensor): Initial observation tensor.
            start_latent (torch.Tensor): Latent tensor for initial observation.
            start_action (torch.Tensor): Policy-chosen action on the start_obs.
            num_samples (int): Number of action trajectories to sample.
            trajectory_length (int): Length of each sampled trajectory.
            horizon_cost_threshold (float): Maximum allowed cumulative costs over the prediction horizon.
            lam_r (float): Discount factor for future rewards.
            lam_c (float): Discount factor for future costs.
            lagrangian_multiplier (float): Current Lagrangian multiplier

        Returns:
            torch.Tensor: The first action in the trajectory with the lowest cumulative cost within threshold,
                          or the best action found if none meet the threshold.
            bool: whether the policy action is overlaid by the safety layer
        """
        safe_action = start_action.clone()
        action_overlay = True  # Flag to check if any trajectory is within the cost threshold

        # First round: try trajectories starting with start_action
        for _ in range(num_samples):
            action, cumulative_reward, cumulative_cost = self.sample_trajectory_reward_cost(
                start_obs,
                start_latent,
                trajectory_length,
                lam_r,
                lam_c,
                start_action,
            )

            # For the given action, only consider safety constraint
            if cumulative_cost <= horizon_cost_threshold:
                action_overlay = False
                break

        # Second round: if no trajectories met the threshold, sample trajectories without start_action
        best_tradeoff_score = -float('inf')
        best_tradeoff_action = start_action.clone()
        best_cost = float('inf')
        if action_overlay:
            for _ in range(num_samples):
                action, cumulative_reward, cumulative_cost = self.sample_trajectory_reward_cost(
                    start_obs,
                    start_latent,
                    trajectory_length,
                    lam_r,
                    lam_c,
                    initial_action=None,
                )

                # Consider both predicted horizon rewards and costs
                # tradeoff_score = cumulative_reward - lagrangian_multiplier * cumulative_cost  # Lagrangian-regulated
                tradeoff_score = cumulative_reward - cumulative_cost  # Direct regulation
                if tradeoff_score > best_tradeoff_score:
                    best_tradeoff_action, best_tradeoff_score = action, tradeoff_score

                # Keep track of best cost for safe fallback
                if cumulative_cost < best_cost:
                    safe_action, best_cost = action, cumulative_cost

            # Use best tradeoff action if the tradeoff score is above threshold, otherwise use safest action
            # The largest immediate cost is about 1 if the cost estimator converges well, and the largest immediate
            # reward is way less than 1, which makes the tradeoff score negative if all sampled initial actions are not
            # good enough. If so choose the action with the lowest cost.
            if best_tradeoff_score > 0:
                safe_action = best_tradeoff_action
            # safe_action = best_tradeoff_action

            # Force negate action overlay if the safe action is no op
            if (isinstance(self.act_space, Discrete) and safe_action.item() == 0 or
                isinstance(self.act_space, MultiDiscrete) and torch.all(safe_action == 1)):
                safe_action = start_action
                action_overlay = False

        if action_overlay:
            print(self._print_obs(start_obs))
            print(
                f'Action overlaid, orig act: {start_action}, safe act: {safe_action}, tradeoff act: {best_tradeoff_action}, '
                f'horizon best cost: {best_cost}, horizon tradeoff score: {best_tradeoff_score}, '
                f'Lagrangian multiplier: {lagrangian_multiplier}.')
            print('-' * 60)

        return safe_action, action_overlay

    def _print_obs(
        self,
        obs: torch.Tensor,
    ) -> str:
        outfile = StringIO()

        obs_np = obs.squeeze().numpy()
        obs_side_len = int(np.sqrt(obs_np.shape[0]))

        for row in range(obs_side_len):
            for col in range(obs_side_len):
                idx = row * obs_side_len + col
                state = obs_np[idx]
                if state == 1:
                    output = ' C '
                else:
                    output = ' o '

                if idx == obs_np.shape[0] // 2:
                    output = ' x '

                if col == 0:
                    output = output.lstrip()
                if col == obs_side_len - 1:
                    output = output.rstrip()
                    output += '\n'

                outfile.write(output)
            # outfile.write('\n')

        with closing(outfile):
            return outfile.getvalue()

    def step(
        self,
        obs: torch.Tensor,
        last_act: torch.Tensor,
        lagrangian_multiplier: float,
        latent: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        enable_safety_layer: bool = False,
        safety_layer_use_reward: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Get necessary network outputs (log_prob, reward_pred, cost_value_pred) to calculate policy loss

        Args:
            obs:
            last_act:
            lagrangian_multiplier:
            latent:
            deterministic:
            enable_safety_layer:
            safety_layer_use_reward:

        Returns:

        """
        with torch.no_grad():
            # Step shared layers of actor and reward estimator
            obs_last_act = torch.cat([obs, last_act], dim=-1)
            gru_output, latent_output = self.gru(obs_last_act, latent)
            # gru_output, latent_output = self.gru(obs, latent)  # Only obs input

            # print(f'{obs.shape=} {gru_output.shape=} {latent_output.shape=}')  # [Sequence x Feature]
            # print(f'{gru_output=}')
            # print(f'{latent_output=}')

            # Step actor
            action = self.actor.predict(gru_output, deterministic=deterministic)
            # print(f'Step action: {action}')

            # Alternate action by action space (disable no_op)
            if self._disable_no_op:
                if isinstance(self.act_space, Discrete):
                    action -= 1
                elif isinstance(self.act_space, MultiDiscrete):
                    pass
                    # TODO
                else:
                    raise NotImplementedError

            log_prob = self.actor.log_prob(action)

            # Step reward (value) estimator
            # reward = self.reward_critic(gru_output)[0]  # assume single estimator
            reward = self.reward_critic(gru_output, action)[0]  # latent+action as input

            # Step SDM
            obs_cur_act = torch.cat([obs, action], dim=-1)
            next_obs_pred = self.sdm.predict(obs_cur_act, round_to_int=True)

            # Step cost estimator
            # value_cost = self.cost_critic(obs)[0]  # assume single cost estimator
            value_cost = self.cost_critic(next_obs_pred)[0]  # predicted (Markovian) cost

            # Overlay action
            action_overlaid = False
            if enable_safety_layer:
                if safety_layer_use_reward:
                    # Action overlay by both reward and cost criteria
                    action, action_overlaid = self.select_safe_action_by_imagined_rewards_and_costs(
                        start_obs=obs,
                        start_latent=latent_output,
                        start_action=action,
                        num_samples=self.model_cfgs.dynamics.traj_sampling_num,
                        trajectory_length=self.model_cfgs.dynamics.horizon,
                        horizon_cost_threshold=self.model_cfgs.dynamics.horizon_cost_threshold,
                        lam_r=self.model_cfgs.dynamics.discount_factor,  # Use the same discount factor for both reward and cost
                        lam_c=self.model_cfgs.dynamics.discount_factor,
                        lagrangian_multiplier=lagrangian_multiplier,
                    )
                else:
                    # Action overlay by only cost criteria
                    action, action_overlaid = self.select_safe_action_by_imagined_costs(
                        start_obs=obs,
                        start_latent=latent_output,
                        start_action=action,
                        num_samples=self.model_cfgs.dynamics.traj_sampling_num,
                        trajectory_length=self.model_cfgs.dynamics.horizon,
                        horizon_cost_threshold=self.model_cfgs.dynamics.horizon_cost_threshold,
                        lam_c=self.model_cfgs.dynamics.discount_factor,
                    )

                # reset the action distribution on the first observation to get the corresponding log_prob of
                # the really executed action
                if action_overlaid:
                    # Update log_prob
                    self.actor.predict(gru_output, deterministic=deterministic)
                    if self._disable_no_op:
                        log_prob = self.actor.log_prob(action - 1)  # TODO only for CliffCircular env
                    else:
                        log_prob = self.actor.log_prob(action)

                    # Update predicted cost (TODO not used yet)
                    obs_cur_act = torch.cat([obs, action], dim=-1)
                    next_obs_pred = self.sdm.predict(obs_cur_act, round_to_int=True)
                    value_cost = self.cost_critic(next_obs_pred)[0]

        return action, log_prob, torch.Tensor([action_overlaid]), reward, value_cost, latent_output

    def _shift_action(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        assert action.dim() == 2, f'Expect action dim to be 2, got {action.dim()}.'

        a_shifted = torch.zeros_like(action)  # (sequence, feature)
        a_shifted[1:] = action[:-1]
        a_shifted = a_shifted.unsqueeze(0)  # (1, sequence, feature)

        return a_shifted

    def forward_gru(
        self,
        obs: Union[torch.Tensor, List[torch.Tensor]],
        act: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        if isinstance(obs, List):
            assert isinstance(act, List)
            assert len(obs) == len(act), f'Obs and act lists should have the same length.'

            # List of tensors (each tensor of shape (sequence, feature))
            gru_outputs = []
            for i, o in enumerate(obs):
                o = o.unsqueeze(0)  # Add batch dimension (1, sequence, feature)

                # shift action backward by 1 step
                a = act[i]  # (sequence, feature)
                a_shifted = self._shift_action(a)

                obs_last_act = torch.cat([o, a_shifted], dim=-1)

                gru_output, _ = self.gru(obs_last_act)

                # gru_output, _ = self.gru(seq)
                # gru_output = gru_output.view(-1, gru_output.size(-1))  # Flatten the output

                gru_output = gru_output.reshape(-1, gru_output.size(-1))  # Flatten the output
                gru_outputs.append(gru_output)

            gru_output = torch.cat(gru_outputs, dim=0)  # Concatenate along the sequence dimension
            # TODO act is not considered

        elif isinstance(obs, torch.Tensor):
            assert isinstance(act, torch.Tensor)

            # Original tensor input
            if obs.dim() == 2:
                obs = obs.unsqueeze(0)

            if act.dim() == 2:
                act = self._shift_action(act)
            elif act.dim() == 3:  # (batch, sequence, feature), batch=1 by default
                act = self._shift_action(act.squeeze(0))

            assert obs.dim() == 3, f'Observation shape should be (Batch, Sequence, Feature), current: {obs.shape}'
            assert act.dim() == 3, f'Action shape should be (Batch, Sequence, Feature), current: {act.shape}'

            obs_last_act = torch.cat([obs, act], dim=-1)
            gru_output, latent = self.gru(obs_last_act)

            # print(f'{obs_last_act.shape=} {gru_output.shape=} {latent.shape=}')
            # print(f'{gru_output[0, -1]=}')
            # print(f'{latent[0, -1]=}')

            # gru_output, _ = self.gru(obs)
            # print(f'{gru_output.shape=}')
            # gru_output = gru_output.view(-1, gru_output.size(-1))  # Flatten the output
            gru_output = gru_output.reshape(-1, gru_output.size(-1))  # Flatten the output
        else:
            raise TypeError(f"Expected input type torch.Tensor or list, got {type(obs)}")

        return gru_output

    def forward_actor(
        self,
        obs: Union[torch.Tensor, List[torch.Tensor]],
        act: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Union[Distribution, List[Distribution]]:
        # Pass obs to shared GRU layers, but no grad for separate pass
        with torch.no_grad():
            gru_output = self.forward_gru(obs, act)

        # Pass the concatenated GRU outputs to the actor MLP
        distribution: Union[Distribution, List[Distribution]] = self.actor(gru_output)

        return distribution

    def predict_actor(
        self,
        obs: Union[torch.Tensor, List[torch.Tensor]],
        last_act: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        obs_last_act = torch.cat([obs, last_act], dim=-1)
        gru_output, self.gru_latent = self.gru(obs_last_act, self.gru_latent)

        # gru_output, self.gru_latent = self.gru(obs, self.gru_latent)

        action = self.actor.predict(gru_output, deterministic=deterministic)
        return action

    def forward_reward(
        self,
        obs: Union[torch.Tensor, List[torch.Tensor]],
        act: Union[torch.Tensor, List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        # Pass obs to shared GRU layers, but no grad for separate pass
        with torch.no_grad():
            gru_output = self.forward_gru(obs, act)

        # print(f'{obs.shape=} {gru_output.shape=} {act.shape=}')

        # Pass the concatenated GRU outputs to the reward estimator MLP
        # act = act.squeeze(0)  # Seq x Feature
        reward_pred = self.reward_critic(gru_output, act)

        return reward_pred

    def forward_actor_reward(
        self,
        obs: Union[torch.Tensor, List[torch.Tensor]],
        act: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Tuple[Union[Distribution, List[Distribution]], List[torch.Tensor]]:
        # Pass obs to shared GRU layers, grad is required for combined pass
        gru_output = self.forward_gru(obs, act)

        reward_features = gru_output.detach()

        # Pass the concatenated GRU outputs to the actor MLP
        distribution: Union[Distribution, List[Distribution]] = self.actor(gru_output)

        # Pass the concatenated GRU outputs to the reward estimator MLP
        # reward_pred = self.reward_critic(gru_output, act)
        reward_pred = self.reward_critic(reward_features, act)  # Reward loss does not affect gru training

        return distribution, reward_pred

    def forward_sdm(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, act], dim=-1)
        delta = self.sdm(obs_act)
        return delta

    def forward(self,
                obs: torch.Tensor,
                latent: torch.Tensor,
                deterministic: bool = False,
                ) -> tuple[torch.Tensor, ...]:
        return self.step(obs, latent, deterministic)
