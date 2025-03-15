from typing import Optional, List, Tuple
import torch

from omnisafe.common.lagrange import Lagrange
from omnisafe.models.dynamics.sdm import SemanticDynamicsModel as SDM
from omnisafe.common.buffer.onpolicy_cade_buffer import OnPolicyCADEBuffer
from omnisafe.models.actor_critic.constraint_actor_dynamics_estimator import ConstraintActorDynamicsEstimator as CADE


class MPCLagrange(Lagrange):
    def __init__(
        self,
        cade: CADE,
        horizon: int,
        rollout_num: int,
        buffer: OnPolicyCADEBuffer,
        alpha_init: float,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound: Optional[float] = None,
    ) -> None:
        super().__init__(
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer,
            lagrangian_upper_bound=lagrangian_upper_bound,
        )

        self.cade = cade

        # Extract the dynamics model from CADE
        self.dynamics: SDM = self.cade.sdm
        self.horizon: int = horizon
        self.rollout_num: int = rollout_num

        # Define the on-policy data buffer for initial state sampling
        self.buffer: OnPolicyCADEBuffer = buffer

        self.alpha_raw = torch.nn.Parameter(torch.tensor(alpha_init, requires_grad=True))
        self.lambda_optimizer = torch.optim.Adam(
            [self.lagrangian_multiplier, self.alpha_raw],  # Include alpha in the optimizer
            lr=lambda_lr,
        )

    def get_alpha(self) -> torch.Tensor:
        """Apply sigmoid to raw_alpha to ensure it's in range [0, 1]."""
        return torch.sigmoid(self.alpha_raw)

    def sample_start_states_from_buffer(self, num_states: int) -> torch.Tensor:
        """
        Uniformly samples start states from the buffer.

        Args:
            num_states (int): Number of states to sample from the buffer.

        Returns:
            torch.Tensor: Sampled start states.
        """
        # Sample uniformly from recent episodes in the buffer
        data = self.buffer.get()
        obs = data['obs']
        obs_size = obs.size(0)

        # Generate random indices without replacement
        rand_indices = torch.randperm(obs_size)[:num_states]

        # Uniformly sample the specified number of states
        sampled_states = obs[rand_indices]

        return sampled_states

    def sample_start_states_from_episode(self, obs: torch.Tensor, num_states: int) -> torch.Tensor:
        """
        Uniformly samples start states from the current episode.
        Args:
            obs: shape (ep_len, feature_len)
            num_states:

        Returns:

        """
        obs_size = obs.size(0)

        if num_states > obs_size:
            print(f'Start state sample num {num_states} is greater than episode len {obs_size}.')
            num_states = obs_size

        # Generate random indices without replacement
        rand_indices = torch.randperm(obs_size)[:num_states]

        # Uniformly sample the specified number of states
        sampled_states = obs[rand_indices]

        return sampled_states

    def mpc_rollout(self, start_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rollout trajectories using MPC starting from the given start states.

        Args:
            start_states (torch.Tensor): The starting states for MPC rollouts.

        Returns
            Tuple: MPC-rolled trajectories and cost values
        """
        trajectories = []
        cost_values = []
        for start_state in start_states:
            traj = [start_state]
            c_vals = []
            gru_latent = None

            for _ in range(self.horizon):
                cur_state = traj[-1].unsqueeze(0)  # shape (1 x feature_dim)

                # Choose action using current policy
                action, _, _, _, value_cost, gru_latent = self.cade.step(cur_state, gru_latent, deterministic=False)

                # Append cost value of current state
                c_vals.append(value_cost)

                # Imagine the next state using SDM
                next_state = self.dynamics.predict(torch.cat((cur_state, action), dim=-1), round_to_int=True)

                # Append next predicted state
                traj.append(next_state.squeeze(0))

            # Update trajectories and cost values
            trajectories.append(traj)
            cost_values.append(c_vals)

        # Convert to tensor, shape (Rollout x Horizon x obs_len)
        trajectories = torch.stack([torch.stack(tl) for tl in trajectories])
        cost_values = torch.stack([torch.stack(cl) for cl in cost_values])

        return trajectories, cost_values

    def mean_horizon_cost(
        self,
        cost_values: torch.Tensor,
        discount_factor: float,
        is_immediate_cost: bool = False
    ) -> float:
        """Calculate the mean horizon cost using Bellman backup for the cost return values, or averaging horizon
        immediate costs.

        Args:
            cost_values (torch.Tensor): The cost values from MPC rollouts, shape (Rollout x Horizon).
            discount_factor (float): Discount factor for future costs (gamma).
            is_immediate_cost (bool): Whether the cost_values are immediate costs or cost returns

        Returns:
            float: The mean horizon cost over the trajectories.
        """
        total_cost = 0.0
        num_trajectories = cost_values.shape[0]  # Number of rollouts
        horizon = cost_values.shape[1]  # Number of steps in the horizon

        for i in range(num_trajectories):
            traj_cost_values = cost_values[i]  # Cost values for this trajectory
            rollout_cost = 0.0

            if is_immediate_cost:
                # Horizon cost is just the sum of all immediate costs along current trajectory
                total_cost += torch.sum(traj_cost_values).item()
            else:
                # Step 1: Compute immediate costs using Bellman backup for all steps except the last one
                for t in range(horizon - 1):
                    cur_value = traj_cost_values[t]
                    next_value = traj_cost_values[t + 1]

                    # Immediate cost calculation using Bellman backup
                    immediate_cost = cur_value - discount_factor * next_value
                    rollout_cost += immediate_cost

                # Step 2: Add the terminal cost (cost-to-go)
                rollout_cost += traj_cost_values[-1]

                # Step 3: Add the total rollout cost to the total cost
                total_cost += rollout_cost

        # Step 4: Compute the mean cost over all trajectories
        mean_horizon_cost = total_cost / num_trajectories

        return mean_horizon_cost

    def compute_lambda_loss(self, mean_ep_cost: float, ep_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Lagrange multiplier loss by combining mean episodic cost
        and MPC-predicted mean horizon cost.

        Args:
            mean_ep_cost (float): Mean episode cost from on-policy buffer.
            ep_states: Episode states tensor for MPC start state sampling

        Returns:
            torch.Tensor: Penalty loss for the Lagrange multiplier.
        """
        # Step 1: Sample start states from the buffer for MPC rollout
        if ep_states is None:
            start_states = self.sample_start_states_from_buffer(num_states=self.rollout_num)
        else:
            start_states = self.sample_start_states_from_episode(ep_states, num_states=self.rollout_num)

        # Step 2: Perform MPC rollouts from the sampled start states
        trajectories, cost_values = self.mpc_rollout(start_states)

        # Step 3: Calculate the mean horizon cost using the cost critic
        mean_horizon_cost = self.mean_horizon_cost(cost_values, discount_factor=0.99, is_immediate_cost=True)

        # Step 4: Combine mean episodic cost and mean horizon cost with sigmoid-constrained alpha
        # TODO trajectory length better be same for fair comparison?
        alpha = self.get_alpha()  # Get alpha in [0, 1] range
        # combined_cost = alpha * mean_ep_cost + (1 - alpha) * mean_horizon_cost

        # Use actual episodic cost only
        # combined_cost = mean_ep_cost

        # Use imagined horizon cost only
        combined_cost = mean_horizon_cost

        # Step 5: Compute the Lagrange loss based on the combined cost
        lagrange_loss = -self.lagrangian_multiplier * (combined_cost - self.cost_limit)

        return lagrange_loss

    def update_lagrange_multiplier(self, Jc: float, ep_states: Optional[torch.Tensor] = None) -> None:
        r"""Update Lagrange multiplier (lambda).

        We update the Lagrange multiplier by minimizing the penalty loss, which is defined as:

        .. math::

            \lambda ^{'} = \lambda + \eta \cdot (J_C - J_C^*)

        where :math:`\lambda` is the Lagrange multiplier, :math:`\eta` is the learning rate,
        :math:`J_C` is the mean episode cost, and :math:`J_C^*` is the cost limit.

        Args:
            Jc (float): mean episode cost.
            ep_states: Episode states tensor for MPC start state sampling
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc, ep_states)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(
            0.0,
            self.lagrangian_upper_bound,
        )  # enforce: lambda in [0, inf]

