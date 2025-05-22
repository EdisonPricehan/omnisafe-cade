import torch
from typing import Tuple
from gymnasium.spaces import MultiBinary, MultiDiscrete

from omnisafe.models.actor_critic.constraint_actor_dynamics_estimator import ConstraintActorDynamicsEstimator as CADE
from omnisafe.utils.config import Config


class CadeWrapper(CADE):
    """
    Inherits directly from ConstraintActorDynamicsEstimator, for model conversion and deployment.
    Initialize it with the same arguments as the base class, then load the pretrained weights.
    Holds its GRU latent state internally and exposes:
      forward(obs, reset_flag) -> action
      reset() -> None
    """
    def __init__(self,
                 obs_space,
                 act_space,
                 model_cfgs,
                 pretrained_weights_path: str,
                 device: torch.device = torch.device('cpu'),
                 ):
        # 1. Initialize the base policy with its normal args
        super().__init__(
            obs_space=obs_space,
            act_space=act_space,
            model_cfgs=model_cfgs,
            epochs=0,  # unused at inference
        )

        # 2. Load the pretrained .pth exactly once
        state_dict = torch.load(pretrained_weights_path, map_location=device)
        self.load_state_dict(state_dict['actor_critic'])

        # 3. Move to device and eval mode
        self.to(device).eval()

        # 4. Initialize internal GRU latent state
        latent_size = model_cfgs.latent_size
        self.latent = torch.zeros((1, latent_size), dtype=torch.float32, device=device)

        # 5. Initialize last action
        self.last_act = torch.tensor([[1] * self.act_space.nvec.shape[0]])  # Nominal action (no_op)

    def forward(
        self,
        obs: torch.Tensor,
        reset_flag: torch.Tensor,
    ) -> torch.Tensor:
        # Reset flag
        # rf = reset_flag.reshape(-1, 1).to(dtype=torch.bool)

        # Update recurrent values based on reset flag
        zero_latent = torch.zeros_like(self.latent)
        nom_last_act = torch.ones_like(self.last_act)
        self.latent = torch.where(reset_flag, zero_latent, self.latent)
        self.last_act = torch.where(reset_flag, nom_last_act, self.last_act)

        # Call the original forward method
        # TODO: assume the policy action is actually executed
        # print(f'{obs.shape=} {self.last_act.shape=} {self.latent.shape=}')
        outs = super().forward(obs=obs, last_act=self.last_act, latent=self.latent, deterministic=True)
        action, *_, next_latent = outs

        # Update internal latent and last action
        self.latent = next_latent
        self.last_act = action

        return action


class CadeWrapperV2(CADE):
    """
    Inherits directly from ConstraintActorDynamicsEstimator, for model conversion and deployment.
    Initialize it with the same arguments as the base class, then load the pretrained weights.
    Exposes its GRU latent state and last action externally, using deterministic policy.
    This is for exporting to onnx or tensorrt models that are stateless but enjoy higher inference throughput.
    With 2 functions:
      forward(obs, last_act, latent) -> action, cur_latent
      reset() -> None
    """
    def __init__(self,
                 obs_space,
                 act_space,
                 model_cfgs,
                 pretrained_weights_path: str,
                 device: torch.device = torch.device('cpu'),
                 ):
        # 1. Initialize the base policy with its normal args
        super().__init__(
            obs_space=obs_space,
            act_space=act_space,
            model_cfgs=model_cfgs,
            epochs=0,  # unused at inference
        )

        # 2. Load the pretrained .pth exactly once
        state_dict = torch.load(pretrained_weights_path, map_location=device)
        self.load_state_dict(state_dict['actor_critic'])

        # 3. Move to device and eval mode
        self.to(device).eval()

        # 4. Initialize internal GRU latent state
        latent_size = model_cfgs.latent_size
        self.latent = torch.zeros((1, latent_size), dtype=torch.float32, device=device)

        # 5. Initialize last action
        self.last_act = torch.tensor([[1] * self.act_space.nvec.shape[0]])  # Nominal action (no_op)

    def forward(
        self,
        obs: torch.Tensor,
        last_act: torch.Tensor,
        latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Call the original forward method with deterministic policy
        # print(f'{obs.shape=} {self.last_act.shape=} {self.latent.shape=}')
        outs = super().forward(obs=obs, last_act=last_act, latent=latent, deterministic=True)
        action, *_, cur_latent = outs

        return action, cur_latent






