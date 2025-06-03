import os
import json
import torch
from typing import Tuple
import onnx
from gymnasium.spaces import MultiBinary, MultiDiscrete

from omnisafe.models.actor_critic.constraint_actor_dynamics_estimator import ConstraintActorDynamicsEstimator as CADE
from omnisafe.models.actor_critic.cade_wrapper import CadeWrapperV2
from omnisafe.utils.config import Config


def dummy_input() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dummy inputs for CadeWrapperV2.

    Returns:

    """
    dummy_obs = torch.zeros((1, *obs_space.shape), dtype=torch.float32)
    dummy_act = torch.tensor([[1] * act_space.nvec.shape[0]])
    dummy_latent = torch.zeros((1, cfgs.model_cfgs.latent_size), dtype=torch.float32)
    return dummy_obs, dummy_act, dummy_latent


def load_cfgs(cfg_path: str) -> Config:
    """
    Load the config from the save directory.

    Args:

    Raises:
        FileNotFoundError: If the config file is not found.
    """
    try:
        with open(cfg_path, encoding='utf-8') as file:
            kwargs = json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f'The config file {cfg_path} is not found.') from error
    return Config.dict2config(kwargs)


def load_model(model_path: str) -> CADE:
    """
    Load CADE model.

    Returns:

    """
    assert os.path.exists(model_path), f'Model path {model_path} does not exist!'

    model_params = torch.load(model_path, map_location='cpu')

    cade: CADE = CADE(
        obs_space=obs_space,
        act_space=act_space,
        model_cfgs=cfgs.model_cfgs,
        epochs=1,  # Not used, for linear lr decay
    )

    # for name, module in cade.named_modules():
    #     print(f'{name=} {module=}')
    #     print('-'*40)

    cade.load_state_dict(model_params['actor_critic'])

    return cade


def load_model_wrapper(model_path: str) -> CadeWrapperV2:
    assert os.path.exists(model_path), f'Model path {model_path} does not exist!'

    cade_wrapper = CadeWrapperV2(
        obs_space=obs_space,
        act_space=act_space,
        model_cfgs=cfgs.model_cfgs,
        pretrained_weights_path=model_path,
        device=torch.device('cpu'),
    )

    return cade_wrapper


def pth2onnx(pth_path: str, onnx_name: str = 'model.onnx') -> None:
    assert os.path.exists(pth_path), f'{pth_path} does not exist!'
    assert onnx_name != '' and os.path.splitext(os.path.basename(onnx_name))[-1] == '.onnx'

    # Load pth model
    # model = load_model(model_path=pth_path)
    # model.eval()

    # Load wrapped pth model
    model = load_model_wrapper(model_path=pth_path)
    model.eval()

    # Get dummy inputs
    dummy_obs, dummy_act, dummy_latent = dummy_input()
    # print(f'{type(dummy_obs)=} {dummy_obs}')
    # exit(0)

    # Export the model to ONNX format
    target_dir: str = os.path.dirname(pth_path)
    target_onnx_path: str = os.path.join(target_dir, onnx_name)
    torch.onnx.export(
        model,
        (dummy_obs, dummy_act, dummy_latent),
        target_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['obs', 'last_act', 'latent'],
        output_names=['action', 'cur_latent'],
    )


def valid_onnx(onnx_path: str):
    assert os.path.exists(onnx_path), f'{onnx_path} does not exist!'

    model = onnx.load(onnx_path)

    for node in model.graph.node:
        if node.op_type == "Reshape":
            print("Reshape node:", node.name)
            # Check if second input is in initializer list
            shape_name = node.input[1]
            is_init = any(init.name == shape_name for init in model.graph.initializer)
            print("  shape constant:", is_init)


if __name__ == '__main__':
    # Define constants
    obs_space = MultiBinary(16 * 16)
    act_space = MultiDiscrete([3, 3, 3, 3])
    latent_shape = (64,)
    cade_dir: str = './runs/FOCOPS_CACD-{medium}/seed-000-2025-03-01-15-06-43'
    model_name: str = 'epoch-350.pt'
    cade_pth_path: str = os.path.join(cade_dir, f'torch_save/{model_name}')
    cade_onnx_name: str = 'policy_v2.onnx'
    # cade_onnx_name: str = 'policy_dyn_ax.onnx'
    cade_cfg_path: str = os.path.join(cade_dir, 'config.json')

    # Load cfg
    cfgs = load_cfgs(cfg_path=cade_cfg_path)

    # Test torch model
    # cade_wrapper = load_model_wrapper(model_path=cade_pth_path)
    # cade_wrapper.eval()
    # for i in range(100):
    #     obs = torch.randint(0, 2, (1, 256), dtype=torch.float32)
    #     reset = torch.tensor([[0]], dtype=torch.bool)
    #     action = cade_wrapper(obs, reset)
    #     print(f'{action=}')
    # print(f'Random inference of pth policy is finished.')

    # Convert to onnx
    pth2onnx(pth_path=cade_pth_path, onnx_name=cade_onnx_name)
    print(f'Conversion finished.')

    # Validate exported onnx model
    # onnx_path = os.path.join(cade_dir, 'torch_save', cade_onnx_name)
    # valid_onnx(onnx_path=onnx_path)
    # print(f'Validation finished.')
