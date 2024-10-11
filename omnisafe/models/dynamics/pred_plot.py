import matplotlib.pyplot as plt
from typing import Union, Type, List, Tuple
import os
from gymnasium.spaces import MultiBinary, MultiDiscrete
import torch

from encoder.vae_v1 import VAE

from omnisafe.typing import OmnisafeSpace
from omnisafe.models.dynamics.sdm import SemanticDynamicsModel as SDM
from omnisafe.models.dynamics.sdm_mlp import SemanticDynamicsModelMLP as SDM_MLP
from omnisafe.models.dynamics.ldm import LatentDynamicsModel as LDM
from omnisafe.models.dynamics.ldm_mlp import LatentDynamicsModelMLP as LDM_MLP
from omnisafe.models.dynamics.train_ldm import encode_input
from omnisafe.utils.riverine_dataset import RiverDataset, get_train_test_datasets
from omnisafe.utils.patchification import get_patchified_mask, inflate_patch_mask
from omnisafe.utils.config import ModelConfig


def load_model(
    model_class: Union[Type[SDM], Type[SDM_MLP], Type[LDM], Type[LDM_MLP]],
    model_path: str,
):
    assert os.path.exists(model_path), f'{model_path} does not exist!'

    # Construct model
    if model_class == SDM:
        model = SDM(
            obs_space=obs_space,
            act_space=act_space,
            model_cfgs=model_config,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
        )
        print(f'SDM is loaded.')
    elif model_class == SDM_MLP:
        model = SDM_MLP(
            obs_space=obs_space,
            act_space=act_space,
            model_cfgs=model_config,
        )
        print(f'SDM-MLP is loaded.')
    elif model_class == LDM:
        model = LDM(
            latent_dim=latent_dim,
            act_dim=action_dim,
            gru_hidden_dim=gru_hidden_dim,
            num_gru_layers=gru_layers,
        )
        print(f'LDM is loaded.')
    elif model_class == LDM_MLP:
        model = LDM_MLP(
            latent_dim=latent_dim,
            act_dim=action_dim,
            model_cfgs=model_config,
        )
        print(f'LDM-MLP is loaded.')
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode

    return model


def get_ground_truth(
    start_idx: int,
    horizon: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Get the ground truth rgb images, masks and actions.
    Assume data points are sequential in the dataset (which is true for test set of riverine env).

    """
    rgb_gt_list: List[torch.Tensor] = []
    mask_gt_list: List[torch.Tensor] = []
    act_list: List[torch.Tensor] = []

    for i in range(horizon + 1):
        cur_rgb, cur_mask, act, next_rgb, next_mask = test_dataset[start_idx + i]
        rgb_gt_list.append(cur_rgb)
        mask_gt_list.append(cur_mask)
        act_list.append(act)

    return rgb_gt_list, mask_gt_list, act_list


def predict_masks(
    model,
    rgbs: List[torch.Tensor],
    masks: List[torch.Tensor],
    actions: List[torch.Tensor],
    horizon: int,
) -> List[torch.Tensor]:
    """
    Predict masks using a given model over a horizon with given actions, starting from the input rgb/mask

    """
    assert len(actions) == horizon + 1
    assert len(actions) == len(rgbs) == len(masks)

    # For SDM and SDM-MLP
    input_mask_patchified: torch.Tensor = get_patchified_mask(
        masks[0],
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
        patch_step=patch_step,
    )

    # For LDM and LDM-MLP
    input_rgb: torch.Tensor = rgbs[0]
    input_mask: torch.Tensor = masks[0]
    h_prev = None  # GRU initial hidden state

    predicted_patchified_masks = [input_mask_patchified]

    with torch.no_grad():
        for i in range(horizon):
            cur_act = actions[i]

            if isinstance(model, SDM | SDM_MLP):
                cur_mask_act = torch.cat((input_mask_patchified, cur_act), dim=-1)

                if isinstance(model, SDM):
                    pred_mask = model.predict(cur_mask_act.unsqueeze(0), round_to_int=True).squeeze(0)
                else:
                    pred_mask = model.predict(cur_mask_act, round_to_int=True)

                predicted_patchified_masks.append(pred_mask)

                input_mask_patchified = pred_mask  # Use the predicted mask as the next input

            elif isinstance(model, LDM | LDM_MLP):
                # Encode the current state using the VAE
                cur_latent = encode_input(vae, input_rgb.unsqueeze(0), input_mask.unsqueeze(0)).squeeze(0)

                if isinstance(model, LDM_MLP):
                    latent_act = torch.cat((cur_latent, cur_act), dim=-1)
                    pred_latent = model(latent_act)
                else:
                    pred_latent, h_prev = model(cur_latent.unsqueeze(0), cur_act.unsqueeze(0), h_prev)
                    pred_latent = pred_latent.squeeze(0)

                # Decode the predicted latent state back to the 4-channel RGB + mask
                reconstructed = vae.decode(pred_latent.unsqueeze(0)).squeeze(0)  # Output shape: (4, H, W)

                # Separate the reconstructed rgb and mask
                pred_rgb = reconstructed[:-1]  # (3, H, W)
                pred_mask = reconstructed[-1].unsqueeze(0)  # Mask is in the 4th channel. (1, H, W)

                # Patchify the reconstructed predicted next mask
                pred_mask_patchified = get_patchified_mask(
                    pred_mask,
                    patch_size_x=patch_size_x,
                    patch_size_y=patch_size_y,
                    patch_step=patch_step,
                )

                predicted_patchified_masks.append(pred_mask_patchified)

                # Use the predicted rgb and mask as the next input
                input_rgb = pred_rgb
                input_mask = pred_mask

            else:
                raise NotImplementedError

    return predicted_patchified_masks


def update_figure(event):
    global start_idx

    # Move the index forward
    start_idx += step

    # Get ground truth
    rgbs, masks, actions = get_ground_truth(start_idx, horizon)

    # Get ground truth of patchified masks
    masks_patchified = [get_patchified_mask(
        mask,
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
        patch_step=patch_step,
    ) for mask in masks]

    # Inflate these gt patchified masks for plot
    masks_patchified_inflated = [inflate_patch_mask(
        m.squeeze().numpy(),
        image_size=image_size,
        patch_dim_x=patch_rows,
        patch_dim_y=patch_cols,
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
    ) for m in masks_patchified]

    # Get patchified mask predictions
    sdm_pred = predict_masks(sdm_model, rgbs, masks, actions, horizon)
    sdm_mlp_pred = predict_masks(sdm_mlp_model, rgbs, masks, actions, horizon)
    ldm_pred = predict_masks(ldm_model, rgbs, masks, actions, horizon)
    ldm_mlp_pred = predict_masks(ldm_mlp_model, rgbs, masks, actions, horizon)

    # Inflate these predicted patchified masks for plot
    sdm_pred_inflated = [inflate_patch_mask(
        m.numpy(),
        image_size=image_size,
        patch_dim_x=patch_rows,
        patch_dim_y=patch_cols,
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
    ) for m in sdm_pred]
    sdm_mlp_pred_inflated = [inflate_patch_mask(
        m.numpy(),
        image_size=image_size,
        patch_dim_x=patch_rows,
        patch_dim_y=patch_cols,
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
    ) for m in sdm_mlp_pred]
    ldm_pred_inflated = [inflate_patch_mask(
        m.numpy(),
        image_size=image_size,
        patch_dim_x=patch_rows,
        patch_dim_y=patch_cols,
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
    ) for m in ldm_pred]
    ldm_mlp_pred_inflated = [inflate_patch_mask(
        m.numpy(),
        image_size=image_size,
        patch_dim_x=patch_rows,
        patch_dim_y=patch_cols,
        patch_size_x=patch_size_x,
        patch_size_y=patch_size_y,
    ) for m in ldm_mlp_pred]

    # Clear previous plots
    for ax in axes.flatten():
        ax.clear()

    # Plot ground truth and predictions for each model
    for i in range(horizon + 1):
        axes[0, i].imshow(rgbs[i].permute(1, 2, 0).numpy())  # Ground truth: rgb
        axes[1, i].imshow(masks[i].squeeze().numpy(), cmap='gray')  # Ground truth: original mask
        axes[2, i].imshow(masks_patchified_inflated[i], cmap='gray')  # Ground truth: patchified and inflated original mask
        axes[3, i].imshow(sdm_pred_inflated[i], cmap='gray')  # SDM
        axes[4, i].imshow(sdm_mlp_pred_inflated[i], cmap='gray')  # SDM-MLP
        axes[5, i].imshow(ldm_pred_inflated[i], cmap='gray')  # LDM
        axes[6, i].imshow(ldm_mlp_pred_inflated[i], cmap='gray')  # LDM-MLP

    # Hide the ticks and spines but keep axis labels
    for ax in axes.flatten():
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # Hide ticks and tick labels
        ax.spines['top'].set_visible(False)  # Hide top spine
        ax.spines['right'].set_visible(False)  # Hide right spine
        ax.spines['bottom'].set_visible(False)  # Hide bottom spine
        ax.spines['left'].set_visible(False)  # Hide left spine

    # Column titles for each step (including input)
    column_titles = [f"Step {i}" for i in range(horizon + 1)]
    for i, ax in enumerate(axes[0, :]):
        ax.set_title(column_titles[i])

    # Row titles for each row
    row_titles = ["GT RGB", "GT Mask", "GT Mask Patch", "SDM", "SDM-MLP", "LDM", "LDM-MLP"]
    for i, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(row_titles[i], rotation=90, size='large', fontweight='bold')

    plt.draw()


def on_key(event):
    """
    Function to be called when the key is pressed
    Args:
        event:

    Returns:

    """
    if event.key == ' ':
        update_figure(None)  # Call the update function when spacebar is pressed


if __name__ == '__main__':
    # Constants of model paths
    cur_dir: str = os.path.dirname(os.path.abspath(__file__))
    sdm_path: str = os.path.join(cur_dir, 'riverine_sdm_iou.pth')
    sdm_mlp_path: str = os.path.join(cur_dir, 'riverine_sdm_mlp_iou.pth')
    ldm_path: str = os.path.join(cur_dir, 'riverine_ldm.pth')
    ldm_mlp_path: str = os.path.join(cur_dir, 'riverine_ldm_mlp.pth')

    # Constants for patchification
    image_size: int = 128  # dimension of raw image (assuming square)
    patch_rows, patch_cols = 16, 16  # dimension of coarsened mask
    patch_size_x, patch_size_y = image_size // patch_rows, image_size // patch_cols
    patch_step = patch_size_x  # no overlap among patches

    # Constants for agent in Safe Riverine Environment
    obs_space: OmnisafeSpace = MultiBinary(patch_rows * patch_cols)
    act_space: OmnisafeSpace = MultiDiscrete([3, 3, 3, 3])

    device: str = 'cpu'
    model_config = ModelConfig(
        dynamics={
            'hidden_sizes': [128, 64],
            'activation': 'relu',
            'lr': 0.001,
        }
    )
    metric: str = 'iou'

    # Constants for LDM
    action_dim = 4  # aligned with the act_space
    gru_hidden_dim = 128
    gru_layers = 1

    # Constants for VAE
    channel_size = 4  # 4-channel rgb+mask
    latent_dim = 64  # observation dim for latent dynamics models
    hidden_dims = [16, 32, 64, 128]
    vae_model_name = 'vae-4channel.pth'

    # Configurable parameters
    horizon = 10  # Number of steps to predict
    step = 1  # Step size when "space" button is pressed
    start_idx = 0

    # Define the VAE model
    vae = VAE(
        in_channels=channel_size,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        target_output_size=2,
        original_height=image_size,
        original_width=image_size,
    )

    # Load the VAE model
    vae_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), vae_model_name)
    vae.load_state_dict(torch.load(vae_model_path))
    vae.eval()
    print(f'VAE is loaded.')

    # Load all visual dynamics models
    sdm_model = load_model(SDM, sdm_path)
    sdm_mlp_model = load_model(SDM_MLP, sdm_mlp_path)
    ldm_model = load_model(LDM, ldm_path)
    ldm_mlp_model = load_model(LDM_MLP, ldm_mlp_path)

    # Initialize datasets
    train_dataset, test_dataset = get_train_test_datasets()

    # Setup figure
    fig, axes = plt.subplots(7, horizon + 1, figsize=(14, 9))
    # fig.subplots_adjust(bottom=0.2)
    # fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.2, right=0.9)

    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial plot
    update_figure(None)

    plt.tight_layout()
    plt.show()

    # TODO plot action
