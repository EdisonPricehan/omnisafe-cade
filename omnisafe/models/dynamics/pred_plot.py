import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, Type, List, Tuple, Dict, Optional
import os
import time
import json
import numpy as np
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


ModelClassType = Union[Type[SDM], Type[SDM_MLP], Type[LDM], Type[LDM_MLP]]
ModelInstanceType = Union[SDM, SDM_MLP, LDM, LDM_MLP]
ModelFamily = Dict[str, ModelInstanceType]


def load_model(
    model_class: ModelClassType,
    model_path: str,
) -> ModelInstanceType:
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


def calculate_inference_time(
    model: ModelInstanceType,
    test_data: RiverDataset,
    horizon: int,
) -> Tuple[float, int, float, float]:
    """
    Calculate the total inference time for the given model across the test dataset, excluding data loading time.

    Args:
        model (ModelInstanceType): The model to be tested.
        test_data (RiverDataset): The test dataset.
        horizon (int): The number of future steps to predict.

    Returns:
        Tuple: total time (float), number of inferences (int), mean inference time (float), and standard deviation (float).
    """
    inference_times = []

    with torch.no_grad():
        for idx in range(len(test_data) - horizon):
            # Pre-load the data outside timing to exclude data loading time
            rgbs, masks, actions = get_ground_truth(idx, horizon)

            # Time only the inference process
            start_time = time.time()
            predict_masks(model, rgbs, masks, actions, horizon)
            end_time = time.time()

            # Record the inference time
            inference_times.append(end_time - start_time)

    total_time = sum(inference_times)
    num_inferences = len(inference_times)
    mean_inference_time = total_time / num_inferences
    std_inference_time = torch.std(torch.tensor(inference_times)).item()

    return total_time, num_inferences, mean_inference_time, std_inference_time


def calculate_model_size(model: ModelInstanceType) -> int:
    return sum(p.numel() for p in model.parameters())


def run_inference_only(models: ModelFamily, test_data: RiverDataset, horizon: int):
    """
    Main function to run inferences without displaying figures.
    Args:
        models:
        test_data:
        horizon:

    Returns:
        A dict summary of inference statistics.

    """
    results = {}
    for model_name, model in models.items():
        print(f'Start inferring for {model_name} ...')

        # Calculate inference time
        total_time, infer_count, mean_infer_time, std_infer_time = calculate_inference_time(model, test_data, horizon)

        # Calculate model size
        model_size = calculate_model_size(model)

        # Store results
        results[model_name] = {
            'total_inference_time': total_time,
            'infer_count': infer_count,
            'infer_time_mean': mean_infer_time,
            'infer_time_std': std_infer_time,
            'model_size': model_size,
        }

    return results


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


def update_figure(auto_save: bool = False, save_dir: Optional[str] = None):
    if auto_save:
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        else:
            print(f'If auto_save is enabled, the save_dir should not be empty.')

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

    # Clear all subplots
    for axes_row in axes:
        for ax in axes_row:
            ax.clear()

    # Plot ground truth and predictions for each model
    for i in range(horizon + 1):
        axes[0][i].imshow(rgbs[i].permute(1, 2, 0).numpy())  # Ground truth: rgb
        axes[1][i].imshow(masks[i].squeeze().numpy(), cmap='gray')  # Ground truth: original mask
        axes[2][i].imshow(masks_patchified_inflated[i],
                          cmap='gray')  # Ground truth: patchified and inflated original mask
        axes[3][i].imshow(sdm_pred_inflated[i], cmap='gray')  # SDM
        axes[4][i].imshow(sdm_mlp_pred_inflated[i], cmap='gray')  # SDM-MLP
        axes[5][i].imshow(ldm_pred_inflated[i], cmap='gray')  # LDM
        axes[6][i].imshow(ldm_mlp_pred_inflated[i], cmap='gray')  # LDM-MLP

        if i == horizon:
            continue
        plot_action_arrow(axes[7][i], actions[i])

    # Hide the ticks and spines but keep axis labels
    for axes_row in axes:
        for ax in axes_row:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # Hide ticks and tick labels
            ax.spines['top'].set_visible(False)  # Hide top spine
            ax.spines['right'].set_visible(False)  # Hide right spine
            ax.spines['bottom'].set_visible(False)  # Hide bottom spine
            ax.spines['left'].set_visible(False)  # Hide left spine

    # Column titles for each step (including input)
    column_titles = ['Current'] + [f"Step {i}" for i in range(1, horizon + 1)]
    for i, ax in enumerate(axes[0]):
        ax.set_title(column_titles[i])

    # Row titles for each row
    row_titles = ["GT RGB", "GT Mask", "GT Mask Patch", "SDM", "SDM-MLP", "LDM", "LDM-MLP", 'Action']
    for i, ax in enumerate([axes_row[0] for axes_row in axes]):
        if i == 7:  # action row
            ax.set_zlabel(row_titles[i])  # not taking any effect
        else:
            ax.set_ylabel(row_titles[i], rotation=90, size='large', fontweight='bold')

    plt.tight_layout()

    if auto_save:
        save_path = os.path.join(save_dir, f"step_{start_idx:03}.png")
        plt.savefig(save_path)
    else:
        plt.draw()


def plot_action_arrow(ax, action, scale=1.0):
    """
    Plots a 3D cross symbol with an arrow indicating the action direction.

    Parameters:
    ax (Axes3D): The 3D axes to plot the action arrow.
    action (list): The action vector [vertical, rotation, forward-backward, left-right].
                   Each element is 0, 1, or 2, where 1 means no movement.
    scale (float): Scaling factor for the arrow size.
    """
    # Draw fixed 3D axes without arrows (for non-movement directions)
    ax.quiver(0, 0, 0, 1.5, 0, 0, arrow_length_ratio=0, color='gray')  # X-axis (forward/backward)
    ax.quiver(0, 0, 0, 0, 1.5, 0, arrow_length_ratio=0, color='gray')  # Y-axis (left/right)
    ax.quiver(0, 0, 0, 0, 0, 1.5, arrow_length_ratio=0, color='gray')  # Z-axis (up/down)

    # Define arrow scaling for each axis
    arrow_scale = 1.5 * scale

    # Plot the action along the triggered axis with an arrow
    if action[0] == 0:  # Upward movement (Z-axis)
        ax.quiver(0, 0, 0, 0, 0, arrow_scale, arrow_length_ratio=0.3, color='blue', alpha=.8, lw=2)
    elif action[0] == 2:  # Downward movement (Z-axis)
        ax.quiver(0, 0, 0, 0, 0, -arrow_scale, arrow_length_ratio=0.3, color='blue', alpha=.8, lw=2)

    if action[2] == 0:  # Forward movement (X-axis)
        ax.quiver(0, 0, 0, arrow_scale, 0, 0, arrow_length_ratio=0.3, color='green', alpha=.8, lw=2)
    elif action[2] == 2:  # Backward movement (X-axis)
        ax.quiver(0, 0, 0, -arrow_scale, 0, 0, arrow_length_ratio=0.3, color='green', alpha=.8, lw=2)

    if action[3] == 0:  # Left movement (Y-axis)
        ax.quiver(0, 0, 0, 0, arrow_scale, 0, arrow_length_ratio=0.3, color='orange', alpha=.8, lw=2)
    elif action[3] == 2:  # Right movement (Y-axis)
        ax.quiver(0, 0, 0, 0, -arrow_scale, 0, arrow_length_ratio=0.3, color='orange', alpha=.8, lw=2)

    # For rotation, add an arc or circular representation (rotation along Z-axis)
    if action[1] == 0:  # Rotate left
        theta = np.linspace(0, np.pi / 2, 100)
        x = 1 * np.cos(theta)
        y = 1 * np.sin(theta)
        ax.plot(x, y, zs=0, zdir='z', color='red', lw=2)  # Plot rotation arc
        ax.quiver(x[-1], y[-1], 0, -0.7, 0, 0, arrow_length_ratio=0.6, color='red', alpha=.8, lw=2)
    elif action[1] == 2:  # Rotate right
        theta = np.linspace(np.pi / 2, 0, 100)
        x = 1 * np.cos(theta)
        y = 1 * np.sin(theta)
        ax.plot(x, y, zs=0, zdir='z', color='red', lw=2)  # Plot rotation arc
        ax.quiver(x[-1], y[-1], 0, 0, -0.7, 0, arrow_length_ratio=0.6, color='red', alpha=.8, lw=2)

    # Adjust the view for better visualization
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=30, azim=-150)  # Set a good 3D view angle


def on_key(event):
    """
    Function to be called when the key is pressed
    Args:
        event:

    Returns:

    """
    if event.key == ' ':
        update_figure()  # Call the update function when spacebar is pressed


def run_headless_inference(save_dir: str):
    for idx in range(0, len(test_dataset) - horizon, step):
        update_figure(auto_save=True, save_dir=save_dir)


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
    inference_only: bool = False  # Only do model inference to get statistics
    headless: bool = True  # Run inference and save model prediction figures without showing them
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

    # Define the family of all models
    all_models: ModelFamily = {
        'sdm': sdm_model,
        'sdm-mlp': sdm_mlp_model,
        'ldm': ldm_model,
        'ldm-mlp': ldm_mlp_model,
    }

    # Initialize datasets
    train_dataset, test_dataset = get_train_test_datasets()

    if inference_only:
        print(f'Running inference only ...')
        results = run_inference_only(all_models, test_dataset, horizon)

        # Save results to json file
        with open("model_comp.json", "w") as file:
            json.dump(results, file, indent=4)

        print(f'Inferences are finished.')

    else:
        # Setup figure
        axes = []
        fig = plt.figure(figsize=(12, 9))
        for row in range(7):
            axes_row = []
            for col in range(horizon + 1):
                ax = fig.add_subplot(8, horizon + 1, (horizon + 1) * row + col + 1)
                axes_row.append(ax)
            axes.append(axes_row)

        # Manually add 3D subplots for the action row
        act_axes = []
        for i in range(horizon):
            action_ax = fig.add_subplot(8, horizon + 1, 7 * (horizon + 1) + i + 1, projection='3d')
            act_axes.append(action_ax)
        axes.append(act_axes)

        if headless:
            print(f'Running in headless mode while saving figures ...')
            figure_save_path: str = 'models_pred'
            run_headless_inference(figure_save_path)
            print(f'Model inference finished, figures saved to {figure_save_path}.')

        else:
            print(f'Running in interactive mode, press "space" key to advance the inference.')

            # Connect the key press event
            fig.canvas.mpl_connect('key_press_event', on_key)

            # Initial plot
            update_figure()

            plt.show()
