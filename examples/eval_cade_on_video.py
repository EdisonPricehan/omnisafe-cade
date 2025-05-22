import os
import json
import csv
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from gymnasium.spaces import MultiDiscrete, MultiBinary
from PIL import Image
from typing import Optional
from tqdm import tqdm

from omnisafe.utils.config import Config
from omnisafe.envs.core import make, CMDP
from omnisafe.typing import OmnisafeSpace
from omnisafe.models.actor_critic import ConstraintActorDynamicsEstimator
from omnisafe.utils.patchification import get_patchified_mask, inflate_patch_mask

from mixed_sequence_reader import MixedSequenceReader
from video_reader import VideoReader
from segmentation import Segmenter


def evaluate_model(
    log_dir: str,
    model_name: str,
    segment_model_path: str,
    video_path: str,
    data_save_path: Optional[str] = None,
    image_save_path: Optional[str] = None,
    mask_save_path: Optional[str] = None,
    deterministic: bool = True,
    enable_safety_layer: bool = False,
):
    # Write header for data save path
    if data_save_path is not None:
        with open(data_save_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['observation', 'action', 'reward_pred', 'cost_pred'])
    if image_save_path is not None:
        assert os.path.exists(image_save_path), f'{image_save_path} does not exist.'
    if mask_save_path is not None:
        assert os.path.exists(mask_save_path), f'{mask_save_path} does not exist.'

    # Define constants
    obs_space = MultiBinary(16 * 16)
    act_space = MultiDiscrete([3, 3, 3, 3])
    nominal_action = torch.tensor([[1] * act_space.nvec.shape[0]])
    print(f'{nominal_action=}')

    # Init utilities
    video_reader = VideoReader(filepath=video_path, resize=(128, 128))
    segmenter = Segmenter(filepath=segment_model_path)

    # Load config
    cfg_path = os.path.join(log_dir, 'config.json')
    try:
        with open(cfg_path, encoding='utf-8') as file:
            kwargs = json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'The config file is not found in the save directory {log_dir}.',
        ) from error
    cfgs = Config.dict2config(kwargs)
    print('Config is loaded.')

    # Load policy model
    assert os.path.exists(log_dir), f'Model dir {log_dir} does not exist!'

    model_path: str = os.path.join(log_dir, 'torch_save', model_name)
    print(f'{model_path=}')

    assert os.path.exists(model_path), f'Model path {model_path} does not exist!'

    model_params = torch.load(model_path, map_location='cpu')

    is_value_critic: bool = False  # Use MGAE, so estimators are used instead of critics
    cade: ConstraintActorDynamicsEstimator = ConstraintActorDynamicsEstimator(
        obs_space=obs_space,
        act_space=act_space,
        model_cfgs=cfgs.model_cfgs,
        epochs=1,  # Not used, for linear lr decay
        is_value_critic=is_value_critic,
    )

    cade.load_state_dict(model_params['actor_critic'])
    print('Model is loaded.')

    # Iterate over video frames to do inference
    print('Start evaluation on video ...')
    latent = None
    last_act = nominal_action.clone()
    for idx, img in tqdm(enumerate(video_reader)):
        mask = segmenter.segment(img=img, binarize=True)
        mask = mask.squeeze()

        obs = get_patchified_mask(
            mask=mask,
            is_uint8=True,
            patch_size_x=8,
            patch_size_y=8,
            patch_step=8,
            binary_threshold=0.5,
            patch_threshold=0.5,  # Tunable
        )
        # Reshape obs
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, 256]
        # print(f'{obs.shape=}')

        # For debug
        # cv2.imshow('Mask', mask.numpy())
        # patchified_mask = inflate_patch_mask(
        #     obs=obs.squeeze().numpy(),
        #     image_size=128,
        #     patch_dim_x=16,
        #     patch_dim_y=16,
        #     patch_size_x=8,
        #     patch_size_y=8,
        # )
        # cv2.imshow('Mask Patch', patchified_mask)
        # cv2.waitKey(1000)

        # Step CAD
        act, logp, act_overlaid, reward_pred, cost_pred, latent = cade.step(
            obs=obs,
            last_act=last_act,
            lagrangian_multiplier=1.0,  # equally weigh reward and cost
            latent=latent,
            deterministic=deterministic,
            enable_safety_layer=enable_safety_layer,
            safety_layer_use_reward=False,
        )
        last_act.copy_(act)

        # Think of every frame as the first frame since the policy action is not actually interacting with video
        # Reset recurrent variables to their initial values
        last_act.copy_(nominal_action)
        latent = None

        # Save inference stat
        if data_save_path is not None:
            with open(data_save_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([
                    obs.squeeze().numpy(),
                    act.squeeze().numpy(),
                    f'{reward_pred.item():.2f}',
                    f'{cost_pred.item():.2f}',
                ])

        # Save image
        if image_save_path is not None:
            img_pil = Image.fromarray(img)
            img_pil.save(f'{image_save_path}/rgb_{idx:04d}.jpg')

        if mask_save_path is not None:
            mask_pil = Image.fromarray(mask.numpy())
            mask_pil.save(f'{mask_save_path}/mask_{idx:04d}.png')

        # print(f'{act=}')
        # if idx == 10:
        #     exit(0)

    print('Video inference finished.')


def plot_video_inference(
    inference_dir: str,
    river: str,
    col_num: int = 8,
    policy_num: int = 6,
    save_fig: bool = False,
):
    assert os.path.exists(inference_dir), f'{inference_dir} does not exist.'
    assert col_num > 0, f'{col_num} should be positive integer.'
    assert policy_num > 0, f'{policy_num} should be positive integer.'

    # Define constants
    image_row_num: int = 3  # rgb, mask, patchified mask
    row_num: int = image_row_num + policy_num
    print(f'Row num: {row_num}, col num: {col_num}')

    images_dir: str = os.path.join(inference_dir, f'images_{river}')
    masks_dir: str = os.path.join(inference_dir, f'masks_{river}')
    eval_stats_dict = {
        'mgae': os.path.join(inference_dir, f'mgae_{river}.csv'),
        'lagrangian': os.path.join(inference_dir, f'lagrangian_{river}.csv'),
        'safety_layer': os.path.join(inference_dir, f'safety_layer_{river}.csv'),
    }
    if save_fig:
        fig_save_dir: str = os.path.join(inference_dir, f'eval_figs_{river}')
        os.makedirs(fig_save_dir, exist_ok=True)

    # Init sequence reader
    msr = MixedSequenceReader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        csv_path_dict=eval_stats_dict,
        horizon=col_num,
    )

    # Setup figure
    axes = []
    fig = plt.figure(figsize=(12, 6), dpi=200)
    # plt.tight_layout(pad=1, w_pad=0.2, h_pad=0.2)
    # plt.style.use("seaborn-v0_8-darkgrid")
    font = {'family': 'STIXGeneral',
            'weight': 'bold',
            'size': 10,
            }

    for row in range(image_row_num):
        axes_row = []
        for col in range(col_num):
            ax = fig.add_subplot(row_num, col_num, col_num * row + col + 1)
            axes_row.append(ax)
        axes.append(axes_row)

    # Manually add 3D subplots for the action rows
    for row in range(policy_num):
        axes_row = []
        for col in range(col_num):
            action_ax = fig.add_subplot(row_num, col_num, (image_row_num + row) * col_num + col + 1, projection='3d')
            axes_row.append(action_ax)
        axes.append(axes_row)

    # Plot axes in sliding window
    for idx, (image_list, mask_list, obs_act_dict) in tqdm(enumerate(msr)):
        # Clear all subplots
        for axes_row in axes:
            for ax in axes_row:
                ax.clear()

        # Plot rgb image row
        for i, image in enumerate(image_list):
            axes[0][i].imshow(image)

        # Plot mask row
        for i, mask, in enumerate(mask_list):
            axes[1][i].imshow(mask, cmap='gray')

        # Plot patchified mask row
        obs_list = obs_act_dict['mgae']['observation']  # Use patchified masks from mgae csv
        for i, pmask in enumerate(obs_list):
            pmask_infla = inflate_patch_mask(
                obs=pmask,
                image_size=128,
                patch_dim_x=16,
                patch_dim_y=16,
                patch_size_x=8,
                patch_size_y=8,
            )
            axes[2][i].imshow(pmask_infla, cmap='gray')

        # Plot action rows
        for i, (method_name, data) in enumerate(obs_act_dict.items()):
            for j, action in enumerate(data['action']):
                # Post-process action since by default multi-discrete action only has one branch active
                # Action branch priority: Vertical Translation > Rotation > Long Translation > Lat Translation
                dim = action.shape[0]
                # print(f'Before: {action=}')
                k = 0
                while k < dim - 1:
                    if action[k] != 1:
                        action[k+1:] = 1
                        break
                    k += 1
                # print(f'After: {action=}')

                plot_action_arrow(ax=axes[3 + i][j], action=action)

        # Hide the ticks and spines but keep axis labels
        for axes_row in axes:
            for ax in axes_row:
                ax.tick_params(left=False, bottom=False, labelleft=False,
                               labelbottom=False)  # Hide ticks and tick labels
                ax.spines['top'].set_visible(False)  # Hide top spine
                ax.spines['right'].set_visible(False)  # Hide right spine
                ax.spines['bottom'].set_visible(False)  # Hide bottom spine
                ax.spines['left'].set_visible(False)  # Hide left spine

        # Column titles for each step (including input)
        column_titles = [f'T={i}' for i in range(idx, idx + col_num)]
        for i, ax in enumerate(axes[0]):
            ax.set_title(column_titles[i])

        # Row titles for each row
        row_titles = ["RGB", "Mask", "Mask Patch", "MGAE", "MGAE+Lag", 'MGAE+SL']
        for i, ax in enumerate([axes_row[0] for axes_row in axes]):
            if i > 2:  # action rows
                ax.text2D(
                    -0.06,
                    0.5,
                    row_titles[i],
                    transform=ax.transAxes,  # interpret x/y in the 2D axes coordinate system
                    rotation=90,  # vertical text
                    verticalalignment='center',
                    horizontalalignment='right',
                    fontdict=font,
                )
            else:
                ax.set_ylabel(row_titles[i], rotation=90, fontdict=font)

        # plt.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        if save_fig:
            save_path = os.path.join(fig_save_dir, f"step_{idx:04}.png")
            plt.savefig(save_path, dpi=200)
        else:
            plt.draw()

        # exit(0)


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


if __name__ == '__main__':
    # Define model paths
    # policy_model_path: str = './runs/FOCOPS_CACD-{medium}/seed-000-2025-03-01-13-10-24'  # MGAE
    # policy_model_path: str = './runs/FOCOPS_CACD-{medium}/seed-000-2025-03-01-15-06-43'  # Lagrangian
    policy_model_path: str = './runs/FOCOPS_CACD-{medium}/seed-000-2025-03-02-20-33-01'  # Safety Layer

    policy_model_name: str = 'epoch-350.pt'

    segmentation_model_path: str = '/home/edison/Research/Aerial-Fluvial-Semantic-Segmentation/src/models/unet-resnet34-128x128.pth'

    # Define river name
    # river_name: str = 'wabash'  # or wildcat
    river_name: str = 'wildcat'  # or wabash

    if river_name == 'wabash':
        video_path: str = '/home/edison/afid_videos/wabash/forward-1-2fps-720x480.mp4'  # wabash forward view
        image_path: str = './evaluations/riverine/video_inference/images_wabash'
        mask_path: str = './evaluations/riverine/video_inference/masks_wabash'

        # data_path: str = './evaluations/riverine/video_inference/mgae_wabash.csv'  # TODO set model data name
        # data_path: str = './evaluations/riverine/video_inference/lagrangian_wabash.csv'
        data_path: str = './evaluations/riverine/video_inference/safety_layer_wabash.csv'
    elif river_name == 'wildcat':
        video_path: str = '/home/edison/afid_videos/wildcat/output_2fps_720x480.mp4'  # wildcat forward view
        image_path: str = './evaluations/riverine/video_inference/images_wildcat'
        mask_path: str = './evaluations/riverine/video_inference/masks_wildcat'

        # data_path: str = './evaluations/riverine/video_inference/mgae_wildcat.csv'  # TODO set model data name
        # data_path: str = './evaluations/riverine/video_inference/lagrangian_wildcat.csv'
        data_path: str = './evaluations/riverine/video_inference/safety_layer_wildcat.csv'
    else:
        raise ValueError(f'{river_name} is not supported.')

    # evaluate_model(
    #     log_dir=policy_model_path,
    #     model_name=policy_model_name,
    #     segment_model_path=segmentation_model_path,
    #     video_path=video_path,
    #     data_save_path=data_path,
    #     image_save_path=image_path,
    #     # image_save_path=None,  # just save stats csv
    #     mask_save_path=mask_path,
    #     # mask_save_path=None,  # just save stats csv
    #     deterministic=True,
    #     enable_safety_layer=False,
    # )

    inference_dir: str = './evaluations/riverine/video_inference'
    plot_video_inference(
        inference_dir=inference_dir,
        river=river_name,
        col_num=10,
        policy_num=3,
        save_fig=True,
        # save_fig=None,
    )
