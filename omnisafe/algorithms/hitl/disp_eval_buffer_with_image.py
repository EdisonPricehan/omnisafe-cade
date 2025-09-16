"""
Comparison visualizer for HITL evaluation buffers with RGB images.

For a given episode number, this script loads evaluation results for all loss types
from a base directory structured as:

base_dir/
  <LossTypeA>/
    ... *episodeXYZ*.csv
    ... <some folder containing RGB frames like rgb001.png, rgb002.png, ...>
  <LossTypeB>/
    ...
  Baseline/
    ...

It creates a comparison animation (MP4 preferred; GIF fallback) with 8 outer subplots
(4 rows x 2 columns, one per loss type). Inside each outer subplot, there are 4 subplots:
- RGB image
- Patchified mask (128x128) reconstructed from the obs vector in CSV
- 3D action arrow (borrowed from disp_buffer.py)
- Cumulative reward up to the current step

If a loss type's episode terminates earlier than others, its four inner subplots
freeze at the last state, and the cumulative reward subplot displays the final
cumulative reward in large text.

Usage:
  python -m omnisafe.algorithms.hitl.disp_eval_buffer_with_image \
      --base-dir /home/edison/Research/omnisafe_zjy/omnisafe/algorithms/hitl/evaluations/non_deterministic_cumu_buffer_level1 \
      --episode 0 \
      --fps 3 \
      --output ./comparison_episode000.mp4

Notes:
- The script attempts to locate the CSV and the RGB frames directory by searching for
  names containing the episode number pattern "episodeNNN".
- If ffmpeg is not available, it falls back to saving a GIF and logs a notice.
"""

from __future__ import annotations

import os
import glob
import csv
import ast
import re
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.gridspec import GridSpec

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # Fallback when tqdm is not installed

# -----------------------------
# Helpers to locate files
# -----------------------------


def natural_key(s: str) -> List[Any]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", os.path.basename(s))]


@dataclass
class EpisodeData:
    loss_name: str
    csv_path: Optional[str]
    images: List[str]
    steps: List[Dict[str, Any]]
    cum_rewards: List[float]


# -----------------------------
# CSV parsing and mask creation
# -----------------------------


def parse_csv(csv_path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if not csv_path or not os.path.exists(csv_path):
        return data
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                step: Dict[str, Any] = {}
                for k, v in row.items():
                    if k in ['obs', 'act', 'act_agent', 'next_obs', 'next_obs_pred']:
                        try:
                            step[k] = np.array(ast.literal_eval(v), dtype=np.float32)
                        except Exception:
                            step[k] = None
                    else:
                        # try float, else keep string
                        try:
                            step[k] = float(v)
                        except Exception:
                            step[k] = v
                data.append(step)
    except Exception as e:
        print(f"Error parsing CSV {csv_path}: {e}")
        return []
    return data


def obs_to_mask128(obs: Optional[np.ndarray]) -> np.ndarray:
    """Convert 1D obs (len 256) -> 16x16 then patchify to 128x128.
    If obs is missing or wrong shape, return zeros(128,128).
    """
    try:
        if obs is None:
            return np.zeros((128, 128), dtype=np.float32)
        flat = np.asarray(obs).ravel()
        if flat.size == 256:
            grid = flat.reshape(16, 16)
        elif flat.size == 128 * 128:
            # Already 128x128
            return flat.reshape(128, 128)
        else:
            # Attempt to infer square and resize via simple nearest tiling
            side = int(np.sqrt(flat.size))
            if side * side == flat.size:
                grid = flat.reshape(side, side)
            else:
                return np.zeros((128, 128), dtype=np.float32)
        # Patchify 16x16 -> 128x128 via upsampling by 8 using Kronecker product
        mask128 = np.kron(grid, np.ones((8, 8), dtype=grid.dtype))
        return mask128.astype(np.float32)
    except Exception:
        return np.zeros((128, 128), dtype=np.float32)


# -----------------------------
# 3D Action plot (borrowed and slightly adapted)
# From disp_buffer.py lines 634–775
# -----------------------------


def plot_action_3d(ax, action: Optional[np.ndarray], title: str, step: int, is_overlaid: bool = False):
    if action is None or len(action) != 4:
        action = np.ones(4, dtype=int)
    action = np.asarray(action).astype(int)
    up_down, rotate, forward_back, left_right = action
    ax.clear()
    origin = [0, 0, 0]
    arrow_length = 1.5
    active_actions: List[str] = []

    # Up/Down (Z)
    if up_down == 0:
        ax.quiver(origin[0], origin[1], origin[2], 0, 0, arrow_length, color='green', arrow_length_ratio=0.2, linewidth=4, label='UP')
        active_actions.append('UP')
    elif up_down == 2:
        ax.quiver(origin[0], origin[1], origin[2], 0, 0, -arrow_length, color='green', arrow_length_ratio=0.2, linewidth=4, label='DOWN')
        active_actions.append('DOWN')

    # Forward/Backward (Y)
    if forward_back == 0:
        ax.quiver(origin[0], origin[1], origin[2], 0, arrow_length, 0, color='red', arrow_length_ratio=0.2, linewidth=4, label='FORWARD')
        active_actions.append('FORWARD')
    elif forward_back == 2:
        ax.quiver(origin[0], origin[1], origin[2], 0, -arrow_length, 0, color='red', arrow_length_ratio=0.2, linewidth=4, label='BACKWARD')
        active_actions.append('BACKWARD')

    # Left/Right (X)
    if left_right == 0:
        ax.quiver(origin[0], origin[1], origin[2], -arrow_length, 0, 0, color='blue', arrow_length_ratio=0.2, linewidth=4, label='LEFT')
        active_actions.append('LEFT')
    elif left_right == 2:
        ax.quiver(origin[0], origin[1], origin[2], arrow_length, 0, 0, color='blue', arrow_length_ratio=0.2, linewidth=4, label='RIGHT')
        active_actions.append('RIGHT')

    # Rotation
    if rotate == 0:
        theta = np.linspace(0, -1.5 * np.pi, 30)
        r = 0.8
        x_curve = r * np.cos(theta)
        y_curve = r * np.sin(theta)
        z_curve = np.ones_like(theta) * 1.0
        ax.plot(x_curve, y_curve, z_curve, color='purple', linewidth=3)
        ax.quiver(x_curve[-2], y_curve[-2], z_curve[-2], x_curve[-1] - x_curve[-2], y_curve[-1] - y_curve[-2], 0, color='purple', arrow_length_ratio=0.5, linewidth=3)
        ax.text(0, 0, 1.3, 'ROTATE LEFT', color='purple', fontweight='bold', ha='center', va='center')
        active_actions.append('ROTATE LEFT')
    elif rotate == 2:
        theta = np.linspace(0, 1.5 * np.pi, 30)
        r = 0.8
        x_curve = r * np.cos(theta)
        y_curve = r * np.sin(theta)
        z_curve = np.ones_like(theta) * 1.0
        ax.plot(x_curve, y_curve, z_curve, color='purple', linewidth=3)
        ax.quiver(x_curve[-2], y_curve[-2], z_curve[-2], x_curve[-1] - x_curve[-2], y_curve[-1] - y_curve[-2], 0, color='purple', arrow_length_ratio=0.5, linewidth=3)
        ax.text(0, 0, 1.3, 'ROTATE RIGHT', color='purple', fontweight='bold', ha='center', va='center')
        active_actions.append('ROTATE RIGHT')

    if not active_actions:
        ax.text(0, 0, 0, 'NO ACTION', color='gray', fontweight='bold', ha='center', va='center', fontsize=12)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('Left/Right')
    ax.set_ylabel('Forward/Back')
    ax.set_zlabel('Up/Down')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    overlay_text = " (OVERLAID)" if is_overlaid else ""
    action_summary = f"[{up_down},{rotate},{forward_back},{left_right}]"
    ax.set_title(f"{title}{overlay_text}\n{action_summary} (Step {step})", fontsize=9, fontweight='bold')

    if active_actions:
        actions_text = " + ".join(active_actions)
        ax.text2D(0.02, 0.98, f"Active: {actions_text}", transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


# -----------------------------
# Data discovery per loss type
# -----------------------------


def find_loss_dirs(base_dir: str) -> List[str]:
    dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    dirs.sort(key=lambda p: os.path.basename(p).lower())
    return dirs


def find_csv_for_episode(loss_dir: str, ep: int) -> Optional[str]:
    pattern = os.path.join(loss_dir, f"*episode{ep:03d}*.csv")
    files = glob.glob(pattern)
    files.sort(key=natural_key)
    return files[0] if files else None


def find_images_for_episode(loss_dir: str, ep: int) -> List[str]:
    # Search for subdirectories containing episode pattern
    cand_dirs = [d for d in glob.glob(os.path.join(loss_dir, f"*episode{ep:03d}*")) if os.path.isdir(d)]
    # If no matching subdir, also check the loss_dir directly
    if not cand_dirs:
        cand_dirs = [loss_dir]
    images: List[str] = []
    for d in cand_dirs:
        imgs = glob.glob(os.path.join(d, 'rgb*.png'))
        if not imgs:
            imgs = glob.glob(os.path.join(d, '*.png'))
        if imgs:
            imgs.sort(key=natural_key)
            # Heuristic: choose the directory with most frames
            if len(imgs) > len(images):
                images = imgs
    return images


# -----------------------------
# Animator
# -----------------------------


class ComparisonAnimator:
    def __init__(self, base_dir: str, episode: int, figsize: Tuple[int, int] = (24, 13.5), fps: int = 3):
        self.base_dir = base_dir
        self.episode = episode
        self.figsize = figsize
        self.fps = fps
        self.loss_dirs = find_loss_dirs(base_dir)
        # Ensure exactly 8; if more, take first 8; if fewer, fill with placeholders
        if len(self.loss_dirs) > 8:
            self.loss_dirs = self.loss_dirs[:8]
        self.loss_names = [os.path.basename(d) for d in self.loss_dirs]
        while len(self.loss_dirs) < 8:
            self.loss_dirs.append(None)  # placeholder
            self.loss_names.append('(None)')
        self.episodes: List[EpisodeData] = []

        for d, name in zip(self.loss_dirs, self.loss_names):
            if d is None:
                self.episodes.append(EpisodeData(name, None, [], [], []))
                continue
            csv_path = find_csv_for_episode(d, episode)
            images = find_images_for_episode(d, episode)
            steps = parse_csv(csv_path) if csv_path else []
            cum_rewards: List[float] = []
            cr = 0.0
            for st in steps:
                r = st.get('reward', 0.0) or 0.0
                try:
                    r = float(r)
                except Exception:
                    r = 0.0
                cr += r
                cum_rewards.append(cr)
            self.episodes.append(EpisodeData(name, csv_path, images, steps, cum_rewards))

        self.max_frames = max([max(len(ep.steps), len(ep.images)) for ep in self.episodes]) if self.episodes else 0

        # Global stats for cumulative reward scaling across all loss types
        all_cum_vals = [val for ep in self.episodes for val in ep.cum_rewards]
        if all_cum_vals:
            self.global_cum_min = float(np.min(all_cum_vals))
            self.global_cum_max = float(np.max(all_cum_vals))
        else:
            self.global_cum_min = 0.0
            self.global_cum_max = 1.0
        # Ensure some padding and sensible defaults
        if self.global_cum_min == self.global_cum_max:
            self.global_cum_max = self.global_cum_min + 1.0
        # Always include zero for visual comparability
        self.global_cum_min = min(0.0, self.global_cum_min)
        self.global_cum_max = max(1.0, self.global_cum_max)
        pad_min = 0.05 * (abs(self.global_cum_min) if self.global_cum_min != 0 else 1.0)
        pad_max = 0.05 * (abs(self.global_cum_max) if self.global_cum_max != 0 else 1.0)
        self.global_cum_ylim = (self.global_cum_min - pad_min, self.global_cum_max + pad_max)
        self.global_max_steps = max([len(ep.cum_rewards) for ep in self.episodes]) if self.episodes else 1
        if self.global_max_steps <= 0:
            self.global_max_steps = 1

        # Matplotlib figure and axes (compact layout, 16:9 overall)
        self.fig = plt.figure(figsize=self.figsize, constrained_layout=False)
        outer = self.fig.add_gridspec(nrows=4, ncols=2, hspace=0.2, wspace=0.05)
        self.axes_per_loss: List[Tuple[Any, Any, Any, Any]] = []  # (rgb_ax, mask_ax, act_ax, cum_ax)

        for i in range(8):
            r = i // 2
            c = i % 2
            inner = outer[r, c].subgridspec(1, 4, wspace=0.08)
            ax_img = self.fig.add_subplot(inner[0, 0])
            ax_mask = self.fig.add_subplot(inner[0, 1])
            ax_act = self.fig.add_subplot(inner[0, 2], projection='3d')
            ax_cum = self.fig.add_subplot(inner[0, 3])
            self.axes_per_loss.append((ax_img, ax_mask, ax_act, ax_cum))

        # For updating artists
        self.image_artists: List[Optional[Any]] = [None] * 8
        self.mask_artists: List[Optional[Any]] = [None] * 8
        self.cum_lines: List[Optional[Any]] = [None] * 8
        self.final_texts: List[Optional[Any]] = [None] * 8  # big text shown after termination
        self.frozen_after: List[int] = [-1] * 8  # index of last frame for each loss when it terminates
        self.outer_title_texts: List[Optional[Any]] = [None] * 8

        self._init_frame()
        # Compact margins
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04, wspace=0.04, hspace=0.18)
        # Place per-loss bold titles at the top center of each outer subplot cell
        self._place_outer_titles()

    def _init_frame(self):
        # Initialize content for frame 0
        for i, ep in enumerate(self.episodes):
            ax_img, ax_mask, ax_act, ax_cum = self.axes_per_loss[i]
            # Titles handled at outer level; keep inner titles minimal
            ax_mask.set_title("Mask", fontsize=9)
            ax_cum.set_title("Cumulative Reward", fontsize=9)

            # RGB
            if ep.images:
                try:
                    img0 = plt.imread(ep.images[0])
                    self.image_artists[i] = ax_img.imshow(img0)
                except Exception as e:
                    self.image_artists[i] = ax_img.text(0.5, 0.5, f"Image error\n{e}", ha='center', va='center')
            else:
                self.image_artists[i] = ax_img.text(0.5, 0.5, "No image", ha='center', va='center')
            ax_img.axis('off')

            # Mask
            if ep.steps:
                m = obs_to_mask128(ep.steps[0].get('obs'))
                self.mask_artists[i] = ax_mask.imshow(m, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            else:
                self.mask_artists[i] = ax_mask.text(0.5, 0.5, "No data", ha='center', va='center')
            ax_mask.axis('off')

            # Action
            if ep.steps:
                act = ep.steps[0].get('act')
                overlaid = False
                if 'act_agent' in ep.steps[0] and ep.steps[0].get('act_agent') is not None and act is not None:
                    overlaid = not np.array_equal(np.asarray(act), np.asarray(ep.steps[0].get('act_agent')))
                plot_action_3d(ax_act, act, "Action", 0, is_overlaid=overlaid)
            else:
                ax_act.text(0.5, 0.5, 1.3, "No action", ha='center', va='center')
                ax_act.set_axis_off()

            # Cum reward
            if ep.cum_rewards:
                line, = ax_cum.plot([0], [ep.cum_rewards[0]], color='tab:blue', linewidth=1.5)
                self.cum_lines[i] = line
                ax_cum.set_xlim(0, self.global_max_steps)
                ax_cum.set_ylim(*self.global_cum_ylim)
                ax_cum.grid(True, alpha=0.3)
            else:
                self.cum_lines[i] = None
                ax_cum.text(0.5, 0.5, "No reward data", ha='center', va='center')
                ax_cum.set_xlim(0, self.global_max_steps)
                ax_cum.set_ylim(*self.global_cum_ylim)
                ax_cum.grid(True, alpha=0.3)

            # Compact ticks for cumulative plot
            ax_cum.tick_params(axis='both', labelsize=8)

        self.fig.suptitle(f"Episode {self.episode:03d} - Frame 0", fontsize=16, fontweight='bold')

    def _update_loss_axes(self, i: int, frame: int):
        ep = self.episodes[i]
        ax_img, ax_mask, ax_act, ax_cum = self.axes_per_loss[i]
        steps_n = len(ep.steps)
        imgs_n = len(ep.images)
        # Determine if terminated
        terminated = (steps_n == 0 and imgs_n == 0) or (frame >= max(steps_n, imgs_n) and max(steps_n, imgs_n) > 0)
        # Use clamped indices for data
        idx_step = min(frame, max(0, steps_n - 1)) if steps_n > 0 else -1
        idx_img = min(frame, max(0, imgs_n - 1)) if imgs_n > 0 else -1

        # Update RGB
        if idx_img >= 0 and isinstance(self.image_artists[i], plt.Axes) is False:
            # artist is an AxesText or AxesImage; handle AxesImage only
            if hasattr(self.image_artists[i], 'set_data'):
                try:
                    self.image_artists[i].set_data(plt.imread(ep.images[idx_img]))
                except Exception:
                    pass
        # Update mask
        if idx_step >= 0 and hasattr(self.mask_artists[i], 'set_data'):
            m = obs_to_mask128(ep.steps[idx_step].get('obs'))
            self.mask_artists[i].set_data(m)

        # Update action only if not terminated before this frame (freeze at last)
        if idx_step >= 0 and (self.frozen_after[i] < 0 or frame <= self.frozen_after[i]):
            act = ep.steps[idx_step].get('act')
            overlaid = False
            if 'act_agent' in ep.steps[idx_step] and ep.steps[idx_step].get('act_agent') is not None and act is not None:
                overlaid = not np.array_equal(np.asarray(act), np.asarray(ep.steps[idx_step].get('act_agent')))
            plot_action_3d(ax_act, act, "Action", idx_step, is_overlaid=overlaid)

        # Update cum reward line
        if self.cum_lines[i] is not None and ep.cum_rewards:
            x = list(range(len(ep.cum_rewards)))
            upto = min(frame + 1, len(ep.cum_rewards))
            self.cum_lines[i].set_data(x[:upto], ep.cum_rewards[:upto])
            ax_cum.set_xlim(0, self.global_max_steps)
            ax_cum.set_ylim(*self.global_cum_ylim)

        # If just terminated at this frame, record and show final text
        if terminated and self.frozen_after[i] < 0:
            self.frozen_after[i] = frame
            # Add big final cumulative reward text
            if self.final_texts[i] is None:
                final_val = ep.cum_rewards[-1] if ep.cum_rewards else 0.0
                self.final_texts[i] = ax_cum.text(
                    0.5, 0.5, f"Final Cum Reward\n{final_val:.2f}", ha='center', va='center', transform=ax_cum.transAxes,
                    fontsize=14, fontweight='bold', color='darkgreen', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
                )

    def animate(self, frame: int):
        for i in range(8):
            self._update_loss_axes(i, frame)
        self.fig.suptitle(f"Episode {self.episode:03d} - Frame {frame}", fontsize=16, fontweight='bold')
        return []

    def _place_outer_titles(self):
        # Place a bold loss-name title centered above each outer subplot area
        for i, ep in enumerate(self.episodes):
            ax_img, _, _, _ = self.axes_per_loss[i]
            bbox = ax_img.get_position()
            x = bbox.x0 + (bbox.x1 - bbox.x0) / 2.0
            y = bbox.y1 + 0.01  # small offset above the inner axes
            if self.outer_title_texts[i] is not None:
                try:
                    self.outer_title_texts[i].remove()
                except Exception:
                    pass
            self.outer_title_texts[i] = self.fig.text(x, y, ep.loss_name, ha='center', va='bottom', fontsize=11, fontweight='bold')

    def save_frames(self, frames_dir: str, dpi: int = 150):
        os.makedirs(frames_dir, exist_ok=True)
        if self.max_frames <= 0:
            print("No frames to save.")
            return
        iterator = tqdm(range(self.max_frames), desc="Rendering frames") if tqdm else range(self.max_frames)
        try:
            for f in iterator:
                self.animate(f)
                out = os.path.join(frames_dir, f"frame_{f:03d}.jpg")
                self.fig.savefig(out, dpi=dpi)
        finally:
            if tqdm and hasattr(iterator, 'close'):
                try:
                    iterator.close()
                except Exception:
                    pass
        print(f"Saved {self.max_frames} frames to {frames_dir}")

    def encode_frames_to_video(self, frames_dir: str, output_path: str, fps: int):
        # Encode pre-rendered frames into a video without regenerating the plots
        pattern = os.path.join(frames_dir, 'frame_*.jpg')
        frames = glob.glob(pattern)
        frames.sort(key=natural_key)
        if not frames:
            print(f"No frames found in {frames_dir}")
            return
        # Create a simple figure for encoding frames
        fig2 = plt.figure(figsize=self.figsize)
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.axis('off')
        img0 = plt.imread(frames[0])
        im_artist = ax2.imshow(img0)
        try:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='ComparisonAnimator'), bitrate=2000)
            pbar = tqdm(total=len(frames), desc="Encoding MP4") if tqdm else None
            try:
                with writer.saving(fig2, output_path, dpi=150):
                    for fp in frames:
                        try:
                            im = plt.imread(fp)
                            im_artist.set_data(im)
                            writer.grab_frame()
                        except Exception as e:
                            print(f"Skipping frame {fp}: {e}")
                        finally:
                            if pbar:
                                pbar.update(1)
                print(f"Saved MP4 to {output_path}")
            finally:
                if pbar:
                    pbar.close()
        except Exception as e:
            print(f"FFMpeg not available or failed ({e}). Falling back to GIF...")
            alt = os.path.splitext(output_path)[0] + '.gif'
            writer = PillowWriter(fps=fps)
            pbar = tqdm(total=len(frames), desc="Encoding GIF") if tqdm else None
            try:
                with writer.saving(fig2, alt, dpi=150):
                    for fp in frames:
                        try:
                            im = plt.imread(fp)
                            im_artist.set_data(im)
                            writer.grab_frame()
                        except Exception as e:
                            print(f"Skipping frame {fp}: {e}")
                        finally:
                            if pbar:
                                pbar.update(1)
                print(f"Saved GIF to {alt}")
            finally:
                if pbar:
                    pbar.close()
        finally:
            plt.close(fig2)

    def save(self, output_path: str):
        if self.max_frames <= 0:
            print("No frames to animate. Nothing saved.")
            return
        anim = FuncAnimation(self.fig, self.animate, frames=self.max_frames, interval=int(1000 / max(1, self.fps)), blit=False, repeat=False)
        # Try mp4 first
        try:
            writer = FFMpegWriter(fps=self.fps, metadata=dict(artist='ComparisonAnimator'), bitrate=2000)
            pbar = tqdm(total=self.max_frames, desc="Encoding MP4") if tqdm else None
            try:
                def _cb(i, n):
                    if pbar:
                        # Set absolute position to avoid double-increment issues
                        pbar.n = i + 1
                        pbar.refresh()
                anim.save(output_path, writer=writer, progress_callback=_cb)
                print(f"Saved MP4 to {output_path}")
            finally:
                if pbar:
                    pbar.close()
        except Exception as e:
            print(f"FFMpeg not available or failed ({e}). Falling back to GIF...")
            alt = os.path.splitext(output_path)[0] + '.gif'
            writer = PillowWriter(fps=self.fps)
            pbar = tqdm(total=self.max_frames, desc="Encoding GIF") if tqdm else None
            try:
                def _cb_gif(i, n):
                    if pbar:
                        pbar.n = i + 1
                        pbar.refresh()
                anim.save(alt, writer=writer, progress_callback=_cb_gif)
                print(f"Saved GIF to {alt}")
            finally:
                if pbar:
                    pbar.close()
        finally:
            plt.close(self.fig)


# -----------------------------
# CLI
# -----------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create comparison video across loss types for a specific episode.')
    parser.add_argument('--base-dir', type=str, default='/home/edison/Research/omnisafe_zjy/omnisafe/algorithms/hitl/evaluations/non_deterministic_cumu_buffer_level1', help='Base evaluations directory with loss-type subfolders.')
    parser.add_argument('--episode', type=int, required=True, help='Episode number (e.g., 0 for episode000).')
    parser.add_argument('--fps', type=int, default=3, help='Frames per second for the output video.')
    parser.add_argument('--output', type=str, default=None, help='Output video path. If not set, saved under base-dir.')
    parser.add_argument('--frames-dir', type=str, default=None, help='Directory to save rendered frames (jpg). If not set, a unique directory will be auto-created under base-dir.')
    parser.add_argument('--encode-from-frames', type=str, default=None, help='If set, encode a video from pre-rendered frames in this directory and exit.')
    parser.add_argument('--dpi', type=int, default=150, help='DPI when saving frames.')
    args = parser.parse_args()

    # If only encoding from frames is requested
    if args.encode_from_frames:
        if args.output is None:
            args.output = os.path.join(args.base_dir, f'comparison_episode{args.episode:03d}.mp4')
        # Instantiate to reuse encoding utility and default figsize (no re-generation of plots)
        encoder = ComparisonAnimator(args.base_dir, args.episode, fps=args.fps)
        encoder.encode_frames_to_video(args.encode_from_frames, args.output, args.fps)
        return

    if args.output is None:
        args.output = os.path.join(args.base_dir, f'comparison_episode{args.episode:03d}.mp4')

    animator = ComparisonAnimator(args.base_dir, args.episode, fps=args.fps)
    print(f"Discovered loss types: {animator.loss_names}")
    for ep in animator.episodes:
        print(f"- {ep.loss_name}: CSV={'yes' if ep.csv_path else 'no'}, steps={len(ep.steps)}, images={len(ep.images)}")

    # Save frames first to allow flexible FPS selection later
    if args.frames_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        auto_dir_base = os.path.join(args.base_dir, '..', f'comparison_episode{args.episode:03d}_{ts}_frames')
        frames_dir = auto_dir_base
        suffix = 1
        while os.path.exists(frames_dir):
            frames_dir = f"{auto_dir_base}_{suffix}"
            suffix += 1
        print(f"No --frames-dir provided. Auto-creating frames directory: {frames_dir}")
    else:
        frames_dir = args.frames_dir
    animator.save_frames(frames_dir, dpi=args.dpi)

    # Save video if an output path is provided (can be regenerated later from saved frames)
    if args.output:
        animator.save(args.output)


if __name__ == '__main__':
    main()
