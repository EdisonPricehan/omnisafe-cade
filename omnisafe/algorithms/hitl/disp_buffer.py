"""Display buffer for HITL RL episode visualization.

This module provides functionality to read CSV files containing HITL RL episode data
and create comprehensive visualizations including observations, actions, rewards, costs,
and predictions. The visualizations can be saved as individual images or compiled into
a video for analysis.
"""

import os
import glob
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import torch
import re


class HITLDisplayBuffer:
    """Display buffer for HITL RL episode data visualization."""

    def __init__(self, figsize: Tuple[int, int] = (20, 12), policy_dir: Optional[str] = None):
        """Initialize the display buffer.

        Args:
            figsize: Figure size for the plots (width, height)
            policy_dir: Directory containing saved policy files for comparison
        """
        self.figsize = figsize
        self.episodes_data = []
        self.policy_dir = policy_dir

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def read_csv_files(self, directory_path: str) -> None:
        """Read all CSV files from the given directory.

        Args:
            directory_path: Path to directory containing CSV files
        """
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
        csv_files.sort()  # Sort for consistent ordering

        print(f"Found {len(csv_files)} CSV files in {directory_path}")

        for csv_file in csv_files:
            episode_data = self._parse_csv_file(csv_file)
            if episode_data:
                self.episodes_data.append({
                    'filename': os.path.basename(csv_file),
                    'data': episode_data
                })
                print(f"Loaded episode: {os.path.basename(csv_file)} with {len(episode_data)} steps")

    def _parse_csv_file(self, csv_file: str) -> List[Dict[str, Any]]:
        """Parse a single CSV file and extract episode data.

        Args:
            csv_file: Path to the CSV file

        Returns:
            List of dictionaries containing step data
        """
        episode_data = []

        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    step_data = {}

                    # Parse arrays from string representation
                    for key, value in row.items():
                        if key in ['obs', 'act', 'act_agent', 'next_obs', 'next_obs_pred']:
                            try:
                                step_data[key] = np.array(ast.literal_eval(value), dtype=np.float32)
                            except (ValueError, SyntaxError):
                                print(f"Warning: Could not parse {key} in {csv_file}")
                                step_data[key] = None
                        else:
                            try:
                                step_data[key] = float(value)
                            except ValueError:
                                step_data[key] = value

                    episode_data.append(step_data)

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            return []

        return episode_data

    def _reshape_observation(self, obs: np.ndarray) -> np.ndarray:
        """Reshape 1D observation to 16x16 grid.

        Args:
            obs: 1D observation array (256 elements)

        Returns:
            16x16 reshaped observation
        """
        if obs is None or len(obs) != 256:
            return np.zeros((16, 16))
        return obs.reshape(16, 16)

    def _plot_observation(self, ax, obs: np.ndarray, title: str, step: int):
        """Plot a single observation as a heatmap.

        Args:
            ax: Matplotlib axis
            obs: Observation array
            title: Plot title
            step: Current step number
        """
        obs_2d = self._reshape_observation(obs)

        # Use black and white colormap for binary masks
        im = ax.imshow(obs_2d, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'{title} (Step {step})', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        # Add grid lines aligned with cell boundaries
        ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
        ax.grid(which='minor', color='red', linestyle='-', linewidth=0.8, alpha=0.7)

        # Set major ticks for readability and disable major grid
        ax.set_xticks(np.arange(0, 16, 2))
        ax.set_yticks(np.arange(0, 16, 2))
        ax.grid(which='major', visible=False)  # Explicitly disable major grid

        return im

    def _plot_action_vector(self, ax, action: np.ndarray, title: str, step: int):
        """Plot action vector as a bar chart.

        Args:
            ax: Matplotlib axis
            action: Action array (4D)
            title: Plot title
            step: Current step number
        """
        if action is None or len(action) != 4:
            action = np.zeros(4)

        action_labels = ['Forward/Back', 'Left/Right', 'Up/Down', 'Rotation']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        bars = ax.bar(action_labels, action, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_title(f'{title} (Step {step})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Action Value')
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, action):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.2,
                   f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

    def _plot_predictions(self, ax, reward_pred: float, cost_pred: float,
                         reward_actual: float, cost_actual: float, step: int):
        """Plot reward and cost predictions vs actual values.

        Args:
            ax: Matplotlib axis
            reward_pred: Predicted reward
            cost_pred: Predicted cost
            reward_actual: Actual reward
            cost_actual: Actual cost
            step: Current step number
        """
        metrics = ['Reward', 'Cost']
        predicted = [reward_pred, cost_pred]
        actual = [reward_actual, cost_actual]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width/2, predicted, width, label='Predicted',
                      color='#FF9999', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, actual, width, label='Actual',
                      color='#66B2FF', alpha=0.8, edgecolor='black')

        ax.set_title(f'Predictions vs Actual (Step {step})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.2,
                       f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    def create_episode_visualization(self, episode_idx: int, step_idx: int,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Create a comprehensive visualization for a single step of an episode.

        Args:
            episode_idx: Index of the episode
            step_idx: Index of the step within the episode
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        if episode_idx >= len(self.episodes_data):
            raise ValueError(f"Episode index {episode_idx} out of range")

        episode = self.episodes_data[episode_idx]
        episode_data = episode['data']

        if step_idx >= len(episode_data):
            raise ValueError(f"Step index {step_idx} out of range for episode {episode_idx}")

        step_data = episode_data[step_idx]

        # Create figure with subplots (use 3D for actions)
        fig = plt.figure(figsize=self.figsize)

        # Create subplot layout
        ax1 = plt.subplot2grid((2, 3), (0, 0))  # Current obs
        ax2 = plt.subplot2grid((2, 3), (0, 1))  # Next obs actual
        ax3 = plt.subplot2grid((2, 3), (0, 2))  # Next obs predicted
        ax4 = plt.subplot2grid((2, 3), (1, 0), projection='3d')  # Agent action 3D
        ax5 = plt.subplot2grid((2, 3), (1, 1), projection='3d')  # Actual action 3D
        ax6 = plt.subplot2grid((2, 3), (1, 2))  # Predictions

        fig.suptitle(f"HITL Episode: {episode['filename']} - Step {step_idx}",
                    fontsize=16, fontweight='bold')

        # Plot current observation
        self._plot_observation(ax1, step_data['obs'], 'Current Observation', step_idx)

        # Plot next observation (peek at next step for actual)
        actual_next_obs = self._get_actual_next_obs(episode_data, step_idx)
        self._plot_observation(ax2, actual_next_obs, 'Next Observation (Actual)', step_idx)

        # Plot next observation (predicted)
        self._plot_observation(ax3, step_data['next_obs_pred'], 'Next Observation (Predicted)', step_idx)

        # Check if action is overlaid (act != act_agent)
        act_overlaid = not np.array_equal(step_data.get('act', []), step_data.get('act_agent', []))

        # Plot agent action in 3D
        self._plot_action_3d(ax4, step_data['act_agent'], 'Agent Action', step_idx, is_overlaid=False)

        # Plot actual action taken in 3D
        self._plot_action_3d(ax5, step_data['act'], 'Actual Action', step_idx, is_overlaid=act_overlaid)

        # Plot predictions vs actual
        self._plot_predictions(ax6,
                             step_data.get('reward_pred', 0),
                             step_data.get('cost_pred', 0),
                             step_data.get('reward', 0),
                             step_data.get('cost', 0),
                             step_idx)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        return fig

    def create_episode_summary(self, episode_idx: int, save_path: Optional[str] = None) -> plt.Figure:
        """Create a summary visualization for an entire episode.

        Args:
            episode_idx: Index of the episode
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        if episode_idx >= len(self.episodes_data):
            raise ValueError(f"Episode index {episode_idx} out of range")

        episode = self.episodes_data[episode_idx]
        episode_data = episode['data']

        # Extract time series data
        steps = list(range(len(episode_data)))
        rewards = [step.get('reward', 0) for step in episode_data]
        costs = [step.get('cost', 0) for step in episode_data]
        reward_preds_original = [step.get('reward_pred', 0) for step in episode_data]
        cost_preds_original = [step.get('cost_pred', 0) for step in episode_data]

        # Try to get updated estimates from corresponding policy
        episode_num = self._extract_episode_number(episode['filename'])
        policy_path = self._find_policy_file(episode_num)

        reward_preds_updated = None
        cost_preds_updated = None
        policy_comparison_available = False

        if policy_path:
            try:
                reward_preds_updated, cost_preds_updated = self._load_policy_and_get_estimates(episode_data, policy_path)
                policy_comparison_available = True
                print(f"Successfully loaded updated estimates from policy for episode {episode_num}")
            except Exception as e:
                print(f"Could not load policy estimates for episode {episode_num}: {e}")
        else:
            if episode_num == 0:
                print(f"No policy file found for episode {episode_num} (expected - episode 0 missing)")
            else:
                print(f"No policy file found for episode {episode_num}")

        # Check overlay status for each step
        overlay_status = []
        for step in episode_data:
            act = step.get('act', [])
            act_agent = step.get('act_agent', [])
            is_overlaid = not np.array_equal(act, act_agent) if act is not None and act_agent is not None else False
            overlay_status.append(is_overlaid)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f"Episode Summary: {episode['filename']}", fontsize=16, fontweight='bold')

        # Reward over time with policy comparison
        axes[0, 0].plot(steps, rewards, 'o-', label='Actual Reward', color='#FF6B6B', linewidth=2)
        axes[0, 0].plot(steps, reward_preds_original, 's--', label='Original Prediction', color='#FFB6C1', alpha=0.7)

        if policy_comparison_available:
            axes[0, 0].plot(steps, reward_preds_updated, '^:', label='Updated Prediction', color='#FF8C42', alpha=0.8, linewidth=2)
            axes[0, 0].set_title('Reward Over Time (Policy Comparison)', fontweight='bold')
        else:
            axes[0, 0].set_title('Reward Over Time (Original Only)', fontweight='bold')

        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Cost over time with policy comparison
        axes[0, 1].plot(steps, costs, 'o-', label='Actual Cost', color='#4ECDC4', linewidth=2)
        axes[0, 1].plot(steps, cost_preds_original, 's--', label='Original Prediction', color='#B2DFDB', alpha=0.7)

        if policy_comparison_available:
            axes[0, 1].plot(steps, cost_preds_updated, '^:', label='Updated Prediction', color='#45B7D1', alpha=0.8, linewidth=2)
            axes[0, 1].set_title('Cost Over Time (Policy Comparison)', fontweight='bold')
        else:
            axes[0, 1].set_title('Cost Over Time (Original Only)', fontweight='bold')

        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Cost')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Action overlay status bar plot - only show overlaid actions
        overlaid_steps = [i for i, overlaid in enumerate(overlay_status) if overlaid]
        if overlaid_steps:
            axes[1, 0].bar(overlaid_steps, [1] * len(overlaid_steps),
                          color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5,
                          label='Overlaid Actions')

        axes[1, 0].set_title('Action Overlay Status', fontweight='bold')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Overlay Status')
        axes[1, 0].set_xlim(-0.5, len(steps) - 0.5)
        axes[1, 0].set_ylim(0, 1.2)
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_yticklabels(['No Overlay', 'Overlaid'])
        axes[1, 0].grid(True, alpha=0.3, axis='both')

        # Add legend and summary text
        if overlaid_steps:
            axes[1, 0].legend(loc='upper right')
            overlay_percentage = len(overlaid_steps) / len(steps) * 100
            axes[1, 0].text(0.02, 0.95, f'Overlaid: {len(overlaid_steps)}/{len(steps)} steps ({overlay_percentage:.1f}%)',
                           transform=axes[1, 0].transAxes, fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           verticalalignment='top')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Overlaid Actions', transform=axes[1, 0].transAxes,
                           fontsize=12, fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # Prediction accuracy analysis
        if policy_comparison_available:
            # Calculate prediction errors
            original_reward_error = np.mean(np.abs(np.array(reward_preds_original) - np.array(rewards)))
            updated_reward_error = np.mean(np.abs(np.array(reward_preds_updated) - np.array(rewards)))
            original_cost_error = np.mean(np.abs(np.array(cost_preds_original) - np.array(costs)))
            updated_cost_error = np.mean(np.abs(np.array(cost_preds_updated) - np.array(costs)))

            # Plot prediction accuracy comparison
            metrics = ['Reward MAE', 'Cost MAE']
            original_errors = [original_reward_error, original_cost_error]
            updated_errors = [updated_reward_error, updated_cost_error]

            x = np.arange(len(metrics))
            width = 0.35

            bars1 = axes[1, 1].bar(x - width/2, original_errors, width, label='Original Policy',
                                  color='#FFB6C1', alpha=0.8, edgecolor='black')
            bars2 = axes[1, 1].bar(x + width/2, updated_errors, width, label='Updated Policy',
                                  color='#FF8C42', alpha=0.8, edgecolor='black')

            axes[1, 1].set_title('Prediction Accuracy Comparison', fontweight='bold')
            axes[1, 1].set_ylabel('Mean Absolute Error')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(metrics)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # Add improvement percentages
            for i, (orig, upd) in enumerate(zip(original_errors, updated_errors)):
                improvement = ((orig - upd) / orig * 100) if orig != 0 else 0
                axes[1, 1].text(i, max(orig, upd) + 0.1, f'{improvement:+.1f}%',
                               ha='center', va='bottom', fontweight='bold',
                               color='green' if improvement > 0 else 'red')
        else:
            # Fallback to cumulative metrics if no policy comparison
            cum_rewards = np.cumsum(rewards)
            cum_costs = np.cumsum(costs)
            axes[1, 1].plot(steps, cum_rewards, 'o-', label='Cumulative Reward', color='#45B7D1', linewidth=2)
            axes[1, 1].plot(steps, cum_costs, 's-', label='Cumulative Cost', color='#96CEB4', linewidth=2)
            axes[1, 1].set_title('Cumulative Metrics', fontweight='bold')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Cumulative Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved episode summary to {save_path}")

        return fig

    def create_video(self, episode_idx: int, output_path: str, fps: int = 2, format: str = 'gif') -> None:
        """Create a video animation of an episode.

        Args:
            episode_idx: Index of the episode
            output_path: Path to save the video file
            fps: Frames per second for the video
            format: Video format ('gif' or 'mp4')
        """
        if episode_idx >= len(self.episodes_data):
            raise ValueError(f"Episode index {episode_idx} out of range")

        episode = self.episodes_data[episode_idx]
        episode_data = episode['data']

        print(f"Creating {format.upper()} video for episode {episode['filename']} with {len(episode_data)} steps...")

        # Create figure with mixed 2D/3D subplots
        fig = plt.figure(figsize=self.figsize)

        def animate(frame):
            """Animation function for video creation."""
            # Clear the figure
            fig.clear()

            # Recreate subplot layout for each frame
            ax1 = plt.subplot2grid((2, 3), (0, 0))  # Current obs
            ax2 = plt.subplot2grid((2, 3), (0, 1))  # Next obs actual
            ax3 = plt.subplot2grid((2, 3), (0, 2))  # Next obs predicted
            ax4 = plt.subplot2grid((2, 3), (1, 0), projection='3d')  # Agent action 3D
            ax5 = plt.subplot2grid((2, 3), (1, 1), projection='3d')  # Actual action 3D
            ax6 = plt.subplot2grid((2, 3), (1, 2))  # Predictions

            step_data = episode_data[frame]

            # Plot current observation
            self._plot_observation(ax1, step_data['obs'], 'Current Observation', frame)

            # Plot next observation (peek at next step for actual)
            actual_next_obs = self._get_actual_next_obs(episode_data, frame)
            self._plot_observation(ax2, actual_next_obs, 'Next Observation (Actual)', frame)

            # Plot next observation (predicted)
            self._plot_observation(ax3, step_data['next_obs_pred'], 'Next Observation (Predicted)', frame)

            # Check if action is overlaid
            act_overlaid = not np.array_equal(step_data.get('act', []), step_data.get('act_agent', []))

            # Plot agent action in 3D
            self._plot_action_3d(ax4, step_data['act_agent'], 'Agent Action', frame, is_overlaid=False)

            # Plot actual action taken in 3D
            self._plot_action_3d(ax5, step_data['act'], 'Actual Action', frame, is_overlaid=act_overlaid)

            # Plot predictions vs actual
            self._plot_predictions(ax6,
                                 step_data.get('reward_pred', 0),
                                 step_data.get('cost_pred', 0),
                                 step_data.get('reward', 0),
                                 step_data.get('cost', 0),
                                 frame)

            fig.suptitle(f"HITL Episode: {episode['filename']} - Step {frame}",
                        fontsize=16, fontweight='bold')
            plt.tight_layout()

        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(episode_data), interval=1000//fps, repeat=False)

        # Save video based on format
        if format.lower() == 'mp4':
            try:
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=fps, metadata=dict(artist='HITLDisplayBuffer'), bitrate=1800)
                anim.save(output_path, writer=writer)
            except ImportError:
                print("FFMpeg not available, falling back to PillowWriter for MP4...")
                writer = PillowWriter(fps=fps)
                # Change extension to .gif if FFMpeg not available
                output_path = output_path.replace('.mp4', '.gif')
                anim.save(output_path, writer=writer)
        else:  # gif
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer)

        plt.close(fig)
        print(f"Video saved to {output_path}")

    def visualize_all_episodes(self, output_dir: str, all_steps: bool = False, video_formats: List[str] = ['gif']) -> None:
        """Create visualizations for all loaded episodes.

        Args:
            output_dir: Directory to save all visualizations
            all_steps: If True, create images for all steps, otherwise only first 5
            video_formats: List of video formats to create ('gif', 'mp4')
        """
        os.makedirs(output_dir, exist_ok=True)

        for episode_idx, episode in enumerate(self.episodes_data):
            episode_name = os.path.splitext(episode['filename'])[0]

            # Create episode summary
            summary_path = os.path.join(output_dir, f"{episode_name}_summary.png")
            self.create_episode_summary(episode_idx, summary_path)

            # Create videos in requested formats
            for fmt in video_formats:
                if fmt.lower() == 'mp4':
                    video_path = os.path.join(output_dir, f"{episode_name}_video.mp4")
                else:
                    video_path = os.path.join(output_dir, f"{episode_name}_video.gif")
                self.create_video(episode_idx, video_path, format=fmt)

            # Create individual step visualizations
            step_dir = os.path.join(output_dir, f"{episode_name}_steps")
            os.makedirs(step_dir, exist_ok=True)

            # Generate images for all steps if requested, otherwise only first 5
            max_steps = len(episode['data']) if all_steps else min(5, len(episode['data']))

            print(f"Creating {max_steps} step visualizations for {episode_name}...")
            for step_idx in range(max_steps):
                step_path = os.path.join(step_dir, f"step_{step_idx:03d}.png")
                fig = self.create_episode_visualization(episode_idx, step_idx, step_path)
                plt.close(fig)

        print(f"All visualizations saved to {output_dir}")

    def _get_actual_next_obs(self, episode_data: List[Dict[str, Any]], step_idx: int) -> np.ndarray:
        """Get the actual next observation by peeking at the next step.

        Args:
            episode_data: Episode data list
            step_idx: Current step index

        Returns:
            Next observation array or zeros if at last step
        """
        if step_idx + 1 < len(episode_data):
            return episode_data[step_idx + 1]['obs']
        else:
            # Last step, return zeros
            return np.zeros(256, dtype=np.float32)

    def _plot_action_3d(self, ax, action: np.ndarray, title: str, step: int, is_overlaid: bool = False):
        """Plot action vector as 3D arrows for discrete drone actions.

        Args:
            ax: Matplotlib 3D axis
            action: Action array (4D): [up/down, rotate, forward/backward, left/right]
                    Each element is 0, 1, or 2 where 1=no-op, 0/2=opposite directions
            title: Plot title
            step: Current step number
            is_overlaid: Whether this action is overlaid
        """
        if action is None or len(action) != 4:
            action = np.ones(4, dtype=int)  # Default to no-op (all 1s)

        # Convert to integer for discrete actions
        action = action.astype(int)

        # Action mapping: [up/down, rotate, forward/backward, left/right]
        up_down, rotate, forward_back, left_right = action

        # Clear the axis content but don't remove it
        ax.clear()

        # Set origin at center
        origin = [0, 0, 0]
        arrow_length = 1.5

        # Track which actions are active for legend
        active_actions = []

        # Up/Down (Z-axis) - 0=up, 1=no-op, 2=down
        if up_down == 0:  # Up
            ax.quiver(origin[0], origin[1], origin[2],
                     0, 0, arrow_length,
                     color='green', arrow_length_ratio=0.2, linewidth=4,
                     label='UP')
            active_actions.append('UP')
        elif up_down == 2:  # Down
            ax.quiver(origin[0], origin[1], origin[2],
                     0, 0, -arrow_length,
                     color='green', arrow_length_ratio=0.2, linewidth=4,
                     label='DOWN')
            active_actions.append('DOWN')

        # Forward/Backward (Y-axis) - 0=forward, 1=no-op, 2=backward
        if forward_back == 0:  # Forward
            ax.quiver(origin[0], origin[1], origin[2],
                     0, arrow_length, 0,
                     color='red', arrow_length_ratio=0.2, linewidth=4,
                     label='FORWARD')
            active_actions.append('FORWARD')
        elif forward_back == 2:  # Backward
            ax.quiver(origin[0], origin[1], origin[2],
                     0, -arrow_length, 0,
                     color='red', arrow_length_ratio=0.2, linewidth=4,
                     label='BACKWARD')
            active_actions.append('BACKWARD')

        # Left/Right (X-axis) - 0=left, 1=no-op, 2=right
        if left_right == 0:  # Left
            ax.quiver(origin[0], origin[1], origin[2],
                     -arrow_length, 0, 0,
                     color='blue', arrow_length_ratio=0.2, linewidth=4,
                     label='LEFT')
            active_actions.append('LEFT')
        elif left_right == 2:  # Right
            ax.quiver(origin[0], origin[1], origin[2],
                     arrow_length, 0, 0,
                     color='blue', arrow_length_ratio=0.2, linewidth=4,
                     label='RIGHT')
            active_actions.append('RIGHT')

        # Rotation - 0=rotate left (counter-clockwise), 1=no-op, 2=rotate right (clockwise)
        if rotate == 0:  # Rotate Left (Counter-clockwise)
            # Draw curved arrow for counter-clockwise rotation (when looking down from above)
            theta = np.linspace(0, -1.5*np.pi, 30)  # Negative for counter-clockwise
            r = 0.8
            x_curve = r * np.cos(theta)
            y_curve = r * np.sin(theta)
            z_curve = np.ones_like(theta) * 1.0
            ax.plot(x_curve, y_curve, z_curve, color='purple', linewidth=3)

            # Add arrow head for counter-clockwise direction
            ax.quiver(x_curve[-2], y_curve[-2], z_curve[-2],
                     x_curve[-1] - x_curve[-2], y_curve[-1] - y_curve[-2], 0,
                     color='purple', arrow_length_ratio=0.5, linewidth=3)

            ax.text(0, 0, 1.3, 'ROTATE LEFT', color='purple', fontweight='bold',
                   ha='center', va='center')
            active_actions.append('ROTATE LEFT')

        elif rotate == 2:  # Rotate Right (Clockwise)
            # Draw curved arrow for clockwise rotation (when looking down from above)
            theta = np.linspace(0, 1.5*np.pi, 30)  # Positive for clockwise
            r = 0.8
            x_curve = r * np.cos(theta)
            y_curve = r * np.sin(theta)
            z_curve = np.ones_like(theta) * 1.0
            ax.plot(x_curve, y_curve, z_curve, color='purple', linewidth=3)

            # Add arrow head for clockwise direction
            ax.quiver(x_curve[-2], y_curve[-2], z_curve[-2],
                     x_curve[-1] - x_curve[-2], y_curve[-1] - y_curve[-2], 0,
                     color='purple', arrow_length_ratio=0.5, linewidth=3)

            ax.text(0, 0, 1.3, 'ROTATE RIGHT', color='purple', fontweight='bold',
                   ha='center', va='center')
            active_actions.append('ROTATE RIGHT')

        # If no actions are active, show "NO ACTION"
        if not active_actions:
            ax.text(0, 0, 0, 'NO ACTION', color='gray', fontweight='bold',
                   ha='center', va='center', fontsize=12)

        # Set axis properties and remove ticks for cleaner look
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('Left/Right')
        ax.set_ylabel('Forward/Back')
        ax.set_zlabel('Up/Down')

        # Remove ticks for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Add overlay information to title
        overlay_text = " (OVERLAID)" if is_overlaid else ""
        action_summary = f"[{up_down},{rotate},{forward_back},{left_right}]"
        ax.set_title(f'{title}{overlay_text}\n{action_summary} (Step {step})',
                    fontsize=10, fontweight='bold')

        # Add text summary of active actions
        if active_actions:
            actions_text = " + ".join(active_actions)
            ax.text2D(0.02, 0.98, f"Active: {actions_text}", transform=ax.transAxes,
                     fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        return ax

    def _extract_episode_number(self, filename: str) -> int:
        """Extract episode number from CSV filename.

        Args:
            filename: CSV filename (e.g., "real_hitlTrue_lossIndirect_episode002_20250630_101832.csv")

        Returns:
            Episode number as integer
        """
        # Extract episode number from filename pattern
        print(f"DEBUG: Trying to extract episode number from filename: '{filename}'")
        # Updated pattern to match your actual filename format: episode002, episode001, etc.
        match = re.search(r'episode(\d+)', filename)
        if match:
            episode_num = int(match.group(1))
            print(f"DEBUG: Successfully extracted episode number: {episode_num}")
            return episode_num
        else:
            print(f"DEBUG: No match found for pattern 'episode(\\d+)' in '{filename}', returning 0")
            return 0  # Default to 0 if no match found

    def _find_policy_file(self, episode_num: int, session_type: str = 'upstream') -> Optional[str]:
        """Find the corresponding policy file for an episode.

        Args:
            episode_num: Episode number
            session_type: 'upstream' or 'downstream'

        Returns:
            Path to policy file or None if not found
        """
        if not self.policy_dir:
            return None

        policy_subdir = os.path.join(self.policy_dir, session_type)
        if not os.path.exists(policy_subdir):
            return None

        # Look for policy file with matching episode number
        pattern = f"real-episode-{episode_num:03d}-*.pt"
        policy_files = glob.glob(os.path.join(policy_subdir, pattern))

        if policy_files:
            return policy_files[0]  # Return first match
        return None

    def _load_policy_and_get_estimates(self, episode_data: List[Dict[str, Any]],
                                     policy_path: str) -> Tuple[List[float], List[float]]:
        """Load policy and get updated reward/cost estimates.

        Args:
            episode_data: Episode data containing observations and actions
            policy_path: Path to the saved policy file

        Returns:
            Tuple of (updated_reward_preds, updated_cost_preds)
        """
        try:
            print(f"Loading policy from {policy_path}...")

            # Check if file exists and is readable
            if not os.path.exists(policy_path):
                raise FileNotFoundError(f"Policy file not found: {policy_path}")

            # Check file size to detect potential corruption
            file_size = os.path.getsize(policy_path)
            if file_size < 1000:  # Very small files are likely corrupted
                raise ValueError(f"Policy file seems corrupted (too small: {file_size} bytes): {policy_path}")

            print(f"Policy file size: {file_size / (1024*1024):.2f} MB")

            # Try to load with better error handling
            try:
                model_params = torch.load(policy_path, map_location='cpu', weights_only=False)
            except Exception as torch_error:
                print(f"PyTorch load error: {torch_error}")
                # Try with different loading options
                try:
                    print("Attempting alternative loading method...")
                    model_params = torch.load(policy_path, map_location='cpu', pickle_module=None)
                except Exception as alt_error:
                    raise ValueError(f"Failed to load model with both methods. Original error: {torch_error}, Alternative error: {alt_error}")

            # Check if the expected key exists
            if 'actor_critic' not in model_params:
                available_keys = list(model_params.keys()) if isinstance(model_params, dict) else "Not a dictionary"
                raise KeyError(f"'actor_critic' key not found in model_params. Available keys: {available_keys}")

            # Check if we have the necessary config from the original HITL CADE
            # For now, we'll skip the complex model recreation and return original predictions with a warning
            print("WARNING: Policy comparison requires proper configuration from original training.")
            print("Falling back to original predictions for now.")
            print("To enable policy comparison, the model configuration from training needs to be saved alongside the model.")

            # Return original predictions since we can't safely load without proper config
            original_rewards = [step.get('reward_pred', 0) for step in episode_data]
            original_costs = [step.get('cost_pred', 0) for step in episode_data]
            return original_rewards, original_costs


        except Exception as e:
            print(f"Error loading policy from {policy_path}: {e}")
            print("Falling back to original predictions...")
            # Return original predictions if loading fails
            original_rewards = [step.get('reward_pred', 0) for step in episode_data]
            original_costs = [step.get('cost_pred', 0) for step in episode_data]
            return original_rewards, original_costs

def main():
    """Main function for command-line usage.

    Example usage:
        # Basic visualization without policy comparison
        python disp_buffer.py /path/to/csv/files

        # With policy comparison (replace paths with your actual paths)
        python disp_buffer.py /path/to/csv/files --policy-dir /home/edison/Research/omnisafe_zjy/examples/models/torch_save/wabash_0630

        # Full example with all options
        python disp_buffer.py /path/to/csv/files \
            --policy-dir /home/edison/Research/omnisafe_zjy/examples/models/torch_save/wabash_0630 \
            --output ./my_visualizations \
            --video-formats mp4 gif \
            --all-steps \
            --fps 3
    """
    import argparse

    parser = argparse.ArgumentParser(description='Visualize HITL RL episode data')
    parser.add_argument('directory', help='Directory containing CSV files')
    parser.add_argument('--output', '-o', default='./hitl_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--policy-dir', '-p', default=None,
                       help='Directory containing saved policy files for comparison (optional)')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second for videos')
    parser.add_argument('--all-steps', action='store_true',
                       help='Create images for all steps (default: only first 5)')
    parser.add_argument('--video-formats', nargs='+', choices=['gif', 'mp4'],
                       default=['mp4'], help='Video formats to create (default: mp4)')

    args = parser.parse_args()

    # Create display buffer and load data with optional policy directory
    display_buffer = HITLDisplayBuffer(policy_dir=args.policy_dir)
    display_buffer.read_csv_files(args.directory)

    if not display_buffer.episodes_data:
        print("No episode data found!")
        return

    # Create visualizations with new options
    display_buffer.visualize_all_episodes(
        args.output,
        all_steps=args.all_steps,
        video_formats=args.video_formats
    )

    print(f"Visualization complete! Check {args.output} for results.")
    if args.policy_dir:
        print(f"Policy comparison enabled using policies from: {args.policy_dir}")
    else:
        print("Policy comparison disabled (no policy directory provided)")


if __name__ == "__main__":
    main()
