import os
import json
import ast
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from gymnasium.spaces import MultiBinary, MultiDiscrete
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42   # TrueType instead of Type 3
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['text.usetex']  = False
mpl.rcParams['axes.labelweight'] = 'bold'  # Make axis labels bold by default

from omnisafe.utils.config import Config
from omnisafe.models.actor_critic import ConstraintActorDynamicsEstimator


def _resolve_path(*parts: str) -> str:
    return os.path.join(os.path.dirname(__file__), *parts)


def _load_cfgs(model_dir: str) -> Config:
    cfg_path = _resolve_path(model_dir, 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'Config not found: {cfg_path}')
    with open(cfg_path, encoding='utf-8') as f:
        kwargs = json.load(f)
    return Config.dict2config(kwargs)


def _load_cade(model_dir: str, model_name: str, device: str = 'cpu') -> ConstraintActorDynamicsEstimator:
    model_path = _resolve_path(model_dir, 'torch_save', model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')

    cfgs = _load_cfgs(model_dir)

    # Spaces match the training setup (16x16 mask, 4-branch action)
    obs_space = MultiBinary(16 * 16)
    act_space = MultiDiscrete([3, 3, 2, 3])

    cade = ConstraintActorDynamicsEstimator(
        obs_space=obs_space,
        act_space=act_space,
        model_cfgs=cfgs.model_cfgs,
        epochs=1,
    )

    # Load weights
    params = torch.load(model_path, map_location=device)
    cade.load_state_dict(params['actor_critic'])
    cade = cade.to(device)
    cade.eval()
    return cade


def _parse_listlike_column(series: pd.Series) -> pd.Series:
    # Convert stringified lists like "[0, 1, 2]" to Python lists; leave numeric values as-is.
    if series.dtype == 'object':
        def parse(x):
            if isinstance(x, str):
                xs = x.strip()
                if xs.startswith('[') and xs.endswith(']'):
                    try:
                        return ast.literal_eval(xs)
                    except Exception:
                        return x
                # try simple numeric parse
                try:
                    return float(xs)
                except Exception:
                    return x
            return x
        return series.map(parse)
    return series


def _to_tensor_2d_listcol(col: pd.Series, dtype: torch.dtype, device: str) -> torch.Tensor:
    # col is a Series of lists (e.g., obs with length-256, action with length-4)
    data = col.tolist()
    arr = np.array(data)
    t = torch.tensor(arr, dtype=dtype)
    return t.to(device)


def _to_tensor_1d_scalar(col: pd.Series, device: str) -> torch.Tensor:
    def to_scalar(x):
        if isinstance(x, list):
            return float(x[0]) if len(x) > 0 else float('nan')
        if isinstance(x, str):
            xs = x.strip()
            if xs.startswith('[') and xs.endswith(']'):
                try:
                    v = ast.literal_eval(xs)
                    if isinstance(v, list):
                        return float(v[0]) if len(v) > 0 else float('nan')
                    return float(v)
                except Exception:
                    pass
            try:
                return float(xs)
            except Exception:
                return float('nan')
        try:
            return float(x)
        except Exception:
            return float('nan')

    vals = col.map(to_scalar).values.astype(np.float32)
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _build_intervention_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Build a boolean mask of human interventions per step.
    Prefer act_overlaid column if present; otherwise, infer via act != act_agent.
    """
    N = len(df)
    mask = np.zeros(N, dtype=bool)
    if 'act_overlaid' in df.columns:
        # Convert to scalar 0/1 via _to_tensor_1d_scalar-like logic on CPU then to numpy bool
        try:
            vals = _to_tensor_1d_scalar(df['act_overlaid'], device='cpu').numpy()
            mask = vals.astype(np.float32) > 0.5
        except Exception:
            # Fallback: try to parse strings directly
            mask = df['act_overlaid'].astype(str).str.strip().isin(['1', '1.0', 'true', 'True', 'TRUE', 'yes', 'y'])
            mask = mask.values
    elif 'act' in df.columns and 'act_agent' in df.columns:
        # Compare list-like actions element-wise string representation
        a = df['act'].astype(str).str.strip()
        b = df['act_agent'].astype(str).str.strip()
        mask = (a != b).values
    return mask


def plot_reward_comparison(
    pred: np.ndarray,
    gt: np.ndarray,
    intervened_mask: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot predicted rewards (CAPER) vs stored reward_pred (CSV) and distinguish intervened steps.
    """
    T = min(len(pred), len(gt), len(intervened_mask))
    x = np.arange(T)
    pred = pred[:T]
    gt = gt[:T]
    intervened_mask = intervened_mask[:T]

    plt.figure(figsize=(8, 5))
    plt.plot(x, pred, label='SPAR-H (Checkpoint 4)', color='tab:purple', linewidth=2)
    plt.plot(x, gt, label='Novice (Before Retraining)', color='tab:blue', alpha=0.8, linewidth=2)

    # Mark intervened steps
    if intervened_mask.any():
        xi = x[intervened_mask]
        yi = pred[intervened_mask]
        plt.scatter(xi, yi, color='red', marker='o', s=30, label='Intervened')

    # Optionally mark unintervened lightly for context
    # xu = x[~intervened_mask]
    # if len(xu) > 0:
    #     yu = pred[~intervened_mask]
    #     plt.scatter(xu, yu, color='gray', marker='.', s=10, alpha=0.5, label='Unintervened')

    plt.xlabel('Step')
    plt.ylabel('Estimated Reward')
    if title:
        plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(prop={'weight': 'bold'})
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower().lstrip('.')
        if ext in ('png', 'pdf', 'svg', 'eps', 'jpg', 'jpeg'):
            plt.savefig(save_path, dpi=300, format=ext)
        else:
            plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def compare_caper_reward_on_episode(
    model_dir: Optional[str] = None,
    model_name: str = 'sim-episode-004-hitl-False-loss-CAPER.pt',
    episode_csv_relpath: str = os.path.join('evaluations', 'hitl_demo', 'medium_hitlTrue_lossNone_episode000.csv'),
    use_agent_actions: bool = True,
    device: str = 'cpu',
    plot: bool = False,
    save_path: Optional[str] = None,
    return_mask: bool = False,
):
    """
    Load the CAPER policy, roll out reward predictions on the novice trajectory from a CSV,
    and compare to stored reward_pred at each step.

    Args:
        model_dir: Directory containing config.json and torch_save/<model_name>. Defaults to the one used in hitl_cade.
        model_name: Checkpoint filename to load.
        episode_csv_relpath: CSV path relative to this file.
        use_agent_actions: If True, use 'act_agent' (novice actions); otherwise, use executed 'act'.
        device: Torch device, e.g., 'cpu' or 'cuda:0'.

    Returns:
        pred: numpy array of predicted rewards per step.
        gt: numpy array of ground-truth stored reward_pred per step from CSV.
    """
    if model_dir is None:
        # Default consistent with hitl_cade.py
        model_dir = _resolve_path('../../examples/runs/FOCOPS_CADE-{medium}/seed-042-2025-08-21-15-47-02')
        # Normalize path
        model_dir = os.path.normpath(model_dir)

    csv_path = _resolve_path(episode_csv_relpath)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Episode CSV not found: {csv_path}')

    # Load model
    cade = _load_cade(model_dir, model_name, device=device)

    # Load CSV and parse columns
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f'CSV is empty: {csv_path}')

    for col in ['obs', 'act', 'act_agent', 'reward_pred']:
        if col in df.columns:
            df[col] = _parse_listlike_column(df[col])

    # Build obs tensor [1, T, 256]
    assert 'obs' in df.columns, 'Missing obs column in CSV.'
    obs = _to_tensor_2d_listcol(df['obs'], dtype=torch.float32, device=device)  # [T, 256]
    T = obs.shape[0]
    obs = obs.unsqueeze(0)  # [1, T, 256]

    # Choose actions
    act_col = 'act_agent' if use_agent_actions and 'act_agent' in df.columns else 'act'
    assert act_col in df.columns, f'Missing action column {act_col}.'
    act = _to_tensor_2d_listcol(df[act_col], dtype=torch.long, device=device)  # [T, 4]
    act = act.unsqueeze(0)  # [1, T, 4]

    # Compute reward predictions using the reward estimator
    with torch.no_grad():
        reward_pred_list = cade.forward_reward(obs, act, ep_lens=None)
        assert isinstance(reward_pred_list, list) and len(reward_pred_list) > 0
        pred_t = reward_pred_list[0]
        if pred_t.dim() > 1:
            pred_t = pred_t.squeeze(-1)
        pred = pred_t.detach().cpu().view(-1).numpy()

    # Load ground-truth stored reward_pred column
    assert 'reward_pred' in df.columns, 'Missing reward_pred column in CSV.'
    gt_t = _to_tensor_1d_scalar(df['reward_pred'], device='cpu')  # keep on cpu
    gt = gt_t.view(-1).numpy()

    # Truncate to the same length if mismatched
    N = min(len(pred), len(gt), T)
    pred = pred[:N]
    gt = gt[:N]

    # Compute metrics
    mask_valid = np.isfinite(pred) & np.isfinite(gt)
    if mask_valid.sum() == 0:
        mae = np.nan
        mse = np.nan
    else:
        mae = np.mean(np.abs(pred[mask_valid] - gt[mask_valid]))
        mse = np.mean((pred[mask_valid] - gt[mask_valid]) ** 2)

    # Build intervention mask (intervened steps)
    intervened_mask = _build_intervention_mask(df) if len(df) > 0 else np.zeros(N, dtype=bool)
    intervened_mask = intervened_mask[:N]

    # Print per-step comparison (first few and last few for brevity)
    print(f'Loaded model: {os.path.join(model_dir, "torch_save", model_name)}')
    print(f'Compared against episode: {csv_path}')
    print(f'Using actions: {act_col}')
    print(f'Total steps compared: {N}  |  Intervened: {int(intervened_mask.sum())}  ({(intervened_mask.mean()*100 if N>0 else 0):.2f}%)')
    print('Step  pred_reward    csv_reward_pred    diff')
    preview = 10
    for i in range(min(N, preview)):
        print(f'{i:4d}  {pred[i]:12.6f}  {gt[i]:16.6f}  {pred[i]-gt[i]:8.6f}')
    if N > preview:
        print(' ...')
        for i in range(max(0, N - 3), N):
            print(f'{i:4d}  {pred[i]:12.6f}  {gt[i]:16.6f}  {pred[i]-gt[i]:8.6f}')
    print(f'MAE: {mae:.6f}, MSE: {mse:.6f}')

    # Optionally plot
    if plot:
        episode_name = os.path.basename(csv_path)
        # title = f'Reward predictions vs CSV ({episode_name})\nModel: {model_name} | Using actions: {act_col}'
        title = None
        plot_save = None
        if save_path is not None:
            plot_save = save_path
            if not os.path.isabs(plot_save):
                plot_save = _resolve_path(plot_save)
        plot_reward_comparison(pred, gt, intervened_mask, title=title, save_path=plot_save, show=(plot_save is None))

    if return_mask:
        return pred, gt, intervened_mask
    return pred, gt


if __name__ == '__main__':
    # Defaults matching the issue description
    try:
        ep_id: int = 4  # Change to [0, 1, 2, 3, 4] to test different episodes

        # Plot and save a figure next to the CSV for convenience
        csv_rel = os.path.join('evaluations', 'hitl_demo', f'medium_hitlTrue_lossNone_episode{ep_id:03d}.csv')

        # fig_rel = os.path.join('evaluations', 'hitl_demo', f'medium_hitlTrue_lossNone_episode{ep_id:03d}_plot.png')
        fig_rel = os.path.join('evaluations', 'hitl_demo', f'medium_hitlTrue_lossNone_episode{ep_id:03d}_plot.pdf')

        compare_caper_reward_on_episode(
            model_dir='/home/edison/Research/omnisafe_zjy/examples/runs/FOCOPS_CADE-{medium}/seed-042-2025-08-21-15-47-02',
            model_name='sim-episode-004-hitl-False-loss-CAPER.pt',
            episode_csv_relpath=csv_rel,
            use_agent_actions=False,
            device='cpu',
            plot=True,
            save_path=fig_rel,
        )
        print(f'Plot saved to: {fig_rel}')
    except Exception as e:
        print(f'Error during comparison: {e}')
