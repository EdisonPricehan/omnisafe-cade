# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the math utils."""

from __future__ import annotations

from typing import Any, Callable, Tuple, Optional, List, Sequence

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Normal, TanhTransform, TransformedDistribution, constraints


class SlidingWindowFilter:
    def __init__(self, window_size: int = 10):
        assert window_size >= 0, f'Sliding window length should be non-negative, given {window_size}.'

        self.window_size: int = window_size
        self.data: list[float] = []

    def mean(self, cur_value: Optional[float] = None) -> float:
        if cur_value is not None:
            self.data.append(cur_value)

            if self.window_size != 0 and len(self.data) > self.window_size:
                self.data.pop(0)

        return sum(self.data) / len(self.data)


def l1_loss(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(mask1, mask2)


def mse_loss(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(mask1, mask2)


def iou(mask1: torch.Tensor, mask2: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    # Check dimension
    if mask1.dim() >= 4:
        assert mask1.shape[1] == 1, f'Channel dim should be 1, {mask1.shape=}'
        mask1 = mask1.squeeze(1)  # Remove the channel dimension
    elif mask1.dim() == 3:
        assert mask1.shape[0] == 1, f'Channel dim should be 1, {mask1.shape=}'
    elif mask1.dim() == 1:
        mask1 = mask1.view(1, -1)  # Treat as a single flat mask

    if mask2.dim() >= 4:
        assert mask2.shape[1] == 1, f'Channel dim should be 1, {mask2.shape=}'
        mask2 = mask2.squeeze(1)  # Remove the channel dimension
    elif mask2.dim() == 3:
        assert mask2.shape[0] == 1, f'Channel dim should be 1, {mask2.shape=}'
    elif mask2.dim() == 1:
        mask2 = mask2.view(1, -1)  # Treat as a single flat mask

    # Ensure masks are boolean tensors
    mask1 = mask1 > threshold
    mask2 = mask2 > threshold

    # Flatten masks for IoU calculation if they aren't already flat
    mask1_flat = mask1.view(mask1.size(0), -1) if mask1.dim() > 1 else mask1
    mask2_flat = mask2.view(mask2.size(0), -1) if mask2.dim() > 1 else mask2

    # Calculate intersection and union
    intersection = torch.sum(mask1_flat & mask2_flat, dim=1).float()
    union = torch.sum(mask1_flat | mask2_flat, dim=1).float()

    # Avoid division by zero
    iou = intersection / (union + 1e-6)

    return iou


def soft_iou_loss(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the differentiable IoU (soft IoU or Jaccard loss) between predicted mask and true mask.

    Args:
        pred_mask (torch.Tensor): Predicted mask with values between 0 and 1 (probabilities).
        true_mask (torch.Tensor): Ground truth binary mask (0 or 1).
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: IoU loss (1 - IoU)
    """
    # Flatten the masks to treat each pixel as an individual element
    pred_mask_flat = pred_mask.view(pred_mask.size(0), -1)
    true_mask_flat = true_mask.view(true_mask.size(0), -1)

    # Compute the intersection and union (using soft masks for the predicted mask)
    intersection = torch.sum(pred_mask_flat * true_mask_flat, dim=1)
    union = torch.sum(pred_mask_flat + true_mask_flat, dim=1) - intersection

    # Compute the IoU
    iou_value = intersection / (union + eps)

    # IoU loss is 1 - IoU
    return 1 - iou_value.mean()


def get_dist_mean_std(dist: Distribution) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dist, Categorical):
        probs = dist.probs
        categories = torch.arange(probs.size(-1)).float()
        mean = torch.sum(probs * categories, dim=-1)  # Shape: (N,)
        mean_squared = torch.sum(probs * categories ** 2, dim=-1)
        variance = mean_squared - mean ** 2  # Shape: (N,)
        stddev = torch.sqrt(variance)  # Shape: (N,)
        return torch.mean(mean), torch.mean(stddev)
    else:
        raise NotImplementedError


def get_multi_dist_mean_std(dists: List[Distribution]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    assert len(dists) > 0, f'Received empty distributions!'

    means = []
    stddevs = []
    for dist in dists:
        mean, stddev = get_dist_mean_std(dist)
        means.append(mean)
        stddevs.append(stddev)

    return means, stddevs


def kld_multi_categorical(
    dist_p: List[Categorical],
    dist_q: List[Categorical],
) -> torch.Tensor:
    assert len(dist_p) == len(dist_q), f'Two distributions size do not match!'

    klds = []
    for dp, dq in zip(dist_p, dist_q):
        kld = torch.distributions.kl.kl_divergence(dp, dq).sum(-1, keepdim=True).mean()
        klds.append(kld)

    return torch.mean(torch.stack(klds), dim=0)


def kld_multi_categorical_flattened(
    dist_p: List[Categorical],     # new/current policy heads
    dist_q: List[Categorical],     # old policy heads (same minibatch)
    no_op_idx: Sequence[int] = (1, 1, 1, 1),  # for [3,3,3,3] or [3,3,2,3]
    non_noop_idx: Sequence[Sequence[int]] = ([0,2],[0,2],[0],[0,2]),  # for [3,3,2,3]
    eps: float = 1e-8,
    direction: str = "new||old",   # or "old||new" — be consistent everywhere
) -> torch.Tensor:
    """
    KL over the executed-action distribution induced by 'first non–no-op wins'.

    Outcomes = [all_noop] + concat_i [ (i, d) for d in non_noop_idx[i] ].
    For [3,3,2,3] with no_op=1 and axis2 having only forward (0), outcomes = 1+(2+2+1+2)=8.
    """

    assert len(dist_p) == len(dist_q) == len(no_op_idx) == len(non_noop_idx)
    A = len(dist_p)

    def _probs2d(d: Categorical) -> torch.Tensor:
        p = d.probs
        return p.reshape(-1, p.size(-1))  # [N, n_classes]

    # Build behavior probs b(s) ∈ R^{N × (1 + Σ_i |non_noop_idx[i]|)}
    def _behavior(dists: List[Categorical]) -> torch.Tensor:
        N = _probs2d(dists[0]).size(0)
        device = _probs2d(dists[0]).device

        # π_j(no-op)
        p_noop_cols = []
        for j, dj in enumerate(dists):
            pj = _probs2d(dj)
            assert 0 <= no_op_idx[j] < pj.size(1), f"no_op_idx[{j}] out of range"
            p_noop_cols.append(pj[:, no_op_idx[j]])
        p_noop = torch.stack(p_noop_cols, dim=1)  # [N, A]

        # prefix[i] = ∏_{k<i} π_k(no-op), with prefix[0] = 1
        prefix = []
        run = torch.ones(N, device=device)
        for i in range(A):
            prefix.append(run)
            run = run * p_noop[:, i]
        prefix = torch.stack(prefix, dim=1)  # [N, A]

        # Assemble outcomes
        comps = [p_noop.prod(dim=1)]  # all_noop, shape [N]
        for i, di in enumerate(dists):
            pi = _probs2d(di)
            # sanity: all non-no-op indices valid for this head
            for k in non_noop_idx[i]:
                assert 0 <= k < pi.size(1) and k != no_op_idx[i], \
                    f"non_noop_idx[{i}] contains invalid class {k}"
                comps.append(prefix[:, i] * pi[:, k])  # [N]
        b = torch.stack(comps, dim=1).clamp_min(eps)     # [N, 1+Σ|non_noop|]
        return b

    b_new = _behavior(dist_p)          # gradients flow through new
    with torch.no_grad():
        b_old = _behavior(dist_q)      # old is a constant target

    if direction == "new||old":
        kl_per = (b_new * (b_new.log() - b_old.log())).sum(dim=1)  # KL(new||old)
    elif direction == "old||new":
        kl_per = (b_old * (b_old.log() - b_new.log())).sum(dim=1)  # KL(old||new)
    else:
        raise ValueError("direction must be 'new||old' or 'old||new'")

    return kl_per.mean()


def logits_from_multi_categorical(
    dists: List[Categorical],
) -> torch.Tensor:
    """
    Get the logits from a list of Categorical distributions.

    :param dists: List of Categorical distributions.
    :return: A tensor of logits concatenated from the distributions.
    """
    assert len(dists) > 0, f'List of distributions is empty!'

    # For a MultiDiscrete action space with shape [3,3,3,3], the logits will be of shape (seq, 3+3+3+3=12)
    # If the backward movement is blocked, action space is [3,3,2,3], the logits will be of shape (seq, 3+3+2+3=11)
    logits = torch.cat([dist.logits for dist in dists], dim=-1)

    return logits


def get_transpose(tensor: torch.Tensor) -> torch.Tensor:
    """Transpose the last two dimensions of a tensor.

    Examples:
        >>> tensor = torch.rand(2, 3)
        >>> get_transpose(tensor).shape
        torch.Size([3, 2])

    Args:
        tensor(torch.Tensor): The tensor to transpose.

    Returns:
        Transposed tensor.
    """
    return tensor.transpose(dim0=-2, dim1=-1)


def get_diagonal(tensor: torch.Tensor) -> torch.Tensor:
    """Get the diagonal of the last two dimensions of a tensor.

    Examples:
        >>> tensor = torch.rand(3, 3)
        >>> get_diagonal(tensor).shape
        torch.Size([1, 3])

    Args:
        tensor (torch.Tensor): The tensor to get the diagonal from.

    Returns:
        Diagonal part of the tensor.
    """
    return tensor.diagonal(dim1=-2, dim2=-1).sum(-1)


def discount_cumsum(vector_x: torch.Tensor, discount: float) -> torch.Tensor:
    """Compute the discounted cumulative sum of vectors.

    Examples:
        >>> vector_x = torch.arange(1, 5)
        >>> vector_x
        tensor([1, 2, 3, 4])
        >>> discount_cumsum(vector_x, 0.9)
        tensor([8.15, 5.23, 2.80, 1.00])

    Args:
        vector_x (torch.Tensor): A sequence of shape (B, T).
        discount (float): The discount factor.

    Returns:
        The discounted cumulative sum of vectors.
    """
    length = vector_x.shape[0]
    vector_x = vector_x.type(torch.float64)
    cumsum = vector_x[-1]
    for idx in reversed(range(length - 1)):
        cumsum = vector_x[idx] + discount * cumsum
        vector_x[idx] = cumsum
    return vector_x


def forward_discount_cumsum(vector_x: torch.Tensor, discount_factor: float) -> torch.Tensor:
    """
    Calculate the discounted return
    Args:
        vector_x: Can be thought as a sequence of rewards
        discount_factor:

    Returns:
        The discounted return
    """
    length = vector_x.shape[0]
    vector_x = vector_x.type(torch.float64)
    cumsum = 0
    discount = 1
    for idx in range(length):
        cumsum += vector_x[idx] * discount
        vector_x[idx] = cumsum
        discount *= discount_factor
    return vector_x


# pylint: disable-next=too-many-locals
def conjugate_gradients(
    fisher_product: Callable[[torch.Tensor], torch.Tensor],
    vector_b: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Implementation of Conjugate gradient algorithm.

    Conjugate gradient algorithm is used to solve the linear system of equations :math:`A x = b`.
    The algorithm is described in detail in the paper `Conjugate Gradient Method`_.

    .. _Conjugate Gradient Method: https://en.wikipedia.org/wiki/Conjugate_gradient_method

    .. note::
        Increasing ``num_steps`` will lead to a more accurate approximation to :math:`A^{-1} b`, and
        possibly slightly-improved performance, but at the cost of slowing things down. Also
        probably don't play with this hyperparameter.

    Args:
        fisher_product (Callable[[torch.Tensor], torch.Tensor]): Fisher information matrix vector
            product.
        vector_b (torch.Tensor): The vector :math:`b` in the equation :math:`A x = b`.
        num_steps (int, optional): The number of steps to run the algorithm for. Defaults to 10.
        residual_tol (float, optional): The tolerance for the residual. Defaults to 1e-10.
        eps (float, optional): A small number to avoid dividing by zero. Defaults to 1e-6.

    Returns:
        The vector x in the equation Ax=b.
    """
    vector_x = torch.zeros_like(vector_b)
    vector_r = vector_b - fisher_product(vector_x)
    vector_p = vector_r.clone()
    rdotr = torch.dot(vector_r, vector_r)

    for _ in range(num_steps):
        vector_z = fisher_product(vector_p)
        alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
        vector_x += alpha * vector_p
        vector_r -= alpha * vector_z
        new_rdotr = torch.dot(vector_r, vector_r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        vector_mu = new_rdotr / (rdotr + eps)
        vector_p = vector_r + vector_mu * vector_p
        rdotr = new_rdotr
    return vector_x


class SafeTanhTransformer(TanhTransform):
    """Safe Tanh Transformer.

    This transformer is used to avoid the error caused by the input of tanh function being too large
    or too small.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform to the input."""
        return torch.clamp(torch.tanh(x), min=-0.999999, max=0.999999)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dtype.is_floating_point:
            eps = torch.finfo(y.dtype).eps
        else:
            raise ValueError('Expected floating point type')
        y = y.clamp(min=-1 + eps, max=1 - eps)
        return super()._inverse(y)


class TanhNormal(TransformedDistribution):  # pylint: disable=abstract-method
    r"""Create a tanh-normal distribution.

    .. math::

        X \sim Normal(loc, scale)

        Y = tanh(X) \sim TanhNormal(loc, scale)

    Examples:
        >>> m = TanhNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # tanh-normal distributed with mean=0 and stddev=1
        tensor([-0.7616])

    Args:
        loc (float or Tensor): The mean of the underlying normal distribution.
        scale (float or Tensor): The standard deviation of the underlying normal distribution.
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        """Initialize an instance of :class:`TanhNormal`."""
        base_dist = Normal(loc, scale)
        super().__init__(base_dist, SafeTanhTransformer())
        self.arg_constraints = {
            'loc': constraints.real,
            'scale': constraints.positive,
        }

    def expand(self, batch_shape: tuple[int, ...], instance: Any | None = None) -> TanhNormal:
        """Expand the distribution."""
        new = self._get_checked_instance(TanhNormal, instance)
        return super().expand(batch_shape, new)

    @property
    def loc(self) -> torch.Tensor:
        """The mean of the normal distribution."""
        return self.base_dist.mean

    @property
    def scale(self) -> torch.Tensor:
        """The standard deviation of the normal distribution."""
        return self.base_dist.stddev

    @property
    def mean(self) -> torch.Tensor:
        """The mean of the tanh normal distribution."""
        return SafeTanhTransformer()(self.base_dist.mean)

    @property
    def stddev(self) -> torch.Tensor:
        """The standard deviation of the tanh normal distribution."""
        return self.base_dist.stddev

    def entropy(self) -> torch.Tensor:
        """The entropy of the tanh normal distribution."""
        return self.base_dist.entropy()

    @property
    def variance(self) -> torch.Tensor:
        """The variance of the tanh normal distribution."""
        return self.base_dist.variance
