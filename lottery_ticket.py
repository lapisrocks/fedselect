# Importing Libraries
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import types
from collections import OrderedDict
from typing import List, Tuple, Dict, OrderedDict, Optional, Union


def eval_per_layer_sparsity(mask: OrderedDict) -> List[Tuple[str, str, str, float]]:
    """Calculate sparsity statistics for each weight layer in the mask.

    Args:
        mask: OrderedDict containing binary masks for model parameters

    Returns:
        List of tuples containing (num ones, num zeros, layer name, sparsity) for each weight layer
    """
    return [
        (
            f"1: {torch.count_nonzero(mask[name])}",
            f"0: {torch.count_nonzero(1-mask[name])}",
            name,
            (
                torch.count_nonzero(1 - mask[name])
                / (
                    torch.count_nonzero(mask[name])
                    + torch.count_nonzero(1 - mask[name])
                )
            ).item(),
        )
        for name in mask.keys()
        if "weight" in name
    ]


def eval_layer_sparsity(
    mask: OrderedDict, layer_name: str
) -> Tuple[str, str, str, float]:
    """Calculate sparsity statistics for a specific layer in the mask.

    Args:
        mask: OrderedDict containing binary masks for model parameters
        layer_name: Name of layer to analyze

    Returns:
        Tuple containing (num ones, num zeros, layer name, sparsity) for specified layer
    """
    return (
        f"1: {torch.count_nonzero(mask[layer_name])}",
        f"0: {torch.count_nonzero(1-mask[layer_name])}",
        layer_name,
        (
            torch.count_nonzero(1 - mask[layer_name])
            / (
                torch.count_nonzero(mask[layer_name])
                + torch.count_nonzero(1 - mask[layer_name])
            )
        ).item(),
    )


def print_nonzeros(
    model: OrderedDict, verbose: bool = False, invert: bool = False
) -> float:
    """Print statistics about non-zero parameters in model.

    Args:
        model: OrderedDict containing model parameters
        verbose: Whether to print detailed statistics
        invert: Whether to count zeros instead of non-zeros

    Returns:
        Percentage of pruned parameters
    """
    nonzero = total = 0
    for name, p in model.items():
        tensor = p.data.cpu().numpy()
        nz_count = (
            np.count_nonzero(tensor) if not invert else np.count_nonzero(1 - tensor)
        )
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        if verbose:
            print(
                f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}"
            )
    if verbose:
        print(
            f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)"
        )
    return 100 * (total - nonzero) / total


def print_lth_stats(mask: OrderedDict, invert: bool = False) -> None:
    """Print lottery ticket hypothesis statistics about mask sparsity.

    Args:
        mask: OrderedDict containing binary masks
        invert: Whether to invert the sparsity calculation
    """
    current_prune = print_nonzeros(mask, invert=invert)
    print(f"Mask Sparsity: {current_prune:.2f}%")


def _violates_bound(
    mask: torch.Tensor, bound: Optional[float] = None, invert: bool = False
) -> bool:
    """Check if mask sparsity violates specified bound.

    Args:
        mask: Binary mask tensor
        bound: Maximum allowed sparsity
        invert: Whether to invert the sparsity calculation

    Returns:
        True if bound is violated, False otherwise
    """
    if invert:
        return (
            torch.count_nonzero(mask)
            / (torch.count_nonzero(mask) + torch.count_nonzero(1 - mask))
        ).item() > bound
    else:
        return (
            torch.count_nonzero(1 - mask)
            / (torch.count_nonzero(mask) + torch.count_nonzero(1 - mask))
        ).item() > bound


def init_mask(model: nn.Module) -> OrderedDict:
    """Initialize binary mask of ones for model parameters.

    Args:
        model: Neural network model

    Returns:
        OrderedDict containing binary masks initialized to ones
    """
    mask = OrderedDict()
    for name, param in model.named_parameters():
        mask[name] = torch.ones_like(param)
    return mask


def init_mask_zeros(model: nn.Module) -> OrderedDict:
    """Initialize binary mask of zeros for model parameters.

    Args:
        model: Neural network model

    Returns:
        OrderedDict containing binary masks initialized to zeros
    """
    mask = OrderedDict()
    for name, param in model.named_parameters():
        mask[name] = torch.zeros_like(param)
    return mask


def get_mask_from_delta(
    prune_percent: float,
    current_state_dict: OrderedDict,
    prev_state_dict: OrderedDict,
    current_mask: OrderedDict,
    bound: float = 0.80,
    invert: bool = True,
) -> OrderedDict:
    """Generate new mask based on parameter changes between states.

    Args:
        prune_percent: Percentage of parameters to prune
        current_state_dict: Current model state
        prev_state_dict: Previous model state
        current_mask: Current binary mask
        bound: Maximum allowed sparsity
        invert: Whether to invert the pruning logic

    Returns:
        Updated binary mask based on parameter changes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return_mask = copy.deepcopy(current_mask)
    for name, param in current_state_dict.items():
        if "weight" in name:
            if _violates_bound(current_mask[name], bound=bound, invert=invert):
                continue
            tensor = param.data.cpu().numpy()
            compare_tensor = prev_state_dict[name].cpu().numpy()
            delta_tensor = np.abs(tensor - compare_tensor)

            delta_percentile_tensor = (
                delta_tensor[current_mask[name].cpu().numpy() == 1]
                if not invert
                else delta_tensor[current_mask[name].cpu().numpy() == 0]
            )
            sorted_weights = np.sort(np.abs(delta_percentile_tensor))
            if not invert:
                cutoff_index = np.round(prune_percent * sorted_weights.size).astype(int)
                cutoff = sorted_weights[cutoff_index]

                # Convert Tensors to numpy and calculate
                new_mask = np.where(
                    abs(delta_tensor) <= cutoff, 0, return_mask[name].cpu().numpy()
                )
                return_mask[name] = torch.from_numpy(new_mask).to(device)
            else:
                cutoff_index = np.round(
                    (1 - prune_percent) * sorted_weights.size
                ).astype(int)
                cutoff = sorted_weights[cutoff_index]

                # Convert Tensors to numpy and calculate
                new_mask = np.where(
                    abs(delta_tensor) >= cutoff, 1, return_mask[name].cpu().numpy()
                )
                return_mask[name] = torch.from_numpy(new_mask).to(device)
    # print(eval_per_layer_sparsity(return_mask))
    print(eval_layer_sparsity(return_mask, "fc.weight"))
    return return_mask


def delta_update(
    prune_percent: float,
    current_state_dict: OrderedDict,
    prev_state_dict: OrderedDict,
    current_mask: OrderedDict,
    bound: float = 0.80,
    invert: bool = False,
) -> OrderedDict:
    """Update mask based on parameter changes between states.

    Args:
        prune_percent: Percentage of parameters to prune
        current_state_dict: Current model state
        prev_state_dict: Previous model state
        current_mask: Current binary mask
        bound: Maximum allowed sparsity
        invert: Whether to invert the pruning logic

    Returns:
        Updated binary mask
    """
    mask = get_mask_from_delta(
        prune_percent,
        current_state_dict,
        prev_state_dict,
        current_mask,
        bound=bound,
        invert=invert,
    )
    print_lth_stats(mask, invert=invert)
    return mask
