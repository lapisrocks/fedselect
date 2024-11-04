import torch
from typing import OrderedDict


def broadcast_server_to_client_initialization(
    server_weights: OrderedDict[str, torch.Tensor],
    mask: OrderedDict[str, torch.Tensor],
    client_initialization: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Broadcasts server weights to client initialization for non-masked parameters.

    Args:
        server_weights: Server model state dict
        mask: Binary mask indicating which parameters are local (1) vs global (0)
        client_initialization: Client model state dict to update

    Returns:
        Updated client model state dict with server weights broadcast to non-masked parameters
    """
    for key in client_initialization.keys():
        # only override client_initialization where mask is non-zero
        if "weight" in key or "bias" in key:
            client_initialization[key][mask[key] == 0] = server_weights[key][
                mask[key] == 0
            ]
    return client_initialization


def div_server_weights(
    server_weights: OrderedDict[str, torch.Tensor],
    server_mask: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Divides server weights by mask values where mask is non-zero.

    Args:
        server_weights: Server model state dict
        server_mask: Mask indicating number of contributions to each parameter

    Returns:
        Server weights normalized by number of contributions
    """
    for key in server_weights.keys():
        # only divide where server_mask is non-zero
        if "weight" in key or "bias" in key:
            server_weights[key][server_mask[key] != 0] /= server_mask[key][
                server_mask[key] != 0
            ]
    return server_weights


def add_masks(
    server_dict: OrderedDict[str, torch.Tensor],
    client_dict: OrderedDict[str, torch.Tensor],
    invert: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Accumulates client masks into server mask dictionary.

    Args:
        server_dict: Server mask accumulator
        client_dict: Client mask to add
        invert: Whether to invert client mask before adding

    Returns:
        Updated server mask accumulator
    """
    for key in client_dict.keys():
        if "weight" in key or "bias" in key:
            if key not in server_dict.keys():
                server_dict[key] = 1 - client_dict[key] if invert else client_dict[key]
            else:
                server_dict[key] += (
                    (1 - client_dict[key]) if invert else client_dict[key]
                )
    return server_dict


def add_server_weights(
    server_weights: OrderedDict[str, torch.Tensor],
    client_weights: OrderedDict[str, torch.Tensor],
    client_mask: OrderedDict[str, torch.Tensor],
    invert: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Accumulates masked client weights into server weights.

    Args:
        server_weights: Server weights accumulator
        client_weights: Client model weights to add
        client_mask: Binary mask indicating which parameters to add
        invert: Whether to invert mask before applying

    Returns:
        Updated server weights accumulator
    """
    for key in client_weights.keys():
        if "weight" in key or "bias" in key:
            mask = 1 - client_mask[key] if invert else client_mask[key]
            if key not in server_weights.keys():
                server_weights[key] = client_weights[key] * mask
            else:
                server_weights[key] += client_weights[key] * mask
    return server_weights
