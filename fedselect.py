# Importing Libraries
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, OrderedDict, Tuple, Optional, Any

# Custom Libraries
from utils.options import lth_args_parser
from utils.train_utils import prepare_dataloaders, get_data
from pflopt.optimizers import MaskLocalAltSGD, local_alt
from lottery_ticket import init_mask_zeros, delta_update
from broadcast import (
    broadcast_server_to_client_initialization,
    div_server_weights,
    add_masks,
    add_server_weights,
)
import random
from torchvision.models import resnet18


def evaluate(
    model: nn.Module, ldr_test: torch.utils.data.DataLoader, args: Any
) -> float:
    """Evaluate model accuracy on test data loader.

    Args:
        model: Neural network model to evaluate
        ldr_test: Test data loader
        args: Arguments containing device info

    Returns:
        float: Average accuracy on test set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_accuracy = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(ldr_test):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).sum().item() / len(data)
            average_accuracy += acc
        average_accuracy /= len(ldr_test)
    return average_accuracy


def train_personalized(
    model: nn.Module,
    ldr_train: torch.utils.data.DataLoader,
    mask: OrderedDict,
    args: Any,
    initialization: Optional[OrderedDict] = None,
    verbose: bool = False,
    eval: bool = True,
) -> Tuple[nn.Module, float]:
    """Train model with personalized local alternating optimization.

    Args:
        model: Neural network model to train
        ldr_train: Training data loader
        mask: Binary mask for parameters
        args: Training arguments
        initialization: Optional initial model state
        verbose: Whether to print training progress
        eval: Whether to evaluate during training

    Returns:
        Tuple containing:
            - Trained model
            - Final training loss
    """
    if initialization is not None:
        model.load_state_dict(initialization)
    optimizer = MaskLocalAltSGD(model.parameters(), mask, lr=args.lr)
    epochs = args.la_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    with tqdm(total=epochs) as pbar:
        for i in range(epochs):
            train_loss = local_alt(
                model,
                criterion,
                optimizer,
                ldr_train,
                device,
                clip_grad_norm=args.clipgradnorm,
            )
            if verbose:
                print(f"Epoch: {i} \tLoss: {train_loss}")
            pbar.update(1)
            pbar.set_postfix({"Loss": train_loss})
    return model, train_loss


def fedselect_algorithm(
    model: nn.Module,
    args: Any,
    dataset_train: torch.utils.data.Dataset,
    dataset_test: torch.utils.data.Dataset,
    dict_users_train: Dict[int, np.ndarray],
    dict_users_test: Dict[int, np.ndarray],
    labels: np.ndarray,
    idxs_users: List[int],
) -> Dict[str, Any]:
    """Main FedSelect federated learning algorithm.

    Args:
        model: Neural network model
        args: Training arguments
        dataset_train: Training dataset
        dataset_test: Test dataset
        dict_users_train: Mapping of users to training data indices
        dict_users_test: Mapping of users to test data indices
        labels: Data labels
        idxs_users: List of user indices

    Returns:
        Dict containing:
            - client_accuracies: Accuracy history for each client
            - labels: Data labels
            - client_masks: Final client masks
            - args: Training arguments
            - cross_client_acc: Cross-client accuracy matrix
            - lth_convergence: Lottery ticket convergence history
    """
    # initialize model
    initial_state_dict = copy.deepcopy(model.state_dict())
    com_rounds = args.com_rounds
    # initialize server
    client_accuracies = [{i: 0 for i in idxs_users} for _ in range(com_rounds)]
    client_state_dicts = {i: copy.deepcopy(initial_state_dict) for i in idxs_users}
    client_state_dict_prev = {i: copy.deepcopy(initial_state_dict) for i in idxs_users}
    client_masks = {i: None for i in idxs_users}
    client_masks_prev = {i: init_mask_zeros(model) for i in idxs_users}
    server_accumulate_mask = OrderedDict()
    server_weights = OrderedDict()
    lth_iters = args.lth_epoch_iters
    prune_rate = args.prune_percent / 100
    prune_target = args.prune_target / 100
    lottery_ticket_convergence = []
    # Begin FL
    for round_num in range(com_rounds):
        round_loss = 0
        for i in idxs_users:
            # initialize model
            model.load_state_dict(client_state_dicts[i])
            # get data
            ldr_train, _ = prepare_dataloaders(
                dataset_train,
                dict_users_train[i],
                dataset_test,
                dict_users_test[i],
                args,
            )
            # Update LTN_i on local data
            client_mask = client_masks_prev.get(i)
            # Update u_i parameters on local data
            # 0s are global parameters, 1s are local parameters
            client_model, loss = train_personalized(model, ldr_train, client_mask, args)
            round_loss += loss
            # Send u_i update to server
            if round_num < com_rounds - 1:
                server_accumulate_mask = add_masks(server_accumulate_mask, client_mask)
                server_weights = add_server_weights(
                    server_weights, client_model.state_dict(), client_mask
                )
            client_state_dicts[i] = copy.deepcopy(client_model.state_dict())
            client_masks[i] = copy.deepcopy(client_mask)

            if round_num % lth_iters == 0 and round_num != 0:
                client_mask = delta_update(
                    prune_rate,
                    client_state_dicts[i],
                    client_state_dict_prev[i],
                    client_masks_prev[i],
                    bound=prune_target,
                    invert=True,
                )
                client_state_dict_prev[i] = copy.deepcopy(client_state_dicts[i])
                client_masks_prev[i] = copy.deepcopy(client_mask)
        round_loss /= len(idxs_users)
        cross_client_acc = cross_client_eval(
            model,
            client_state_dicts,
            dataset_train,
            dataset_test,
            dict_users_train,
            dict_users_test,
            args,
        )

        accs = torch.diag(cross_client_acc)
        for i in range(len(accs)):
            client_accuracies[round_num][i] = accs[i]
        print("Client Accs: ", accs, " | Mean: ", accs.mean())

        if round_num < com_rounds - 1:
            # Server averages u_i
            server_weights = div_server_weights(server_weights, server_accumulate_mask)
            # Server broadcasts non lottery ticket parameters u_i to every device
            for i in idxs_users:
                client_state_dicts[i] = broadcast_server_to_client_initialization(
                    server_weights, client_masks[i], client_state_dicts[i]
                )
            server_accumulate_mask = OrderedDict()
            server_weights = OrderedDict()

    cross_client_acc = cross_client_eval(
        model,
        client_state_dicts,
        dataset_train,
        dataset_test,
        dict_users_train,
        dict_users_test,
        args,
        no_cross=False,
    )

    out_dict = {
        "client_accuracies": client_accuracies,
        "labels": labels,
        "client_masks": client_masks,
        "args": args,
        "cross_client_acc": cross_client_acc,
        "lth_convergence": lottery_ticket_convergence,
    }

    return out_dict


def cross_client_eval(
    model: nn.Module,
    client_state_dicts: Dict[int, OrderedDict],
    dataset_train: torch.utils.data.Dataset,
    dataset_test: torch.utils.data.Dataset,
    dict_users_train: Dict[int, np.ndarray],
    dict_users_test: Dict[int, np.ndarray],
    args: Any,
    no_cross: bool = True,
) -> torch.Tensor:
    """Evaluate models across clients.

    Args:
        model: Neural network model
        client_state_dicts: Client model states
        dataset_train: Training dataset
        dataset_test: Test dataset
        dict_users_train: Mapping of users to training data indices
        dict_users_test: Mapping of users to test data indices
        args: Evaluation arguments
        no_cross: Whether to only evaluate on own data

    Returns:
        torch.Tensor: Matrix of cross-client accuracies
    """
    cross_client_acc_matrix = torch.zeros(
        (len(client_state_dicts), len(client_state_dicts))
    )
    idx_users = list(client_state_dicts.keys())
    for _i, i in enumerate(idx_users):
        model.load_state_dict(client_state_dicts[i])
        for _j, j in enumerate(idx_users):
            if no_cross:
                if i != j:
                    continue
            # eval model i on data from client j
            _, ldr_test = prepare_dataloaders(
                dataset_train,
                dict_users_train[j],
                dataset_test,
                dict_users_test[j],
                args,
            )
            acc = evaluate(model, ldr_test, args)
            cross_client_acc_matrix[_i, _j] = acc
    return cross_client_acc_matrix


def get_cross_correlation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Get cross correlation between two tensors using F.conv2d.

    Args:
        A: First tensor
        B: Second tensor

    Returns:
        torch.Tensor: Cross correlation result
    """
    # Normalize A
    A = A.cuda() if torch.cuda.is_available() else A
    B = B.cuda() if torch.cuda.is_available() else B
    A = A.unsqueeze(0).unsqueeze(0)
    B = B.unsqueeze(0).unsqueeze(0)
    A = A / (A.max() - A.min()) if A.max() - A.min() != 0 else A
    B = B / (B.max() - B.min()) if B.max() - B.min() != 0 else B
    return F.conv2d(A, B)


def run_base_experiment(model: nn.Module, args: Any) -> None:
    """Run base federated learning experiment.

    Args:
        model: Neural network model
        args: Experiment arguments
    """
    dataset_train, dataset_test, dict_users_train, dict_users_test, labels = get_data(
        args
    )
    idxs_users = np.arange(args.num_users * args.frac)
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    idxs_users = [int(i) for i in idxs_users]
    fedselect_algorithm(
        model,
        args,
        dataset_train,
        dataset_test,
        dict_users_train,
        dict_users_test,
        labels,
        idxs_users,
    )


def load_model(args: Any) -> nn.Module:
    """Load and initialize model.

    Args:
        args: Model arguments

    Returns:
        nn.Module: Initialized model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    model = resnet18(pretrained=args.pretrained_init)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_classes)
    model = model.to(device)
    return model.to(device)


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # Argument Parser
    args = lth_args_parser()

    # Set the seed
    setup_seed(args.seed)
    model = load_model(args)

    run_base_experiment(model, args)
