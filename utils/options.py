import argparse


def lth_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.05, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--lth_epoch_iters", default=3, type=int)
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
    )
    parser.add_argument(
        "--arch_type",
        default="resnet18",
        type=str,
    )
    parser.add_argument(
        "--setting",
        default="",
        type=str,
    )
    parser.add_argument(
        "--prune_percent", default=25, type=float, help="Pruning percent"
    )
    parser.add_argument("--prune_target", default=80, type=int, help="Pruning target")
    parser.add_argument(
        "--com_rounds", type=int, default=4, help="rounds of fedavg training"
    )
    parser.add_argument(
        "--la_epochs",
        type=int,
        default=15,
        help="rounds of training for local alt optimization",
    )
    parser.add_argument("--iid", action="store_true", help="whether i.i.d or not")
    parser.add_argument("--num_users", type=int, default=100, help="number of users: K")
    parser.add_argument(
        "--shard_per_user", type=int, default=2, help="classes per user"
    )
    parser.add_argument("--local_bs", type=int, default=32, help="local batch size: B")
    parser.add_argument(
        "--frac", type=float, default=0.1, help="the fraction of clients: C"
    )
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--model", type=str, default="mlp", help="model name")
    parser.add_argument("--bs", type=int, default=128, help="test batch size")
    parser.add_argument("--lth_freq", type=int, default=1, help="frequency of lth")
    parser.add_argument("--pretrained_init", action="store_true")
    parser.add_argument("--rewind", action="store_true")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--clipgradnorm", action="store_true")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--exp_name", type=str, default="prune_rate_vary")
    parser.add_argument(
        "--server_data_ratio",
        type=float,
        default=0.0,
        help="The percentage of data that servers also have across data of all clients.",
    )

    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

    args = parser.parse_args()
    return args
