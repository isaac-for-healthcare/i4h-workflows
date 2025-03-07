import argparse
import os

from policy_runner.config import get_config
from policy_runner.utils import compute_normalization_stats

from openpi import train
from openpi.training.config import DataConfigFactory


def ensure_norm_stats_exist(config):
    """Ensure normalization statistics exist, computing them if necessary."""
    data_config = config.data
    if isinstance(data_config, DataConfigFactory):
        data_config = data_config.create(config.assets_dirs, config.model)

    output_path = config.assets_dirs / data_config.repo_id
    stats_file = output_path / "norm_stats.json"

    if not os.path.exists(stats_file):
        print(f"Normalization statistics not found at {stats_file}. Computing...")
        compute_normalization_stats(config)
    else:
        print(f"Normalization statistics found at {stats_file}. Skipping computation.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a PI-Zero model")
    parser.add_argument(
        "--config", type=str, default="robotic_ultrasound", help="Configuration name to use for training"
    )
    parser.add_argument(
        "--exp_name", type=str, required=True, help="Name of the experiment for logging and checkpointing"
    )
    parser.add_argument("--repo_id", type=str, default="i4h/robotic_ultrasound", help="Repository ID for the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Get configuration using the provided config name
    config = get_config(name=args.config, repo_id=args.repo_id, exp_name=args.exp_name)
    # Ensure we have normalization stats
    ensure_norm_stats_exist(config)
    # Begin training
    train.main(config)
