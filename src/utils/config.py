import os
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> DictConfig | ListConfig:
    """
    Load configuration from a YAML file using OmegaConf.

    Parameters
    ----------
    config_path : str
        Path to the config YAML file.

    Returns
    -------
    DictConfig | ListConfig
        Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)
    return config


def merge_cli_args(
    config: DictConfig | ListConfig, cli_args: Dict[str, Any]
) -> DictConfig | ListConfig:
    """
    Merge CLI arguments with loaded config, giving CLI arguments precedence.

    Parameters
    ----------
    config : DictConfig | ListConfig
        Configuration loaded from file.
    cli_args : Dict[str, Any]
        Command line arguments.

    Returns
    -------
    DictConfig | ListConfig
        Merged configuration.
    """
    # Filter out None values from CLI args to prevent overriding config values with None
    filtered_cli_args = {k: v for k, v in cli_args.items() if v is not None}
    cli_config = OmegaConf.create(filtered_cli_args)
    merged_config = OmegaConf.merge(config, cli_config)
    return merged_config
