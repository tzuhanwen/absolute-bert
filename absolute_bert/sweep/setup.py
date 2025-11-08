import argparse
import logging
import os
from typing import Any

from omegaconf import OmegaConf
from absolute_bert.utils import init_logging
from .config import ExperimentUnresolved

logger = logging.getLogger(__name__)


def parse_cli_unknown_args(xs: list[str]) -> dict[str, str]:
    logger.debug(f"start of parse_cli_unknown_args, {xs=}")
    d = {}
    key = None
    for x in xs:
        if x.startswith("--"):
            if "=" in x:
                key, value = x[2:].split("=")
                d[key] = value
                continue
            key = x[2:]
        else:
            if key is None:
                raise ValueError(f"Unexpected CLI value without key: {x}")
            if d.get(key, None) is not None:
                raise ValueError(f"key `{key}` already set to `{x}`")
            
            d[key] = x
            key = None
    return d

def get_config() -> ExperimentUnresolved:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Optional YAML config override")
    args, unknown = parser.parse_known_args()
    options = parse_cli_unknown_args(unknown)
    logger.debug(f"cli options parsed: `{options=}`")
    cli_config = OmegaConf.from_dotlist([f"{k}={v}" for k, v in options.items()])
    logger.info(f"configs from cli: `{cli_config=}`")

    # 試著讀 config.yaml；沒指定或失敗就給空 config
    user_config = (
        OmegaConf.load(args.config)
        if args.config and os.path.exists(args.config)
        else OmegaConf.create()
    )

    # 最終設定：default < config.yaml < CLI
    omega_config = OmegaConf.merge(user_config, cli_config)
    config_dict = OmegaConf.to_container(omega_config, resolve=True)
    logger.info(f"parsing configs: {config_dict=}")

    config_unresolved = ExperimentUnresolved.from_dict(config_dict)
    return config_unresolved


def main():
    init_logging()
    config = get_config()
    print(config.to_dict())

if __name__ == "__main__":
    main()