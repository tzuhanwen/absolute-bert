import argparse
import logging
import os

from omegaconf import OmegaConf
from absolute_bert.utils import init_logging
from .config import ExperimentUnresolved

logger = logging.getLogger(__name__)


def get_config() -> ExperimentUnresolved:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Optional YAML config override")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--k_temperature", type=float)
    parser.add_argument("--masking_probability", type=float)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--job_type", type=str)

    args = parser.parse_args()

    # 試著讀 config.yaml；沒指定或失敗就給空 config
    user_config = (
        OmegaConf.load(args.config)
        if args.config and os.path.exists(args.config)
        else OmegaConf.create()
    )

    # 把 argparse 的參數抓出來轉成 config（排除掉 None）
    cli_config = OmegaConf.create(
        {k: v for k, v in vars(args).items() if v is not None and k != "config"}
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