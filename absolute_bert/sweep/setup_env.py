import argparse
import os
from omegaconf import OmegaConf

# -------- argparse CLI 定義 --------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Optional YAML config override")
parser.add_argument("--lr", type=float)
parser.add_argument("--warmup_ratio", type=float)
parser.add_argument("--k_temperature", type=float)
parser.add_argument("--masking_probability", type=float)
parser.add_argument("--scheduler", type=str)
parser.add_argument("--job_type", type=str)

args = parser.parse_args()

# -------- config loading with fallback + override --------
default_cfg = OmegaConf.load("configs/default.yaml")

# 試著讀 config.yaml；沒指定或失敗就給空 config
user_cfg = (
    OmegaConf.load(args.config)
    if args.config and os.path.exists(args.config)
    else OmegaConf.create()
)

# 把 argparse 的參數抓出來轉成 config（排除掉 None）
cli_cfg = OmegaConf.create({k: v for k, v in vars(args).items() if v is not None and k != "config"})

# 最終設定：default < config.yaml < CLI
config = OmegaConf.merge(default_cfg, user_cfg, cli_cfg)
