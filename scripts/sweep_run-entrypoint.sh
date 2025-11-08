#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

if [ -z "${WANDB_API_KEY}" ]; then
    echo "[ERROR] WANDB_API_KEY is not set."
    exit 1
fi

if [ -z "${SWEEP_ID}" ]; then
    echo "[ERROR] SWEEP_ID is not set."
    exit 1
fi

echo "[INFO] WANDB ready."
exec python -m src/train.py
