#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

if [ -z "${WANDB_API_KEY}" ]; then
    echo "[ERROR] WANDB_API_KEY is not set."
    exit 1
fi

if [ -z "$WANDB_SWEEP_ID" ]; then
  echo "[ERROR] WANDB_SWEEP_ID not set"
  exit 1
fi

exec wandb agent "$WANDB_SWEEP_ID" --count 1