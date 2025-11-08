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

if [ -n "$ABSOLUTE_BERT_BRANCH_NAME" ]; then
  # env 有值 → 執行這個 block
  echo "[INFO] switching branch to: $ABSOLUTE_BERT_BRANCH_NAME"
  exec git switch -f "$ABSOLUTE_BERT_BRANCH_NAME"
fi

echo "[INFO] perform git pull"
exec git pull -f

exec wandb agent "$WANDB_SWEEP_ID" --count 1