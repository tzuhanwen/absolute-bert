#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."   # 自動回 repo root

docker build -f dockerfiles/base_env.dockerfile -t sweep-base:latest .
docker build -f dockerfiles/sweep_run.dockerfile -t sweep-run:latest .
