# Sweep

## Usage
1. startup a VM with docker installed
1. build "sweep agent" docker image via `scripts/build_sweep_run.sh`
1. start a new sweep process in cli with `wandb sweep sweep/sweep.yaml`
1. pass WANDB_API_KEY and WANDB_SWEEP_ID to start a sweep agent:
   `docker run --gpus all  -e WANDB_API_KEY=your_api_key -e WANDB_SWEEP_ID=<entity_name>/<project_name>/<sweep_id> --rm sweep-run:latest `
1. if the repo is fixed but not affect the image to run successfully, 
   start the container with command `scripts/sweep_run-pull_and_run.sh`:
   `docker run --gpus all  -e WANDB_API_KEY=your_api_key -e WANDB_SWEEP_ID=<entity_name>/<project_name>/<sweep_id> --rm sweep-run:latest scripts/sweep_run-pull_and_run.sh`