FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Stop tzdata from asking timezone questions
ENV DEBIAN_FRONTEND=noninteractive

# Basic dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add Python 3.11 (from deadsnakes PPA)
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
        python3.11 python3.11-dev python3.11-venv \
        python3.11-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as system default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Poetry (no venv mode)
ARG POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir poetry==$POETRY_VERSION && \
    poetry config virtualenvs.create false

WORKDIR /workspace

# Default run (change to your entry point)
CMD ["bash"]
