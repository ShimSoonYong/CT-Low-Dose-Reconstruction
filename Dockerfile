FROM nvcr.io/nvidia/pytorch:26.10-py3

RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    tmux \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir pytest monai[all] opencv-python-headless
