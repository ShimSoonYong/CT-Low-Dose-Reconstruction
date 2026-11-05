FROM nvcr.io/nvidia/pytorch:26.10-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  curl \
  vim \
  tmux \
  build-essential \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
  pip install --no-cache-dir -r requirements.txt
