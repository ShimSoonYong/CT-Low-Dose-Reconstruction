#!/bin/bash

MAX_EPOCHS="${MAX_EPOCHS:-100}"
docker container rm -f U_Net_NoRes 2>/dev/null

echo "[*] Running Model: U_Net_NoRes | Seed: 42 | Aug: none | MaxEpochs: $MAX_EPOCHS"

docker run --gpus all --rm -d --ipc=host \
  -v "$(pwd)":/workspace \
  --name U_Net_NoRes \
  ai4mia \
  bash -c \
  "python -m src.main \
              --model U_Net_NoRes \
              --epochs '$MAX_EPOCHS' \
              --lr 2e-5 \
              --grad_norm 1.0 \
              --patience 2 \
              --seed 42"

docker wait U_Net_NoRes
wait
