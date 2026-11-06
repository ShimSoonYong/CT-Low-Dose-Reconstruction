#!/bin/bash

models=("U_Net" "R2U_Net" "R2AttU_Net" "AttU_Net" "NestedUNet")
seeds=(42)

augs=(
  "none"
  "RandAffined"
  "Affine_Noise_Combo"
  "Rot90_Scale_Combo"
)

# Grid Search 반복문 실행
for seed in "${seeds[@]}"; do
  for model in "${models[@]}"; do
    for aug in "${augs[@]}"; do

      docker container rm -f "$model" 2>/dev/null

      echo "[*] Running Model: $model | Seed: $seed | Aug: $aug"

      if [ "$aug" == "none" ]; then
        docker run --gpus all --rm -d --ipc=host \
          -v "$(pwd)":/workspace \
          --name "$model" \
          ai4mia \
          bash -c \
          "python -m src.main \
                    --model '$model' \
                    --epochs 500 \
                    --lr 2e-5 \
                    --grad_norm 0.1 \
                    --patience 1 \
                    --seed '$seed'"
      else
        docker run --gpus all --rm -d --ipc=host \
          -v "$(pwd)":/workspace \
          --name "$model" \
          ai4mia \
          bash -c \
          "python -m src.main \
                    --model '$model' \
                    --epochs 500 \
                    --lr 2e-5 \
                    --grad_norm 0.1 \
                    --augment '$aug' \
                    --patience 1 \
                    --seed '$seed'"
      fi

      docker wait "$model"
      wait
    done
  done
done
