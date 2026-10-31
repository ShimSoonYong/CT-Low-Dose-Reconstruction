#!/bin/bash

models=("U_Net" "R2U_Net" "R2AttU_Net" "AttU_Net" "NestedUNet")
seeds=(42 123 554 9543 329102412049)

augs=(
    "none"
    "RandAffined"
    "RandRot90d"
    "RandFlipd_0"
    "RandFlipd_1"
    "RandScaleIntensityd"
    "RandShiftIntensityd"
    "RandGaussianNoised"
    "RandGaussianSmoothd"
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
                    --lr 1e-5 \
                    --patience 10 \
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
                    --lr 1e-5 \
                    --augment '$aug' \
                    --patience 10 \
                    --seed '$seed'" 
            fi

            docker wait "$model"
            wait
        done
    done
done