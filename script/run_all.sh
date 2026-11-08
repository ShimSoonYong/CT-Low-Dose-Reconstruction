#!/bin/bash

models=("U_Net_NoRes" "U_Net" "AttU_Net" "NestedUNet" "R2AttU_Net" "R2U_Net")
seeds=(42)

augs=(
  "none"
  "Rot90_Flip_Combo"
)

MAX_EPOCHS="${MAX_EPOCHS:-100}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

has_valid_best_ckpt() {
  local ckpt_path="$1"
  if [ ! -f "$ckpt_path" ]; then
    return 1
  fi
  if [ -x ".venv/bin/python" ]; then
    CUDA_VISIBLE_DEVICES="" .venv/bin/python - "$ckpt_path" <<'PY' >/dev/null 2>&1
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
ok = bool(sd) and all(torch.isfinite(t).all().item() for t in sd.values() if isinstance(t, torch.Tensor) and t.is_floating_point())
bs = ckpt.get("best_score", None)
if bs is not None and bs <= 0:
    ok = False
sys.exit(0 if ok else 1)
PY
    return $?
  fi
  return 0
}

get_batch_size() {
  local m="$1"
  case "$m" in
  NestedUNet | R2U_Net | R2AttU_Net)
    echo 1
    ;;
  *)
    echo 4
    ;;
  esac
}

# Grid Search 반복문 실행
for seed in "${seeds[@]}"; do
  for aug in "${augs[@]}"; do
    for model in "${models[@]}"; do

      # U_Net_NoRes는 Global Residual 제거 효과를 대조하기 위한 전용 Ablation 베이스라인이므로 none만 수행
      if [ "$model" == "U_Net_NoRes" ] && [ "$aug" != "none" ]; then
        continue
      fi

      run_dir="log/${model}_${aug}"
      if [ "$SKIP_COMPLETED" == "1" ] && has_valid_best_ckpt "${run_dir}/best_model.pth"; then
        echo "[=] Skipping already converged run: ${run_dir} (set SKIP_COMPLETED=0 to force rerun)"
        continue
      fi

      batch_size=$(get_batch_size "$model")

      docker container rm -f "$model" 2>/dev/null

      echo "[*] Running Model: $model | Seed: $seed | Aug: $aug | BatchSize: $batch_size | MaxEpochs: $MAX_EPOCHS"

      if [ "$aug" == "none" ]; then
        docker run --gpus all --rm -d --ipc=host \
          -v "$(pwd)":/workspace \
          --name "$model" \
          ai4mia \
          bash -c \
          "python -m src.main \
                    --model '$model' \
                    --batch_size '$batch_size' \
                    --epochs '$MAX_EPOCHS' \
                    --lr 2e-5 \
                    --grad_norm 1.0 \
                    --patience 2 \
                    --seed '$seed'"
      else
        docker run --gpus all --rm -d --ipc=host \
          -v "$(pwd)":/workspace \
          --name "$model" \
          ai4mia \
          bash -c \
          "python -m src.main \
                    --model '$model' \
                    --batch_size '$batch_size' \
                    --epochs '$MAX_EPOCHS' \
                    --lr 2e-5 \
                    --grad_norm 1.0 \
                    --augment '$aug' \
                    --patience 2 \
                    --seed '$seed'"
      fi

      docker wait "$model"
      wait
    done
  done
done
