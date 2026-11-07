#!/usr/bin/env bash
set -euo pipefail

# 1-Click Non-Intrusive Report Builder for AI4MIA Assignment 3
# Defaults to CPU-only evaluation (CUDA_VISIBLE_DEVICES="") so ongoing Docker GPU training is never disturbed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${WORKSPACE_ROOT}/.venv/bin/python"

STUDENT_ID="${STUDENT_ID:-StudentID}"
AUTHOR_NAME="${AUTHOR_NAME:-Name}"
DEVICE="cpu"
FORCE_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --student-id)
      STUDENT_ID="$2"
      shift 2
      ;;
    --name)
      AUTHOR_NAME="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --force)
      FORCE_FLAG="--force"
      shift 1
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: bash report/build_report.sh [--student-id 2026XXXXXX] [--name HongGilDong] [--device cpu|cuda] [--force]"
      exit 1
      ;;
  esac
done

mkdir -p "${SCRIPT_DIR}/generated"
cat > "${SCRIPT_DIR}/generated/author_info.tex" <<EOF
\newcommand{\ReportStudentID}{${STUDENT_ID}}
\newcommand{\ReportAuthorName}{${AUTHOR_NAME}}
EOF

export MLFLOW_DISABLE_AGENT_HINT=1
if [[ "${DEVICE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

echo "[1/4] Rendering IDE syntax-highlighted code snapshots..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/render_code_snapshots.py" ${FORCE_FLAG}

echo "[2/4] Extracting TensorBoard logs & evaluating checkpoints (device=${DEVICE}, mtime-cached)..."
if [[ -n "${FORCE_FLAG}" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/extract_logs_and_eval.py" --device "${DEVICE}" --force-eval
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/extract_logs_and_eval.py" --device "${DEVICE}"
fi

echo "[3/4] Generating [-100, 300] HU figures & LaTeX tables..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_figures_and_tables.py"

echo "[4/4] Compiling Korean PDF report via xelatex..."
pushd "${SCRIPT_DIR}" >/dev/null
xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
popd >/dev/null

SAFE_ID="${STUDENT_ID// /_}"
SAFE_NAME="${AUTHOR_NAME// /_}"
OUT_PDF="${WORKSPACE_ROOT}/${SAFE_ID}_${SAFE_NAME}_Report.pdf"
cp "${SCRIPT_DIR}/main.pdf" "${OUT_PDF}"

echo "[+] Successfully built PDF report: ${OUT_PDF}"
