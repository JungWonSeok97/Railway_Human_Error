#!/bin/bash
#SBATCH -J enu-ft
#SBATCH -p natStr
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -t 02:30:00
#SBATCH --comment="field=nlp;appl=misc"
#SBATCH -o logs/%x-%j.out
# LoRA fine-tuning of OpenJev on the railway survey answers (4 GPUs, DDP), then candidate scoring.
# Submit from the project root:  cd /scratch/$USER/enu-ai && sbatch ft/finetune.sbatch [extra args]
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit this job from the project root}"
source "$ROOT/cluster/slurm/common.sh"
use_cuda_env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ROOT/hf_cache" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$ROOT/ft/out-$SLURM_JOB_ID"
echo "[ft] node=$(hostname -s) job=$SLURM_JOB_ID out=$OUT ($(date -Is))"
"$ROOT/envs/openjev/bin/torchrun" --standalone --nproc_per_node 4 "$ROOT/ft/finetune_openjev.py" \
  --model "$ROOT/models/openjev" --data "$ROOT/ft" --out "$OUT" "$@"
echo "[ft] done ($(date -Is))"
