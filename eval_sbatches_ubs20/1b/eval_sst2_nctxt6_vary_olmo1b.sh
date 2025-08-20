#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearn_eval

ICUL_DIR="/home/c01yomi/CISPA-home/youssef/ICUL"
cd "$ICUL_DIR"

echo "ubs20: eval_sst2_nctxt6_vary_olmo1b.sh: Running evaluation script"

export MASTER_PORT=$(shuf -i 20000-60000 -n 1)

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT eval.py \
  --dataset_name "sst2" \
  --lfm "first-k" \
  --batch_sizes 20 \
  --n_ctxt 6 \
  --ctxt_style "vary" \
  --K_models 1 \
  --model_path "allenai/OLMo-2-0425-1B" \
  --rng_offset ${SLURM_ARRAY_TASK_ID} \
  --config configs/config_eval.json
