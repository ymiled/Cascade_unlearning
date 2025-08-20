#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearn_eval

ICUL_DIR="/home/c01yomi/CISPA-home/youssef/ICUL"
cd "$ICUL_DIR"

echo "ubs20: run sst2"

export MASTER_PORT=$(shuf -i 10000-65535 -n 1) 

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT run.py --dataset_name "sst2" --dataset_size 25000 --config configs/config_run_ubs20_olmo1b.json
