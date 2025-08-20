#!/bin/bash
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=tmp,gpu,xe8545
#SBATCH -t 3-00:00
#SBATCH --job-name=run_ubs20
#SBATCH --output=/home/c01yomi/CISPA-home/youssef/ICUL/slurm_jobs/outputs/run_sst2_ubs20_olmo1b.out
#SBATCH --error=/home/c01yomi/CISPA-home/youssef/ICUL/slurm_jobs/errors/run_sst2_ubs20_olmo1b.err
#SBATCH --exclude=xe8545-a100-06,xe8545-a100-05


JOBDATADIR=/home/c01yomi/CISPA-work/c01yomi/stor_ICUL


srun --container-image=projects.cispa.saarland:5005#c01yomi/test_project:pytorch-unlearn-eval-v3 \
     --container-mounts=${JOBDATADIR}":/tmp" bash /home/c01yomi/CISPA-home/youssef/ICUL/run_sbatches_ubs20/1b/run_sst2_ubs20_olmo1b.sh