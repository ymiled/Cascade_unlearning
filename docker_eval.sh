#!/bin/bash
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=tmp,gpu,xe8545
#SBATCH -t 3-00:00
#SBATCH --job-name=sst_ubs20_n6
#SBATCH --array=0-9
#SBATCH --output=/home/c01yomi/CISPA-home/youssef/ICUL/slurm_jobs/outputs/eval_sst2_ubs20_n6_%a.out
#SBATCH --error=/home/c01yomi/CISPA-home/youssef/ICUL/slurm_jobs/errors/eval_sst2_ubs20_n6_%a.err
#SBATCH --exclude=xe8545-a100-06,xe8545-a100-21,xe8545-a100-01


JOBDATADIR=/home/c01yomi/CISPA-work/c01yomi/stor_ICUL


srun --container-image=projects.cispa.saarland:5005#c01yomi/test_project:pytorch-unlearn-eval-v3 \
     --container-mounts=${JOBDATADIR}":/tmp" bash /home/c01yomi/CISPA-home/youssef/ICUL/eval_sbatches_ubs20/1b/eval_sst2_nctxt6_vary_olmo1b.sh