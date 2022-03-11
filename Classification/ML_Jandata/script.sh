#!/bin/bash

#SBATCH -p pi_ohern
#SBATCH --job-name=ml
#SBATCH --array=1-36
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=01-00:00:00

sed -n "${SLURM_ARRAY_TASK_ID}p" joblist | /bin/bash
