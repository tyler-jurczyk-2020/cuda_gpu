#!/bin/bash
#SBATCH --job-name="Milestone3"
#SBATCH --output="Milestone3.out"
#SBATCH --error="Milestone3.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="scratch,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# srun ./m3 > m3.out
{ time srun ./m3 100 > m31.out; } 2> time1.out
{ time srun ./m3 1000 > m32.out; } 2> time2.out
{ time srun ./m3 10000 > m33.out; } 2> time3.out

# Generate nsys profile information, output will be in profile.out
srun nsys profile --stats=true ./m3 > profile.out

# Generate analysis_file.ncu-rep with a detailed profiling result
ncu --set full -o analysis_file ./m3 > m3_analysis.out
