#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100 -C a100_40
#SBATCH --time=00:59:00
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH --error=./SLURM_ERR_FILES/%j_%x.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load cuda nvhpc cmake
cmake -S . -B build
cmake --build build -j

echo "# datasize H2DBW D2HBW kernelBW" > benchmark_$SLURM_JOB_ID

for i in $(seq 1 111)
do
    ./executable/STREAM_BENCHMARK $i \
    | tee -a /dev/tty \
    | awk '{print $3, $8, $13, $18}' >> benchmark_$SLURM_JOB_ID
done

