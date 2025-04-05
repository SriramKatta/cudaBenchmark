#!/bin/bash -l
#SBATCH -J GPU_Bandwidths
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100 -C a100_40
#SBATCH --time=00:59:00
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH --error=./SLURM_ERR_FILES/%j_%x.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load cuda nvhpc cmake
#cmake -S . -B build
cmake --build build -j

dirname="RESULTS"
[ ! -d "$dirname" ] && mkdir -p "$dirname"

fname="$dirname/benchmark_$SLURM_JOB_ID"

echo "# datasize H2DBW D2HBW kernelBW" > $fname

for i in $(seq 1 111)
do
    ./executable/STREAM_BENCHMARK $i \
    | tee >(awk '{print $3, $8, $13, $18}' \
    >> $fname)
done

