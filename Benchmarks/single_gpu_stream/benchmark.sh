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

#needed for sbatch to work
if [ -n "$SLURM_JOB_ID" ]; then
    module load cuda nvhpc cmake
    export http_proxy=http://proxy.nhr.fau.de:80
    export https_proxy=http://proxy.nhr.fau.de:80
fi

cmake -S . -B build
cmake --build build -j

dirname="RESULTS"
[ ! -d "$dirname" ] && mkdir -p "$dirname"

fname="$dirname/${SLURM_JOB_ID}_benchmark"

echo "# datasize H2DBW D2HBW kernelBW" > $fname

for i in $(seq 8 111)
do
    ./executable/STREAM_BENCHMARK $i \
    | tee >(awk '{print $3, $8, $13, $18}' \
    >> $fname)
done

module load python
python BWplotting.py $fname
