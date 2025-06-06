#!/bin/bash -l
#
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH -J stream_GPU_bw100
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

gpgpu=$(nvidia-smi --query-gpu=gpu_name \
    --format=csv |
    tail -n 1 |
    tr '-' ' ' |
    awk '{print $2}')

description="
benchmark of simple single gpu streaming kernel on $gpgpu  
"

echo "$description"

module purge
module load cuda cmake

[ ! -d simdata ] && mkdir -p simdata
resfile=./simdata/${SLURM_JOBID}_resfile
touch $resfile

echo "--------------------------------------------------------------"
echo "varying problem size ":
echo "--------------------------------------------------------------"

executable=./executable/stream_$SLURM_JOB_ID

cp ./executable/stream $executable

val12pow=116
for ((i = 0; i <= $val12pow; i += 4)); do
    echo "$i of $val12pow start"
    numelem=$(echo "12^$i/10^$i" | bc)
    $executable -CV -N $numelem -R 24 -B 3456 -T 256 -S 1 >>$resfile
done

module load python
source ~/plotspace/bin/activate
python parseplotBW.py $resfile
