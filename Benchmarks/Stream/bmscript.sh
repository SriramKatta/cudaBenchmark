#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100 -C a100_40
#SBATCH --time=00:30:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load cuda
module load nvhpc
module load cmake

mkdir simdata
fname=./simdata/resfile_$SLURM_JOBID
touch $fname

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
cmake -S . -B build/ -DCMAKE_BUILD_TYPE:STRING=Release >/dev/null
cmake --build build/ -j >/dev/null

echo "--------------------------------------------------------------"
echo "varying problem size ":
echo "--------------------------------------------------------------"

val12pow=111
for ((i = 0; i <= $val12pow; i += 1)); do
    echo "$i of $val12pow start"
    numelem=$(echo "12^$i/10^$i" | bc)
    ./executable/stream -CV -N $numelem -R 24 -B 3456 -T 256 -S 15 >> $fname
done

module load python
source ~/plotspace/bin/activate
python parseplotBW.py $fname