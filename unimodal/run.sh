#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:2
#SBATCH --account=shrikann_35


module purge
module load nvidia-hpc-sdk
eval "$(conda shell.bash hook)"
conda activate /home1/rajatheb/.conda/envs/astmm2

imagenetpretrain=True
audiosetpretrain=True
lr=1e-5
epoch=15

project_dir=<BASEPATH>/mica-subtitle-aligned-movie-sounds
tr_data=${project_dir}/data/train.json
val_data=${project_dir}/data/val.json
test_data=${project_dir}/data/test.json
label_csv=${project_dir}/data/movie_sounds.csv
num_workers=32
n_class=120
freqm=48
timem=192
mixup=0.5
# corresponding to overlap of 6 for 16*16 patches
fstride=10
tstride=10
batch_size=20
exp_dir=${project_dir}/exp/nclass${n_class}-fm${freqm}-tm${timem}-mix${mixup}-p${audiosetpretrain}-b${batch_size}-lr${lr}-warmup

if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ./run.py --model ${model} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${test_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class ${n_class} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
--audioset_pretrain $audiosetpretrain --num-workers ${num_workers} > ${exp_dir}/log.txt
