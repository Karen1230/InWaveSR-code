#!/bin/bash

set -e

cuda_devices="1"
basic_log_name="./log/Exp_hfunet_attimnet3_eage1_tpro"
train_datas_str="c307.npz,c308.npz,c322.npz,c329.npz,c332.npz,c340.npz,c350.npz"
eval_datas_str="c324.npz"
eval_dataset=$eval_datas_str
ckpt_name="checkpoint_latest.pth.tar_pdenet_best_psnr.pth.tar"
batch_size_per_gpu=6
eval_tres=256
epochs=100
pseudo_epoch_size=8000
n_samp_pts_per_crop=1024
log_dir=$basic_log_name
downsamp_x=4
lr=0.0008

use_eage_samp=True
use_terrain_pred_proc=True

unet_name="att_unet_fft"
imnet_name="att_imnet3"

terrain_zero=False
fluid_mean=True

#test
train_datas_str=$eval_dataset
epochs=1
pseudo_epoch_size=500
n_samp_pts_per_crop=64
eval_tres=32
log_dir=$basic_log_name"_tem"

# train
ckpt=$log_dir"/"$ckpt_name
save_path=$log_dir"/eval"
# resume=$log_dir$ckpt_name

python train.py --epochs=$epochs --pseudo_epoch_size=$pseudo_epoch_size\
 --n_samp_pts_per_crop=$n_samp_pts_per_crop --cuda_devices=$cuda_devices\
 --unet_name=$unet_name --imnet_name=$imnet_name\
 --log_dir=$log_dir --train_datas_str=$train_datas_str --eval_datas_str=$eval_datas_str\
 --all_eval_tres=$eval_tres\
 --resume=$resume --batch_size_per_gpu=$batch_size_per_gpu --terrain_zero=$terrain_zero\
 --fluid_mean=$fluid_mean --downsamp_x=$downsamp_x --lr=$lr --use_eage_samp=$use_eage_samp\
 --use_terrain_pred_proc=$use_terrain_pred_proc

# eval

python evaluation.py --eval_tres=$eval_tres --cuda_devices=$cuda_devices\
 --ckpt=$ckpt --save_path=$save_path --eval_dataset=$eval_dataset\
 --eval_downsamp_x=$downsamp_x

# viusal

python visual.py --eval_folder=$save_path --eval_downsamp_x=$downsamp_x

# remake floder

echo "finish!"


