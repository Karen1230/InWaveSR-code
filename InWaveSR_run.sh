#!/bin/bash

set -e  # Exit immediately if any command fails

# ========== Basic Configuration Parameters ==========
cuda_devices="1"  # CUDA device IDs to use (GPU 1)
basic_log_name="./log/Exp_hfunet_attimnet3_eage1_tpro"  # Base path for experiment logs and outputs
train_datas_str="c307.npz,c308.npz,c322.npz,c329.npz,c332.npz,c340.npz,c350.npz"  # Training dataset files
eval_datas_str="c324.npz"  # Validation dataset files
eval_dataset=$eval_datas_str  # Dataset used for evaluation
ckpt_name="checkpoint_latest.pth.tar_pdenet_best_psnr.pth.tar"  # Model filename
batch_size_per_gpu=6  # Batch size per GPU
eval_tres=256  # Temporal resolution for evaluation
epochs=100  # Total training epochs
pseudo_epoch_size=8000  # Number of spatio-temporal blocks per epoch
n_samp_pts_per_crop=1024  # Number of sampling points per spatio-temporal block
log_dir=$basic_log_name  # Log directory path
downsamp_x=4  # Downsampling factor in x-direction
lr=0.0008  # Learning rate

# ========== Feature Switches ==========
use_eage_samp=True  # Whether to use edge sampling
use_terrain_pred_proc=True  # Whether to use terrain prediction processing

# ========== Model Architecture Configuration ==========
unet_name="att_unet_fft"  #  HF-FEN
imnet_name="att_imnet3"  # Implicit network architecture name (corresponds to PCN in paper)

## For testing only (currently commented out)
#train_datas_str=$eval_dataset
#epochs=1
#pseudo_epoch_size=500
#n_samp_pts_per_crop=64
#eval_tres=32
#log_dir=$basic_log_name"_tem"

# ========== Path Settings ==========
ckpt=$log_dir"/"$ckpt_name  # Full model file path
save_path=$log_dir"/eval"  # Evaluation results save path
# resume=$log_dir$ckpt_name  # Resume training path (use when training is interrupted)

# ========== Training Phase ==========
python train.py --epochs=$epochs --pseudo_epoch_size=$pseudo_epoch_size\
 --n_samp_pts_per_crop=$n_samp_pts_per_crop --cuda_devices=$cuda_devices\
 --unet_name=$unet_name --imnet_name=$imnet_name\
 --log_dir=$log_dir --train_datas_str=$train_datas_str --eval_datas_str=$eval_datas_str\
 --all_eval_tres=$eval_tres\
 --resume=$resume --batch_size_per_gpu=$batch_size_per_gpu\
 --downsamp_x=$downsamp_x --lr=$lr --use_eage_samp=$use_eage_samp\
 --use_terrain_pred_proc=$use_terrain_pred_proc

# ========== Evaluation Phase ==========
python evaluation.py --eval_tres=$eval_tres --cuda_devices=$cuda_devices\
 --ckpt=$ckpt --save_path=$save_path --eval_dataset=$eval_dataset\
 --eval_downsamp_x=$downsamp_x

# ========== Visualization Phase ==========
python visual.py --eval_folder=$save_path --eval_downsamp_x=$downsamp_x

# ========== End Flag ==========
echo "finish!"  # Output completion message