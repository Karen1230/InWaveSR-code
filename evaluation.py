import argparse
from collections import defaultdict
import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import shutil
import os
# import our modules

current_dir = os.path.dirname(os.path.realpath(__file__))
# import our modules
import sys
sys.path.append(os.path.join(current_dir, "src"))
sys.path.append(os.path.join(current_dir, "meshfreeflownet"))
sys.path.append(os.path.join(current_dir, "meshfreeflownet/src"))
sys.path.append(os.path.join(current_dir, "octree"))

# from unet3d import UNet3d
# from implicit_net import ImNet
# from pde import PDELayer
from nonlinearities import NONLINEARITIES
from physics import get_rb2_pde_layer
from torch_flow_stats import *
import revised

def frames_to_video(frames_pattern, save_video_to, frame_rate=10, keep_frames=False):
    """Create video from frames.

    frames_pattern: str, glob pattern of frames.
    save_video_to: str, path to save video to.
    keep_frames: bool, whether to keep frames after generating video.
    """
    cmd = ("ffmpeg -framerate {frame_rate} -pattern_type glob -i '{frames_pattern}' "
           "-c:v libx264 -r 30 -pix_fmt yuv420p {save_video_to}"
           .format(frame_rate=frame_rate, frames_pattern=frames_pattern,
                   save_video_to=save_video_to))
    os.system(cmd)
    # print
    print("Saving videos to {}".format(save_video_to))
    # delete frames if keep_frames is not needed
    if not keep_frames:
        frames_dir = os.path.dirname(frames_pattern)
        shutil.rmtree(frames_dir)


def calculate_flow_stats(args, pred, hres, visc=0.0001):
    data = pred
    uw = np.transpose(data[2:4,:,:,1:1+args.eval_zres], (1, 0, 2, 3))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    uw = torch.tensor(uw, device=device).float()
    stats = compute_all_stats(uw[2:,:,:,:], viscosity=visc, description=False)
    s = [stats[..., i].item() for i in range(stats.shape[0])]

    file = open("REPORT___FlowStats_Pred_vs_GroundTruth.txt", "w")

    file.write("***** Pred Data Flow Statistics ******\n")
    file.write("Total Kinetic Energy     : {}\n".format(s[0]))
    file.write("Dissipation              : {}\n".format(s[1]))
    file.write("Rms velocity             : {}\n".format(s[2]))
    file.write("Taylor Micro. Scale      : {}\n".format(s[3]))
    file.write("Taylor-scale Reynolds    : {}\n".format(s[4]))
    file.write("Kolmogorov time sclae    : {}\n".format(s[5]))
    file.write("Kolmogorov length sclae  : {}\n".format(s[6]))
    file.write("Integral scale           : {}\n".format(s[7]))
    file.write("Large eddy turnover time : {}\n\n\n\n\n".format(s[8]))

    data = hres
    uw = np.transpose(data[2:4,:,:,1:1+args.eval_zres], (1, 0, 2, 3))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    uw = torch.tensor(uw, device=device).float()
    stats = compute_all_stats(uw[2:,:,:,:], viscosity=visc, description=False)
    s = [stats[..., i].item() for i in range(stats.shape[0])]

    file.write("***** Ground Truth Data Flow Statistics ******\n")
    file.write("Total Kinetic Energy     : {}\n".format(s[0]))
    file.write("Dissipation              : {}\n".format(s[1]))
    file.write("Rms velocity             : {}\n".format(s[2]))
    file.write("Taylor Micro. Scale      : {}\n".format(s[3]))
    file.write("Taylor-scale Reynolds    : {}\n".format(s[4]))
    file.write("Kolmogorov time sclae    : {}\n".format(s[5]))
    file.write("Kolmogorov length sclae  : {}\n".format(s[6]))
    file.write("Integral scale           : {}\n".format(s[7]))
    file.write("Large eddy turnover time : {}\n".format(s[8]))

def export_video(args, res_dict, hres, lres, dataset):
    """Export inference result as a video.
    """
    def get_phys_channels():
        def origin():
            phys_channels = ["p", "b", "u", "w"]
            return phys_channels

        def revised():
            return args.phy_fea_names

        return revised()
    phys_channels = get_phys_channels()
    os.makedirs(args.save_path, exist_ok=True)
    if dataset:
        # hres = dataset.denormalize_grid(hres.copy())
        lres = dataset.denormalize_grid(lres.copy())
        pred = np.stack([res_dict[key] for key in phys_channels], axis=0)
        pred = dataset.denormalize_grid(pred)

        # revised: grid information


        # calculate_flow_stats(args, pred, hres)       # Warning: only works with pytorch > v1.3 and CUDA >= v10.1
        np.savez_compressed(os.path.join(args.save_path, 'highres_lowres_pred'), hres=hres, lres=lres, pred=pred)
        
        interp_hres = revised.get_interp_hres_grid(args, lres)

        revised.evaluation_print_ind(args.save_path, hres, pred, interp_hres, args.phy_fea_names)


    # enumerate through physical channels first

    # for idx, name in enumerate(phys_channels):
    #     frames_dir = os.path.join(args.save_path, f'frames_{name}')
    #     os.makedirs(frames_dir, exist_ok=True)
    #     hres_frames = hres[idx]
    #     lres_frames = lres[idx]
    #     pred_frames = pred[idx]

    #     # loop over each timestep in pred_frames
    #     max_val = np.max(hres_frames)
    #     min_val = np.min(hres_frames)

    #     for pid in range(pred_frames.shape[0]):
    #         hid = int(np.round(pid / (pred_frames.shape[0] - 1) * (hres_frames.shape[0] - 1)))
    #         lid = int(np.round(pid / (pred_frames.shape[0] - 1) * (lres_frames.shape[0] - 1)))

    #         fig, axes = plt.subplots(3, figsize=(10, 10))#, 1, sharex=True)
    #         # high res ground truth
    #         im0 = axes[0].imshow(hres_frames[hid], cmap='RdBu',interpolation='spline16')
    #         axes[0].set_title(f'{name} channel, high res ground truth.')
    #         im0.set_clim(min_val, max_val)
    #         # low res input
    #         im1 = axes[1].imshow(lres_frames[lid], cmap='RdBu',interpolation='none')
    #         axes[1].set_title(f'{name} channel, low  res ground truth.')
    #         im1.set_clim(min_val, max_val)
    #         # prediction
    #         im2 = axes[2].imshow(pred_frames[pid], cmap='RdBu',interpolation='spline16')
    #         axes[2].set_title(f'{name} channel, predicted values.')
    #         im2.set_clim(min_val, max_val)
    #         # add shared colorbar
    #         cbaxes = fig.add_axes([0.1, 0, .82, 0.05])
    #         fig.colorbar(im2, orientation="horizontal", pad=0, cax=cbaxes)
    #         frame_name = 'frame_{:03d}.png'.format(pid)
    #         fig.savefig(os.path.join(frames_dir, frame_name))

    #     # stitch frames into video (using ffmpeg)
    #     frames_to_video(
    #         frames_pattern=os.path.join(frames_dir, "*.png"),
    #         save_video_to=os.path.join(args.save_path, f"video_{name}.mp4"),
    #         frame_rate=args.frame_rate, keep_frames=args.keep_frames)

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument("--eval_xres", type=int, default=512, metavar="X",
                        help="x resolution during evaluation (default: 512)")
    parser.add_argument("--eval_zres", type=int, default=128, metavar="Z",
                        help="z resolution during evaluation (default: 128)")
    parser.add_argument("--eval_tres", type=int, default=192, metavar="T",
                        help="t resolution during evaluation (default: 192)")
    parser.add_argument("--save_path", type=str, default='eval')
    parser.add_argument("--lres_interp", type=str, default='linear',
                        help="str, interpolation scheme for generating low res. choices of 'linear', 'nearest'")
    parser.add_argument("--lres_filter", type=str, default='none',
                        help=" str, filter to apply on original high-res image before \
                        interpolation. choices of 'none', 'gaussian', 'uniform', 'median', 'maximum'")
    parser.add_argument("--frame_rate", type=int, default=10, metavar="N",
                        help="frame rate for output video (default: 10)")
    parser.add_argument("--keep_frames", dest='keep_frames', action='store_true')
    parser.add_argument("--no_keep_frames", dest='keep_frames', action='store_false')
    parser.add_argument("--eval_pseudo_batch_size", type=int, default=10000,
                        help="psudo batch size for querying the grid. set to a smaller"
                             " value if OOM error occurs")
    parser.set_defaults(keep_frames=False)

    revised.evaluation_args_supplement(parser)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    param_file = os.path.join(os.path.dirname(args.ckpt), "params.json")
    with open(param_file, 'r') as fh:
        args.__dict__.update(json.load(fh))

    print(args)
    eqn_names, eqn_strs = revised.load_equation(args)
    # prepare dataset
    dataset = revised.evaluation_dataset_loader(args)

    # select inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # construct model
    print(f"Loading model parameters from {args.ckpt}...")
    igres = (int(args.nt/args.downsamp_t),
             int(args.nz/args.downsamp_z),
             int(args.nx/args.downsamp_x),)
    unet = revised.create_unet(args, igres)
    imnet = revised.create_imnet(args, NONLINEARITIES)

    # load model params
    resume_dict = torch.load(args.ckpt)
    unet.load_state_dict(resume_dict["unet_state_dict"])
    imnet.load_state_dict(resume_dict["imnet_state_dict"])

    unet.to(device)
    imnet.to(device)
    unet.eval()
    imnet.eval()
    all_model_params = list(unet.parameters())+list(imnet.parameters())

    # get pdelayer for the RB2 equations
    if args.normalize_channels:
        mean = dataset.channel_mean
        std = dataset.channel_std
    else:
        mean = std = None
    pde_layer = revised.evaluation_get_rb2_pde_layer(args, get_rb2_pde_layer, mean, std, eqn_names, eqn_strs)

    hres, lres, pred = revised.get_hres_lred_pred(args, dataset, pde_layer, unet, imnet, device,
                                                  args.eval_tres, args.eval_zres, args.eval_xres, 
                                                  args.eval_pseudo_batch_size, use_tqdm=True)
    
    # calculate_flow_stats(args, pred, hres)       # Warning: only works with pytorch > v1.3 and CUDA >= v10.1
    os.makedirs(args.save_path, exist_ok=True)

    if args.with_dis_file:
        
        npdata = np.load(os.path.join(args.data_folder, args.eval_dataset))
        grid = npdata["grid"]
        np.savez_compressed(os.path.join(args.save_path, 'highres_lowres_pred'), hres=hres, lres=lres, pred=pred, grid=grid)
    else:
        np.savez_compressed(os.path.join(args.save_path, 'highres_lowres_pred'), hres=hres, lres=lres, pred=pred)
    
    interp_hres = revised.get_interp_hres_grid(args, lres)

    revised.log_ind(args.save_path, hres, pred, interp_hres, args.phy_fea_names, args.ssim_psnr_normalize)

if __name__ == '__main__':
    main()
