"""Training script for RB2 experiment.
"""
import argparse
import json
import os
from glob import glob
import numpy as np
from collections import defaultdict
np.set_printoptions(precision=4)

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import linregress

# get current python file's path
current_dir = os.path.dirname(os.path.realpath(__file__))
# import our modules
import sys
sys.path.append(os.path.join(current_dir, "src"))
sys.path.append(os.path.join(current_dir, "meshfreeflownet"))
sys.path.append(os.path.join(current_dir, "meshfreeflownet/src"))
sys.path.append(os.path.join(current_dir, "octree"))

import train_utils as utils
from local_implicit_grid import query_local_implicit_grid
from nonlinearities import NONLINEARITIES
# import dataloader_spacetime as loader
from physics import get_rb2_pde_layer
import revised

# pylint: disable=no-member

def loss_functional(loss_type):
    """Get loss function given function type names."""
    if loss_type == 'l1':
        return F.l1_loss
    if loss_type == 'l2':
        return F.mse_loss
    # else (loss_type == 'huber')
    return F.smooth_l1_loss

def flag_point(pred_value, point_value, flag):
    shape = flag.shape
    num = shape[0] * shape[1]
    flag = flag.reshape(num)
    pred_value = pred_value.detach().cpu().numpy().reshape(num, -1)
    point_value = point_value.detach().cpu().numpy().reshape(num, -1)
    flag_pred_value, flag_point_value = [], []
    for i in range(num):
        if flag[i] == 1:
            flag_pred_value.append(pred_value[i])
            flag_point_value.append(point_value[i])

    flag_point_num = len(flag_pred_value)
    
    flag_pred_value = np.stack(flag_pred_value, axis=0) if flag_point_num > 0.05 * num else "none"
    flag_point_value = np.stack(flag_point_value, axis=0) if flag_point_num > 0.05 * num  else "none"

    return flag_pred_value, flag_point_value

def get_loss_distrib(pred_value, point_value, point_eage, loss_func):
    eage_pred_value, eage_point_value = flag_point(pred_value, point_value, point_eage)
    eage_loss = "none" if eage_pred_value == "none" else np.mean(np.square(eage_pred_value, eage_point_value))

    glob_pred_value, glob_point_value = flag_point(pred_value, point_value, 1-point_eage)
    glob_loss = "none" if glob_pred_value == "none" else np.mean(np.square(glob_pred_value, glob_point_value))

    return eage_loss, glob_loss

def get_samp_alpha(eage_loss, glob_loss):
    if eage_loss == "none" or glob_loss == "none":
        return None
    
    else:
        # adaptive edge sampling para
        alpha = eage_loss / glob_loss
        alpha = 0.8 if alpha < 0.8 else alpha
        alpha = 3.0 if alpha > 3.0 else alpha
        alpha = (alpha - 0.8) / 10 + 0.05
        return alpha
        
    
def train(args, unet, imnet, train_loader, epoch, global_step, device,
          logger, writer, optimizer, pde_layer, model_D=None, optimizer_D=None):
    """Training function."""
    unet.train()
    imnet.train()
    if args.gan_mode:
        model_D.train()
    tot_loss = 0
    tot_loss_D = 0 # revised: gan
    count = 0
    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)
    loss_func = loss_functional(args.reg_loss_type)
    for batch_idx, data_tensors in enumerate(train_loader):
        # send tensors to device
        
        data_tensors = [t.to(device) for t in data_tensors]
        # revised: sod
        input_grid, point_coord, point_value, point_sod, point_eage, lres_sod, point_diss = data_tensors
        optimizer.zero_grad()
        latent_grid = unet(input_grid, lres_sod)  # [batch, N, C, T, X, Y]
        # permute such that C is the last channel for local implicit grid query
        latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, N, T, X, Y, C]

        # define lambda function for pde_layer
        fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

        # update pde layer and compute predicted values + pde residues
        pde_layer.update_forward_method(fwd_fn)
        pred_value, residue_dict = pde_layer(point_coord, return_residue=True, diss=point_diss)

        # function value regression loss
        reg_loss = loss_func(pred_value, point_value)

        # pde residue loss
        pde_tensors = torch.stack([d for d in residue_dict.values()], dim=0)

        # print([torch.sum(d) for d in residue_dict.values()])

        if args.sigmoid_pde:
            pde_tensors = F.sigmoid(pde_tensors) - 0.5

        if args.use_double_pde:
            pde_tensors_with_sod = pde_tensors * point_sod.reshape(1, point_sod.shape[0], point_sod.shape[1], 1)
            pde_loss = loss_func(pde_tensors_with_sod, torch.zeros_like(pde_tensors_with_sod))
        else:
            pde_loss = loss_func(pde_tensors, torch.zeros_like(pde_tensors))

        if args.gan_mode:
            from torch.autograd import Variable
            cuda = torch.cuda.is_available()
            Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
            imgs_hr = Variable(point_value.type(Tensor))

            valid = Variable(Tensor(np.ones((imgs_hr.size(0), args.n_samp_pts_per_crop, len(args.phy_fea_names)+3))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_hr.size(0), args.n_samp_pts_per_crop, len(args.phy_fea_names)+3))), requires_grad=False) # imgs_hr.size(0)指batch

            criterion_GAN = torch.nn.MSELoss()

            loss_G = criterion_GAN(model_D(pred_value,point_coord), valid)#pred_value相当于生成的，point_coord相当于条件

            loss = args.alpha_reg * reg_loss + args.alpha_pde * pde_loss + args.alpha_loss_g * loss_G

        else:
            loss = args.alpha_reg * reg_loss + args.alpha_pde * pde_loss

        loss.backward()

        # gradient clipping
        if device.type == "cpu":
            torch.nn.utils.clip_grad_value_(unet.parameters(), args.clip_grad)
            torch.nn.utils.clip_grad_value_(imnet.parameters(), args.clip_grad)
        else:
            torch.nn.utils.clip_grad_value_(unet.module.parameters(), args.clip_grad)
            torch.nn.utils.clip_grad_value_(imnet.module.parameters(), args.clip_grad)

        optimizer.step()
        tot_loss += loss.item()
        count += input_grid.size()[0]

        if args.gan_mode:
            optimizer_D.zero_grad()

            # Loss of real and fake images
            
            loss_real = criterion_GAN(model_D(imgs_hr,point_coord), valid)   # 1 point_value==imgs_hr
            loss_fake = criterion_GAN(model_D(pred_value.detach(),point_coord), fake)    # 0
            # print(model_D(imgs_hr,point_coord).shape)
            loss_D = (loss_real + loss_fake)/2
            loss_D.backward()
            if device.type == "cpu":
                torch.nn.utils.clip_grad_value_(model_D.parameters(), args.clip_grad)
            else:
                torch.nn.utils.clip_grad_value_(model_D.module.parameters(), args.clip_grad)
            optimizer_D.step()

            tot_loss_D += loss_D.item()
        
        eage_loss, glob_loss = get_loss_distrib(pred_value, point_value, point_eage, loss_func) if args.use_eage_samp else (0.0, 0.0)

        eage_samp_num_alpha = get_samp_alpha(eage_loss, glob_loss) if args.use_eage_samp else 0.0
        train_loader.dataset.eage_samp_num_alpha = eage_samp_num_alpha if eage_samp_num_alpha else train_loader.dataset.eage_samp_num_alpha

        if batch_idx % args.log_interval == 0:
            # logger log
            info_str = ("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss Sum: {:.6f}\t"
                "Loss Reg: {:.6f}\tLoss Pde: {:.6f}".format(
                    epoch, batch_idx * len(input_grid), len(train_loader) * len(input_grid),
                    100. * batch_idx / len(train_loader), loss.item(),
                    args.alpha_reg * reg_loss, args.alpha_pde * pde_loss))
            if args.gan_mode:
                info_str += (
                    "\tLoss G: {:.6f}\tLoss D: {:.6f}".format(
                        loss_G.item(), loss_D.item()
                    )
                )

            logger.info(info_str)
            # tensorboard log
            writer.add_scalar('train/reg_loss_unweighted', reg_loss, global_step=int(global_step))
            writer.add_scalar('train/pde_loss_unweighted', pde_loss, global_step=int(global_step))
            writer.add_scalar('train/sum_loss', loss, global_step=int(global_step))
            if args.gan_mode:
                writer.add_scalar('train/loss_g_unweighted', loss_G, global_step=int(global_step))
                writer.add_scalar('train/loss_d_unweighted', loss_D, global_step=int(global_step))

            scalars_dict = {
                "reg_loss": args.alpha_reg * reg_loss,
                "pde_loss": args.alpha_pde * pde_loss,
                "sum_loss": loss
            }

            if args.gan_mode:
                scalars_dict["loss_g"] = args.alpha_loss_g * loss_G
                scalars_dict["loss_d"] = loss_D

            writer.add_scalars('train/losses_weighted',
                               scalars_dict, global_step=int(global_step))

        global_step += 1
    tot_loss /= count
    tot_loss_D /= count

    # logger.info([torch.sum(d) for d in residue_dict.values()])

    return tot_loss, tot_loss_D

def all_eval(args, unet, imnet, all_eval_evalset, epoch, global_step, device,
         logger, writer, optimizer, pde_layer, model_D=None):
    
    hres, _, pred = revised.get_hres_lred_pred(args, all_eval_evalset, pde_layer, unet, imnet, device, 
                                               args.all_eval_tres, args.all_eval_zres, args.all_eval_xres,
                                               args.pseudo_batch_size, use_tqdm=False)

    average_ssim_value, average_psnr_value = revised.get_average_psnr_ssim(
        hres, pred, args.phy_fea_names, args.ssim_psnr_normalize)
    return average_ssim_value, average_psnr_value

def eval(args, unet, imnet, eval_loader, epoch, global_step, device,
         logger, writer, optimizer, pde_layer, model_D=None):
    """Eval function. Used for evaluating entire slices and comparing to GT."""
    unet.eval()
    imnet.eval()
    if args.gan_mode:
        model_D.eval()
    # phys_channels = ["p", "b", "u", "w"]
    phys_channels = args.phy_fea_names
    phys2id = dict(zip(phys_channels, range(len(phys_channels))))
    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)

    ssim_value_list = []
    psnr_value_list = []

    for data_id, data_tensors in enumerate(eval_loader):
        # only need the first batch
        # break
        # send tensors to device
        data_tensors = [t.to(device) for t in data_tensors]
        hres_grid, lres_grid, _, _, _,lres_sod , _ = data_tensors
        latent_grid = unet(lres_grid ,lres_sod)  # [batch, C, T, Z, X]
        nb, nc, nt, nz, nx = hres_grid.shape

        # permute such that C is the last channel for local implicit grid query
        latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, Z, X, C]

        # define lambda function for pde_layer
        fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

        # update pde layer and compute predicted values + pde residues
        pde_layer.update_forward_method(fwd_fn)

        # layout query points for the desired slices
        eps = 1e-6
        # t_seq = torch.linspace(eps, 1-eps, nt)[::int(nt/8)]  # temporal sequences
        t_seq = torch.linspace(eps, 1-eps, nt)  # temporal sequences
        z_seq = torch.linspace(eps, 1-eps, nz)  # z sequences
        x_seq = torch.linspace(eps, 1-eps, nx)  # x sequences

        query_coord = torch.stack(torch.meshgrid(t_seq, z_seq, x_seq), axis=-1)  # [nt, nz, nx, 3]
        query_coord = query_coord.reshape([-1, 3]).to(device)  # [nt*nz*nx, 3]
        n_query = query_coord.shape[0]

        res_dict = defaultdict(list)

        n_iters = int(np.ceil(n_query/args.pseudo_batch_size))

        for idx in range(n_iters):
            sid = idx * args.pseudo_batch_size
            eid = min(sid+args.pseudo_batch_size, n_query)
            query_coord_batch = query_coord[sid:eid]
            query_coord_batch = query_coord_batch[None].expand(*(nb, eid-sid, 3))  # [nb, eid-sid, 3]

            pred_value, residue_dict = pde_layer(query_coord_batch, return_residue=True)
            pred_value = pred_value.detach()
            for key in residue_dict.keys():
                residue_dict[key] = residue_dict[key].detach()
            for name, chan_id in zip(phys_channels, range(len(phys_channels))):
                res_dict[name].append(pred_value[..., chan_id])  # [b, pb]
            for name, val in residue_dict.items():
                res_dict[name].append(val[..., 0])   # [b, pb]

        for key in res_dict.keys():
            res_dict[key] = (torch.cat(res_dict[key], axis=1)
                            .reshape([nb, len(t_seq), len(z_seq), len(x_seq)]))
        
        # log the imgs sample-by-sample

        if args.train_eval_norm_way == "norm":
            flag = False
        else:
            flag = True

        if not flag:
            normalize_hres_grid = [eval_loader.dataset.normalize_grid(singlle_hres_grid) for singlle_hres_grid in hres_grid]

        for samp_id in range(nb):
            # revised
            image_id = eval_loader.batch_size * data_id + samp_id
            if flag:
                pred_grid = np.stack([res_dict[key][samp_id].detach().cpu().numpy() for key in phys_channels], axis=0)
                denormalize_pred_grid = eval_loader.dataset.denormalize_grid(pred_grid)
            image_id = data_id
            for key in res_dict.keys():
                

                field = res_dict[key][samp_id]  # [nt, nz, nx]
                if key in phys_channels:
                    if flag:
                        denormalize_field = denormalize_pred_grid[phys2id[key], :] 
                        ground_truth = hres_grid[samp_id][phys2id[key], :]
                        ssim_value = revised.fea_npdata_ssim(ground_truth.detach().cpu().numpy(), denormalize_field, args.ssim_psnr_normalize)
                        psnr_value = revised.fea_npdata_psnr(ground_truth.detach().cpu().numpy(), denormalize_field, args.ssim_psnr_normalize)

                    else:
                        normalize_ground_truth = normalize_hres_grid[samp_id][phys2id[key], :]
                        ssim_value = revised.fea_npdata_ssim(normalize_ground_truth.cpu().numpy(), field.detach().cpu().numpy(), args.ssim_psnr_normalize)
                        psnr_value = revised.fea_npdata_psnr(normalize_ground_truth.cpu().numpy(), field.detach().cpu().numpy(), args.ssim_psnr_normalize)


                    ssim_value_list.append(ssim_value)
                    psnr_value_list.append(psnr_value)
                
                # add predicted slices
                images = utils.batch_colorize_scalar_tensors(field)  # [nt, nz, nx, 3]

                if samp_id == 0:
                    writer.add_images('sample_{}/{}/predicted'.format(image_id, key), images,
                        dataformats='NHWC', global_step=int(global_step))
                    # add ground truth slices (only for phys channels)
                    if key in phys_channels:
                        # gt_fields = hres_grid[samp_id, phys2id[key], ::int(nt/8)]  # [nt, nz, nx]
                        gt_fields = hres_grid[samp_id, phys2id[key], :]  # [nt, nz, nx]
                        gt_images = utils.batch_colorize_scalar_tensors(gt_fields)  # [nt, nz, nx, 3]

                        writer.add_images('sample_{}/{}/ground_truth'.format(image_id, key), gt_images,
                            dataformats='NHWC', global_step=int(global_step))
    
    average_ssim_value = sum(ssim_value_list) / len(ssim_value_list)
    average_psnr_value = sum(psnr_value_list) / len(psnr_value_list)
    return average_ssim_value, average_psnr_value


def get_args():

    # Training settings
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument("--batch_size_per_gpu", type=int, default=10, metavar="N",
                        help="input batch size for training (default: 10)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--pseudo_epoch_size", type=int, default=3000, metavar="N",
                        help="number of samples in an pseudo-epoch. (default: 3000)")
    parser.add_argument("--lr", type=float, default=1e-2, metavar="R",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--no_cuda", type=revised.str2bool, default=False)
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--data_folder", type=str, default="./data",
                        help="path to data folder (default: ./data)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint if resume is needed")
    parser.add_argument("--nt", default=16, type=int, help="resolution of high res crop in t.")
    parser.add_argument("--nx", default=128, type=int, help="resolution of high res crop in x.")
    parser.add_argument("--nz", default=128, type=int, help="resolution of high res crop in z.")
    parser.add_argument("--downsamp_t", default=4, type=int,
                        help="down sampling factor in t for low resolution crop.")
    parser.add_argument("--n_samp_pts_per_crop", default=1024, type=int,
                        help="number of sample points to draw per crop.")
    parser.add_argument("--lat_dims", default=32, type=int, help="number of latent dimensions.")
    parser.add_argument("--unet_nf", default=16, type=int,
                        help="number of base number of feature layers in unet.")
    parser.add_argument("--unet_mf", default=256, type=int,
                        help="a cap for max number of feature layers throughout the unet.")
    parser.add_argument("--imnet_nf", default=32, type=int,
                        help="number of base number of feature layers in implicit network.")
    parser.add_argument("--reg_loss_type", default="l1", type=str,
                        choices=["l1", "l2", "huber"],
                        help="number of base number of feature layers in implicit network.")
    parser.add_argument("--alpha_reg", default=1., type=float, help="weight of regression loss.")
    parser.add_argument("--alpha_pde", default=1., type=float, help="weight of pde residue loss.")
    parser.add_argument("--num_log_images", default=2, type=int, help="number of images to log.")
    parser.add_argument("--pseudo_batch_size", default=1024, type=int,
                        help="size of pseudo batch during eval.")
    parser.add_argument("--normalize_channels", dest='normalize_channels', action='store_true')
    parser.add_argument("--no_normalize_channels", dest='normalize_channels', action='store_false')
    parser.set_defaults(normalize_channels=True)
    parser.add_argument("--lr_scheduler", dest='lr_scheduler', action='store_true')
    parser.add_argument("--no_lr_scheduler", dest='lr_scheduler', action='store_false')
    parser.set_defaults(lr_scheduler=True)
    parser.add_argument("--clip_grad", default=1., type=float,
                        help="clip gradient to this value. large value basically deactivates it.")
    parser.add_argument("--lres_filter", default='none', type=str,
                        help=("type of filter for generating low res input data. "
                              "choice of 'none', 'gaussian', 'uniform', 'median', 'maximum'."))
    parser.add_argument("--lres_interp", default='linear', type=str,
                        help=("type of interpolation scheme for generating low res input data."
                              "choice of 'linear', 'nearest'"))
    parser.add_argument('--nonlin', type=str, default='softplus', choices=list(NONLINEARITIES.keys()),
                        help='Nonlinear activations for continuous decoder.')
    parser.add_argument('--use_continuity', type=revised.str2bool, nargs='?', default=True, const=True,
                        help='Whether to enforce continuity equation (mass conservation) or not')
    
    # other parameter
    revised.train_args_supplement(parser)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    # adjust batch size based on the number of gpus available
    args.batch_size = int(torch.cuda.device_count()) * args.batch_size_per_gpu

    revised.train_args_transform(args)
    eqn_names, eqn_strs = revised.load_equation(args)

    # log and create snapshots
    os.makedirs(args.log_dir, exist_ok=True)
    filenames_to_snapshot = glob("*.py") + glob("*.sh")
    utils.snapshot_files(filenames_to_snapshot, args.log_dir)
    logger = utils.get_logger(log_dir=args.log_dir)
    with open(os.path.join(args.log_dir, "params.json"), 'w') as fh:
        json.dump(args.__dict__, fh, indent=2)
    logger.info("%s", repr(args))

    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))

    # random seed for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create dataloaders
    trainset = revised.train_trainset_loader(args)
    train_sampler = RandomSampler(trainset, replacement=True, num_samples=args.pseudo_epoch_size)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                              sampler=train_sampler, **kwargs)
    
    if args.all_eval_mode:
        all_eval_evalset = revised.train_all_dataset_loader(args)
    else:
        evalset = revised.train_evalset_loader(args)
        eval_sampler = RandomSampler(evalset, replacement=True, num_samples=args.num_log_images)
        eval_loader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                sampler=eval_sampler, **kwargs)

    # setup model
    unet = revised.create_unet(args, trainset.scale_lres)
    imnet = revised.create_imnet(args, NONLINEARITIES)

    if args.gan_mode:
        model_D = revised.create_discriminater(args)
    else:
        model_D = None

    all_model_params = list(unet.parameters())+list(imnet.parameters())

    if args.optim == "sgd":
        optimizer = optim.SGD(all_model_params, lr=args.lr)
    else:
        optimizer = optim.Adam(all_model_params, lr=args.lr)
    
    if args.gan_mode:
        optimizer_D = torch.optim.SGD(model_D.parameters(), lr=args.lr_D)
    else:
        optimizer_D = None

    start_ep = 0
    global_step = np.zeros(1, dtype=np.uint32)
    tracked_stats = np.inf

    last_ssim = -1.0
    last_psnr = 0.0

    if args.resume:
        resume_dict = torch.load(args.resume)
        start_ep = resume_dict["epoch"]
        global_step = resume_dict["global_step"]
        tracked_stats = resume_dict["tracked_stats"]
        unet.load_state_dict(resume_dict["unet_state_dict"])
        imnet.load_state_dict(resume_dict["imnet_state_dict"])
        last_ssim = resume_dict["last_ssim"]
        last_psnr = resume_dict["last_psnr"]
        tracked_stats = np.inf
        last_ssim = 0
        last_psnr = 0
        if args.gan_mode:
            model_D.load_state_dict(resume_dict["model_D_state_dict"])
            optimizer_D.load_state_dict(resume_dict["optim_D_state_dict"])
        optimizer.load_state_dict(resume_dict["optim_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # revised.create_ind(args)

    if use_cuda:
        unet = nn.DataParallel(unet)
    unet.to(device)
    if use_cuda:
        imnet = nn.DataParallel(imnet)
    imnet.to(device)
    if args.gan_mode:
        if use_cuda:
            model_D = nn.DataParallel(model_D)
        model_D.to(device)

    model_param_count = lambda model: sum(x.numel() for x in model.parameters())
    logger.info("{}(unet) + {}(imnet) paramerters in total".format(model_param_count(unet),
                                                                   model_param_count(imnet)))
    if args.gan_mode:
        logger.info("{}(discriminater) paramerters in total".format(model_param_count(model_D)))

    checkpoint_path = os.path.join(args.log_dir, "checkpoint_latest.pth.tar")

    # get pdelayer for the RB2 equations
    if args.normalize_channels:
        mean = trainset.channel_mean
        std = trainset.channel_std
    else:
        mean = std = None
    pde_layer = revised.train_get_rb2_pde_layer(args, get_rb2_pde_layer, mean, std, eqn_names, eqn_strs)

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    if args.gan_mode and args.lr_scheduler_D:
        scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min')

    # training loop
    for epoch in range(start_ep + 1, args.epochs + 1):
        loss, loss_D = train(args, unet, imnet, train_loader, epoch, global_step, device, logger, writer,
                     optimizer, pde_layer, model_D, optimizer_D)
        if args.all_eval_mode:
            average_ssim_value, average_psnr_value = all_eval(
                args, unet, imnet, all_eval_evalset, epoch, global_step, device, logger, writer, optimizer,
                pde_layer, model_D)
        else:
            average_ssim_value, average_psnr_value = eval(
                args, unet, imnet, eval_loader, epoch, global_step, device, logger, writer, optimizer,
                pde_layer, model_D)
        info = "Loss Sum={:.6f}, ".format(
            loss*args.batch_size
        )
        if args.gan_mode:
            info += "Loss D={:.6f}, ".format(
                loss_D*args.batch_size
            )

        info += "ssim={:.6f}, psnr={:.6f}".format(
            average_ssim_value, average_psnr_value
        )

        logger.info(info)
        if args.lr_scheduler:
            scheduler.step(loss)
        if args.gan_mode and args.lr_scheduler_D:
            scheduler_D.step(loss_D)

        # revised.updata_ind(args, loss, average_ssim_value, average_psnr_value)

        if average_ssim_value > last_ssim:
            last_ssim = average_ssim_value

        if average_psnr_value > last_psnr:
            is_best_psnr = True
            last_psnr = average_psnr_value
        else:
            is_best_psnr = False

        if loss < tracked_stats:
            tracked_stats = loss
            is_best_loss = True
        else:
            is_best_loss = False

        save_checkpoint_dict = {
            "epoch": epoch,
            "unet_state_dict": unet.module.state_dict() if use_cuda else unet.state_dict(),
            "imnet_state_dict": imnet.module.state_dict() if use_cuda else imnet.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "tracked_stats": tracked_stats,
            "global_step": global_step,
            "last_ssim": last_ssim,
            "last_psnr": last_psnr,
        }
        if args.gan_mode:
            save_checkpoint_dict["model_D_state_dict"] = model_D.module.state_dict() if use_cuda else model_D.state_dict()
            save_checkpoint_dict["optim_D_state_dict"] = optimizer_D.state_dict()

        utils.save_checkpoint(save_checkpoint_dict, is_best_psnr, is_best_loss, epoch, checkpoint_path, "_pdenet", logger)

if __name__ == "__main__":
    main()
