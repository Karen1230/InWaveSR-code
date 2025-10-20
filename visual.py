import argparse
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import our modules 222
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "meshfreeflownet"))
sys.path.append(os.path.join(current_dir, "src"))
import data_flie_transform as dft
import npdata_visualize as nvl
import visual_origin

class MyFramesToMovie:
    def __call__(self, frame_folder, movie_folder, movie_name, frame_rate):
        frame_pattern = os.path.join(frame_folder, "*.png")
        os.system(f"rm -rf {os.path.join(movie_folder, movie_name)}")
        # os.system(f"/home/hanpeng/anaconda3/envs/linuxconda/bin/ffmpeg"
        os.system(f"ffmpeg"
                  f" -framerate {frame_rate} -pattern_type glob"
                  f" -i \"{frame_pattern}\" -c:v libx264 -r 30 -pix_fmt yuv420p"
                  f" {os.path.join(movie_folder, movie_name)}")
        print(f"saving movie to {os.path.join(movie_folder, movie_name)}")

def expand_t_4(lres_grid):
    expand_lres_grid = []
    for t_id in range(lres_grid.shape[1]):
        for _ in range(4):
            lres_t_id = lres_grid[:,t_id,:,:]
            expand_lres_grid.append(lres_t_id)
    expand_lres_grid = np.stack(expand_lres_grid, axis=1)
    return expand_lres_grid

def revised(args, npdata_dict):
    lres_grid = npdata_dict["lres"]
    lres_grid = expand_t_4(lres_grid)
    truth_hres_grid = npdata_dict["hres"]
    model_hres_grid = npdata_dict["pred"]
    
    args.figsize = (args.figsize_x * (truth_hres_grid.shape[-1] / 1024), args.figsize_y)
    
    draw_fun_lres = nvl.DrawHeatImage(aspect=int(args.eval_downsamp_z/args.eval_downsamp_x))
    draw_fun_hres = nvl.DrawHeatImage(interpolation="spline16", aspect=1)
    draw_funs = [draw_fun_hres, draw_fun_lres, draw_fun_hres]
    tick_info_fun = nvl.CreatePlanarTickInfo()
    reduce_alpha_x, reduce_alpha_z = 64, 32
    tick_info_hres1 = tick_info_fun(truth_hres_grid[0][0], 1, 1, (reduce_alpha_x, reduce_alpha_z), None, None)
    tick_info_hres2 = tick_info_fun(truth_hres_grid[0][0], 1, 1, (reduce_alpha_x, reduce_alpha_z), None, None)
    tick_info_lres = tick_info_fun(lres_grid[0][0], 1, 1, 
                                   (int(reduce_alpha_x/4), int(reduce_alpha_z/8)), None, None)
    tick_infos = [tick_info_hres1, tick_info_lres, tick_info_hres2]

    image_generator = nvl.ComeparingColorbar2ImageGenerator(
        args.figsize, args.fontsize, draw_funs, tick_infos
    )
    movie_generator = nvl.MovieGenerator(image_generator, args.frame_rate)
    movie_generator.frames_to_movie = MyFramesToMovie()
    
    for fea_id, fea_name in enumerate(args.phy_fea_names):

        npdata_list = [[truth_hres_grid[fea_id][t_id], lres_grid[fea_id][t_id], model_hres_grid[fea_id][t_id]] for t_id in range(truth_hres_grid.shape[1])]
        movie_generator(npdata_list, args.eval_folder, f"{fea_name}.mp4")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_folder"    , type=str,   default="./log/Exp_Tem_/eval" )
    parser.add_argument("--frame_rate"            , type=int,   default=10           )
    parser.add_argument("--figsize_x"             , type=int,   default=22           )
    parser.add_argument("--figsize_y"             , type=int,   default=10           )
    parser.add_argument("--fontsize"              , type=int,   default=18           )
    parser.add_argument("--saved_result_file_name", type=str,   default="highres_lowres_pred")
    parser.add_argument("--eval_downsamp_t", default=4, type=int,
                        help="down sampling factor in t for low resolution crop.")
    parser.add_argument("--eval_downsamp_z", default=8, type=int,
                        help="down sampling factor in z for low resolution crop.")
    parser.add_argument("--eval_downsamp_x", default=4, type=int,
                        help="down sampling factor in x for low resolution crop.")
    parser.add_argument("--use_origin", action='store_true')

    parser.set_defaults( eval_folder="log/Exp_argo/eval_2901615" )
    parser.set_defaults( use_origin=False )

    def set_exp(basic_log_name):
        parser.set_defaults(eval_folder=f"{basic_log_name}/eval")

    def set_test(basic_log_name):
        parser.set_defaults(eval_folder=f"{basic_log_name}_tem/eval")

    # basic_log_name
    basic_log_name = "./log/Exp_Tem"
    # set_exp(basic_log_name)
    # set_test(basic_log_name)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    param_file = os.path.join(os.path.dirname(args.eval_folder), "params.json")
    with open(param_file, 'r') as fh:
        args.__dict__.update(json.load(fh))

    transform_fun = dft.NpzFileToNpdataDict()
    npdata_dict = transform_fun(file_folder=args.eval_folder, file_name=f"{args.saved_result_file_name}.npz")

    if args.use_origin:
        visual_origin.visual(args, npdata_dict)
    else:
        revised(args, npdata_dict)

if __name__ == "__main__":
    main()