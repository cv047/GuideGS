#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from simple_lama_inpainting.models.model import SimpleLama
from utils.loss_utils import l1_loss, ssim, l2_loss, nearMean_map, locality_loss, normalize0, pearson_depth_loss,local_pearson_loss, pearson_depth_loss_weight
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import torch.nn.functional as F
from warp import inverse_warp
from simple_knn._C import distCUDA2
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    # init stage
    gaussians.training_setup_init()
    ema_loss_for_log = 0.0
    viewpoint_stack = scene.getTrainCameras().copy()
    progress_bar = tqdm(range(0, 2000), desc="Init progress")
    
    best_state_dict = None
    min_loss_state = None
    for iteration in range(0, 2000): 
        
        if iteration in [500, 1000, 1500]:
            gaussians.update_learning_rate_init(0.5)
        
        iter_start.record()
        
        matchloss, loss_state = gaussians.get_matchloss_from_base() 
        
        loss = 5 * matchloss
        
        if best_state_dict is None:
            best_state_dict = gaussians.get_z_val() 
            min_loss_state = loss_state
        else:
            curr_state_dict = gaussians.get_z_val() 
            for key, v in loss_state.items():
                for key1, v1 in loss_state[key].items():
                    best_state_dict[key][key1] = torch.where((min_loss_state[key][key1]<loss_state[key][key1]).unsqueeze(-1), best_state_dict[key][key1], curr_state_dict[key][key1]) # 记录每一个匹配对的最小loss对应的z值
                    min_loss_state[key][key1] = torch.where(min_loss_state[key][key1]<loss_state[key][key1], min_loss_state[key][key1], loss_state[key][key1]) # 记录每一个匹配对的最小loss值
        
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log 
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration >= 1999:
                progress_bar.close()
            
            # Optimizer step
            gaussians.optimizer_init.step()
            gaussians.optimizer_init.zero_grad(set_to_none = True)
            
    gaussians.load_z_val(best_state_dict) 
    print("\n[Init Stage] Saving Gaussians")

    gaussians.optimize_mono_depth(scene.getTrainCameras())
    scene.save_init(2000)
    for i,pseudo_cam in enumerate(scene.getPseudoCameras()):
        nearest_img_name = pseudo_cam.nearest_image
        nearest_cam = next(camera for camera in scene.train_cameras[1.0] if camera.image_name == nearest_img_name)
        nearest_img = nearest_cam.original_image
        nearest_depth = nearest_cam.depth_mono
        warp_img, mask = inverse_warp(nearest_img, nearest_depth, nearest_cam.w2c, pseudo_cam.w2c, nearest_cam.intr)
        pseudo_cam.warp_image = warp_img
        pseudo_cam.inpainting_mask = mask
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.create_from_pcd(min_loss_state)
    gaussians.training_setup(opt) 
    viewpoint_stack = None
    pseudo_stack = None
    # match_data = scene.get_match_data()
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    lama = SimpleLama()

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, depth, opacity, color = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["rendered_depth"].squeeze(0), render_pkg["opacity"], render_pkg["color"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        bg_mask = None
        # if "dtu" in args.source_path:
        #     if 'scan110' not in args.source_path:
        #         bg_mask = (gt_image.max(0, keepdim=True).values < 30/255)
        #     else:
        #         bg_mask = (gt_image.max(0, keepdim=True).values < 15/255)
        #     bg_mask_clone = bg_mask.clone()
        #     for i in range(1, 50):
        #         bg_mask[:, i:] *= bg_mask_clone[:, :-i]
        #     gt_image[bg_mask.repeat(3,1,1)] = 0.
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) # SSIM loss

        render_match_loss ,pts  = gaussians.get_matchloss_from_renderdepth(viewpoint_cam, depth, min_loss_state)
        loss += render_match_loss * 0.3 # L_rg

        # opacity_loss
        loss += torch.abs(opacity).mean() * 0.001

        mono_depth = viewpoint_cam.depth_mono_alin.cuda()

        pearson_loss = pearson_depth_loss(depth.squeeze(0), mono_depth)
        loss += 0.03 * pearson_loss  
 
        # if "dtu" in args.source_path:
        #     loss += render_pkg["rendered_alpha"][bg_mask].mean()

        # -----------------------------pseudo_cam_loss----------------------------------------
        if iteration > 800 and iteration< (opt.iterations-100):
            if not pseudo_stack:
                pseudo_stack = scene.getPseudoCameras().copy()

            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
            nearest_cam = next(camera for camera in scene.train_cameras[1.0] if pseudo_cam.nearest_image == camera.image_name)

            mono_warp_img = pseudo_cam.warp_image
            render_pseudo_pkg = render(pseudo_cam, gaussians, pipe, bg)
            pseudo_image = render_pseudo_pkg["render"]


            mono_warp_img = mono_warp_img.cpu().numpy().transpose(1,2,0)*255.0
            mono_mask = pseudo_cam.inpainting_mask.cpu().numpy()*255.0
            mono_result = lama(mono_warp_img, mono_mask)
            mono_result_tensor = torch.from_numpy(np.array(mono_result)[:mono_warp_img.shape[0], :mono_warp_img.shape[1]]).permute(2, 0, 1).float() / 255.0
            mono_result_tensor = mono_result_tensor.cuda()

            pseudo_Ll1 = l1_loss(pseudo_image, mono_result_tensor)
            
            pseudo_loss = (1.0 - opt.lambda_dssim) * pseudo_Ll1 + opt.lambda_dssim * (1.0 - ssim(pseudo_image, mono_result_tensor))
            
            loss = loss + 0.7*pseudo_loss
   
        loss.backward()


        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:  #500次后剪枝
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration < opt.opacity_reset_until_iter and (iteration % 200 == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                gaussians.optimizer_bg.step()
                gaussians.optimizer_bg.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving {} Checkpoint".format(iteration,gaussians.get_xyz().shape[0]))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    if viewpoint.dtumask is not None:
                        mask = viewpoint.dtumask > 0
                        l1_test += l1_loss(image[:, mask], gt_image[:, mask]).mean().double()
                        psnr_test += psnr(image[:, mask], gt_image[:, mask]).mean().double()
                    else:
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1_000, 1_500, 2_000, 2_500, 3_000, 4_000, 5_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 1_000, 1_500, 2_000, 2_500, 3_000, 4_000, 5_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")