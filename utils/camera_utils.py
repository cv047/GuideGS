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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import cv2
import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, match_data):
    
    if  cam_info.image is not None:
        orig_w, orig_h = cam_info.image.size

        if args.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        else:  # should be a type that converts to float
            if args.resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        resized_image_rgb = PILtoTorch(cam_info.image, resolution)

        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None
        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
        blendermask = cam_info.blendermask
        if blendermask is not None:
            blendermask = cv2.resize(blendermask, resolution)
        dtumask = cam_info.dtumask
        if dtumask is not None:
            dtumask = cv2.resize(dtumask, resolution)
            
    else:
        gt_image = None
        loaded_mask = None
        dtumask = None
        blendermask = None
    
    height_in = cam_info.height
    width_in = cam_info.width
    if args.resolution in [1, 2, 4, 8]:
        height_in /= args.resolution
        width_in /= args.resolution
    
    if cam_info.depth_mono is not None:
        resized_depth_mono = cv2.resize(cam_info.depth_mono, (resolution[0], resolution[1]),interpolation=cv2.INTER_LINEAR)
        reversed_depth_tensor = torch.from_numpy(np.array(resized_depth_mono))
    else: 
        reversed_depth_tensor = None

    
    name = cam_info.image_name
    if name in match_data:
        weight = compute_match_density(match_data[name], num_blocks=8)
    else:
        weight = None
    

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, weight=weight,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, near_far=cam_info.near_far, height_in=height_in, width_in=width_in,
                  image=gt_image, gt_alpha_mask=loaded_mask, dtumask=dtumask, blendermask=blendermask, depth_mono=reversed_depth_tensor, depth_mono_alin=reversed_depth_tensor,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, match_data, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, match_data))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def compute_match_density(match_data_list, num_blocks=8):
    density = np.zeros((num_blocks, num_blocks))
    total_matches = sum(len(match_data) for match_data in match_data_list.values()) if match_data_list else 1

    for match_data in match_data_list.values():
        for x, y in match_data:
            if 0 <= x <= 1 and 0 <= y <= 1:
                bx = min(int(x * num_blocks), num_blocks - 1)
                by = min(int(y * num_blocks), num_blocks - 1)
                density[by, bx] += 1

    beta = 100.0  # 缩放因子
    scaled_density = np.log(1 + beta * (density / total_matches))
     # Sigmoid 归一化
    density_sum = np.sum(scaled_density)
    density = scaled_density / density_sum if density_sum > 0 else np.ones_like(scaled_density) / (num_blocks * num_blocks)
        # 自适应 Sigmoid 映射
    mean_density = np.mean(density)
    std_density = np.std(density) if np.std(density) > 1e-6 else 1e-6  # 避免除零
    alpha = 10.0  # 放大因子
    weights = 1 / (1 + np.exp(-alpha * (density - mean_density) / std_density))
    
    return weights