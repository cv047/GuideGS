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

import torch
import math
import numpy as np
from typing import NamedTuple
import torchvision
import torch.nn.functional as F

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose() # c2w
    Rt[:3, 3] = t # w2c
    Rt[3, 3] = 1.0 # c2w

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def project_matches_to_novel_view2(pixels, depth, intr, w2c, pseudo_intr, pseudo_w2c):

    H, W = depth.shape

    # 先计算depth
    norm_x = (pixels[:, 0] / W) * 2 - 1
    norm_y = (pixels[:, 1] / H) * 2 - 1
    norm_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0)
    match_depth = F.grid_sample(depth.unsqueeze(0).unsqueeze(0), norm_grid, mode="bilinear")
    match_depth = match_depth.reshape(-1)

    # 验证---没问题
    # homo_pixels = torch.cat([pixels,torch.ones_like(pixels[:,0]).unsqueeze(1)],dim=1)
    # normalized_coords = torch.matmul(homo_pixels, torch.inverse(intr).T)
    # camera_coords = normalized_coords * match_depth.unsqueeze(1)

    # 像素点投影到相机坐标系
    x = (pixels[:, 0] - intr[0][2]) * match_depth / intr[0][0]
    y = (pixels[:, 1] - intr[1][2]) * match_depth / intr[1][1]
    z = match_depth
    kpi_3d_i = torch.stack([x, y, z], dim=-1) # torch.Size([N, 3])
    kpi_3d_j = torch.linalg.inv(w2c) @ (to_homogeneous(kpi_3d_i).T) # [4, N]
    kpi_3d_j = from_homogeneous(kpi_3d_j.T) # [N, 3] 匹配点的世界坐标系坐标

    pseudo_cam_3d_pts = (pseudo_w2c @ to_homogeneous(kpi_3d_j).T)[:3] # pseudo相机坐标系  [3, N]
    pseudo_2d_pixel = (pseudo_intr @ pseudo_cam_3d_pts).T # 像素坐标 [N,3]
    pseudo_gt_depth_in_sampled_img = pseudo_2d_pixel[:,2]
    pts_in_sampled_img = pseudo_2d_pixel[:,:2] / (pseudo_2d_pixel[:,2:] + 1e-8)

    # pts_in_sampled_img, pseudo_gt_depth_in_sampled_img = batch_project(kpi_3d_j, pose_w2c=pseudo_w2c, intr=pseudo_intr, return_depth=True) #(N, 2)
    
    return pts_in_sampled_img, pseudo_gt_depth_in_sampled_img, kpi_3d_j

def batch_project(points, pose_w2c, intr, return_depth=False):
    pass




    # R_pseudo_w2c = pseudo_w2c[:3, :3]  # 旋转矩阵 (w2c)
    # T_pseudo_w2c = pseudo_w2c[:3, 3]   # 平移向量 (w2c)

    # R_train_w2c = train_w2c[:3, :3]    # 旋转矩阵 (w2c)
    # T_train_w2c = train_w2c[:3, 3]     # 平移向量 (w2c)

    # R_train_c2w = R_train_w2c.T        # 旋转矩阵 (c2w)
    # T_train_c2w = -R_train_c2w @ T_train_w2c  
    # # 平移向量 (c2w)
    
    # K = torch.tensor(K, dtype=torch.float32, device=pixels.device) # mast3r分辨率的内参矩阵
    
    # fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # orig_W, orig_H = cx*2, cy*2

    # scale_w = train_resolution[1]/orig_W
    # scale_h = train_resolution[0]/orig_H
    # fx_new = fx * scale_w
    # fy_new = fy * scale_h
    # cx_new = cx * scale_w
    # cy_new = cy * scale_h

    # # 构建训练分辨率分辨率下的内参矩阵
    # K_adjusted = torch.tensor([
    #     [fx_new, 0, cx_new],
    #     [0, fy_new, cy_new],
    #     [0, 0, 1]
    # ], dtype=torch.float32, device=pixels.device)

    # #缩放match坐标到训练尺寸
    # pixels_scaled = pixels * torch.tensor([scale_w, scale_h], device=pixels.device)
    # u = torch.round(pixels_scaled[:, 0]).long()  # x 坐标，四舍五入
    # v = torch.round(pixels_scaled[:, 1]).long()  # y 坐标，四舍五入

    # # u = torch.clamp(u, 0, image.shape[2] - 1)  # 确保不超出图像宽度
    # # v = torch.clamp(v, 0, image.shape[1] - 1)

    # # 齐次像素坐标
    # pixels_homo = torch.cat((pixels, torch.ones((pixels.shape[0], 1), device=pixels.device)),dim=1)  
    # K_inv = torch.inverse(K)
    # points_camera = (K_inv @ pixels_homo.T).T  # 像素坐标转相机坐标
    # # 使用插值提取深度
    # depth = depth.squeeze(0) # 训练分辨率下的深度图

    # depths_ = depth[v, u]

    # pixels_normalized = pixels.clone()
    # pixels_normalized[:, 0] = (pixels[:, 0] / (orig_W - 1)) * 2 - 1
    # pixels_normalized[:, 1] = (pixels[:, 1] / (orig_H - 1)) * 2 - 1
    # pixels_normalized = pixels_normalized.view(1, 1, -1, 2)
    # depth = depth.unsqueeze(0).unsqueeze(0)
    # depths = torch.nn.functional.grid_sample(
    #     depth,
    #     pixels_normalized,
    #     mode='bilinear',
    #     align_corners=True
    # ).squeeze()  # [N]

    # # depths_ = depth[v, u]

    # points3d_camera = points_camera * depths.unsqueeze(1)  # [N, 3]
    # # 相机坐标系转到世界坐标系
    # points3d_world = (R_train_c2w @ points3d_camera.T + T_train_c2w.unsqueeze(1)).T
    # # 投影到新视图
    # points_3d_pseudo_camera = (R_pseudo_w2c @ points3d_world.T + T_pseudo_w2c.unsqueeze(1)).T
    # points_2d_pseudo = (K_adjusted @ points_3d_pseudo_camera.T).T
    # z = points_2d_pseudo[:, 2] # 新视图下匹配点的深度
    # points_2d_pseudo = points_2d_pseudo[:, :2] / z.unsqueeze(1).clamp(min=1e-6)
    # # 提取颜色值
    # colors_warped_ = (image[:,v, u]).permute(1,0)
    
    # image = image.unsqueeze(0)
    # colors_warped = torch.nn.functional.grid_sample(
    #     image,
    #     pixels_normalized,
    #     mode='bilinear',
    #     align_corners=True
    # ).squeeze().T  # [N, C]

    # # warp_img
    # image = image.squeeze(0).permute(1,2,0)
    # depth = depth.squeeze(0).squeeze(0)
    # warp_img = torch.zeros_like(image)
    # grid_y, grid_x = torch.meshgrid(
    #     torch.linspace(0, train_resolution[0]-1, train_resolution[0], device=pixels.device),
    #     torch.linspace(0, train_resolution[1]-1, train_resolution[1], device=pixels.device),
    #     indexing='ij'
    # )
    # pixel_coords = torch.stack((grid_x, grid_y), dim=2).view(-1, 2).int()
    # pixels_homo = torch.cat((
    #     pixel_coords,
    #     torch.ones((pixel_coords.shape[0], 1), device=pixels.device)
    # ), dim=1).T
    # K_adjusted_inv = torch.inverse(K_adjusted)
    # points_camera_source = (K_adjusted_inv @ pixels_homo).T
    # depth_source = depth[pixel_coords[:,1],pixel_coords[:,0]]
    # points3d_camera_source = points_camera_source * depth_source.unsqueeze(1)
    # points_world_source = (R_train_c2w @ points3d_camera_source.T + T_train_c2w.unsqueeze(1)).T
    # point3d_target_cam = (R_pseudo_w2c@points_world_source.T+T_pseudo_w2c.unsqueeze(1)).T
    # points_2d_target = (K_adjusted@point3d_target_cam.T).T
    # x,y = (points_2d_target[:,:2]/points_2d_target[:,2:3])[:,0].int(),(points_2d_target[:,:2]/points_2d_target[:,2:3])[:,1].int()
    # mask = (x>=0)&(x<train_resolution[1])&(0 <= y) & (y < train_resolution[0])
    # valid_x = x[mask]
    # valid_y = y[mask]
    # valid_pixel_coords = pixel_coords[mask]
    # warp_img[valid_y, valid_x] = image[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]]
    # warp_img = warp_img.permute(2,0,1)
    # torchvision.utils.save_image(warp_img, 'warp_img.png')
 
    # return points_2d_pseudo, z, colors_warped, points3d_world

def to_homogeneous(points):
    ones_shape = list(points.shape[:-1]) + [1]
    ones = torch.ones(ones_shape, dtype=points.dtype, device=points.device)
    return torch.cat([points, ones], dim=-1)

def from_homogeneous(points_homo, eps=1e-8):
    scale = points_homo[..., -1:]  # 取最后一维
    # scale = torch.where(torch.abs(scale) < eps, 
    #                    torch.ones_like(scale), 
    #                    scale)
    return points_homo[..., :-1] / (scale+eps)