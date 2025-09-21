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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal
import torch.nn.functional as F
# from utils.loss_utils import image2canny
import cv2


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, weight,
                 image_name, uid, dtumask, near_far, blendermask, height_in, width_in, depth_mono,depth_mono_alin,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R # c2w
        self.T = T # w2c
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.depth_mono = None # 深度图
        if depth_mono is not None:
            self.depth_mono = depth_mono

        self.depth_mono_alin = None
        if depth_mono_alin is not None:
            self.depth_mono_alin = depth_mono_alin
        self.warp_flag = False

        self.weight = weight

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.near_far = near_far
        self.dtumask = torch.tensor(dtumask).float().cuda() if dtumask is not None else None
        self.blendermask = torch.tensor(blendermask).float().cuda() if blendermask is not None else None
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device) if image is not None else None
        self.image_canny_mask = image2canny(self.original_image.permute(1,2,0), 100, 200, isEdge1=False).detach().to(self.data_device)
        self.depth_canny_mask = image2canny(self.depth_mono, 5, 10, isEdge1=False).detach().to(self.data_device) if self.depth_mono is not None else None
        self.image_width = self.original_image.shape[2] if image is not None else width_in
        self.image_height = self.original_image.shape[1] if image is not None else height_in
        
        if image is not None:
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda() #w2c
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        fx, fy = fov2focal(FoVx, self.image_width), fov2focal(FoVy, self.image_height)
        cx, cy = self.image_width / 2.0, self.image_height / 2.0
        self.intr = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().cuda()
        
        self.w2c = torch.zeros((4, 4)).float().cuda()
        self.w2c[:3, :3] = torch.from_numpy(R).float().cuda().transpose(0, 1)
        self.w2c[:3, 3] = torch.from_numpy(T).float().cuda()
        self.w2c[3, 3] = 1.0

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class PseudoCamera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, width, height, nearest_image, trans=np.array([0.0, 0.0, 0.0]), scale=1.0 ):
        super(PseudoCamera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height
        self.warp_image = None
        self.inpainting_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.nearest_image = nearest_image

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        fx, fy = fov2focal(FoVx, self.image_width), fov2focal(FoVy, self.image_height)
        cx, cy = self.image_width / 2.0, self.image_height / 2.0
        self.intr = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().cuda()
        
        self.w2c = torch.zeros((4, 4)).float().cuda()
        self.w2c[:3, :3] = torch.from_numpy(R).float().cuda().transpose(0, 1)
        self.w2c[:3, 3] = torch.from_numpy(T).float().cuda()
        self.w2c[3, 3] = 1.0


def image2canny(image, thres1, thres2, isEdge1=True):
    """ image: (H, W, 3)"""
    canny_mask = torch.from_numpy(cv2.Canny((image.detach().cpu().numpy()*255.).astype(np.uint8), thres1, thres2)/255.)
    if not isEdge1:
        canny_mask = 1. - canny_mask
    return canny_mask.float()