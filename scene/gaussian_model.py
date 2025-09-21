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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F
from scene.dataset_readers import storePly
import cv2
# from pytorch3d.ops.points_normals import estimate_pointcloud_normals 
import random
# from utils.loss_utils import patch_norm_mse_loss, ranking_loss, cons_loss
import random
import torchvision
# from sklearn.linear_model import LinearRegression 
from scipy.interpolate import interp1d


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        
        scal = self.scaling_activation(self._scaling)
        if hasattr(self, "bg_scaling") and self.bg_scaling.shape[0]>0:
            bg_scal = self.scaling_activation(self.bg_scaling)
            scal = torch.cat([scal, bg_scal])
        
        return scal
    
    @property
    def get_rotation(self):
        
        rot = self.rotation_activation(self._rotation)
        if hasattr(self, "bg_rotation") and self.bg_rotation.shape[0]>0:
            bg_rot = self.rotation_activation(self.bg_rotation)
            rot = torch.cat([rot, bg_rot])
        
        return rot
    
    @property
    def get_xyz(self):
        xyz = self._rayo + self._rayd * self._zval
        if hasattr(self, "bg_xyz") and self.bg_xyz.shape[0]>0:
            xyz = torch.cat([xyz, self.bg_xyz])
        
        return xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        
        if hasattr(self, "bg_features_dc") and self.bg_features_dc.shape[0]>0:
            features_dc = torch.cat([features_dc, self.bg_features_dc])
            features_rest = torch.cat([features_rest, self.bg_features_rest])
        
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        
        opa = self.opacity_activation(self._opacity)
        if hasattr(self, "bg_opacity") and self.bg_opacity.shape[0]>0:
            bg_opa = self.opacity_activation(self.bg_opacity)
            opa = torch.cat([opa, bg_opa])
        
        return opa
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    def get_z_val(self):
        state_dict = {}
        for key, vgs in self.view_gs.items():
            state_dict[key] = {}
            for key1, v in vgs["match_infos"].items():
                state_dict[key][key1] = v["z_val"].data
            
        return state_dict
    
    def load_z_val(self, state_dicts):
        for key in self.view_gs.keys():
            for key1, v in self.view_gs[key]["match_infos"].items():
                self.view_gs[key]["match_infos"][key1]["z_val"] = nn.Parameter(state_dicts[key][key1], requires_grad=True)
    
    def get_matchloss_from_base(self): # L_gp
        
        # key0, key1 = random.sample(list(self.view_gs.keys()), 2)
        
        keys = list(self.view_gs.keys())
        loss_state = {}
        
        match_loss = 0
        for ik, key0 in enumerate(keys[:-1]):
            for key1 in keys[ik+1:]:
        
                intr0, w2c0 = self.view_gs[key0]["intr"], self.view_gs[key0]["w2c"]
                intr1, w2c1 = self.view_gs[key1]["intr"], self.view_gs[key1]["w2c"]
            
                width, height = self.view_gs[key0]["width"], self.view_gs[key0]["height"]
                
                rayso0 = self.view_gs[key0]["match_infos"][key1]["rays_o"]
                raysd0 = self.view_gs[key0]["match_infos"][key1]["rays_d"]
                zvald0 = self.view_gs[key0]["match_infos"][key1]["z_val"] # z是随机初始化的
                cam_rays_d0  = self.view_gs[key0]["match_infos"][key1]["cam_rays_d"]
                depth0 = zvald0.squeeze(-1) * cam_rays_d0[:, 2]
                uv0 = self.view_gs[key0]["match_infos"][key1]["uv"].permute(1, 0)
                mask0 = self.view_gs[key0]["match_infos"][key1]["blender_mask"]
                
                rayso1 = self.view_gs[key1]["match_infos"][key0]["rays_o"]
                raysd1 = self.view_gs[key1]["match_infos"][key0]["rays_d"]
                zvald1 = self.view_gs[key1]["match_infos"][key0]["z_val"]
                cam_rays_d1  = self.view_gs[key1]["match_infos"][key0]["cam_rays_d"]
                depth1 = zvald1.squeeze(-1) * cam_rays_d1[:, 2]
                uv1 = self.view_gs[key1]["match_infos"][key0]["uv"].permute(1, 0)
                mask1 = self.view_gs[key1]["match_infos"][key0]["blender_mask"]
                
                valid_mask = (mask0 * mask1) > 0
                
                world_pts0 = (rayso0 + raysd0 * zvald0).permute(1, 0) # 世界坐标系
                world_pts1 = (rayso1 + raysd1 * zvald1).permute(1, 0)
                # wrap 0->1
                cam_pts_0to1 = torch.matmul(w2c1, torch.cat([world_pts0, torch.ones_like(world_pts0[:1])]))[:3] # 相机坐标系
                xyz_0to1 = torch.matmul(intr1, cam_pts_0to1)
                depth_0to1 = xyz_0to1[2]
                xy_0to1 = xyz_0to1[:2] / (xyz_0to1[2:] + 1e-8) # 像素坐标
                mask_0to1 = (xy_0to1[0] > 0) & (xy_0to1[0] < width) & (xy_0to1[1] > 0) & (xy_0to1[1] < height)
                # wrap 1->0
                cam_pts_1to0 = torch.matmul(w2c0, torch.cat([world_pts1, torch.ones_like(world_pts1[:1])]))[:3]
                xyz_1to0 = torch.matmul(intr0, cam_pts_1to0)
                depth_1to0 = xyz_1to0[2]
                xy_1to0 = xyz_1to0[:2] / (xyz_1to0[2:] + 1e-8)
                mask_1to0 = (xy_1to0[0] > 0) & (xy_1to0[0] < width) & (xy_1to0[1] > 0) & (xy_1to0[1] < height)
                
                # match_loss = (((xy_0to1 - xy1).abs()[:, mask_0to1]/ torch.tensor([width, height]).type_as(xy1).reshape(2, 1)).mean() + ((xy_1to0 - xy0).abs()[:, mask_1to0]) / torch.tensor([width, height]).type_as(xy1).reshape(2, 1))
                
                ml_0t1 = ((xy_0to1 - uv1).abs() / torch.tensor([width, height]).type_as(uv1).reshape(2, 1)).mean(dim=0) # + (depth_0to1 - depth1).abs()
                ml_1t0 = ((xy_1to0 - uv0).abs() / torch.tensor([width, height]).type_as(uv0).reshape(2, 1)).mean(dim=0) # + (depth_1to0 - depth0).abs()
                
                if key0 not in loss_state:
                    loss_state[key0] = {}
                if key1 not in loss_state:
                    loss_state[key1] = {}
                loss_state[key0][key1] = ml_0t1 #+ (depth_0to1 - depth1).abs() / depth1
                loss_state[key1][key0] = ml_1t0 #+ (depth_1to0 - depth0).abs() / depth0
                
                match_loss += ml_0t1[valid_mask].mean() + ml_1t0[valid_mask].mean()
                # match_loss += ml_0t1.mean() + ml_1t0.mean()
        
        return match_loss, loss_state
    
    def get_matchloss_from_renderdepth(self, cam0, depth0, loss_state):
        intr0, w2c0, img_name0 = cam0.intr, cam0.w2c, cam0.image_name
       
        width, height = self.view_gs[img_name0]["width"], self.view_gs[img_name0]["height"]
        
        depth0 = depth0.squeeze(0)
        
        match_loss = 0
        for img_name1, match_data in self.view_gs[img_name0]["match_infos"].items():
            
            mask0 = self.view_gs[img_name0]["match_infos"][img_name1]["blender_mask"]
            mask1 = self.view_gs[img_name1]["match_infos"][img_name0]["blender_mask"]
            valid_mask = (mask0 * mask1) > 0
            
            xy0 = match_data["uv"]
            norm_x = (xy0[:, 0] / width) * 2 - 1
            norm_y = (xy0[:, 1] / height) * 2 - 1
            norm_grid0 = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0)
            match_depth0 = F.grid_sample(depth0.unsqueeze(0).unsqueeze(0), norm_grid0, mode="bilinear") # 从渲染中的深度图提取深度值
            match_depth0 = match_depth0.reshape(-1)
            
            rayso0, raysd0, cam_rays_d0 = match_data["rays_o"], match_data["rays_d"], match_data["cam_rays_d"]
            zval0 = (match_depth0 / cam_rays_d0[:, 2]).unsqueeze(-1) # 获得z值
            world_pts0 = (rayso0 + raysd0 * zval0).permute(1, 0) # 获得3D点
            
            intr1, w2c1 = self.view_gs[img_name1]["intr"], self.view_gs[img_name1]["w2c"]
            
            cam_pts_0to1 = torch.matmul(w2c1, torch.cat([world_pts0, torch.ones_like(world_pts0[:1])]))[:3]
            xyz_0to1 = torch.matmul(intr1, cam_pts_0to1)
            depth_0to1 = xyz_0to1[2]
            xy_0to1 = xyz_0to1[:2] / (xyz_0to1[2:] + 1e-8)
            mask_0to1 = (xy_0to1[0] > 0) & (xy_0to1[0] < width) & (xy_0to1[1] > 0) & (xy_0to1[1] < height)
            
            xy1 = self.view_gs[img_name1]["match_infos"][img_name0]["uv"].permute(1, 0)
            
            # ls_mask = (loss_state[img_name0][img_name1] < 0.1).float()
            
            match_loss_curr = ((xy_0to1 - xy1).abs() / torch.tensor([width, height]).type_as(xy1).reshape(2, 1)).mean(dim=0)
            # match_loss += (match_loss_curr * mask_0to1.float() * ls_mask).sum() / ((mask_0to1.float() * ls_mask).sum() + 1e-8)
            match_loss += (match_loss_curr * mask_0to1.float() * valid_mask.float()).sum() / ((mask_0to1.float() * valid_mask.float()).sum() + 1e-8)
        
        return match_loss,world_pts0.permute(1,0)
        
    def create_from_mono(self, cams, match_data, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale # 包含所有相机的最小范围
        
        self.match_data = match_data
        
        self.view_gs = {}
        
        for ci, cam in enumerate(cams):
            image, image_name, R, T, FovX, FovY = cam.image, cam.image_name, cam.R, cam.T, cam.FovX, cam.FovY
            near_far = torch.tensor(cam.near_far).float().cuda()
            image = torch.from_numpy(np.array(image)).float().cuda() / 255.0    # (h, w, 3)
            height, width = image.shape[:2]
            if cam.blendermask is not None:
                blender_mask = torch.tensor(cam.blendermask).float().cuda()
            else:
                blender_mask = torch.ones_like(image[..., 0])
            
            fx, fy = fov2focal(FovX, width), fov2focal(FovY, height)
            cx, cy = width / 2.0, height / 2.0
            intr = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().cuda() # 内参
            
            R = torch.tensor(R).float().cuda()
            T = torch.tensor(T).float().cuda()
            w2c = torch.zeros((4, 4)).float().cuda()
            w2c[:3, :3] = R.transpose(0, 1)
            w2c[:3, 3] = T
            w2c[3, 3] = 1.0 # w2c矩阵
            
            match_infos = {}
            
            for cj in range(len(cams)):
                if cj == ci:
                    continue
                image_name1 = cams[cj].image_name
                match_pixel = torch.tensor(match_data[image_name][image_name1]).type_as(image)  # (n, 2) 
                pixels_x = match_pixel[:, 0] * width # 全分辨率下的像素坐标x
                pixels_y = match_pixel[:, 1] * height # 全分辨率下的像素坐标y
                uv = torch.stack([pixels_x, pixels_y], dim=-1) # x,y组合起来
                
                norm_grid = (match_pixel * 2 - 1).unsqueeze(0).unsqueeze(0) # -1 - 1
                warp_color = F.grid_sample(image.permute(2, 0, 1).unsqueeze(0), norm_grid, mode="bilinear") # 在图像上采样这些点的颜色
                warp_color = warp_color.reshape(3, -1).permute(1, 0)
                
                warp_mask = F.grid_sample(blender_mask.unsqueeze(0).unsqueeze(0), norm_grid, mode="bilinear")
                warp_mask = warp_mask.reshape(-1)
                
                homo_pixel = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # [n, 3] 2维齐次坐标
                p = torch.matmul(intr.inverse()[None, :3, :3], homo_pixel[:, :, None]).squeeze() # [n, 3] 相机坐标系下的坐标
                rays_d = p / (torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) + 1e-8)    # [n, 3] 先计算p的模，然后除以模得到单位方向向量
                rays_d = torch.matmul(w2c.inverse()[None, :3, :3], rays_d[:, :, None]).squeeze()  # [n, 3] 将单位方向向量转到世界坐标系下
                rays_o = w2c.inverse()[None, :3, 3].expand(rays_d.shape) # [n, 3] 平移部分表示原点位置，扩展到n个，表示所有的方向向量的原点，
                
                cam_rays_d = torch.matmul(w2c[None, :3, :3], rays_d[:, :, None]).squeeze() # [n, 3] 将单位方向向量转到相机坐标系下
                # z_val = (warp_depth.reshape(-1) / (cam_rays_d[:, 2] + 1e-8)).unsqueeze(-1)
                
                z_val = torch.rand(cam_rays_d.shape[0], 1).type_as(cam_rays_d) * (near_far[1] - near_far[0]) + near_far[0] # 在near_far范围内随机采样
                z_val = nn.Parameter(z_val.requires_grad_(True)) # 作为优化参数
                
                match_infos[image_name1] = {}
                match_infos[image_name1]["z_val"] = z_val
                match_infos[image_name1]["rays_o"] = rays_o # 世界坐标系下的
                match_infos[image_name1]["rays_d"] = rays_d # 世界坐标系下的
                match_infos[image_name1]["cam_rays_d"] = cam_rays_d
                match_infos[image_name1]["color"] = warp_color
                match_infos[image_name1]["uv"] = uv
                match_infos[image_name1]["match_pixel"] = match_pixel
                match_infos[image_name1]["blender_mask"] = warp_mask
            
            self.view_gs[image_name] = {
                "image_color": image,
                "intr": intr,
                "w2c": w2c,
                "height": height,
                "width": width,
                "match_infos": match_infos,
                "near_far": near_far,
                "image_name": image_name
            }
            
    def create_from_pcd(self, min_loss_state):
        
        intrs = []
        w2cs = []
        img_colors = []
        sparse_depths = []
        near_fars = []
        
        colors = []
        points = []
        rayos = []
        rayds = []
        zvals = []
        
        exist_keys = []
        
        for idx, (key, vgs) in enumerate(self.view_gs.items()):
            
            exist_keys.append(key)
            
            intrs.append(vgs["intr"])
            w2cs.append(vgs["w2c"])
            img_colors.append(vgs["image_color"].reshape(vgs["height"], vgs["width"], 3).permute(2, 0, 1))
            near_fars.append(vgs["near_far"])
            
            view_num_points = 0
            sparse_depth = torch.zeros(vgs["height"], vgs["width"]).type_as(vgs["image_color"])
            for key1, v in vgs["match_infos"].items():

                if min_loss_state is not None:
                    vmask = min_loss_state[key][key1] <  0.005 #0.1 #0.001
                else:
                    vmask = torch.ones_like(v["rays_o"][..., 0]) > 0

                rayso = v["rays_o"][vmask]
                raysd = v["rays_d"][vmask]
                z_val = v["z_val"][vmask]
                color = v["color"][vmask]
                uv = v["uv"][vmask]
                cam_rays_d = v["cam_rays_d"][vmask]
                
                depth = z_val.squeeze(-1) * cam_rays_d[:, 2]
                sparse_depth[(uv[:, 1].clamp(0, vgs["height"]-1).to(torch.int64), uv[:, 0].clamp(0, vgs["width"]-1).to(torch.int64))] = depth
                
                xyz = rayso + raysd * z_val
                
                points.append(xyz)
                colors.append(color)
                rayos.append(rayso)
                rayds.append(raysd)
                zvals.append(z_val)
                view_num_points += z_val.shape[0]
            
            sparse_depths.append(sparse_depth)

        
        self.intrs = torch.stack(intrs)
        self.w2cs = torch.stack(w2cs)
        self.sparse_depths = torch.stack(sparse_depths)
        self.img_colors = torch.stack(img_colors)
        self.masks = self.sparse_depths > 0
        self.near_fars = torch.stack(near_fars)
        
        self.curr_scale = 1
        self.curr_patch_size = 5
        
        colors = torch.cat(colors)
        points = torch.cat(points)
        rayos = torch.cat(rayos)
        rayds = torch.cat(rayds)
        zvals = torch.cat(zvals)
        
        fused_color = RGB2SH(colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", zvals.shape[0])

        dist2 = torch.clamp_min(distCUDA2(points)[0], 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((zvals.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((zvals.shape[0], 1), dtype=torch.float, device="cuda"))

        # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._zval = nn.Parameter(zvals.requires_grad_(True))
        self._rayo = rayos
        self._rayd = rayds
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # bg_xyz = []
        # bg_colors = []
        # for i,(k,bg) in enumerate(self.new_points_dict.items()):
        #     bg_xyz.append(bg["world_pts"])
        #     bg_colors.append(bg["color"])

        # bg_xyz = torch.from_numpy(np.concatenate(bg_xyz)).float().cuda()
        # bg_colors = torch.from_numpy(np.concatenate(bg_colors)).float().cuda()
        # bg_fused_color = RGB2SH(colors)
        # bg_features = torch.zeros((bg_fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # bg_features[:, :3, 0 ] = bg_fused_color
        # bg_features[:, 3:, 1:] = 0.0

        # bg_dist2 = torch.clamp_min(distCUDA2(bg_xyz)[0], 0.0000001)
        # bg_scales = torch.log(torch.sqrt(bg_dist2))[...,None].repeat(1, 3)
        # bg_rots = torch.zeros((bg_xyz.shape[0], 4), device="cuda")
        # bg_rots[:, 0] = 1

        # bg_opacities = inverse_sigmoid(0.1 * torch.ones((bg_xyz.shape[0], 1), dtype=torch.float, device="cuda"))

        # self.bg_xyz = nn.Parameter(bg_xyz.requires_grad_(True))
        # self.bg_features_dc = nn.Parameter(bg_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self.bg_features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self.bg_scaling = nn.Parameter(bg_scales.requires_grad_(True))
        # self.bg_rotation = nn.Parameter(bg_rots.requires_grad_(True))
        # self.bg_opacity = nn.Parameter(bg_opacities.requires_grad_(True))

        self.bg_xyz = nn.Parameter(torch.empty(0).cuda())
        self.bg_features_dc = nn.Parameter(torch.empty(0).cuda())
        self.bg_features_rest = nn.Parameter(torch.empty(0).cuda())
        self.bg_scaling = nn.Parameter(torch.empty(0).cuda())
        self.bg_rotation = nn.Parameter(torch.empty(0).cuda())
        self.bg_opacity = nn.Parameter(torch.empty(0).cuda())
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup_init(self, ratio=1.0):
        l_init = []
        print("len(self.view_gs):", len(self.view_gs))
        for key, vgs in self.view_gs.items():
            
            l_init.append({'params': [v["z_val"] for k, v in vgs["match_infos"].items()], 'lr': 0.5*ratio, "name": f"z_val_{key}"})
            
        self.optimizer_init = torch.optim.Adam(l_init, lr=0.0, eps=1e-15)
        
    def update_learning_rate_init(self, ratio):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer_init.param_groups:
            if "z_val" in param_group["name"]:
                lr = param_group['lr']
                param_group['lr'] = lr * ratio

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._zval], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "zval"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        l_bg = [
            {'params': [self.bg_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "bg_xyz"},
            {'params': [self.bg_features_dc], 'lr': training_args.feature_lr, "name": "bg_f_dc"},
            {'params': [self.bg_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "bg_f_rest"},
            {'params': [self.bg_opacity], 'lr': training_args.opacity_lr, "name": "bg_opacity"},
            {'params': [self.bg_scaling], 'lr': training_args.scaling_lr, "name": "bg_scaling"},
            {'params': [self.bg_rotation], 'lr': training_args.rotation_lr, "name": "bg_rotation"}
        ]
        self.optimizer_bg = torch.optim.Adam(l_bg, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "zval":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
        for param_group in self.optimizer_bg.param_groups:
            if param_group["name"] == "bg_xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                break

    def construct_list_of_attributes_all(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._zval.shape[1]):
            l.append('zval_{}'.format(i))
        for i in range(self._rayo.shape[1]):
            l.append('rayo_{}'.format(i))
        for i in range(self._rayd.shape[1]):
            l.append('rayd_{}'.format(i))
        return l
    
    def construct_list_of_attributes_bg(self):
        l = ['bx', 'by', 'bz', 'bnx', 'bny', 'bnz']
        # All channels except the 3 DC
        for i in range(self.bg_features_dc.shape[1]*self.bg_features_dc.shape[2]):
            l.append('bf_dc_{}'.format(i))
        for i in range(self.bg_features_rest.shape[1]*self.bg_features_rest.shape[2]):
            l.append('bf_rest_{}'.format(i))
        l.append('bopacity')
        for i in range(self.bg_scaling.shape[1]):
            l.append('bscale_{}'.format(i))
        for i in range(self.bg_rotation.shape[1]):
            l.append('brot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        # xyz = self._xyz.detach().cpu().numpy()
        xyz = (self._rayo + self._rayd * self._zval).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        zval = self._zval.detach().cpu().numpy()
        rayo = self._rayo.detach().cpu().numpy()
        rayd = self._rayd.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, zval, rayo, rayd), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        if self.bg_xyz.shape[0]>0:
            bg_xyz = self.bg_xyz.detach().cpu().numpy()
            bg_normals = np.zeros_like(bg_xyz)
            bg_dc = self.bg_features_dc.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            bg_rest = self.bg_features_rest.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            bg_opacity = self.bg_opacity.detach().cpu().numpy()
            bg_scale = self.bg_scaling.detach().cpu().numpy()
            bg_rotation = self.bg_rotation.detach().cpu().numpy()
            bg_dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_bg()]
            bg_elements = np.empty(bg_xyz.shape[0], dtype=bg_dtype_full)
            bg_attributes = np.concatenate((bg_xyz, bg_normals, bg_dc, bg_rest, bg_opacity, bg_scale, bg_rotation), axis=1)
            bg_elements[:] = list(map(tuple, bg_attributes))
            bg_el = PlyElement.describe(bg_elements, 'vertex')
            PlyData([bg_el]).write(os.path.join(os.path.dirname(path), "point_cloud_bg.ply"))
        
        if self.bg_xyz.shape[0]>0:
            xyz = np.concatenate([xyz, bg_xyz])
            normals = np.concatenate([normals, bg_normals])
            f_dc = np.concatenate([f_dc, bg_dc])
            f_rest = np.concatenate([f_rest, bg_rest])
            opacities = np.concatenate([opacities, bg_opacity])
            scale = np.concatenate([scale, bg_scale])
            rotation = np.concatenate([rotation, bg_rotation])
        else:
            xyz = xyz
            normals = normals
            f_dc = f_dc
            f_rest = f_rest
            opacities = opacities
            scale = scale
            rotation = rotation

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_all()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(os.path.dirname(path), "point_cloud_all.ply"))
        
    def save_ply_at_matchpoint(self, path):
        dirname = os.path.dirname(path)
        mkdir_p(dirname)
        
        xyzs = []
        colors = []
        
        for key, vgs in self.view_gs.items():
            sparse_depth = torch.zeros(vgs["height"], vgs["width"]).float().cuda()
            for key1, v in vgs["match_infos"].items():
                rayso = v["rays_o"]
                raysd = v["rays_d"]
                z_val = v["z_val"]
                color = v["color"]
                xyz = rayso + raysd * z_val # 坐标
                xyzs.append(xyz)
                colors.append(color)
                
                uv = v["uv"]
                cam_rays_d = v["cam_rays_d"]
                
                depth = z_val.squeeze(-1) * cam_rays_d[:, 2]
                sparse_depth[(uv[:, 1].clamp(0, vgs["height"]-1).to(torch.int64), uv[:, 0].clamp(0, vgs["width"]-1).to(torch.int64))] = depth
            
            np.save(os.path.join(dirname, f"{key}.npy"), sparse_depth.detach().cpu().numpy()) # 稀疏深度图
            sparse_depth = (sparse_depth - sparse_depth.min()) / (sparse_depth.max() - sparse_depth.min()) # 归一化
            torchvision.utils.save_image(sparse_depth, os.path.join(dirname, f"sparsedepth_{key}.png"))
        
        for i,(k,bg) in enumerate(self.new_points_dict.items()):
            xyzs.append(torch.from_numpy(bg["world_pts"]).cuda())
            colors.append(torch.from_numpy(bg["color"]).cuda())

        xyz = torch.cat(xyzs)
        color = torch.cat(colors)
                
        storePly(path, xyz.detach().cpu().numpy(), color.detach().cpu().numpy() * 255)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)) # 不透明度超过0.01都设为0.01
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new[:self._zval.shape[0]], "opacity", self.optimizer)
        self._opacity = optimizable_tensors["opacity"]
        if self.bg_opacity.shape[0]>0:
            # print("self.bg_opacity:", self.bg_opacity.shape, opacities_new.shape)
            optimizable_tensors_bg = self.replace_tensor_to_optimizer(opacities_new[self._zval.shape[0]:], "bg_opacity", self.optimizer_bg)
            self.bg_opacity = optimizable_tensors_bg["bg_opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        zval_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("zval")]
        zval_names = sorted(zval_names, key = lambda x: int(x.split('_')[-1]))
        zval = np.zeros((xyz.shape[0], len(zval_names)))
        for idx, attr_name in enumerate(zval_names):
            zval[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        rayo_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rayo")]
        rayo_names = sorted(rayo_names, key = lambda x: int(x.split('_')[-1]))
        rayo = np.zeros((xyz.shape[0], len(rayo_names)))
        for idx, attr_name in enumerate(rayo_names):
            rayo[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        rayd_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rayd")]
        rayd_names = sorted(rayd_names, key = lambda x: int(x.split('_')[-1]))
        rayd = np.zeros((xyz.shape[0], len(rayd_names)))
        for idx, attr_name in enumerate(rayd_names):
            rayd[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._zval = nn.Parameter(torch.tensor(zval, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rayo = torch.tensor(rayo, dtype=torch.float, device="cuda")
        self._rayd = torch.tensor(rayd, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree
        
        if os.path.exists(os.path.join(os.path.dirname(path), "point_cloud_bg.ply")):
            plydata = PlyData.read(os.path.join(os.path.dirname(path), "point_cloud_bg.ply"))

            bg_xyz = np.stack((np.asarray(plydata.elements[0]["bx"]),
                            np.asarray(plydata.elements[0]["by"]),
                            np.asarray(plydata.elements[0]["bz"])),  axis=1)
            bg_opacities = np.asarray(plydata.elements[0]["bopacity"])[..., np.newaxis]

            bg_features_dc = np.zeros((bg_xyz.shape[0], 3, 1))
            bg_features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["bf_dc_0"])
            bg_features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["bf_dc_1"])
            bg_features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["bf_dc_2"])

            bg_extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("bf_rest_")]
            bg_extra_f_names = sorted(bg_extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(bg_extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            bg_features_extra = np.zeros((bg_xyz.shape[0], len(bg_extra_f_names)))
            for idx, attr_name in enumerate(bg_extra_f_names):
                bg_features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            bg_features_extra = bg_features_extra.reshape((bg_features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

            bg_scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("bscale_")]
            bg_scale_names = sorted(bg_scale_names, key = lambda x: int(x.split('_')[-1]))
            bg_scales = np.zeros((bg_xyz.shape[0], len(bg_scale_names)))
            for idx, attr_name in enumerate(bg_scale_names):
                bg_scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

            bg_rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("brot")]
            bg_rot_names = sorted(bg_rot_names, key = lambda x: int(x.split('_')[-1]))
            bg_rots = np.zeros((bg_xyz.shape[0], len(bg_rot_names)))
            for idx, attr_name in enumerate(bg_rot_names):
                bg_rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
                
            self.bg_xyz = nn.Parameter(torch.tensor(bg_xyz, dtype=torch.float, device="cuda").requires_grad_(True))
            self.bg_features_dc = nn.Parameter(torch.tensor(bg_features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self.bg_features_rest = nn.Parameter(torch.tensor(bg_features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self.bg_opacity = nn.Parameter(torch.tensor(bg_opacities, dtype=torch.float, device="cuda").requires_grad_(True))
            self.bg_scaling = nn.Parameter(torch.tensor(bg_scales, dtype=torch.float, device="cuda").requires_grad_(True))
            self.bg_rotation = nn.Parameter(torch.tensor(bg_rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] == name:
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask[:self._zval.shape[0]], self.optimizer)
        optimizable_tensors_bg = self._prune_optimizer(valid_points_mask[self._zval.shape[0]:], self.optimizer_bg)

        self._rayo = self._rayo[valid_points_mask[:self._zval.shape[0]]]
        self._rayd = self._rayd[valid_points_mask[:self._zval.shape[0]]]
        
        self._zval = optimizable_tensors["zval"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.bg_xyz = optimizable_tensors_bg["bg_xyz"]
        self.bg_features_dc = optimizable_tensors_bg["bg_f_dc"]
        self.bg_features_rest = optimizable_tensors_bg["bg_f_rest"]
        self.bg_opacity = optimizable_tensors_bg["bg_opacity"]
        self.bg_scaling = optimizable_tensors_bg["bg_scaling"]
        self.bg_rotation = optimizable_tensors_bg["bg_rotation"]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"bg_xyz": new_xyz,
        "bg_f_dc": new_features_dc,
        "bg_f_rest": new_features_rest,
        "bg_opacity": new_opacities,
        "bg_scaling" : new_scaling,
        "bg_rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d, self.optimizer_bg)
        self.bg_xyz = optimizable_tensors["bg_xyz"]
        self.bg_features_dc = optimizable_tensors["bg_f_dc"]
        self.bg_features_rest = optimizable_tensors["bg_f_rest"]
        self.bg_opacity = optimizable_tensors["bg_opacity"]
        self.bg_scaling = optimizable_tensors["bg_scaling"]
        self.bg_rotation = optimizable_tensors["bg_rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def proximity(self, scene_extent, N = 3):
        dist, nearest_indices = distCUDA2(self.get_xyz)
        selected_pts_mask = torch.logical_and(dist > ( 5.0*scene_extent),
                                              torch.max(self.get_scaling, dim=1).values > (scene_extent))

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self.get_xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self.get_xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = torch.cat([self._scaling, self.bg_scaling])[new_indices]
        new_rotation = torch.zeros_like(torch.cat([self._rotation, self.bg_rotation])[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(torch.cat([self._features_dc, self.bg_features_dc])[new_indices])
        new_features_rest = torch.zeros_like(torch.cat([self._features_rest, self.bg_features_rest])[new_indices])
        new_opacity = torch.cat([self._opacity, self.bg_opacity])[new_indices]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        dist, _ = distCUDA2(self.get_xyz)
        selected_pts_mask2 = torch.logical_and(dist > (10. * scene_extent),
                                               torch.max(self.get_scaling, dim=1).values > ( scene_extent))
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask2)
        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(torch.cat([self._rotation, self.bg_rotation])[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = torch.cat([self._rotation, self.bg_rotation])[selected_pts_mask].repeat(N,1)
        new_features_dc = torch.cat([self._features_dc, self.bg_features_dc])[selected_pts_mask].repeat(N,1,1)
        new_features_rest = torch.cat([self._features_rest, self.bg_features_rest])[selected_pts_mask].repeat(N,1,1)
        new_opacity = torch.cat([self._opacity, self.bg_opacity])[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        split_num = selected_pts_mask.sum()
        
        # change scaling and don't prune ray gs
        new_scaling_ = self._scaling.clone()
        new_scaling_[selected_pts_mask[:self._zval.shape[0]]] /= 0.8*N
        optimizable_tensors = self.replace_tensor_to_optimizer(new_scaling_, "scaling", self.optimizer)
        self._scaling = optimizable_tensors["scaling"]
        selected_pts_mask[:self._zval.shape[0]] = False
        
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * split_num, device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # print("clone selected_pts_mask:", selected_pts_mask.float().mean().item())
        
        new_xyz = self.get_xyz[selected_pts_mask]
        new_features_dc = torch.cat([self._features_dc, self.bg_features_dc])[selected_pts_mask]
        new_features_rest = torch.cat([self._features_rest, self.bg_features_rest])[selected_pts_mask]
        new_opacities = torch.cat([self._opacity, self.bg_opacity])[selected_pts_mask]
        new_scaling = torch.cat([self._scaling, self.bg_scaling])[selected_pts_mask]
        new_rotation = torch.cat([self._rotation, self.bg_rotation])[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        self.proximity(extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze() # 低不透明度
        if max_screen_size:
            big_points_vs = self.max_radii2D > 1.5 * max_screen_size # 超大点
            big_points_ws = self.get_scaling.max(dim=1).values > 0.2 * extent 
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        prune_mask[:self._zval.shape[0]] = False # 不修改初始高斯
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_split_KNN(self, KNN_index, N=2):
        n_init_points = self.get_xyz.shape[0]
        all_scaling = self.get_scaling
        knn_scaling_avg = torch.zeros_like(all_scaling, device="cuda")
        for i in range(n_init_points):
            knn_indices = KNN_index[i].long()
            knn_scaling_avg[i] = all_scaling[knn_indices].mean(dim=0)
            # print("knn_scaling_avg:", knn_scaling_avg[i])
        # import pdb; pdb.set_trace()
        selected_pts_mask = (all_scaling > knn_scaling_avg).all(dim=1)
        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(torch.cat([self._rotation, self.bg_rotation])[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = torch.cat([self._rotation, self.bg_rotation])[selected_pts_mask].repeat(N,1)
        new_features_dc = torch.cat([self._features_dc, self.bg_features_dc])[selected_pts_mask].repeat(N,1,1)
        new_features_rest = torch.cat([self._features_rest, self.bg_features_rest])[selected_pts_mask].repeat(N,1,1)
        new_opacity = torch.cat([self._opacity, self.bg_opacity])[selected_pts_mask].repeat(N,1)
        # print("split:", new_xyz.shape[0])
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        split_num = selected_pts_mask.sum()
        
        # change scaling and don't prune ray gs
        new_scaling_ = self._scaling.clone()
        new_scaling_[selected_pts_mask[:self._zval.shape[0]]] /= 0.8*N
        optimizable_tensors = self.replace_tensor_to_optimizer(new_scaling_, "scaling", self.optimizer)
        self._scaling = optimizable_tensors["scaling"]
        selected_pts_mask[:self._zval.shape[0]] = False
        
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * split_num, device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def optimize_mono_depth(self, viewpoint_stack):
        xyzs = []
        
        for key, vgs in self.view_gs.items():
            xyzs = []
            sparse_depth = torch.zeros(vgs["height"], vgs["width"]).float().cuda()
            image_name = vgs["image_name"]
            for key1, v in vgs["match_infos"].items():
                rayso = v["rays_o"]
                raysd = v["rays_d"]
                z_val = v["z_val"]
                color = v["color"]
                xyz = rayso + raysd * z_val # 坐标
                xyzs.append(xyz)

            xyz = torch.cat(xyzs).detach().cpu().numpy() # N,3
            self.new_points_dict = {}
            cam = next(camera for camera in viewpoint_stack if camera.image_name == image_name)

        # for i,cam in enumerate(viewpoint_stack):
            # near_far = cam.near_far
            source_depth = cam.depth_mono.cpu().numpy().copy()
            # image = cam.original_image.cpu().numpy().copy()
            H, W = source_depth.shape
            depthmap = np.zeros_like(source_depth)
            
            
            K = cam.intr.cpu().numpy() # 3,3
            R = cam.R # c2w 3,3
            T = cam.T # w2c 3,1
            # w2c = cam.w2c # w2c 4,4
            cam_coord = np.matmul(K, np.matmul(R.transpose(), xyz.transpose()) + T.reshape(3,1)) #相机坐标系
            valid_idx = np.where(np.logical_and.reduce((cam_coord[2]>0, cam_coord[0]/cam_coord[2]>=0, cam_coord[0]/cam_coord[2]<=W-1, cam_coord[1]/cam_coord[2]>=0, cam_coord[1]/cam_coord[2]<=H-1)))[0]
            pts_depths = cam_coord[-1:, valid_idx]
            cam_coord = cam_coord[:2, valid_idx]/cam_coord[-1:, valid_idx] # 2,N

            depthmap[np.round(cam_coord[1]).astype(np.int32).clip(0,H-1), np.round(cam_coord[0]).astype(np.int32).clip(0,W-1)] = pts_depths

            # depthmap2, depthloss = optimize_depth(source=source_depth, target=depthmap, mask=depthmap>0.0)
            d_mono_alin = optimize_depth2(source_depth, cam_coord, depthmap)

            # pearson_loss = pearson_depth_loss(torch.from_numpy(d_mono_alin).cuda(),torch.from_numpy(source_depth).cuda())

            cam.depth_mono = torch.from_numpy(d_mono_alin).cuda()



def optimize_depth(source, target, mask, prune_ratio=0.001):
    """
    Arguments
    =========
    source: np.array(h,w)
    target: np.array(h,w)
    mask: np.array(h,w):
        array of [True if valid pointcloud is visible.]
    depth_weight: np.array(h,w):
        weight array at loss.
    Returns
    =======
    refined_source: np.array(h,w)
        literally "refined" source.
    loss: float
    """
    source = torch.from_numpy(source).cuda()
    target = torch.from_numpy(target).cuda()
    mask = torch.from_numpy(mask).cuda()
    # depth_weight = torch.from_numpy(depth_weight).cuda()

    # Prune some depths considered "outlier"     
    with torch.no_grad():
        target_depth_sorted = target[target>1e-7].sort().values
        min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*prune_ratio)]
        max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*(1.0-prune_ratio))]

        mask2 = target > min_prune_threshold
        mask3 = target < max_prune_threshold
        mask = torch.logical_and( torch.logical_and(mask, mask2), mask3)

    source_masked = source[mask]
    target_masked = target[mask]
    # depth_weight_masked = depth_weight[mask]
    # tmin, tmax = target_masked.min(), target_masked.max()

    # # Normalize
    # target_masked = target_masked - tmin 
    # target_masked = target_masked / (tmax-tmin)

    scale = torch.ones(1).cuda().requires_grad_(True)
    shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

    optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
    loss = torch.ones(1).cuda() * 1e5

    iteration = 1
    loss_prev = 1e6
    loss_ema = 0.0
    
    while abs(loss_ema - loss_prev) > 1e-5:
        source_hat = scale*source_masked + shift
        loss = torch.mean(((target_masked - source_hat)**2))

        # penalize depths not in [0,1]
        loss_hinge1 = loss_hinge2 = 0.0
        if (source_hat<=0.0).any():
            loss_hinge1 = 2.0*((source_hat[source_hat<=0.0])**2).mean()
        # if (source_hat>=1.0).any():
        #     loss_hinge2 = 0.3*((source_hat[source_hat>=1.0])**2).mean() 
        
        loss = loss + loss_hinge1 + loss_hinge2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        iteration+=1
        if iteration % 1000 == 0:
            print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            loss_prev = loss.item()
        loss_ema = loss.item() * 0.2 + loss_ema * 0.8

    loss = loss.item()
    print(f"loss ={loss:10.5f}")

    with torch.no_grad():
        refined_source = (scale*source + shift) 
    torch.cuda.empty_cache()
    return refined_source.cpu().numpy(), loss

from scipy.interpolate import PchipInterpolator

def optimize_depth2(d_mono, pts_2d, d_gt, num_bins = 5):
    H, W = d_mono.shape
    # pts_2d.transpose(1,0)
    x = np.round(pts_2d[0]).astype(np.int32).clip(0,W-1)
    y = np.round(pts_2d[1]).astype(np.int32).clip(0,H-1)

    d_mono_samples = d_mono[y, x]
    d_gt_samples = d_gt[y, x]

    sort_idx = np.argsort(d_mono_samples)
    mono_sorted = d_mono_samples[sort_idx]
    gt_sorted = d_gt_samples[sort_idx]
    valid_mask = (mono_sorted > 3) & (mono_sorted < 253)
    mono_filtered = mono_sorted[valid_mask]
    gt_filtered = gt_sorted[valid_mask]

    bin_edges = np.linspace(0, len(mono_filtered) - 1, num_bins).astype(int)
    mono_sampled = mono_filtered[bin_edges]
    gt_sampled = gt_filtered[bin_edges]
    fill_min = gt_sampled.min()
    fill_max = gt_sampled.max()

    interp_fn = interp1d(mono_sampled, gt_sampled, kind='linear', fill_value='extrapolate')

    d_mono_aligned = interp_fn(d_mono)
    
    # interp_fn = PchipInterpolator(mono_sampled, gt_sampled, extrapolate=True)
    # d_mono_aligned = interp_fn(d_mono)

    # 保存深度图
    depth_vis = d_mono_aligned.copy()
    depth_vis[np.isnan(depth_vis)] = 0  # 防止 NaN
    depth_vis[np.isinf(depth_vis)] = 0  # 防止 Inf
    max_val = np.percentile(depth_vis, 100)
    depth_norm = np.clip(depth_vis / max_val, 0, 1)

    min_val = np.min(depth_vis)
    depth_norm = np.clip((depth_vis - min_val) / (max_val - min_val), 0, 1)  # 归一化到0-1
    depth_uint8 = (depth_norm * 255).astype(np.uint8)  # 转换为8位灰度图
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    cv2.imwrite("depth_gray.png", depth_color)

    # depth_uint8 = (depth_norm * 255).astype(np.uint8)
    # depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    # cv2.imwrite("depth_colored.png", depth_colored)

    return d_mono_aligned

def pearson_depth_loss(depth_src, depth_target):
    
    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    src_target = src * target

    co = (src_target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co