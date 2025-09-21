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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import random
import torch.nn as nn
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
import numpy as np
import cv2
import math


def pearson_depth_loss(depth_src, depth_target):
    
    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    src_target = src * target

    co = (src_target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co
def pearson_depth_loss_weight(depth_src, depth_target):
    threshold = 0.4
    scale_factor = 0.1
    
    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    src_target = src * target
    weights = torch.ones_like(src_target)
    weights[(src_target < threshold)& (src_target > 0)] = scale_factor
    weights[src_target < 0 ] = 1.5

    co = (src_target * weights).mean() / (weights.mean() + 1e-6)

    assert not torch.any(torch.isnan(co))
    return 1 - co

def local_pearson_loss(depth_src, depth_target, box_p, p_corr):
    # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
    num_box_h = math.floor(depth_src.shape[0]/box_p)
    num_box_w = math.floor(depth_src.shape[1]/box_p)
    max_h = depth_src.shape[0] - box_p #  计算了在选择一个局部区域时，右下角的最大可选位置
    max_w = depth_src.shape[1] - box_p
    _loss = torch.tensor(0.0,device='cuda')
    n_corr = int(p_corr * num_box_h * num_box_w) # n_corr 计算出需要选择的局部区域的数量，等于图像上所有可能区域的 p_corr 百分比。
    x_0 = torch.randint(0, max_h, size=(n_corr,), device = 'cuda')
    y_0 = torch.randint(0, max_w, size=(n_corr,), device = 'cuda')
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p
    _loss = torch.tensor(0.0,device='cuda')
    for i in range(len(x_0)):
        _loss += pearson_depth_loss(depth_src[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), depth_target[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1))
    return _loss/n_corr

def get_smooth_loss(depth, guide=None):
    grad_disp_x = torch.abs(depth[:, :-1] - depth[:, 1:])
    grad_disp_y = torch.abs(depth[:-1, :] - depth[1:, :])
    
    if guide is None:
        guide = torch.ones_like(depth).detach()
    
    if len(guide.shape)==3:
        grad_img_x = torch.abs(guide[:, :, :-1] - guide[:, :, 1:]).mean(dim=0)
        grad_img_y = torch.abs(guide[:, :-1, :] - guide[:, 1:, :]).mean(dim=0)
    else:
        grad_img_x = torch.abs(guide[:, :-1] - guide[:, 1:])
        grad_img_y = torch.abs(guide[:-1, :] - guide[1:, :])

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    
    smooth_loss = grad_disp_x.mean() + grad_disp_y.mean()
        
    return smooth_loss

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    if mask is not None:
        if len(mask.shape)==2:
            mask = mask.unsqueeze(0)
        mask = F.conv2d(mask, window[:1], padding=window_size // 2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        ssim_map = ssim_map * mask
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def margin_l2_loss(network_output, gt, mask_patches, margin, return_mask=False):
    network_output = network_output[mask_patches]
    gt = gt[mask_patches]
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask

def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def patch_norm_mse_loss(input, target, fore_mask, patch_size, margin=0.2, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    mask_patches = patchify(fore_mask, patch_size).sum(dim=1) < (patch_size*patch_size / 3)
    return margin_l2_loss(input_patches, target_patches, mask_patches, margin, return_mask)


def ranking_loss(input, target, patch_size, margin=1e-4):
    input_patches = patchify(input, patch_size)
    target_patches = patchify(target, patch_size)
    
    rand_idxes = random.sample(list(range(input_patches.shape[1])), 6)
    
    input_pixels = input_patches[:, rand_idxes].reshape(-1, 2)
    target_pixels = target_patches[:, rand_idxes].reshape(-1, 2)
    
    g = target_pixels[:, 0] - target_pixels[:, 1]
    t = input_pixels[:, 0] - input_pixels[:, 1]
    
    t = torch.where(g < 0, t, -t)
    
    t = t + margin
    
    l = torch.mean(t[t>0])
    
    return l

def cons_loss(input, target, patch_size, margin=1e-4):
    input_patches = patchify(input, patch_size)
    target_patches = patchify(target, patch_size)
    
    
    tmp = (target_patches[:, :, None] - target_patches[:, None, :]).abs()
    tmp1 = torch.eye(target_patches.shape[1]).unsqueeze(0).repeat(target_patches.shape[0], 1, 1).type_as(tmp)
    tmp[tmp1>1] = 1e5
    
    sorted_args = torch.argsort(tmp, dim=-1)[:, :, :2]
    tmp_t = torch.gather(tmp, -1, sorted_args)
    t = (input_patches[:, :, None] - input_patches[:, None, :]).abs()
    t = torch.gather(t, -1, sorted_args)
    
    t = t - margin
    
    l = torch.mean(t[(t>0) & (tmp_t<0.01)])
    
    return l
    
    
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(5, 1)
        self.mu_y_pool = nn.AvgPool2d(5, 1)
        self.sig_x_pool = nn.AvgPool2d(5, 1)
        self.sig_y_pool = nn.AvgPool2d(5, 1)
        self.sig_xy_pool = nn.AvgPool2d(5, 1)
        self.mask_pool = nn.AvgPool2d(5, 1)
        self.refl = nn.ReflectionPad2d(2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask=None):
        x = self.refl(x)
        y = self.refl(y)
        if mask is not None:
            mask = self.refl(mask)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        if mask is not None:
            SSIM_mask = self.mask_pool(mask)
            output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        else:
            output = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output
    
def get_pixel_loss(image, gt_image):
    
    l1 = (image - gt_image).abs().mean(dim=0)
    
    ssim_func = SSIM()
    
    ssim_l = ssim_func(image.unsqueeze(0), gt_image.unsqueeze(0)).squeeze(0)
    
    pl = l1 * 0.5 + ssim_l.mean(dim=0) * 0.5
    
    return pl
    
    
def get_virtual_warp_loss(virtual_img, virtual_depth, vir_c2w, intrs, w2cs, img_colors, vir_mask):
    height, width = virtual_img.shape[1:]
    virtual_c2w = torch.eye(4, 4).type_as(w2cs)
    virtual_c2w[:3, :4] = torch.from_numpy(vir_c2w).type_as(w2cs)
    intr = intrs[0]
    nv = intrs.shape[0]
    
    py, px = torch.meshgrid(torch.arange(height), torch.arange(width))
    px, py = px.reshape(-1).type_as(w2cs), py.reshape(-1).type_as(w2cs)
    
    cam_pts = torch.matmul(intr.inverse()[:3, :3], torch.stack([px, py, torch.ones_like(px)]) * virtual_depth.reshape(1, -1))
    world_pts = torch.matmul(virtual_c2w, torch.cat([cam_pts, torch.ones_like(cam_pts[:1])]))   # 4, npts
    cam_pts = torch.matmul(w2cs, world_pts[None])[:, :3]    # nv, 3, npts
    cam_xyz = torch.matmul(intrs[:, :3, :3], cam_pts)   # nv, 3, npts
    cam_xy = cam_xyz[:, :2] / (cam_xyz[:, 2:] + 1e-8)
    
    norm_x = 2 * cam_xy[:, 0] / (width - 1) - 1
    norm_y = 2 * cam_xy[:, 1] / (height - 1) - 1
    norm_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)  # nv, 1, npts, 2
    
    mask = (norm_grid.abs() <= 1).all(dim=-1).reshape(nv, height, width)
    
    warp_img = F.grid_sample(img_colors, norm_grid, mode="bilinear")
    warp_img = warp_img.reshape(nv, 3, height, width)
    
    l1 = (virtual_img.unsqueeze(0) - warp_img).abs().mean(dim=1)
    
    ssim_func = SSIM()
    
    ssim_l = ssim_func(virtual_img.unsqueeze(0), warp_img).mean(dim=1)
    
    loss = ssim_l
    
    loss[~mask] = 1000
    
    loss = torch.min(loss, dim=0)[0]
    loss[(loss >= 1000) | (~vir_mask.squeeze(0))] = 0.0
    
    return loss.mean()

def locality_loss(color, position, index): # position:[N,3] index[N,4], color[N,3]
    N, K = index.shape
    colors_i = color.unsqueeze(1).expand(-1, K, -1)
    position_i = position.unsqueeze(1).expand(-1, K, -1)

    colors_k = color[index]
    position_k = position[index]
    valid_mask = (index >= 0) & (index < color.shape[0])

    distances = torch.norm(position_k - position_i, dim=2)
    weights = torch.exp(-2 * distances)
    color_diffs = torch.norm(colors_k - colors_i, dim=2)
    weighted_loss = (weights * color_diffs).sum(dim=1)

    return weighted_loss.mean()

# def locality_loss(pc: GaussianModel,viewpoint_camera,KNN_index,delta=2.0):
#     N = pc.get_xyz.shape[0]
#     k_neighbors = KNN_index[1]
#     neighbor_means = pc.get_xyz[KNN_index]

#     shs_view = pc.get_features.transpose(1, 2).view(
#         -1, 3, (pc.max_sh_degree + 1) ** 2
#     )
#     dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
#         pc.get_features.shape[0], 1
#     )
#     dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
#     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#     colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
#     neighbor_colors = colors[KNN_index]

#     distances = torch.norm(neighbor_means - pc.get_xyz.unsqueeze(1), dim=2)
#     weights = torch.exp(-delta * distances)
#     color_diffs = torch.norm(neighbor_colors - colors.unsqueeze(1), dim=2)

#     weighted_diffs = weights * color_diffs
#     per_gaussian_loss = torch.sum(weighted_diffs)
#     # locality_loss = torch.mean(per_gaussian_loss)

#     return per_gaussian_loss

def smooth_depth_loss(color_image, depth_image, lambda_color=1.0, kernel_size=3):
    device = color_image.device
    _, H, W = color_image.shape
    color_image = color_image.unsqueeze(0)
    depth_image = depth_image.unsqueeze(0)
    # 定义卷积核以计算水平和垂直方向的梯度

    if kernel_size == 3:
        # 3x3卷积核，用于计算4邻域梯度
        kernel_x = torch.tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]], 
                               dtype=torch.float32, device=device)
        kernel_y = torch.tensor([[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], 
                               dtype=torch.float32, device=device)
    elif kernel_size == 5:
        # 5x5卷积核，用于更广的邻域
        kernel_x = torch.tensor([[[[0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0], 
                                  [-1, 1, 0, 0, 0], 
                                  [0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0]]]], 
                               dtype=torch.float32, device=device)
        kernel_y = torch.tensor([[[[0, 0, -1, 0, 0], 
                                  [0, 0, 1, 0, 0], 
                                  [0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0]]]], 
                               dtype=torch.float32, device=device)
    else:
        raise ValueError("kernel_size must be 3 or 5")

    # 计算颜色和深度的梯度
    color_grad_x = F.conv2d(color_image, kernel_x, padding=kernel_size//2)
    color_grad_y = F.conv2d(color_image, kernel_y, padding=kernel_size//2)
    depth_grad_x = F.conv2d(depth_image, kernel_x, padding=kernel_size//2)
    depth_grad_y = F.conv2d(depth_image, kernel_y, padding=kernel_size//2)

    # 计算颜色差异（L2范数）
    color_diff = torch.sqrt(torch.sum(color_grad_x**2 + color_grad_y**2, dim=1, keepdim=True) + 1e-8)  # (1, H, W)
    
    # 计算权重：exp(-lambda * ||color_diff||_2)
    weights = torch.exp(-lambda_color * color_diff)  # (1, H, W)

    # 计算深度梯度的L2范数
    depth_grad = torch.sqrt(depth_grad_x**2 + depth_grad_y**2 + 1e-8)  # (1, H, W)

    # 计算加权深度平滑损失
    weighted_loss = weights * depth_grad  # (1, H, W)
    loss = torch.mean(weighted_loss)

    return loss


def image2canny(image, thres1, thres2, isEdge1=True):
    """ image: (H, W, 3)"""
    canny_mask = torch.from_numpy(cv2.Canny((image.detach().cpu().numpy()*255.).astype(np.uint8), thres1, thres2)/255.)
    if not isEdge1:
        canny_mask = 1. - canny_mask
    return canny_mask.float()


with torch.no_grad():
    kernelsize=3
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernelsize, padding=(kernelsize//2))
    kernel = torch.tensor([[0.,1.,0.],[1.,1.,1.],[0.,1.,0.]]).reshape(1,1,kernelsize,kernelsize)
    conv.weight.data = kernel #torch.ones((1,1,kernelsize,kernelsize))
    conv.bias.data = torch.tensor([0.])
    conv.requires_grad_(False)
    conv = conv.cuda()
    
def nearMean_map(array, mask, kernelsize=3):
    """ array: (H,W) / mask: (H,W) """
    cnt_map = torch.ones_like(array)

    nearMean_map = conv((array * mask)[None,None])
    cnt_map = conv((cnt_map * mask)[None,None])
    nearMean_map = (nearMean_map / (cnt_map+1e-8)).squeeze()
        
    return nearMean_map


def compute_match_density(match_data_list, num_blocks=8):
    density = np.zeros((num_blocks, num_blocks))
    total_matches = sum(len(match_data) for match_data in match_data_list) if match_data_list else 1

    for match_data in match_data_list:
        for x, y in match_data:
            if 0 <= x <= 1 and 0 <= y <= 1:
                bx = min(int(x * num_blocks), num_blocks - 1)
                by = min(int(y * num_blocks), num_blocks - 1)
                density[by, bx] += 1

    density = density / total_matches
    alpha = 5.0  # 控制 sigmoid 敏感度
    weights = 1 / (1 + np.exp(-alpha * density))  # Sigmoid 归一化
    return weights

def normalize0(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=0, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=0, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))