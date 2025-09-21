import torch
import torchvision
import torch_scatter

def inverse_warp(img, depth, pose1, pose2, K, bg_mask=None): # 把nearest投影到pseudo视图

    '''
    img: origin image of closest view
    depth: rendered depth of closest view
    depth_pseudo: rendered depth of pseudo view
    pose1: camera pose of closest view
    pose2: camera pose of pseudo view
    K: camera intrinsic matrix
    '''

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    C, H, W = img.shape
    y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    x = x.float().to(img.device)
    y = y.float().to(img.device)
    z = depth
    
    x = (x - cx) / fx
    y = (y - cy) / fy
    coordinates = torch.stack([x, y, torch.ones_like(z)], dim=0)
    coordinates = coordinates * z
    
    coordinates = coordinates.view(3, -1)
    coordinates = torch.cat([coordinates, torch.ones_like(z).view(1, -1)], dim=0)
    
    pose = torch.matmul(pose2, torch.inverse(pose1))
    coordinates = torch.matmul(pose, coordinates) # pseudo相机坐标系
    
    coordinates = coordinates[:3, :]
    coordinates = coordinates.view(3, H, W)
    z_pseudo = coordinates[2, :, :] # 深度
    x_pseudo = fx * coordinates[0, :] / coordinates[2, :] + cx # pseudo像素坐标系x
    y_pseudo = fy * coordinates[1, :] / coordinates[2, :] + cy # pseudo像素坐标系y

    # grid = torch.stack([2.0*x_pseudo/W - 1.0, 2.0*y_pseudo/H - 1.0], dim=-1).unsqueeze(0).to('cuda')
    z_buffer = torch.full((H * W,), float('inf')).to(img.device)

    x_pseudo_flat = x_pseudo.view(-1) # H*W
    y_pseudo_flat = y_pseudo.view(-1) # H*W
    z_pseudo_flat = z_pseudo.view(-1) # H*W
    img_flat = img.view(C, -1) # [C, H*W]

    valid = (z_pseudo_flat > 0) & (x_pseudo_flat >= 0) & (x_pseudo_flat < W) & (y_pseudo_flat >= 0) & (y_pseudo_flat < H) #投影有效点
    valid_idx = valid.nonzero(as_tuple=True)[0]
    x_valid = x_pseudo_flat[valid_idx].long()
    y_valid = y_pseudo_flat[valid_idx].long()
    z_valid = z_pseudo_flat[valid_idx]
    img_valid = img_flat[:, valid_idx] # 有效的像素点
    idx = y_valid * W + x_valid

    z_buffer, argmin = torch_scatter.scatter_min(z_valid, idx, dim=0, out=z_buffer)

    # 初始化 warped 图像
    warped_img = torch.zeros_like(img)
    mask = torch.ones(H,W)

    # 将有效像素填充到 warped_img
    valid_scatter = (z_buffer != float('inf')).nonzero(as_tuple=True)[0]
    if valid_scatter.shape[0] > 0:
        selected_indices = argmin[valid_scatter]
        warped_idx = valid_scatter
        warped_img_flat = warped_img.view(C, -1)
        mask_flat = mask.view(-1)
        warped_img_flat[:, warped_idx] = img_valid[:, selected_indices]
        mask_flat[warped_idx] = 0.0
        warped_img = warped_img_flat.view(C, H, W)
    
    mask = mask_flat.view(H, W)

    mask2 = mask_clean(mask, 3)
    
    return warped_img, mask2

import torch.nn.functional as F

def mask_clean(mask, w=5):
    """
    Clean the warped mask by removing outliers using convolution for faster processing.
    
    Args:
        mask: input binary mask (H,W) where 0 indicates valid pixels and 1 indicates holes
        w: window size for neighborhood checking (odd number recommended)
    
    Returns:
        cleaned_mask: mask after removing outliers
    """
    assert w % 2 == 1, "Window size w should be odd"
    
    # Convert mask to binary (1 for holes, 0 for valid)
    binary_mask = mask.clone()
    
    # Create a kernel for counting valid neighbors
    kernel = torch.ones(1, 1, w, w, device=mask.device)
    
    # Pad the mask
    pad = w // 2
    padded_mask = F.pad(binary_mask.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='constant', value=1)
    
    # Count valid neighbors using convolution
    neighbor_count = F.conv2d(1 - padded_mask, kernel).squeeze()
    
    # Threshold: valid if at least half the window is valid
    threshold = (w * w) // 2
    valid_regions = (neighbor_count >= threshold).float() # 有效点
    
    # Only keep valid pixels that are also in the original mask
    cleaned_mask = torch.where((binary_mask == 0) & (valid_regions == 1), 
                              torch.zeros_like(binary_mask), 
                              torch.ones_like(binary_mask))
    
    return cleaned_mask


