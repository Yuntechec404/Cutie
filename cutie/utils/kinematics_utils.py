import torch
import torch.nn.functional as F

def filter_realsense_depth(depth: torch.Tensor, prev_depth: torch.Tensor = None, 
                           min_depth=0.15, max_depth=10.0, jump_thresh=0.5):
    """
    過濾 RealSense 深度圖的雜訊與無效區域。
    參數:
      depth: 當前幀深度圖, B * 1 * H * W (單位: 公尺)
      prev_depth: 前一幀深度圖 (選填，用於過濾時序突波)
      min_depth: 低於 15cm (0.15m) 的盲區不採信
      max_depth: 超過有效測距範圍的極大值
      jump_thresh: 同一像素深度變化超過此值視為雜訊 (突波)
    返回:
      過濾後的深度圖與有效遮罩 (valid_mask)
    """
    valid_mask = torch.ones_like(depth, dtype=torch.bool)
    
    # 1. 過濾測距盲區 (< 15cm) 與極大值
    valid_mask &= (depth >= min_depth)
    valid_mask &= (depth <= max_depth)
    
    # 2. 過濾同一點數值劇烈跳動 (時序突波)
    if prev_depth is not None:
        valid_mask &= (torch.abs(depth - prev_depth) < jump_thresh)
        
    return depth, valid_mask

def depth_aware_warp(feature_map: torch.Tensor, depth: torch.Tensor, 
                     intrinsics: torch.Tensor, se3_matrix: torch.Tensor,
                     stride: int = 16): # [新增] stride 參數以匹配特徵尺度
    B, C, H, W = feature_map.shape
    device = feature_map.device
    
    # 建立 2D 像素座標網格 (u, v)
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    uv_grid = torch.stack([x * stride, y * stride, torch.ones_like(x)], dim=0).unsqueeze(0).repeat(B, 1, 1, 1).float()
    
    inv_K = torch.inverse(intrinsics) 
    uv_flat = uv_grid.view(B, 3, -1)
    rays = torch.bmm(inv_K, uv_flat) 
    
    depth_flat = depth.view(B, 1, -1)
    points_3d = rays * depth_flat 
    
    points_3d_homo = torch.cat([points_3d, torch.ones(B, 1, H*W, device=device)], dim=1) 
    points_3d_warped = torch.bmm(se3_matrix, points_3d_homo) 
    
    Z_warped = points_3d_warped[:, 2:3, :]
    Z_warped = torch.clamp(Z_warped, min=1e-5) 
    points_2d_homo = torch.bmm(intrinsics, points_3d_warped[:, :3, :] / Z_warped)
    u_warped = points_2d_homo[:, 0, :].view(B, H, W) / stride
    v_warped = points_2d_homo[:, 1, :].view(B, H, W) / stride
    u_norm = (u_warped / (W - 1)) * 2 - 1
    v_norm = (v_warped / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1) 
    
    # 指定 align_corners=True 以對齊座標
    warped_feature = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return warped_feature, grid
