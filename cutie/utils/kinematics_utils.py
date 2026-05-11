import torch
import torch.nn.functional as F

def filter_realsense_depth(depth: torch.Tensor, prev_depth: torch.Tensor = None, 
                           min_depth=0.15, max_depth=10.0, jump_thresh=0.5):
    """
    過濾 RealSense 深度圖的雜訊與無效區域。
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
                     stride: int = 16):
    """
    利用 RGB-D 深度資訊與相機內參，進行 3D 反投影與特徵空間扭曲 (Warping)。
    """
    B, C, H, W = feature_map.shape
    device = feature_map.device
    
    # 建立 2D 像素座標網格 (u, v)
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    # 必須將特徵圖的座標乘以 stride，轉換回原始全解析度相機座標系
    uv_grid = torch.stack([x * stride, y * stride, torch.ones_like(x)], dim=0).unsqueeze(0).repeat(B, 1, 1, 1).float()
    
    # 利用內參矩陣的逆矩陣 (Inverse K) 轉換到歸一化相機平面
    inv_K = torch.inverse(intrinsics) 
    uv_flat = uv_grid.view(B, 3, -1)
    rays = torch.bmm(inv_K, uv_flat) 
    
    # 乘上真實深度，生成 3D 點雲 (X, Y, Z)
    depth_flat = depth.view(B, 1, -1)
    points_3d = rays * depth_flat 
    
    # 轉換為齊次座標，並套用機器的 SE(3) 剛體變換矩陣
    points_3d_homo = torch.cat([points_3d, torch.ones(B, 1, H*W, device=device)], dim=1) 
    points_3d_warped = torch.bmm(se3_matrix, points_3d_homo) 
    
    # 再投影回 2D 影像平面 (除以 Z，並乘上內參 K)
    Z_warped = points_3d_warped[:, 2:3, :]
    Z_warped = torch.clamp(Z_warped, min=1e-5) # 避免除以零
    points_2d_homo = torch.bmm(intrinsics, points_3d_warped[:, :3, :] / Z_warped)
    
    # 投影回來後，必須除以 stride 轉回底層特徵圖座標系
    u_warped = points_2d_homo[:, 0, :].view(B, H, W) / stride
    v_warped = points_2d_homo[:, 1, :].view(B, H, W) / stride
    
    # 將座標歸一化到 [-1, 1] 區間 (使用 align_corners=True 標準)
    u_norm = (u_warped / (W - 1)) * 2 - 1
    v_norm = (v_warped / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1) 
    
    # 進行 Feature Warping (雙線性插值，填零補齊邊界)
    warped_feature = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return warped_feature, grid
