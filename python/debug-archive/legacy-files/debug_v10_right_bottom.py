#!/usr/bin/env python3
"""
V10右下影問題のデバッグ
光側のu値と移動距離を詳しく調べる
"""

import cv2
import numpy as np
import sys
import os
# import matplotlib.pyplot as plt  # 使用しないのでコメントアウト

# v10 improved push pointsをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), 'python', 'perspective_shadow'))
from v10_pushpoints import long_shadow_push_points_v10, _u01_along_x, _rotate_expand, _rotate_back

# v3_fixed height mapをインポート  
from v3_longshadow_fixed import make_height_map_fixed

# 接地点検出をインポート
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))
from contact_detection.v3_sliding_window import detect_ground_contact_points_v3

def debug_right_bottom_issue():
    """右下の影が動かない問題をデバッグ"""
    
    # テスト画像読み込み
    test_image = "sample/test1.png"
    src = cv2.imread(test_image, cv2.IMREAD_UNCHANGED)
    if src is None:
        raise FileNotFoundError(f"Test image not found: {test_image}")
    
    print(f"=== V10 Right Bottom Debug ===")
    
    # アルファチャンネル抽出
    if src.ndim == 3 and src.shape[2] == 4:
        alpha = src[:, :, 3]
        rgb = src[:, :, :3]
    else:
        alpha = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    
    # 接地点検出と高さマップ生成（簡略版）
    contact_mask = detect_ground_contact_points_v3(alpha)
    if contact_mask is not None and contact_mask.dtype == bool:
        contact_mask = contact_mask.astype(np.float32)
    
    height_results = make_height_map_fixed(
        alpha, contact_mask, w_boundary=0.35, w_contact=0.65, 
        gamma=1.10, quantile_low=0.02, quantile_high=0.98,
        spread_sigma=1.0, dilate_iter=1, fallback_bottom_is_contact=True
    )
    height_map = height_results["final_height"]
    
    # v10と同じパラメータでデバッグ
    azimuth_deg = 50
    elevation_deg = 7
    scale = 21.5
    h_gamma = 1.3
    push_dir = 'opposite'
    beta_u = 0.75
    p_u = 1.4
    
    # v10のロジックを再現してデバッグ
    M = (alpha > 0).astype(np.uint8)
    H, W = M.shape
    hmap = np.clip(height_map.astype(np.float32), 0.0, 1.0)
    if h_gamma != 1.0:
        hmap = np.power(hmap, float(h_gamma))
    
    # 影方向計算
    import math
    phi = math.radians(float(azimuth_deg))
    Lx, Ly = math.cos(phi), -math.sin(phi)
    dirx, diry = (-Lx, -Ly) if push_dir == 'opposite' else (Lx, Ly)
    angle_to_x = -math.degrees(math.atan2(diry, dirx))
    
    print(f"Shadow direction: dirx={dirx:.3f}, diry={diry:.3f}")
    print(f"Rotation angle: {angle_to_x:.1f}°")
    
    # 回転してu01を計算
    M_r, M0 = _rotate_expand(M, angle_to_x, interp=cv2.INTER_NEAREST, border=0)
    inside = M_r.astype(bool)
    u01_r = _u01_along_x(inside)
    u01 = _rotate_back(u01_r, M0, (H, W), interp=cv2.INTER_NEAREST, border=0).astype(np.float32)
    
    # 右下領域のu値を確認（より広い範囲で確認）
    h_start = H * 2 // 3  # より広い範囲
    w_start = W * 2 // 3
    right_bottom_region = u01[h_start:, w_start:]
    mask_right_bottom = M[h_start:, w_start:]
    
    valid_u = right_bottom_region[mask_right_bottom > 0]
    print(f"\\n=== Right Bottom Region Analysis (broader) ===")
    print(f"Region: [{h_start}:{H}, {w_start}:{W}]")
    print(f"Pixels in region: {np.sum(mask_right_bottom > 0)}")
    
    if valid_u.size > 0:
        print(f"\\n=== Right Bottom Region Analysis ===")
        print(f"Region: [{h_quarter}:{H}, {w_quarter}:{W}]")
        print(f"u01 values in right bottom: min={valid_u.min():.3f}, max={valid_u.max():.3f}, mean={valid_u.mean():.3f}")
        print(f"Number of pixels: {valid_u.size}")
        
        # u_effの計算
        u_min = 0.25
        u_eff = u_min + (1.0 - u_min) * valid_u
        print(f"u_eff values: min={u_eff.min():.3f}, max={u_eff.max():.3f}, mean={u_eff.mean():.3f}")
        
        # 移動距離計算
        theta = math.radians(max(3.0, min(89.9, float(elevation_deg))))
        px_per_unit = (1.0 / math.tan(theta)) * float(scale)
        
        # 右下の高さ値も確認
        height_right_bottom = hmap[h_quarter:, w_quarter:][mask_right_bottom > 0]
        if height_right_bottom.size > 0:
            print(f"Height values in right bottom: min={height_right_bottom.min():.3f}, max={height_right_bottom.max():.3f}")
            
            # 実際の移動距離計算
            D_base = height_right_bottom * px_per_unit
            D_boosted = D_base * (1.0 + beta_u * np.power(u_eff, p_u))
            
            print(f"\\n=== Movement Distance Analysis ===")
            print(f"D_base: min={D_base.min():.1f}px, max={D_base.max():.1f}px, mean={D_base.mean():.1f}px")
            print(f"D_boosted: min={D_boosted.min():.1f}px, max={D_boosted.max():.1f}px, mean={D_boosted.mean():.1f}px")
            print(f"Boost factor: min={D_boosted.min()/D_base.min():.2f}x, max={D_boosted.max()/D_base.max():.2f}x")
    
    # u01マップを保存して視覚的に確認
    cv2.imwrite("debug_u01_map.png", (u01 * 255).astype(np.uint8))
    print(f"\\n=== Debug Files Saved ===")
    print(f"u01 map: debug_u01_map.png (white=far from light, black=near light)")
    
    # 全体のu01統計
    valid_u_all = u01[M > 0]
    if valid_u_all.size > 0:
        print(f"\\n=== Overall u01 Statistics ===")
        print(f"All u01 values: min={valid_u_all.min():.3f}, max={valid_u_all.max():.3f}, mean={valid_u_all.mean():.3f}")
        
        # u01の分布を確認
        hist, bins = np.histogram(valid_u_all, bins=10, range=(0, 1))
        print("u01 distribution (0.0-1.0):")
        for i in range(len(hist)):
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} pixels ({hist[i]/len(valid_u_all)*100:.1f}%)")

if __name__ == "__main__":
    debug_right_bottom_issue()