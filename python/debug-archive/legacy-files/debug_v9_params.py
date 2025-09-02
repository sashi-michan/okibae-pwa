#!/usr/bin/env python3
"""
V9パラメータデバッグテスト
実際に計算されている方向を確認する
"""

import math
import numpy as np

def debug_v9_direction_calc(azimuth_deg, push_dir='opposite'):
    """V9の方向計算をデバッグ"""
    print(f"=== V9 Direction Calculation Debug ===")
    print(f"Input: azimuth_deg={azimuth_deg}, push_dir='{push_dir}'")
    
    # V9と同じ計算
    phi = math.radians(float(azimuth_deg))
    Lx, Ly = math.cos(phi), math.sin(phi)
    dirx, diry = (-Lx, -Ly) if push_dir == 'opposite' else (Lx, Ly)
    angle_to_x = -math.degrees(math.atan2(diry, dirx))
    
    print(f"phi (radians): {phi:.3f}")
    print(f"Light direction (Lx, Ly): ({Lx:.3f}, {Ly:.3f})")
    print(f"Shadow direction (dirx, diry): ({dirx:.3f}, {diry:.3f})")
    print(f"Angle to rotate to +X: {angle_to_x:.1f}°")
    
    # 実際の影方向を逆算
    shadow_angle_deg = math.degrees(math.atan2(diry, dirx))
    print(f"Final shadow direction: {shadow_angle_deg:.1f}° (0°=right, 90°=up, 180°=left, 270°=down)")
    
    # 期待される方向
    if -45 <= shadow_angle_deg <= 45 or shadow_angle_deg >= 315:
        print("→ Going RIGHT")
    elif 45 <= shadow_angle_deg <= 135:
        print("→ Going UP")
    elif 135 <= shadow_angle_deg <= 225:
        print("→ Going LEFT") 
    elif 225 <= shadow_angle_deg <= 315:
        print("→ Going DOWN")
    
    # 左下方向（-135°から-45°の間、または225°から315°）かチェック
    if 225 <= shadow_angle_deg <= 315:
        print("OK: This should give LEFT-BOTTOM shadow")
    else:
        print("NG: This will NOT give left-bottom shadow")
    
    print()

# テストケース
test_cases = [
    (225, 'opposite'),
    (45, 'along'),
    (135, 'opposite'),
    (315, 'along'),
    (315, 'opposite'),  # 新しいケース
]

for azimuth, push_dir in test_cases:
    debug_v9_direction_calc(azimuth, push_dir)