"""
高さマップのデバッグ用単体テスト
Height Map Debug Unit Test for OKIBAE v3 Shadow System
"""

import cv2
import numpy as np
import os
from typing import Dict, Optional, Tuple
import sys

# v3の高さマップ関数をインポート（修正版）
sys.path.append(os.path.join(os.path.dirname(__file__), 'perspective_shadow'))
from v3_longshadow import make_height_map
from v3_longshadow_fixed import make_height_map_fixed

# V3スタイルの接地点検出をインポート
sys.path.append(os.path.dirname(__file__))
from contact_detection.v3_sliding_window import detect_ground_contact_points_v3


def save_height_map_debug_image(
    alpha_mask: np.ndarray,
    contact_mask: Optional[np.ndarray],
    height_results: Dict[str, np.ndarray],
    output_path: str,
    title: str = "Height Map Debug",
    show_raw_combined: bool = False
) -> None:
    """
    高さマップのデバッグ画像を保存
    """
    boundary_height = height_results["boundary_height"]
    contact_height = height_results["contact_height"] 
    final_height = height_results["final_height"]
    
    h, w = alpha_mask.shape
    
    # 画像配置を決定: raw_combinedがある場合は2x3、ない場合は2x2
    if show_raw_combined and "raw_combined" in height_results:
        debug_canvas = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        raw_combined = height_results["raw_combined"]
    else:
        debug_canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # 1. アルファマスク（左上）
    alpha_vis = cv2.cvtColor((alpha_mask).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    debug_canvas[0:h, 0:w] = alpha_vis
    
    # 2. Boundary height（右上）
    boundary_vis = (boundary_height * 255).astype(np.uint8)
    boundary_vis = cv2.applyColorMap(boundary_vis, cv2.COLORMAP_JET)
    debug_canvas[0:h, w:w*2] = boundary_vis
    
    # 3. Contact height（左下）
    contact_vis = (contact_height * 255).astype(np.uint8) 
    contact_vis = cv2.applyColorMap(contact_vis, cv2.COLORMAP_JET)
    debug_canvas[h:h*2, 0:w] = contact_vis
    
    # 4. Final height（右下）
    final_vis = (final_height * 255).astype(np.uint8)
    final_vis = cv2.applyColorMap(final_vis, cv2.COLORMAP_JET)
    debug_canvas[h:h*2, w:w*2] = final_vis
    
    # 5. Raw combined（右上の右、修正版のみ）
    if show_raw_combined and "raw_combined" in height_results:
        raw_vis = (raw_combined * 255).astype(np.uint8)
        raw_vis = cv2.applyColorMap(raw_vis, cv2.COLORMAP_JET)
        debug_canvas[0:h, w*2:w*3] = raw_vis
    
    # オプション: 接地マスクオーバーレイ
    if contact_mask is not None:
        contact_overlay = (contact_mask > 0.1).astype(np.uint8) * 255
        # 左上のアルファ画像に緑色で接地点をオーバーレイ
        green_overlay = np.zeros_like(alpha_vis)
        green_overlay[:, :, 1] = contact_overlay  # Green channel
        alpha_blend = cv2.addWeighted(debug_canvas[0:h, 0:w], 0.7, green_overlay, 0.3, 0)
        debug_canvas[0:h, 0:w] = alpha_blend
    
    # テキストラベル追加
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    color = (255, 255, 255)
    
    cv2.putText(debug_canvas, "Alpha + Contact", (5, 25), font, font_scale, color, thickness)
    cv2.putText(debug_canvas, "Boundary Height", (w + 5, 25), font, font_scale, color, thickness)
    cv2.putText(debug_canvas, "Contact Height", (5, h + 25), font, font_scale, color, thickness)
    cv2.putText(debug_canvas, "Final Height", (w + 5, h + 25), font, font_scale, color, thickness)
    
    # Raw combined用ラベル
    if show_raw_combined and "raw_combined" in height_results:
        cv2.putText(debug_canvas, "Raw Combined", (w*2 + 5, 25), font, font_scale, color, thickness)
    
    # 統計情報を追加
    stats_text = [
        f"Boundary: min={boundary_height.min():.3f}, max={boundary_height.max():.3f}",
        f"Contact: min={contact_height.min():.3f}, max={contact_height.max():.3f}",
        f"Final: min={final_height.min():.3f}, max={final_height.max():.3f}"
    ]
    
    if show_raw_combined and "raw_combined" in height_results:
        stats_text.append(f"Raw: min={raw_combined.min():.3f}, max={raw_combined.max():.3f}")
    
    y_start = h * 2 - 60
    for i, text in enumerate(stats_text):
        cv2.putText(debug_canvas, text, (5, y_start + i * 20), font, 0.4, (255, 255, 0), 1)
    
    cv2.imwrite(output_path, debug_canvas)
    print(f"Height map debug saved: {output_path}")


def test_height_map_generation(test_name: str, alpha_path: str, output_dir: str):
    """
    高さマップ生成の単体テスト
    """
    print(f"\n=== {test_name} ===")
    
    # アルファマスク読み込み
    src = cv2.imread(alpha_path, cv2.IMREAD_UNCHANGED)
    if src is None:
        print(f"ERROR: Could not load {alpha_path}")
        return
        
    if src.ndim == 3 and src.shape[2] == 4:
        alpha_mask = src[:, :, 3]
    else:
        alpha_mask = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    print(f"Alpha mask shape: {alpha_mask.shape}")
    print(f"Alpha range: {alpha_mask.min()}-{alpha_mask.max()}")
    
    # V3接地点検出
    contact_mask = detect_ground_contact_points_v3(alpha_mask)
    
    # ブール配列をfloat配列に変換
    if contact_mask is not None and contact_mask.dtype == bool:
        contact_mask = contact_mask.astype(np.float32)
    
    if contact_mask is not None:
        print(f"Contact mask shape: {contact_mask.shape}")
        print(f"Contact range: {contact_mask.min():.3f}-{contact_mask.max():.3f}")
        print(f"Contact points detected: {np.sum(contact_mask > 0.1)}")
    else:
        print("No contact mask detected")
    
    # オリジナル版と修正版のテスト設定
    test_configs = [
        # オリジナル版
        {
            "name": "original_default",
            "use_fixed": False,
            "params": {
                "w_boundary": 0.5,
                "w_contact": 0.5,
                "gamma": 1.25,
                "spread_sigma": 1.0,
                "dilate_iter": 1
            }
        },
        # 修正版：デフォルト設定
        {
            "name": "fixed_default",
            "use_fixed": True,
            "params": {
                "w_boundary": 0.35,
                "w_contact": 0.65,
                "gamma": 1.10,
                "quantile_low": 0.02,
                "quantile_high": 0.98,
                "spread_sigma": 1.0,
                "dilate_iter": 1
            }
        },
        # 修正版：より強いコントラスト
        {
            "name": "fixed_high_contrast",
            "use_fixed": True,
            "params": {
                "w_boundary": 0.35,
                "w_contact": 0.65,
                "gamma": 1.20,
                "quantile_low": 0.05,
                "quantile_high": 0.95,
                "spread_sigma": 1.0,
                "dilate_iter": 1
            }
        },
        # 修正版：境界重視
        {
            "name": "fixed_boundary_focused",
            "use_fixed": True,
            "params": {
                "w_boundary": 0.60,
                "w_contact": 0.40,
                "gamma": 1.10,
                "quantile_low": 0.02,
                "quantile_high": 0.98,
                "spread_sigma": 1.0,
                "dilate_iter": 1
            }
        }
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for config in test_configs:
        config_name = config["name"]
        use_fixed = config["use_fixed"]
        params = config["params"]
        
        print(f"\nTesting config: {config_name}")
        print(f"  Using fixed version: {use_fixed}")
        print(f"  Parameters: {params}")
        
        # 高さマップ生成（バージョン選択）
        if use_fixed:
            height_results = make_height_map_fixed(
                alpha_mask,
                contact_mask,
                **params
            )
        else:
            height_results = make_height_map(
                alpha_mask,
                contact_mask,
                **params
            )
        
        # 結果統計
        boundary_height = height_results["boundary_height"]
        contact_height = height_results["contact_height"]
        final_height = height_results["final_height"]
        
        print(f"  Boundary height: min={boundary_height.min():.3f}, max={boundary_height.max():.3f}")
        print(f"  Contact height: min={contact_height.min():.3f}, max={contact_height.max():.3f}")
        print(f"  Final height: min={final_height.min():.3f}, max={final_height.max():.3f}")
        
        # デバッグ画像保存
        output_path = os.path.join(output_dir, f"{test_name}_{config_name}_heightmap.png")
        save_height_map_debug_image(
            alpha_mask,
            contact_mask,
            height_results,
            output_path,
            title=f"{test_name} - {config_name}",
            show_raw_combined=use_fixed  # 修正版の場合のみraw_combinedを表示
        )
        
        # 高さマップの個別保存（グレースケール）
        final_gray = (final_height * 255).astype(np.uint8)
        final_path = os.path.join(output_dir, f"{test_name}_{config_name}_final_height.png")
        cv2.imwrite(final_path, final_gray)


def main():
    """
    メインテスト実行
    """
    test_cases = [
        {
            "name": "orca_ring",
            "alpha_path": "sample/test1.png"
        }
    ]
    
    output_dir = "python/debug_output_height_maps"
    
    for test_case in test_cases:
        test_height_map_generation(
            test_case["name"],
            test_case["alpha_path"],
            output_dir
        )
    
    print(f"\nAll height map tests completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()