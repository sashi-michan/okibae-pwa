#!/usr/bin/env python3
"""
Shadow system testing utilities
Separate file for debug/test functions to keep main code clean
"""

import sys
import os
import cv2
import numpy as np
from shadow_enhancer import base64_to_cv2
from contact_detection import (
    detect_ground_contact_points_v1,
    detect_ground_contact_points_v2, 
    detect_ground_contact_points_v3,
    detect_ground_contact_points_v4,
    generate_contact_map
)


def visualize_contact_points_debug(img_rgba, contact_points, output_path="contact_points_debug.png"):
    """
    Debug function: Visualize detected contact points as red dots on the original image
    """
    try:
        # Create visualization image
        debug_img = img_rgba.copy()
        h, w = contact_points.shape
        
        # Draw contact points as red dots
        contact_coords = np.where(contact_points)
        for y, x in zip(contact_coords[0], contact_coords[1]):
            # Draw a small red circle
            cv2.circle(debug_img, (x, y), 2, (0, 0, 255, 255), -1)  # Red dot
        
        # Save debug image
        cv2.imwrite(output_path, debug_img)
        contact_count = len(contact_coords[0])
        sys.stderr.write(f"DEBUG: Saved contact points visualization to {output_path} ({contact_count} points)\n")
        
        return debug_img
        
    except Exception as e:
        sys.stderr.write(f"visualize_contact_points_debug failed: {e}\n")
        return img_rgba


def test_contact_point_comparison_v3(cutout_base64, output_dir="debug_output"):
    """
    Compare v1, v2 and v3 contact point detection systems side by side
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert input image
        image = base64_to_cv2(cutout_base64)
        alpha = image[:, :, 3]
        
        # Test all versions
        contact_points_v1 = detect_ground_contact_points_v1(alpha)
        contact_points_v2 = detect_ground_contact_points_v2(alpha)
        contact_points_v3 = detect_ground_contact_points_v3(alpha)
        
        # Visualize all results
        debug_img_v1 = visualize_contact_points_debug(
            image, contact_points_v1, 
            os.path.join(output_dir, "contact_points_v1.png")
        )
        
        debug_img_v2 = visualize_contact_points_debug(
            image, contact_points_v2, 
            os.path.join(output_dir, "contact_points_v2_convex.png")
        )
        
        debug_img_v3 = visualize_contact_points_debug(
            image, contact_points_v3, 
            os.path.join(output_dir, "contact_points_v3_sliding_window.png")
        )
        
        # Print comparison stats
        contact_count_v1 = np.sum(contact_points_v1)
        contact_count_v2 = np.sum(contact_points_v2)
        contact_count_v3 = np.sum(contact_points_v3)
        total_pixels = np.sum(alpha > 0)
        
        sys.stderr.write(f"3-WAY COMPARISON RESULTS:\n")
        sys.stderr.write(f"  - V1 (basic): {contact_count_v1} points\n")
        sys.stderr.write(f"  - V2 (convex hull): {contact_count_v2} points\n")
        sys.stderr.write(f"  - V3 (sliding window + SDF): {contact_count_v3} points\n")
        sys.stderr.write(f"  - V2 reduction: {((contact_count_v1-contact_count_v2)/contact_count_v1*100):.1f}%\n")
        sys.stderr.write(f"  - V3 reduction: {((contact_count_v1-contact_count_v3)/contact_count_v1*100):.1f}%\n")
        sys.stderr.write(f"  - Total object pixels: {total_pixels}\n")
        sys.stderr.write(f"  - V1 image: {output_dir}/contact_points_v1.png\n")
        sys.stderr.write(f"  - V2 image: {output_dir}/contact_points_v2_convex.png\n")
        sys.stderr.write(f"  - V3 image: {output_dir}/contact_points_v3_sliding_window.png\n")
        
        return {
            "v1_count": contact_count_v1,
            "v2_count": contact_count_v2,
            "v3_count": contact_count_v3,
            "total_pixels": total_pixels
        }
        
    except Exception as e:
        sys.stderr.write(f"test_contact_point_comparison_v3 failed: {e}\n")
        return None


def test_contact_point_comparison(cutout_base64, output_dir="debug_output"):
    """
    Compare v1 and v2 contact point detection systems side by side
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert input image
        image = base64_to_cv2(cutout_base64)
        alpha = image[:, :, 3]
        
        # Test both versions
        contact_points_v1 = detect_ground_contact_points_v1(alpha)
        contact_points_v2 = detect_ground_contact_points_v2(alpha)
        
        # Visualize both results
        debug_img_v1 = visualize_contact_points_debug(
            image, contact_points_v1, 
            os.path.join(output_dir, "contact_points_v1.png")
        )
        
        debug_img_v2 = visualize_contact_points_debug(
            image, contact_points_v2, 
            os.path.join(output_dir, "contact_points_v2_adaptive.png")
        )
        
        # Print comparison stats
        contact_count_v1 = np.sum(contact_points_v1)
        contact_count_v2 = np.sum(contact_points_v2)
        total_pixels = np.sum(alpha > 0)
        
        sys.stderr.write(f"COMPARISON RESULTS:\n")
        sys.stderr.write(f"  - V1 (basic): {contact_count_v1} points\n")
        sys.stderr.write(f"  - V2 (adaptive): {contact_count_v2} points\n")
        sys.stderr.write(f"  - Reduction: {((contact_count_v1-contact_count_v2)/contact_count_v1*100):.1f}%\n")
        sys.stderr.write(f"  - Total object pixels: {total_pixels}\n")
        sys.stderr.write(f"  - V1 image: {output_dir}/contact_points_v1.png\n")
        sys.stderr.write(f"  - V2 image: {output_dir}/contact_points_v2_adaptive.png\n")
        
        return {
            "v1_count": contact_count_v1,
            "v2_count": contact_count_v2,
            "reduction_percent": ((contact_count_v1-contact_count_v2)/contact_count_v1*100) if contact_count_v1 > 0 else 0,
            "total_pixels": total_pixels
        }
        
    except Exception as e:
        sys.stderr.write(f"test_contact_point_comparison failed: {e}\n")
        return None


def test_contact_point_detection(cutout_base64, output_dir="debug_output"):
    """
    Test the ground contact point detection system
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert input image
        image = base64_to_cv2(cutout_base64)
        alpha = image[:, :, 3]
        
        # Detect contact points
        contact_points = detect_ground_contact_points_v1(alpha)
        
        # Visualize results
        debug_img = visualize_contact_points_debug(
            image, contact_points, 
            os.path.join(output_dir, "contact_points_test.png")
        )
        
        # Print stats
        contact_count = np.sum(contact_points)
        total_pixels = np.sum(alpha > 0)
        coverage = contact_count / total_pixels if total_pixels > 0 else 0
        
        sys.stderr.write(f"TEST RESULTS:\n")
        sys.stderr.write(f"  - Contact points: {contact_count}\n")
        sys.stderr.write(f"  - Total object pixels: {total_pixels}\n")
        sys.stderr.write(f"  - Coverage ratio: {coverage:.3f}\n")
        sys.stderr.write(f"  - Debug image: {output_dir}/contact_points_test.png\n")
        
        return {
            "contact_count": contact_count,
            "total_pixels": total_pixels,
            "coverage": coverage,
            "debug_image_path": os.path.join(output_dir, "contact_points_test.png")
        }
        
    except Exception as e:
        sys.stderr.write(f"test_contact_point_detection failed: {e}\n")
        return None


def test_v4_shadow_analysis(fg_img_path, bg_img_path, output_dir="debug_output_v4"):
    """
    Test v4 shadow analysis approach with foreground and background image pair
    
    Args:
        fg_img_path: Path to cutout image (PNG with alpha)
        bg_img_path: Path to original image with shadows  
        output_dir: Directory for debug output
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load images
        fg_img = cv2.imread(fg_img_path, cv2.IMREAD_UNCHANGED)  # With alpha
        bg_img = cv2.imread(bg_img_path, cv2.IMREAD_COLOR)      # RGB
        
        if fg_img is None:
            sys.stderr.write(f"Failed to load foreground image: {fg_img_path}\n")
            return None
            
        if bg_img is None:
            sys.stderr.write(f"Failed to load background image: {bg_img_path}\n")
            return None
        
        # Ensure both images have same size
        if fg_img.shape[:2] != bg_img.shape[:2]:
            sys.stderr.write(f"Size mismatch: FG {fg_img.shape[:2]} vs BG {bg_img.shape[:2]}\n")
            # Resize foreground to match background
            fg_img = cv2.resize(fg_img, (bg_img.shape[1], bg_img.shape[0]))
            sys.stderr.write(f"Resized foreground to match background\n")
        
        sys.stderr.write(f"Testing v4 shadow analysis with images: {fg_img.shape[:2]}\n")
        
        # Generate contact map
        contact_map = generate_contact_map(fg_img, bg_img)
        
        # Convert contact map to visualization (0-255)
        contact_map_vis = (contact_map * 255).astype(np.uint8)
        contact_map_colored = cv2.applyColorMap(contact_map_vis, cv2.COLORMAP_HOT)
        
        # Generate binary contact points  
        contact_points = detect_ground_contact_points_v4(fg_img, bg_img, threshold=0.3)
        
        # Save debug images
        cv2.imwrite(os.path.join(output_dir, "v4_contact_map.png"), contact_map_vis)
        cv2.imwrite(os.path.join(output_dir, "v4_contact_map_colored.png"), contact_map_colored)
        
        # Create visualization with contact points overlaid on original
        debug_img = bg_img.copy()
        contact_coords = np.where(contact_points)
        for y, x in zip(contact_coords[0], contact_coords[1]):
            cv2.circle(debug_img, (x, y), 2, (0, 0, 255), -1)  # Red dots
        
        cv2.imwrite(os.path.join(output_dir, "v4_contact_points_overlay.png"), debug_img)
        
        # Print results
        contact_count = np.sum(contact_points)
        max_contact = np.max(contact_map)
        significant_areas = np.sum(contact_map > 0.1)
        
        sys.stderr.write(f"V4 SHADOW ANALYSIS RESULTS:\n")
        sys.stderr.write(f"  - Contact map max value: {max_contact:.3f}\n")  
        sys.stderr.write(f"  - Significant contact areas: {significant_areas}\n")
        sys.stderr.write(f"  - Binary contact points: {contact_count}\n")
        sys.stderr.write(f"  - Contact map: {output_dir}/v4_contact_map.png\n")
        sys.stderr.write(f"  - Contact map (colored): {output_dir}/v4_contact_map_colored.png\n")
        sys.stderr.write(f"  - Contact points overlay: {output_dir}/v4_contact_points_overlay.png\n")
        
        return {
            "contact_map": contact_map,
            "contact_points": contact_points,
            "contact_count": contact_count,
            "max_contact": max_contact
        }
        
    except Exception as e:
        sys.stderr.write(f"test_v4_shadow_analysis failed: {e}\n")
        return None


def test_v4_debug_parameters(fg_img_path, bg_img_path, output_dir="debug_output_v4_params"):
    """
    Debug V4 algorithm with various parameter combinations
    """
    try:
        from contact_detection.v4_shadow_analysis import (
            detect_object_boundary, 
            estimate_background_without_object,
            detect_shadow_by_luminance_diff,
            find_boundary_shadow_intersections,
            generate_weighted_contact_map
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load images
        fg_img = cv2.imread(fg_img_path, cv2.IMREAD_UNCHANGED)
        bg_img = cv2.imread(bg_img_path, cv2.IMREAD_COLOR)
        
        # Resize if needed
        if fg_img.shape[:2] != bg_img.shape[:2]:
            fg_img = cv2.resize(fg_img, (bg_img.shape[1], bg_img.shape[0]))
        
        h, w = bg_img.shape[:2]
        object_mask = fg_img[:, :, 3] > 0
        
        sys.stderr.write(f"DEBUG: Testing different V4 parameters\n")
        sys.stderr.write(f"Object pixels: {np.sum(object_mask)}\n")
        
        # Step 1: Detect boundary
        boundary = detect_object_boundary(object_mask)
        boundary_count = np.sum(boundary)
        sys.stderr.write(f"Boundary pixels: {boundary_count}\n")
        
        # Step 2: Estimate background
        virtual_bg = estimate_background_without_object(bg_img, object_mask)
        
        # Step 3: Try different shadow detection thresholds
        shadow_thresholds = [5, 10, 15, 20, 25, 30]
        search_radii = [3, 8, 15, 25, 40]
        
        best_result = {"intersections": 0, "threshold": 0, "radius": 0}
        
        for threshold in shadow_thresholds:
            shadow_regions = detect_shadow_by_luminance_diff(bg_img, virtual_bg, threshold=threshold)
            shadow_count = np.sum(shadow_regions)
            
            for radius in search_radii:
                intersections = find_boundary_shadow_intersections(boundary, shadow_regions, search_radius=radius)
                intersection_count = np.sum(intersections)
                
                sys.stderr.write(f"  Threshold={threshold}, Radius={radius}: {shadow_count} shadows, {intersection_count} intersections\n")
                
                if intersection_count > best_result["intersections"]:
                    best_result = {
                        "intersections": intersection_count,
                        "threshold": threshold,
                        "radius": radius,
                        "shadow_count": shadow_count
                    }
                    
                    # Save best intersection visualization
                    debug_img = bg_img.copy()
                    int_coords = np.where(intersections)
                    for y, x in zip(int_coords[0], int_coords[1]):
                        cv2.circle(debug_img, (x, y), 2, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(output_dir, f"best_intersections_t{threshold}_r{radius}.png"), debug_img)
        
        sys.stderr.write(f"BEST PARAMETERS: threshold={best_result['threshold']}, radius={best_result['radius']}, intersections={best_result['intersections']}\n")
        
        return best_result
        
    except Exception as e:
        sys.stderr.write(f"test_v4_debug_parameters failed: {e}\n")
        return None


def test_v3_contact_shadow(cutout_base64, light_angle_deg=45, output_dir="debug_output_v3_shadow"):
    """
    V3接地点検出 + 接地影生成のテスト
    """
    try:
        from contact_detection.v3_sliding_window import detect_ground_contact_points_v3
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 画像変換
        image = base64_to_cv2(cutout_base64)
        alpha = image[:, :, 3]
        h, w = alpha.shape
        
        # V3接地点検出
        contact_points = detect_ground_contact_points_v3(alpha)
        
        # 光ベクトル計算（UI角度から画像座標へ）
        light_rad = np.radians(light_angle_deg - 90)
        light_vector = (np.cos(light_rad), np.sin(light_rad))
        
        sys.stderr.write(f"test_v3_contact_shadow: light_angle={light_angle_deg}°, light_vector=({light_vector[0]:.2f}, {light_vector[1]:.2f})\n")
        
        # 接地影生成
        shadow_mask = create_contact_shadow_from_v3_points(
            contact_points, light_vector, (h, w),
            shadow_intensity=0.8, patch_size=12, blur_radius=8
        )
        
        # デバッグ画像保存
        debug_img = image.copy()
        contact_coords = np.where(contact_points)
        for y, x in zip(contact_coords[0], contact_coords[1]):
            cv2.circle(debug_img, (x, y), 2, (0, 0, 255, 255), -1)
        cv2.imwrite(os.path.join(output_dir, "v3_contact_points.png"), debug_img)
        
        cv2.imwrite(os.path.join(output_dir, "v3_shadow_mask.png"), shadow_mask)
        
        shadow_colored = cv2.applyColorMap(shadow_mask, cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(output_dir, "v3_shadow_colored.png"), shadow_colored)
        
        overlay_img = image[:, :, :3].copy()
        shadow_alpha = shadow_mask.astype(np.float32) / 255.0
        for c in range(3):
            overlay_img[:, :, c] = overlay_img[:, :, c] * (1 - shadow_alpha * 0.5)
        cv2.imwrite(os.path.join(output_dir, "v3_shadow_overlay.png"), overlay_img)
        
        # 結果出力
        contact_count = np.sum(contact_points)
        shadow_pixels = np.sum(shadow_mask > 10)
        max_shadow = np.max(shadow_mask)
        
        sys.stderr.write(f"V3 CONTACT SHADOW TEST RESULTS:\n")
        sys.stderr.write(f"  - Contact points: {contact_count}\n")
        sys.stderr.write(f"  - Shadow pixels: {shadow_pixels}\n")
        sys.stderr.write(f"  - Max shadow intensity: {max_shadow}\n")
        sys.stderr.write(f"  - Light angle: {light_angle_deg}°\n")
        sys.stderr.write(f"  - Contact points: {output_dir}/v3_contact_points.png\n")
        sys.stderr.write(f"  - Shadow mask: {output_dir}/v3_shadow_mask.png\n")
        sys.stderr.write(f"  - Shadow colored: {output_dir}/v3_shadow_colored.png\n")
        sys.stderr.write(f"  - Shadow overlay: {output_dir}/v3_shadow_overlay.png\n")
        
        return {
            "contact_count": contact_count,
            "shadow_pixels": shadow_pixels,
            "max_shadow_intensity": max_shadow,
            "shadow_mask": shadow_mask
        }
        
    except Exception as e:
        sys.stderr.write(f"test_v3_contact_shadow failed: {e}\n")
        return None


def create_contact_shadow_from_v3_points(contact_points, light_vector, image_shape, shadow_intensity=0.8, patch_size=8, blur_radius=6):
    """
    V3接地点検出結果から接地影を生成
    """
    try:
        h, w = image_shape
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        
        # V3接地点の座標を抽出
        contact_coords = np.where(contact_points)
        contact_count = len(contact_coords[0])
        
        if contact_count == 0:
            sys.stderr.write("create_contact_shadow_from_v3_points: No contact points found\n")
            return np.zeros((h, w), dtype=np.uint8)
        
        sys.stderr.write(f"create_contact_shadow_from_v3_points: Creating shadows for {contact_count} contact points\n")
        
        # 光の逆方向ベクトル（影が伸びる方向）
        lx, ly = light_vector
        shadow_dir_x = -lx  # 光の逆方向
        shadow_dir_y = -ly
        
        # 影の伸び設定
        shadow_length = patch_size * 2.5  # 影の長さ
        num_segments = 6  # 影のセグメント数
        
        sys.stderr.write(f"Shadow extends in direction: ({shadow_dir_x:.2f}, {shadow_dir_y:.2f}) over {shadow_length:.1f}px in {num_segments} segments\n")
        
        # 各接地点から影を伸ばす
        for contact_y, contact_x in zip(contact_coords[0], contact_coords[1]):
            
            # 接地点から複数のセグメントで影を伸ばす
            for segment in range(num_segments):
                # セグメントの距離（接地点からの距離）
                segment_distance = (segment / (num_segments - 1)) * shadow_length if num_segments > 1 else 0
                
                # セグメントの中心位置
                segment_center_x = contact_x + shadow_dir_x * segment_distance
                segment_center_y = contact_y + shadow_dir_y * segment_distance
                
                # 距離に応じて影を薄く、大きく
                distance_factor = segment / (num_segments - 1) if num_segments > 1 else 0
                segment_intensity = shadow_intensity * (1.0 - distance_factor * 0.7)  # 遠いほど薄く
                segment_size_h = int(patch_size * (1.0 + distance_factor * 0.5))  # 遠いほど大きく
                segment_size_w = int(segment_size_h * 1.8)  # より横長に
                
                # セグメント楕円の範囲計算
                y_start = max(0, int(segment_center_y - segment_size_h // 2))
                y_end = min(h, int(segment_center_y + segment_size_h // 2 + 1))
                x_start = max(0, int(segment_center_x - segment_size_w // 2))
                x_end = min(w, int(segment_center_x + segment_size_w // 2 + 1))
                
                actual_h = y_end - y_start
                actual_w = x_end - x_start
                
                if actual_h > 0 and actual_w > 0:
                    center_y = actual_h // 2
                    center_x = actual_w // 2
                    
                    # 楕円形パッチを描画
                    for py in range(actual_h):
                        for px in range(actual_w):
                            norm_y = (py - center_y) / (segment_size_h / 2) if segment_size_h > 1 else 0
                            norm_x = (px - center_x) / (segment_size_w / 2) if segment_size_w > 1 else 0
                            ellipse_dist = norm_y**2 + norm_x**2
                            
                            if ellipse_dist <= 1.0:
                                # 楕円内部での強度計算
                                ellipse_intensity = segment_intensity * (1.0 - ellipse_dist) * 0.8
                                shadow_mask[y_start + py, x_start + px] += ellipse_intensity
        
        # クリップとブラー
        shadow_mask = np.clip(shadow_mask, 0, 1)
        
        if blur_radius > 0:
            blur_size = max(1, blur_radius * 2 + 1)
            shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_size, blur_size), blur_radius / 2)
        
        shadow_mask_255 = (shadow_mask * 255).astype(np.uint8)
        
        non_zero_pixels = np.sum(shadow_mask_255 > 10)
        max_intensity = np.max(shadow_mask_255)
        sys.stderr.write(f"create_contact_shadow_from_v3_points: Generated shadow with {non_zero_pixels} pixels, max_intensity={max_intensity}\n")
        
        return shadow_mask_255
        
    except Exception as e:
        sys.stderr.write(f"create_contact_shadow_from_v3_points failed: {e}\n")
        return np.zeros(image_shape, dtype=np.uint8)


if __name__ == "__main__":
    # Simple test with command line arguments
    import json
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
            test_contact_point_detection(data.get("cutoutImageBase64"))
    else:
        print("Usage: python shadow_test_utils.py <input_json_file>")