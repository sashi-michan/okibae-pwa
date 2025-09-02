#!/usr/bin/env python3
"""
Visual composite test for perspective shadows
Creates proper composite images with original RGBA + shadow for better visualization
"""

import cv2
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perspective_shadow.v1_basic import create_perspective_cast_shadow_v1
from contact_detection.v3_sliding_window import detect_ground_contact_points_v3


def load_rgba_image(sample_path):
    """Load sample image as RGBA"""
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample image not found: {sample_path}")
    
    image = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image: {sample_path}")
    
    print(f"Loaded {sample_path}: shape={image.shape}")
    
    # Convert to RGBA if needed
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            # BGR to BGRA
            bgra = np.concatenate([image, np.full((image.shape[0], image.shape[1], 1), 255, dtype=image.dtype)], axis=2)
            image = bgra
        elif image.shape[2] == 4:
            # Already BGRA
            pass
    
    return image


def create_visual_composite(original_rgba, shadow_mask, shadow_color=(60, 60, 60), shadow_opacity=0.7, background_color=(255, 255, 255)):
    """
    Create visual composite of original image + shadow for better visualization
    
    Args:
        original_rgba: Original RGBA image (BGRA format for OpenCV)
        shadow_mask: Shadow mask (0-1 float)
        shadow_color: RGB color for shadow (default: dark gray)
        shadow_opacity: Shadow opacity multiplier
        background_color: RGB background color (default: white)
    
    Returns:
        Composite RGB image with white background
    """
    # Handle size differences
    oh, ow = original_rgba.shape[:2]
    sh, sw = shadow_mask.shape
    
    if (sh, sw) == (oh, ow):
        # Same size - direct composite
        rgba_canvas = original_rgba.copy().astype(np.float32)
        shadow_expanded = shadow_mask
        obj_y, obj_x = 0, 0
    else:
        # Expanded canvas - place original in center
        rgba_canvas = np.zeros((sh, sw, 4), dtype=np.float32)
        
        # Calculate placement (center the original image)
        obj_y = (sh - oh) // 2
        obj_x = (sw - ow) // 2
        
        # Place original image
        rgba_canvas[obj_y:obj_y+oh, obj_x:obj_x+ow] = original_rgba.astype(np.float32)
        shadow_expanded = shadow_mask
    
    # Create final RGB composite with white background
    composite_rgb = np.full((sh, sw, 3), background_color[::-1], dtype=np.float32)  # RGB to BGR, white background
    
    # Add shadows first (behind the object)
    shadow_areas = shadow_expanded > 0.01
    shadow_bgr = np.array(shadow_color[::-1], dtype=np.float32)  # RGB to BGR
    shadow_intensity = shadow_expanded * shadow_opacity
    
    # Blend shadow with background
    for c in range(3):
        composite_rgb[shadow_areas, c] = (
            composite_rgb[shadow_areas, c] * (1 - shadow_intensity[shadow_areas]) +
            shadow_bgr[c] * shadow_intensity[shadow_areas]
        )
    
    # Then add the object on top
    object_mask = rgba_canvas[:, :, 3] > 0
    object_alpha = rgba_canvas[:, :, 3] / 255.0
    
    for c in range(3):
        composite_rgb[object_mask, c] = (
            composite_rgb[object_mask, c] * (1 - object_alpha[object_mask]) +
            rgba_canvas[object_mask, c] * object_alpha[object_mask]
        )
    
    return composite_rgb.astype(np.uint8)


def test_visual_composite():
    """Test visual composites with proper shadow rendering"""
    print("=== Visual Composite Test ===")
    
    # Test with both samples
    samples = [
        ("orca_ring", "../sample/test1.png"),
        # ("macrame_glasses", "../sample/test2.png")  # Skip large one for now
    ]
    
    for sample_name, sample_path in samples:
        print(f"\n--- Testing {sample_name} ---")
        
        # Load original RGBA image
        original_rgba = load_rgba_image(sample_path)
        alpha_mask = original_rgba[:, :, 3]
        
        h, w = alpha_mask.shape
        print(f"Original: {w}x{h}, non-zero pixels: {np.sum(alpha_mask > 0)}")
        
        # V3 contact detection
        contact_points_v3 = detect_ground_contact_points_v3(alpha_mask)
        contact_mask = np.zeros((h, w), dtype=np.float32)
        if contact_points_v3 is not None and len(contact_points_v3) > 0:
            for point in contact_points_v3:
                y, x = int(point[0]), int(point[1])
                if 0 <= y < h and 0 <= x < w:
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            py, px = y + dy, x + dx
                            if 0 <= py < h and 0 <= px < w:
                                dist = np.sqrt(dy*dy + dx*dx)
                                intensity = max(0, 1.0 - dist/4.0)
                                contact_mask[py, px] = max(contact_mask[py, px], intensity)
        
        # Test different shadow angles with visual composites
        test_angles = [135, 180, 225]  # Focus on most visible angles
        max_distance = min(w, h) * 0.4  # 40% for much longer, more visible shadows
        
        output_dir = f"debug_output_visual_{sample_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating visual composites with max_distance={max_distance:.1f}...")
        
        for ui_angle in test_angles:
            print(f"  Angle {ui_angle}°...", end="")
            
            # Convert UI angle to shadow vector
            math_deg = (90 - ui_angle) % 360
            math_rad = np.radians(math_deg)
            dx = np.cos(math_rad) * max_distance
            dy = -np.sin(math_rad) * max_distance
            
            # Generate perspective shadow
            shadow_mask = create_perspective_cast_shadow_v1(
                alpha_mask, contact_mask, (dx, dy), max_distance
            )
            
            # Create visual composite with white background
            visual_composite = create_visual_composite(
                original_rgba, shadow_mask, 
                shadow_color=(30, 30, 30),  # Darker shadow
                shadow_opacity=0.8,  # Higher opacity for more visible shadows
                background_color=(255, 255, 255)  # White background
            )
            
            # Save results
            cv2.imwrite(os.path.join(output_dir, f"shadow_only_{ui_angle:03d}deg.png"), 
                       (shadow_mask * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(output_dir, f"visual_composite_{ui_angle:03d}deg.png"), 
                       visual_composite)
            
            shadow_pixels = np.sum(shadow_mask > 0.01)
            print(f" shadow_pixels={shadow_pixels}, composite_shape={visual_composite.shape}")
        
        print(f"\nResults saved to {output_dir}/")
        print("Key files:")
        for angle in test_angles:
            print(f"  visual_composite_{angle:03d}deg.png - Original + Shadow")
    
    print("\n=== Visual Test Complete ===")


if __name__ == "__main__":
    try:
        test_visual_composite()
    except Exception as e:
        print(f"Visual test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)