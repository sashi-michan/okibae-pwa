#!/usr/bin/env python3
"""
Standalone test for perspective shadow v1
Tests the height map generation and perspective projection algorithms
"""

import cv2
import numpy as np
import sys
import os
from PIL import Image
import json

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perspective_shadow.v1_basic import create_perspective_cast_shadow_v1, create_height_map_debug


def create_test_mask():
    """Create a simple test mask for testing"""
    mask = np.zeros((400, 400), dtype=np.uint8)
    
    # Create a simple rectangular object
    cv2.rectangle(mask, (150, 200), (250, 350), 255, -1)  # Rectangle at bottom
    
    # Add a small circle on top
    cv2.circle(mask, (200, 180), 30, 255, -1)
    
    return mask


def create_contact_mask_simple(alpha_mask):
    """Create a simple contact mask for testing (bottom edge contact)"""
    h, w = alpha_mask.shape
    contact_mask = np.zeros((h, w), dtype=np.float32)
    
    # Find bottom edge of object
    bin_mask = (alpha_mask > 0)
    for x in range(w):
        col = bin_mask[:, x]
        if np.any(col):
            # Find last True value (bottom edge)
            bottom_y = np.where(col)[0][-1]
            if bottom_y < h - 5:  # Not at very edge
                # Create contact area
                for dy in range(-2, 3):
                    if 0 <= bottom_y + dy < h:
                        contact_mask[bottom_y + dy, x] = 1.0
    
    return contact_mask


def visualize_results(alpha_mask, contact_mask, height_maps, shadow_results, output_dir):
    """Save visualization images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original mask
    cv2.imwrite(os.path.join(output_dir, "01_original_mask.png"), alpha_mask)
    
    # Save contact mask
    contact_vis = (contact_mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "02_contact_mask.png"), contact_vis)
    
    # Save height maps
    for name, height_map in height_maps.items():
        height_vis = (height_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"03_height_{name}.png"), height_vis)
    
    # Save shadow results
    for angle, shadow in shadow_results.items():
        shadow_vis = (shadow * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"04_shadow_{angle:03d}deg.png"), shadow_vis)
        
        # Create composite view (original + shadow)
        composite = np.zeros_like(alpha_mask)
        composite = np.maximum(alpha_mask, shadow_vis)
        cv2.imwrite(os.path.join(output_dir, f"05_composite_{angle:03d}deg.png"), composite)


def test_perspective_shadow():
    """Run comprehensive test of perspective shadow generation"""
    print("=== Testing Perspective Shadow V1 ===")
    
    # Create test data
    print("Creating test mask...")
    alpha_mask = create_test_mask()
    contact_mask = create_contact_mask_simple(alpha_mask)
    
    print(f"Test mask: {alpha_mask.shape}, non-zero pixels: {np.sum(alpha_mask > 0)}")
    print(f"Contact mask: non-zero pixels: {np.sum(contact_mask > 0.01)}")
    
    # Test height map generation
    print("\nTesting height map generation...")
    height_maps = create_height_map_debug(alpha_mask, contact_mask)
    
    for name, hmap in height_maps.items():
        print(f"  {name}: max={np.max(hmap):.3f}, mean={np.mean(hmap[hmap > 0]):.3f}")
    
    # Test shadow generation at different angles
    print("\nTesting perspective shadow generation...")
    test_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # UI degrees
    shadow_results = {}
    max_distance = 80  # pixels
    
    for ui_angle in test_angles:
        print(f"  Testing angle {ui_angle}°...")
        
        # Convert UI angle to shadow vector (image coordinates)
        # UI: 0°=up, clockwise; Math: 0°=right, counterclockwise
        math_deg = (90 - ui_angle) % 360
        math_rad = np.radians(math_deg)
        
        # Shadow vector in image coordinates (y↓ positive)
        dx = np.cos(math_rad) * max_distance
        dy = -np.sin(math_rad) * max_distance  # Invert for image coordinates
        
        print(f"    UI {ui_angle}° -> Math {math_deg}° -> Vector ({dx:.1f}, {dy:.1f})")
        
        # Generate shadow
        shadow = create_perspective_cast_shadow_v1(
            alpha_mask, contact_mask, (dx, dy), max_distance
        )
        
        shadow_results[ui_angle] = shadow
        print(f"    Result: max={np.max(shadow):.3f}, non-zero={np.sum(shadow > 0.01)}")
    
    # Save visualization
    output_dir = "debug_output_perspective_v1"
    print(f"\nSaving visualization to {output_dir}/")
    visualize_results(alpha_mask, contact_mask, height_maps, shadow_results, output_dir)
    
    # Generate summary report
    report = {
        "test_summary": {
            "mask_size": alpha_mask.shape,
            "mask_pixels": int(np.sum(alpha_mask > 0)),
            "contact_pixels": int(np.sum(contact_mask > 0.01)),
            "test_angles": test_angles,
            "max_distance": max_distance
        },
        "height_maps": {
            name: {
                "max": float(np.max(hmap)),
                "mean": float(np.mean(hmap[hmap > 0])) if np.sum(hmap > 0) > 0 else 0,
                "non_zero_pixels": int(np.sum(hmap > 0))
            }
            for name, hmap in height_maps.items()
        },
        "shadow_results": {
            str(angle): {
                "max_intensity": float(np.max(shadow)),
                "non_zero_pixels": int(np.sum(shadow > 0.01)),
                "mean_intensity": float(np.mean(shadow[shadow > 0.01])) if np.sum(shadow > 0.01) > 0 else 0
            }
            for angle, shadow in shadow_results.items()
        }
    }
    
    report_path = os.path.join(output_dir, "test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Test report saved to {report_path}")
    print("\n=== Test Complete ===")
    
    return True


if __name__ == "__main__":
    try:
        success = test_perspective_shadow()
        if success:
            print("All tests passed!")
            sys.exit(0)
        else:
            print("Tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)