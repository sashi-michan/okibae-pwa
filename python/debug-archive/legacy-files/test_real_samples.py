#!/usr/bin/env python3
"""
Test perspective shadows with real sample images
Tests with test1.png (orca ring) and test2.png (macrame glasses cord)
"""

import cv2
import numpy as np
import sys
import os
import json

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perspective_shadow.v1_basic import create_perspective_cast_shadow_v1, create_height_map_debug
from contact_detection.v3_sliding_window import detect_ground_contact_points_v3


def load_sample_image(sample_path):
    """Load sample image and extract alpha mask"""
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample image not found: {sample_path}")
    
    # Load with alpha channel
    image = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Could not load image: {sample_path}")
    
    print(f"Loaded {sample_path}: shape={image.shape}, dtype={image.dtype}")
    
    # Extract alpha channel
    if len(image.shape) == 3 and image.shape[2] == 4:
        alpha_mask = image[:, :, 3]
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, alpha_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    elif len(image.shape) == 2:
        alpha_mask = image
    else:
        raise ValueError(f"Unsupported image format: {image.shape}")
    
    return alpha_mask


def convert_v3_contacts_to_mask(contact_points_v3, shape):
    """Convert v3 contact points to a float mask for perspective shadow"""
    contact_mask = np.zeros(shape, dtype=np.float32)
    
    if contact_points_v3 is None or len(contact_points_v3) == 0:
        return contact_mask
    
    h, w = shape
    for point in contact_points_v3:
        if len(point) >= 2:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < h and 0 <= x < w:
                # Create small contact area around each point
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        py, px = y + dy, x + dx
                        if 0 <= py < h and 0 <= px < w:
                            # Distance-based falloff
                            dist = np.sqrt(dy*dy + dx*dx)
                            intensity = max(0, 1.0 - dist/4.0)
                            contact_mask[py, px] = max(contact_mask[py, px], intensity)
    
    return contact_mask


def test_real_sample(sample_name, sample_path, output_dir):
    """Test perspective shadow with a real sample image"""
    print(f"\n=== Testing {sample_name} ===")
    
    # Load sample
    alpha_mask = load_sample_image(sample_path)
    h, w = alpha_mask.shape
    print(f"Alpha mask: {w}x{h}, non-zero pixels: {np.sum(alpha_mask > 0)}")
    
    # Test v3 contact detection
    print("Running v3 contact detection...")
    contact_points_v3 = detect_ground_contact_points_v3(alpha_mask)
    
    if contact_points_v3 is not None and len(contact_points_v3) > 0:
        print(f"V3 detected {len(contact_points_v3)} contact points")
        # Show some contact points
        for i, point in enumerate(contact_points_v3[:3]):
            print(f"  Contact {i+1}: y={point[0]:.1f}, x={point[1]:.1f}")
    else:
        print("No contact points detected")
        contact_points_v3 = np.array([])
    
    # Convert to contact mask
    contact_mask = convert_v3_contacts_to_mask(contact_points_v3, alpha_mask.shape)
    print(f"Contact mask: non-zero pixels: {np.sum(contact_mask > 0.01)}")
    
    # Generate height maps for visualization
    print("Generating height maps...")
    height_maps = create_height_map_debug(alpha_mask, contact_mask)
    
    # Test shadow generation at multiple angles
    test_angles = [45, 90, 135, 180, 225, 270, 315]  # Skip 0 (often problematic)
    max_distance = min(w, h) * 0.3  # 30% of smaller dimension
    
    print(f"Testing shadows with max_distance={max_distance:.1f}...")
    
    results = {
        'without_contact': {},
        'with_contact': {},
        'height_maps': height_maps
    }
    
    for ui_angle in test_angles:
        print(f"  Angle {ui_angle}°...", end="")
        
        # Convert UI angle to shadow vector
        math_deg = (90 - ui_angle) % 360
        math_rad = np.radians(math_deg)
        dx = np.cos(math_rad) * max_distance
        dy = -np.sin(math_rad) * max_distance
        
        # Test without contact mask
        shadow_no_contact = create_perspective_cast_shadow_v1(
            alpha_mask, None, (dx, dy), max_distance
        )
        
        # Test with contact mask
        shadow_with_contact = create_perspective_cast_shadow_v1(
            alpha_mask, contact_mask, (dx, dy), max_distance
        )
        
        results['without_contact'][ui_angle] = shadow_no_contact
        results['with_contact'][ui_angle] = shadow_with_contact
        
        no_pixels = np.sum(shadow_no_contact > 0.01)
        with_pixels = np.sum(shadow_with_contact > 0.01)
        print(f" no_contact={no_pixels} (size: {shadow_no_contact.shape}), with_contact={with_pixels} (size: {shadow_with_contact.shape})")
    
    # Save visualization
    sample_output = os.path.join(output_dir, sample_name)
    os.makedirs(sample_output, exist_ok=True)
    
    print(f"Saving results to {sample_output}/")
    
    # Save source images
    cv2.imwrite(os.path.join(sample_output, "01_original_mask.png"), alpha_mask)
    cv2.imwrite(os.path.join(sample_output, "02_contact_mask.png"), 
               (contact_mask * 255).astype(np.uint8))
    
    # Save height maps
    for name, hmap in height_maps.items():
        cv2.imwrite(os.path.join(sample_output, f"03_height_{name}.png"), 
                   (hmap * 255).astype(np.uint8))
    
    # Save shadow results
    for angle in test_angles:
        shadow_no = results['without_contact'][angle]
        shadow_with = results['with_contact'][angle]
        
        # Individual shadows
        cv2.imwrite(os.path.join(sample_output, f"04_no_contact_{angle:03d}deg.png"), 
                   (shadow_no * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(sample_output, f"05_with_contact_{angle:03d}deg.png"), 
                   (shadow_with * 255).astype(np.uint8))
        
        # Composite views (original + shadow) - need to handle size differences
        if shadow_no.shape == alpha_mask.shape:
            # Same size - direct composite
            composite_no = np.maximum(alpha_mask.astype(np.float32) / 255.0, shadow_no)
            composite_with = np.maximum(alpha_mask.astype(np.float32) / 255.0, shadow_with)
        else:
            # Expanded canvas - place original in center
            composite_no = shadow_no.copy()
            composite_with = shadow_with.copy()
            
            # Calculate placement position (center the original image)
            sh, sw = shadow_no.shape
            ah, aw = alpha_mask.shape
            start_y = (sh - ah) // 2
            start_x = (sw - aw) // 2
            
            # Overlay original image
            alpha_float = alpha_mask.astype(np.float32) / 255.0
            composite_no[start_y:start_y+ah, start_x:start_x+aw] = np.maximum(
                composite_no[start_y:start_y+ah, start_x:start_x+aw], alpha_float)
            composite_with[start_y:start_y+ah, start_x:start_x+aw] = np.maximum(
                composite_with[start_y:start_y+ah, start_x:start_x+aw], alpha_float)
        
        cv2.imwrite(os.path.join(sample_output, f"06_composite_no_{angle:03d}deg.png"), 
                   (composite_no * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(sample_output, f"07_composite_with_{angle:03d}deg.png"), 
                   (composite_with * 255).astype(np.uint8))
    
    # Generate report for this sample
    report = {
        "sample_info": {
            "name": sample_name,
            "path": sample_path,
            "size": {"width": w, "height": h},
            "mask_pixels": int(np.sum(alpha_mask > 0)),
            "max_distance": max_distance
        },
        "v3_contact_detection": {
            "total_points": len(contact_points_v3) if contact_points_v3 is not None else 0,
            "contact_mask_pixels": int(np.sum(contact_mask > 0.01)),
            "sample_points": [
                {"y": float(p[0]), "x": float(p[1])} 
                for p in contact_points_v3[:5]
            ] if contact_points_v3 is not None and len(contact_points_v3) > 0 else []
        },
        "height_maps": {
            name: {
                "max": float(np.max(hmap)),
                "mean": float(np.mean(hmap[hmap > 0])) if np.sum(hmap > 0) > 0 else 0,
                "non_zero_pixels": int(np.sum(hmap > 0))
            }
            for name, hmap in height_maps.items()
        },
        "shadow_results": {}
    }
    
    for angle in test_angles:
        shadow_no = results['without_contact'][angle]
        shadow_with = results['with_contact'][angle]
        
        report["shadow_results"][str(angle)] = {
            "without_contact": {
                "max": float(np.max(shadow_no)),
                "pixels": int(np.sum(shadow_no > 0.01)),
                "mean": float(np.mean(shadow_no[shadow_no > 0.01])) if np.sum(shadow_no > 0.01) > 0 else 0
            },
            "with_contact": {
                "max": float(np.max(shadow_with)),
                "pixels": int(np.sum(shadow_with > 0.01)),
                "mean": float(np.mean(shadow_with[shadow_with > 0.01])) if np.sum(shadow_with > 0.01) > 0 else 0
            }
        }
    
    # Save report
    report_path = os.path.join(sample_output, "sample_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved: {report_path}")
    
    return report


def main():
    """Test both sample images"""
    print("=== Real Sample Testing ===")
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    sample_dir = os.path.join(project_dir, "sample")
    output_dir = os.path.join(script_dir, "debug_output_real_samples")
    
    samples = [
        ("orca_ring", os.path.join(sample_dir, "test1.png")),
        ("macrame_glasses", os.path.join(sample_dir, "test2.png"))
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_reports = {}
    
    for sample_name, sample_path in samples:
        try:
            report = test_real_sample(sample_name, sample_path, output_dir)
            all_reports[sample_name] = report
        except Exception as e:
            print(f"Error testing {sample_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined report
    combined_path = os.path.join(output_dir, "combined_test_report.json")
    with open(combined_path, 'w') as f:
        json.dump(all_reports, f, indent=2)
    
    print(f"\n=== Testing Complete ===")
    print(f"Combined report: {combined_path}")
    print(f"Individual results in: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)