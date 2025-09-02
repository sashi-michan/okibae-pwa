#!/usr/bin/env python3
"""
Integrated test for perspective shadow v1 + contact detection v3
Tests the combination of ground contact detection and perspective cast shadows
"""

import cv2
import numpy as np
import sys
import os
import json

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perspective_shadow.v1_basic import create_perspective_cast_shadow_v1
from contact_detection.v3_sliding_window import detect_ground_contact_points_v3


def create_test_mask():
    """Create a more complex test mask for testing"""
    mask = np.zeros((400, 400), dtype=np.uint8)
    
    # Create a complex object with multiple potential contact points
    # Main body (ellipse)
    cv2.ellipse(mask, (200, 280), (80, 60), 0, 0, 360, 255, -1)
    
    # Extended parts (simulate jaw/tail)
    cv2.ellipse(mask, (120, 320), (30, 20), -30, 0, 360, 255, -1)  # Left extension
    cv2.ellipse(mask, (280, 310), (25, 15), 15, 0, 360, 255, -1)   # Right extension
    
    # Top part (head/neck)
    cv2.ellipse(mask, (200, 200), (50, 40), 0, 0, 360, 255, -1)
    
    return mask


def convert_v3_contacts_to_mask(contact_points_v3, shape):
    """Convert v3 contact points to a float mask for perspective shadow"""
    contact_mask = np.zeros(shape, dtype=np.float32)
    
    if contact_points_v3 is None:
        return contact_mask
    
    h, w = shape
    for point in contact_points_v3:
        if len(point) >= 2:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < h and 0 <= x < w:
                # Create small contact area around each point
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        py, px = y + dy, x + dx
                        if 0 <= py < h and 0 <= px < w:
                            # Distance-based falloff
                            dist = np.sqrt(dy*dy + dx*dx)
                            intensity = max(0, 1.0 - dist/3.0)
                            contact_mask[py, px] = max(contact_mask[py, px], intensity)
    
    return contact_mask


def test_integrated_shadows():
    """Test integration of v3 contact detection + perspective shadows"""
    print("=== Testing Integrated Shadows (V3 + Perspective) ===")
    
    # Create test data
    print("Creating complex test mask...")
    alpha_mask = create_test_mask()
    print(f"Test mask: {alpha_mask.shape}, non-zero pixels: {np.sum(alpha_mask > 0)}")
    
    # Test v3 contact detection
    print("\nTesting v3 contact detection...")
    contact_points_v3 = detect_ground_contact_points_v3(alpha_mask)
    
    # v3 returns a numpy array or None
    if contact_points_v3 is not None and len(contact_points_v3) > 0:
        print(f"V3 detected {len(contact_points_v3)} contact points")
        for i, point in enumerate(contact_points_v3[:5]):  # Show first 5
            print(f"  Point {i+1}: y={point[0]:.1f}, x={point[1]:.1f}")
    else:
        print("V3 detection returned None or empty")
        contact_points_v3 = np.array([])  # Ensure it's a numpy array
    
    # Convert to mask format
    contact_mask = convert_v3_contacts_to_mask(contact_points_v3, alpha_mask.shape)
    print(f"Contact mask: non-zero pixels: {np.sum(contact_mask > 0.01)}, max: {np.max(contact_mask):.3f}")
    
    # Test shadow generation with different approaches
    test_angles = [90, 135, 180, 225]  # Right, down-right, down, down-left
    max_distance = 100
    
    results = {}
    
    for ui_angle in test_angles:
        print(f"\nTesting angle {ui_angle}°...")
        
        # Convert UI angle to shadow vector
        math_deg = (90 - ui_angle) % 360
        math_rad = np.radians(math_deg)
        dx = np.cos(math_rad) * max_distance
        dy = -np.sin(math_rad) * max_distance
        
        # Test 1: No contact mask (fallback)
        shadow_no_contact = create_perspective_cast_shadow_v1(
            alpha_mask, None, (dx, dy), max_distance
        )
        
        # Test 2: With v3 contact mask
        shadow_with_contact = create_perspective_cast_shadow_v1(
            alpha_mask, contact_mask, (dx, dy), max_distance
        )
        
        results[ui_angle] = {
            'no_contact': shadow_no_contact,
            'with_contact': shadow_with_contact
        }
        
        print(f"  No contact: max={np.max(shadow_no_contact):.3f}, pixels={np.sum(shadow_no_contact > 0.01)}")
        print(f"  With V3:    max={np.max(shadow_with_contact):.3f}, pixels={np.sum(shadow_with_contact > 0.01)}")
    
    # Save visualization
    output_dir = "debug_output_integrated_shadows"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving visualization to {output_dir}/")
    
    # Save source images
    cv2.imwrite(os.path.join(output_dir, "01_source_mask.png"), alpha_mask)
    cv2.imwrite(os.path.join(output_dir, "02_contact_mask.png"), (contact_mask * 255).astype(np.uint8))
    
    # Save comparison results
    for angle in test_angles:
        shadow_no = results[angle]['no_contact']
        shadow_with = results[angle]['with_contact']
        
        # Individual results
        cv2.imwrite(os.path.join(output_dir, f"03_shadow_no_contact_{angle:03d}deg.png"), 
                   (shadow_no * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, f"04_shadow_with_v3_{angle:03d}deg.png"), 
                   (shadow_with * 255).astype(np.uint8))
        
        # Side-by-side comparison
        comparison = np.hstack([shadow_no, shadow_with])
        cv2.imwrite(os.path.join(output_dir, f"05_comparison_{angle:03d}deg.png"), 
                   (comparison * 255).astype(np.uint8))
        
        # Composite with original
        composite = np.maximum(alpha_mask.astype(np.float32) / 255.0, shadow_with)
        cv2.imwrite(os.path.join(output_dir, f"06_composite_{angle:03d}deg.png"), 
                   (composite * 255).astype(np.uint8))
    
    # Generate analysis report
    report = {
        "test_summary": {
            "mask_shape": alpha_mask.shape,
            "mask_pixels": int(np.sum(alpha_mask > 0)),
            "v3_contact_points": len(contact_points_v3) if contact_points_v3 is not None and len(contact_points_v3) > 0 else 0,
            "contact_mask_pixels": int(np.sum(contact_mask > 0.01)),
            "test_angles": test_angles,
            "max_distance": max_distance
        },
        "v3_contact_points": [
            {"y": float(p[0]), "x": float(p[1])} 
            for p in contact_points_v3[:10]  # Save first 10
        ] if contact_points_v3 is not None and len(contact_points_v3) > 0 else [],
        "shadow_comparison": {}
    }
    
    for angle in test_angles:
        shadow_no = results[angle]['no_contact']
        shadow_with = results[angle]['with_contact']
        
        report["shadow_comparison"][str(angle)] = {
            "no_contact": {
                "max": float(np.max(shadow_no)),
                "pixels": int(np.sum(shadow_no > 0.01)),
                "mean": float(np.mean(shadow_no[shadow_no > 0.01])) if np.sum(shadow_no > 0.01) > 0 else 0
            },
            "with_v3": {
                "max": float(np.max(shadow_with)),
                "pixels": int(np.sum(shadow_with > 0.01)),
                "mean": float(np.mean(shadow_with[shadow_with > 0.01])) if np.sum(shadow_with > 0.01) > 0 else 0
            }
        }
    
    report_path = os.path.join(output_dir, "integration_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Integration test report saved to {report_path}")
    print("\n=== Integration Test Complete ===")
    
    return True


if __name__ == "__main__":
    try:
        success = test_integrated_shadows()
        if success:
            print("Integration test passed!")
            sys.exit(0)
        else:
            print("Integration test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Integration test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)