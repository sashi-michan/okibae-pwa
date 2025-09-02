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


def load_test_image(image_path):
    """Load and process test image to extract alpha mask"""
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return None
    
    print(f"Loading test image: {image_path}")
    original_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if original_img is None:
        print(f"Failed to load {image_path}")
        return None
    
    print(f"Original image shape: {original_img.shape}")
    
    # Handle different image formats
    if len(original_img.shape) >= 3 and original_img.shape[2] == 4:
        # RGBA image - extract alpha channel
        alpha_mask = original_img[:, :, 3]
    elif len(original_img.shape) >= 3:
        # RGB image - convert to grayscale then binary
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        _, alpha_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    else:
        # Already grayscale
        _, alpha_mask = cv2.threshold(original_img, 1, 255, cv2.THRESH_BINARY)
    
    return alpha_mask


def test_integrated_perspective_shadows():
    """Test integration of v3 contact detection + perspective shadows"""
    print("=== Testing Integrated Perspective Shadows (V3 + Perspective) ===")
    
    # Test with both generated mask and real images
    test_cases = [
        ("generated", create_test_mask()),
        ("test1", load_test_image("../sample/test1.png")),
        ("test2", load_test_image("../sample/test2.png"))
    ]
    
    # Filter out failed loads
    valid_test_cases = [(name, mask) for name, mask in test_cases if mask is not None]
    
    if not valid_test_cases:
        print("No valid test cases available!")
        return False
    
    # Test parameters
    test_angles = [90, 135, 180, 225]  # Right, down-right, down, down-left
    max_distance = 150
    
    # Create output directory
    output_dir = "debug_output_integrated_perspective_shadows"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for test_name, alpha_mask in valid_test_cases:
        print(f"\n--- Testing {test_name} ---")
        print(f"Mask: {alpha_mask.shape}, non-zero pixels: {np.sum(alpha_mask > 0)}")
        
        # Test v3 contact detection
        print("Testing v3 contact detection...")
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
        results = {}
        
        for ui_angle in test_angles:
            print(f"Testing angle {ui_angle}°...")
            
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
        test_output_dir = os.path.join(output_dir, test_name)
        os.makedirs(test_output_dir, exist_ok=True)
        
        print(f"Saving visualization to {test_output_dir}/")
        
        # Save source images with white background
        alpha_on_white = np.zeros((alpha_mask.shape[0], alpha_mask.shape[1], 3), dtype=np.uint8)
        alpha_on_white[:, :] = (255, 255, 255)  # White background
        alpha_on_white[alpha_mask > 0] = (0, 0, 0)  # Object as black
        cv2.imwrite(os.path.join(test_output_dir, "01_source_mask.png"), alpha_on_white)
        
        contact_on_white = np.zeros((contact_mask.shape[0], contact_mask.shape[1], 3), dtype=np.uint8)
        contact_on_white[:, :] = (255, 255, 255)  # White background
        contact_intensity = (contact_mask * 255).astype(np.uint8)
        for i in range(3):
            contact_on_white[:, :, i] = 255 - contact_intensity  # Contact as dark on white
        cv2.imwrite(os.path.join(test_output_dir, "02_contact_mask.png"), contact_on_white)
        
        # Save comparison results with white background
        for angle in test_angles:
            shadow_no = results[angle]['no_contact']
            shadow_with = results[angle]['with_contact']
            
            # Handle expanded canvas case - crop if needed
            if shadow_no.shape != alpha_mask.shape:
                print(f"  Note: Shadow canvas expanded to {shadow_no.shape} (original: {alpha_mask.shape})")
            if shadow_with.shape != alpha_mask.shape:
                print(f"  Note: Shadow canvas expanded to {shadow_with.shape} (original: {alpha_mask.shape})")
            
            # Individual results with white background
            shadow_no_on_white = np.zeros((shadow_no.shape[0], shadow_no.shape[1], 3), dtype=np.uint8)
            shadow_no_on_white[:, :] = (255, 255, 255)  # White background
            shadow_no_intensity = (shadow_no * 255).astype(np.uint8)
            for i in range(3):
                shadow_no_on_white[:, :, i] = 255 - shadow_no_intensity
            cv2.imwrite(os.path.join(test_output_dir, f"03_perspective_shadow_no_contact_{angle:03d}deg.png"), 
                       shadow_no_on_white)
            
            shadow_with_on_white = np.zeros((shadow_with.shape[0], shadow_with.shape[1], 3), dtype=np.uint8)
            shadow_with_on_white[:, :] = (255, 255, 255)  # White background
            shadow_with_intensity = (shadow_with * 255).astype(np.uint8)
            for i in range(3):
                shadow_with_on_white[:, :, i] = 255 - shadow_with_intensity
            cv2.imwrite(os.path.join(test_output_dir, f"04_perspective_shadow_with_v3_{angle:03d}deg.png"), 
                       shadow_with_on_white)
            
            # Composite with original (V3-style with white background)
            alpha_normalized = alpha_mask.astype(np.float32) / 255.0
            
            # Handle canvas size mismatch for composite
            if shadow_with.shape != alpha_mask.shape:
                # Crop shadow to original size for composite
                h_orig, w_orig = alpha_mask.shape
                h_shadow, w_shadow = shadow_with.shape
                y_offset = (h_shadow - h_orig) // 2
                x_offset = (w_shadow - w_orig) // 2
                shadow_cropped = shadow_with[y_offset:y_offset+h_orig, x_offset:x_offset+w_orig]
                composite = np.maximum(alpha_normalized, shadow_cropped)
            else:
                composite = np.maximum(alpha_normalized, shadow_with)
            
            composite_on_white = np.zeros((composite.shape[0], composite.shape[1], 3), dtype=np.uint8)
            composite_on_white[:, :] = (255, 255, 255)  # White background
            composite_intensity = (composite * 255).astype(np.uint8)
            for i in range(3):
                composite_on_white[:, :, i] = 255 - composite_intensity
            cv2.imwrite(os.path.join(test_output_dir, f"05_composite_{angle:03d}deg.png"), 
                       composite_on_white)
        
        all_results[test_name] = results
    
    # Generate analysis report
    report = {
        "test_summary": {
            "test_cases": list(all_results.keys()),
            "test_angles": test_angles,
            "max_distance": max_distance
        },
        "results": {}
    }
    
    for test_name, results in all_results.items():
        test_mask = dict(valid_test_cases)[test_name]
        contact_points_v3 = detect_ground_contact_points_v3(test_mask)
        contact_mask = convert_v3_contacts_to_mask(contact_points_v3, test_mask.shape)
        
        report["results"][test_name] = {
            "mask_shape": list(test_mask.shape),
            "mask_pixels": int(np.sum(test_mask > 0)),
            "v3_contact_points": len(contact_points_v3) if contact_points_v3 is not None and len(contact_points_v3) > 0 else 0,
            "contact_mask_pixels": int(np.sum(contact_mask > 0.01)),
            "v3_contact_points_data": [
                {"y": float(p[0]), "x": float(p[1])} 
                for p in contact_points_v3[:10]  # Save first 10
            ] if contact_points_v3 is not None and len(contact_points_v3) > 0 else [],
            "shadow_comparison": {}
        }
        
        for angle in test_angles:
            shadow_no = results[angle]['no_contact']
            shadow_with = results[angle]['with_contact']
            
            report["results"][test_name]["shadow_comparison"][str(angle)] = {
                "no_contact": {
                    "max": float(np.max(shadow_no)),
                    "pixels": int(np.sum(shadow_no > 0.01)),
                    "mean": float(np.mean(shadow_no[shadow_no > 0.01])) if np.sum(shadow_no > 0.01) > 0 else 0,
                    "shape": list(shadow_no.shape)
                },
                "with_v3": {
                    "max": float(np.max(shadow_with)),
                    "pixels": int(np.sum(shadow_with > 0.01)),
                    "mean": float(np.mean(shadow_with[shadow_with > 0.01])) if np.sum(shadow_with > 0.01) > 0 else 0,
                    "shape": list(shadow_with.shape)
                }
            }
    
    report_path = os.path.join(output_dir, "perspective_integration_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nIntegration test report saved to {report_path}")
    print("=== Perspective Integration Test Complete ===")
    
    return True


if __name__ == "__main__":
    try:
        success = test_integrated_perspective_shadows()
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