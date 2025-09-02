#!/usr/bin/env python3
"""
Test V1 vs V2 vs V3 long shadow implementations
Compare all three approaches with V3 contact detection
"""

import cv2
import numpy as np
import sys
import os
import json

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perspective_shadow.v1_basic import create_perspective_cast_shadow_v1
from perspective_shadow.v2_longshadow import build_long_shadow
from perspective_shadow.v3_longshadow import build_long_shadow_v3
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


def save_with_white_background(image_array, output_path, is_float=True):
    """Save image array with white background"""
    if is_float:
        # Float array (0-1) -> grayscale with white background
        image_on_white = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
        image_on_white[:, :] = (255, 255, 255)  # White background
        intensity = (image_array * 255).astype(np.uint8)
        for i in range(3):
            image_on_white[:, :, i] = 255 - intensity  # Dark on white
    else:
        # Binary array (0-255) -> black on white
        image_on_white = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
        image_on_white[:, :] = (255, 255, 255)  # White background
        image_on_white[image_array > 0] = (0, 0, 0)  # Object as black
    
    cv2.imwrite(output_path, image_on_white)


def crop_to_original_size(shadow, original_shape, expanded_info=None):
    """Crop expanded shadow to original size"""
    if shadow.shape == original_shape:
        return shadow
    
    h_orig, w_orig = original_shape
    h_shadow, w_shadow = shadow.shape
    
    if expanded_info:
        # Use provided offsets
        start_y, start_x = expanded_info
        return shadow[start_y:start_y+h_orig, start_x:start_x+w_orig]
    else:
        # Center crop
        y_offset = (h_shadow - h_orig) // 2
        x_offset = (w_shadow - w_orig) // 2
        return shadow[y_offset:y_offset+h_orig, x_offset:x_offset+w_orig]


def test_v1_v2_v3_comparison():
    """Test and compare v1 vs v2 vs v3 shadow implementations"""
    print("=== Testing V1 vs V2 vs V3 Long Shadow Comparison ===")
    
    # Test with both generated mask and real images
    test_cases = [
        ("generated", create_test_mask()),
        ("test1", load_test_image("../sample/test1.png")),
    ]
    
    # Filter out failed loads
    valid_test_cases = [(name, mask) for name, mask in test_cases if mask is not None]
    
    if not valid_test_cases:
        print("No valid test cases available!")
        return False
    
    # Test parameters
    test_angle = 180  # Downward shadow for clear comparison
    max_distance = 150
    
    # Convert UI angle to shadow vector
    math_deg = (90 - test_angle) % 360
    math_rad = np.radians(math_deg)
    dx = np.cos(math_rad) * max_distance
    dy = -np.sin(math_rad) * max_distance
    
    # Create output directory
    output_dir = "debug_output_v1_v2_v3_comparison"
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
        else:
            print("V3 detection returned None or empty")
            contact_points_v3 = np.array([])
        
        # Convert to mask format
        contact_mask = convert_v3_contacts_to_mask(contact_points_v3, alpha_mask.shape)
        print(f"Contact mask: non-zero pixels: {np.sum(contact_mask > 0.01)}, max: {np.max(contact_mask):.3f}")
        
        # Test V1 shadow generation
        print("Generating V1 shadow...")
        shadow_v1 = create_perspective_cast_shadow_v1(
            alpha_mask, contact_mask, (dx, dy), max_distance
        )
        print(f"V1 shadow: max={np.max(shadow_v1):.3f}, pixels={np.sum(shadow_v1 > 0.01)}, shape={shadow_v1.shape}")
        
        # Test V2 shadow generation
        print("Generating V2 shadow...")
        v2_result = build_long_shadow(
            alpha_mask,
            contact_mask,
            (dx, dy),
            max_distance,
            return_debug=True
        )
        shadow_v2 = v2_result['shadow_combined']
        print(f"V2 shadow: max={np.max(shadow_v2):.3f}, pixels={np.sum(shadow_v2 > 0.01)}, shape={shadow_v2.shape}")
        
        # Test V3 shadow generation (NEW!)
        print("Generating V3 shadow...")
        v3_result = build_long_shadow_v3(
            alpha_mask,
            contact_mask,
            (dx, dy),
            max_distance,
            return_debug=True
        )
        shadow_v3 = v3_result['shadow_combined']
        print(f"V3 shadow: max={np.max(shadow_v3):.3f}, pixels={np.sum(shadow_v3 > 0.01)}, shape={shadow_v3.shape}")
        
        # Save visualization
        test_output_dir = os.path.join(output_dir, test_name)
        os.makedirs(test_output_dir, exist_ok=True)
        
        print(f"Saving visualization to {test_output_dir}/")
        
        # Save source images
        save_with_white_background(alpha_mask, os.path.join(test_output_dir, "01_source_mask.png"), is_float=False)
        save_with_white_background(contact_mask, os.path.join(test_output_dir, "02_contact_mask.png"), is_float=True)
        
        # Crop all shadows to original size for comparison
        shadow_v1_cropped = crop_to_original_size(shadow_v1, alpha_mask.shape)
        shadow_v2_cropped = crop_to_original_size(shadow_v2, alpha_mask.shape, 
                                                  v2_result['offsets'] if 'offsets' in v2_result else None)
        shadow_v3_cropped = crop_to_original_size(shadow_v3, alpha_mask.shape, 
                                                  v3_result['offsets'] if 'offsets' in v3_result else None)
        
        # Save individual results
        save_with_white_background(shadow_v1_cropped, os.path.join(test_output_dir, "03_shadow_v1.png"), is_float=True)
        save_with_white_background(shadow_v2_cropped, os.path.join(test_output_dir, "04_shadow_v2.png"), is_float=True)
        save_with_white_background(shadow_v3_cropped, os.path.join(test_output_dir, "05_shadow_v3.png"), is_float=True)
        
        # Save V3 debug components
        if 'shadow_core' in v3_result:
            core_cropped = crop_to_original_size(v3_result['shadow_core'], alpha_mask.shape, v3_result['offsets'])
            save_with_white_background(core_cropped, os.path.join(test_output_dir, "06_v3_core.png"), is_float=True)
        
        if 'seed' in v3_result:
            seed_cropped = crop_to_original_size(v3_result['seed'], alpha_mask.shape, v3_result['offsets'])
            save_with_white_background(seed_cropped, os.path.join(test_output_dir, "07_v3_seed.png"), is_float=True)
        
        # Save composites (object + shadow)
        alpha_normalized = alpha_mask.astype(np.float32) / 255.0
        
        composite_v1 = np.maximum(alpha_normalized, shadow_v1_cropped)
        save_with_white_background(composite_v1, os.path.join(test_output_dir, "08_composite_v1.png"), is_float=True)
        
        composite_v2 = np.maximum(alpha_normalized, shadow_v2_cropped)
        save_with_white_background(composite_v2, os.path.join(test_output_dir, "09_composite_v2.png"), is_float=True)
        
        composite_v3 = np.maximum(alpha_normalized, shadow_v3_cropped)
        save_with_white_background(composite_v3, os.path.join(test_output_dir, "10_composite_v3.png"), is_float=True)
        
        # Side-by-side comparison
        comparison = np.hstack([shadow_v1_cropped, shadow_v2_cropped, shadow_v3_cropped])
        save_with_white_background(comparison, os.path.join(test_output_dir, "11_comparison_v1_v2_v3.png"), is_float=True)
        
        all_results[test_name] = {
            'v1': shadow_v1_cropped,
            'v2': shadow_v2_cropped,
            'v3': shadow_v3_cropped,
            'v3_result': v3_result
        }
    
    # Generate analysis report
    report = {
        "test_summary": {
            "test_cases": list(all_results.keys()),
            "test_angle": test_angle,
            "max_distance": max_distance,
            "shadow_vector": [dx, dy]
        },
        "comparison": {}
    }
    
    for test_name, results in all_results.items():
        shadow_v1 = results['v1']
        shadow_v2 = results['v2']
        shadow_v3 = results['v3']
        
        report["comparison"][test_name] = {
            "v1": {
                "max": float(np.max(shadow_v1)),
                "pixels": int(np.sum(shadow_v1 > 0.01)),
                "mean": float(np.mean(shadow_v1[shadow_v1 > 0.01])) if np.sum(shadow_v1 > 0.01) > 0 else 0,
                "shape": list(shadow_v1.shape)
            },
            "v2": {
                "max": float(np.max(shadow_v2)),
                "pixels": int(np.sum(shadow_v2 > 0.01)),
                "mean": float(np.mean(shadow_v2[shadow_v2 > 0.01])) if np.sum(shadow_v2 > 0.01) > 0 else 0,
                "shape": list(shadow_v2.shape)
            },
            "v3": {
                "max": float(np.max(shadow_v3)),
                "pixels": int(np.sum(shadow_v3 > 0.01)),
                "mean": float(np.mean(shadow_v3[shadow_v3 > 0.01])) if np.sum(shadow_v3 > 0.01) > 0 else 0,
                "shape": list(shadow_v3.shape)
            }
        }
    
    report_path = os.path.join(output_dir, "v1_v2_v3_comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nThree-way comparison test report saved to {report_path}")
    print("=== V1 vs V2 vs V3 Comparison Test Complete ===")
    
    return True


if __name__ == "__main__":
    try:
        success = test_v1_v2_v3_comparison()
        if success:
            print("Three-way comparison test passed!")
            sys.exit(0)
        else:
            print("Three-way comparison test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Three-way comparison test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)