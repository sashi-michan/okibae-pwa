#!/usr/bin/env python3
"""
Debug perspective shadow zero output issue
Step-by-step analysis to identify the root cause
"""

import cv2
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perspective_shadow.v1_basic import create_perspective_cast_shadow_v1


def debug_perspective_shadow_step_by_step():
    """Debug each step of perspective shadow generation"""
    print("=== Debugging Perspective Shadow Generation ===")
    
    # Test 1: Simple synthetic mask (known working case)
    print("\n1. Testing with simple synthetic mask...")
    simple_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(simple_mask, (80, 120), (120, 180), 255, -1)
    
    max_distance = 50
    shadow_vector = (max_distance, 0)  # Right direction
    
    shadow_simple = create_perspective_cast_shadow_v1(
        simple_mask, None, shadow_vector, max_distance
    )
    print(f"   Simple mask result: max={np.max(shadow_simple):.3f}, pixels={np.sum(shadow_simple > 0.01)}")
    
    if np.max(shadow_simple) > 0:
        print("   ✓ Simple case works - algorithm is functional")
    else:
        print("   ✗ Simple case fails - fundamental algorithm issue")
        return
    
    # Test 2: Load real image and analyze step by step
    print("\n2. Loading real image for analysis...")
    sample_path = "../sample/test1.png"  # Orca ring
    
    if not os.path.exists(sample_path):
        print(f"   Sample not found: {sample_path}")
        return
    
    image = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("   Failed to load image")
        return
    
    alpha_mask = image[:, :, 3]
    h, w = alpha_mask.shape
    print(f"   Loaded real image: {w}x{h}, non-zero pixels: {np.sum(alpha_mask > 0)}")
    
    # Test 3: Scale down real image to manageable size
    print("\n3. Testing with scaled-down real image...")
    scale_factor = 200 / min(h, w)  # Scale to 200px on shorter side
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    scaled_mask = cv2.resize(alpha_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled_max_distance = max_distance * scale_factor
    
    print(f"   Scaled image: {new_w}x{new_h}, max_distance={scaled_max_distance:.1f}")
    
    shadow_scaled = create_perspective_cast_shadow_v1(
        scaled_mask, None, (scaled_max_distance, 0), scaled_max_distance
    )
    print(f"   Scaled result: max={np.max(shadow_scaled):.3f}, pixels={np.sum(shadow_scaled > 0.01)}")
    
    # Test 4: Full size real image
    print("\n4. Testing with full-size real image...")
    full_max_distance = min(h, w) * 0.1  # 10% of smaller dimension
    
    shadow_full = create_perspective_cast_shadow_v1(
        alpha_mask, None, (full_max_distance, 0), full_max_distance
    )
    print(f"   Full-size result: max={np.max(shadow_full):.3f}, pixels={np.sum(shadow_full > 0.01)}")
    
    # Test 5: Debug internal steps manually
    print("\n5. Manual step-by-step debugging...")
    
    # Use scaled image for detailed debugging
    test_mask = scaled_mask
    test_h, test_w = test_mask.shape
    
    # Step 5a: Binary mask
    bin_mask = (test_mask > 0).astype(np.uint8) * 255
    print(f"   Binary mask: {np.sum(bin_mask > 0)} pixels")
    
    # Step 5b: Distance transform
    try:
        dist_transform = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 3)
        print(f"   Distance transform: max={np.max(dist_transform):.3f}, mean={np.mean(dist_transform[dist_transform > 0]):.3f}")
    except Exception as e:
        print(f"   Distance transform failed: {e}")
        return
    
    # Step 5c: Height map
    if np.max(dist_transform) > 0:
        height_map = dist_transform / np.max(dist_transform)
        height_map = height_map * (bin_mask > 0).astype(np.float32)
        print(f"   Height map: max={np.max(height_map):.3f}, non_zero={np.sum(height_map > 0)}")
    else:
        print("   Height map failed - distance transform is zero")
        return
    
    # Step 5d: Coordinate grids
    try:
        y_indices, x_indices = np.mgrid[0:test_h, 0:test_w].astype(np.float32)
        print(f"   Coordinate grids created: {y_indices.shape}")
    except Exception as e:
        print(f"   Coordinate grid creation failed: {e}")
        return
    
    # Step 5e: Offset calculation
    test_distance = scaled_max_distance
    offset_x = test_distance * height_map  # Simplified - no contact mask
    offset_y = np.zeros_like(offset_x)     # Only horizontal shadow for testing
    
    print(f"   Offsets: offset_x max={np.max(offset_x):.3f}, mean={np.mean(offset_x[offset_x > 0]):.3f}")
    
    # Step 5f: Source coordinates
    src_x = x_indices - offset_x
    src_y = y_indices - offset_y
    
    print(f"   Source coords: src_x range=[{np.min(src_x):.1f}, {np.max(src_x):.1f}], src_y range=[{np.min(src_y):.1f}, {np.max(src_y):.1f}]")
    
    # Check if source coordinates are out of bounds
    valid_x = (src_x >= 0) & (src_x < test_w)
    valid_y = (src_y >= 0) & (src_y < test_h)
    valid_coords = valid_x & valid_y
    
    print(f"   Valid coordinates: {np.sum(valid_coords)} / {valid_coords.size} ({np.sum(valid_coords)/valid_coords.size*100:.1f}%)")
    
    if np.sum(valid_coords) == 0:
        print("   ✗ PROBLEM FOUND: All source coordinates are out of bounds!")
        print("      This means the offset is too large or coordinate calculation is wrong")
        return
    
    # Step 5g: Remapping
    alpha_float = test_mask.astype(np.float32) / 255.0
    
    try:
        shadow_result = cv2.remap(
            alpha_float,
            src_x.astype(np.float32),
            src_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )
        
        print(f"   Remap result: max={np.max(shadow_result):.3f}, non_zero={np.sum(shadow_result > 0.01)}")
        
        if np.max(shadow_result) > 0:
            print("   ✓ Remapping successful")
        else:
            print("   ✗ Remapping produced zero result")
            
            # Additional debugging for remap
            print("      Debug: Checking input alpha_float...")
            print(f"         alpha_float: max={np.max(alpha_float):.3f}, non_zero={np.sum(alpha_float > 0.01)}")
            
    except Exception as e:
        print(f"   Remap failed: {e}")
        return
    
    print("\n=== Debug Summary ===")
    print("Simple synthetic case: Works")
    print("Scaled real image: ", "Works" if np.max(shadow_scaled) > 0 else "Fails")
    print("Full-size real image: ", "Works" if np.max(shadow_full) > 0 else "Fails")
    
    if np.max(shadow_full) == 0 and np.max(shadow_scaled) > 0:
        print("CONCLUSION: Issue is related to image size - large images cause problems")
    elif np.max(shadow_scaled) == 0:
        print("CONCLUSION: Issue is with real image characteristics or algorithm parameters")
    else:
        print("CONCLUSION: Algorithm works correctly")


if __name__ == "__main__":
    debug_perspective_shadow_step_by_step()