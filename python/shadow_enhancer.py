#!/usr/bin/env python3
"""
OpenCV-based shadow enhancement for OKIBAE
Clean SSOT (Single Source of Truth) implementation with preview_fast system
"""

import cv2
import numpy as np
import base64
import json
import sys
from io import BytesIO
from PIL import Image
from math import cos, sin, radians, ceil

"""
STABLE_BASELINE MODE + SSOT SYSTEM:
- Clean, minimal shadow generation with perfect preview/final consistency
- SSOT: Parameters calculated once from original dimensions, scaled for preview
- Feature flags for gradual enhancement rollout
- SSIM consistency validation in development mode

Clean architecture:
1. Single entry point: generate_shadow_layer()
2. SSOT parameter calculation: compute_shadow_params()
3. Preview scaling: scale_for_preview()
4. Unified rendering: render_shadow_stable_from_params()
5. Enhancement framework ready for gradual rollout
"""
STABLE_BASELINE = True
CONSISTENCY_CHECK = True  # Enable SSIM check in development

# Feature flags for gradual enhancement rollout
ENABLE_AO_SPOTS = False
ENABLE_SKELETON_SWEEP = False  
ENABLE_PERSPECTIVE = False


def base64_to_cv2(base64_string):
    """Convert base64 string to OpenCV image"""
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # If image has 3 channels (RGB), add alpha channel
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Add alpha channel (fully opaque)
        alpha = np.full((image_array.shape[0], image_array.shape[1], 1), 255, dtype=image_array.dtype)
        image_array = np.concatenate([image_array, alpha], axis=2)
    
    # Convert RGBA to BGRA for OpenCV (but keep alpha in position 3)
    if image_array.shape[2] == 4:
        # Swap R and B channels, keep G and A in place
        bgra = image_array.copy()
        bgra[:, :, 0] = image_array[:, :, 2]  # B = R
        bgra[:, :, 2] = image_array[:, :, 0]  # R = B
        return bgra
    else:
        return image_array


def cv2_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


def ui_to_math_deg(ui_deg, ui_zero_is_up=True):
    """
    Convert UI angle (0°=up, clockwise) to math angle (0°=right, counterclockwise)
    """
    if ui_zero_is_up:
        return (90 - ui_deg) % 360
    else:
        return ui_deg


def compute_shadow_params(h, w, ui_deg, distance_norm, softness, opacity_long, opacity_ground, ui_zero_is_up=True):
    """
    Compute unified shadow parameters (SSOT - Single Source of Truth)
    
    Args:
        h, w: Image dimensions  
        ui_deg: Shadow direction in UI coordinates (0°=up by default)
        distance_norm: Shadow distance as fraction of shorter side
        softness: Blur softness factor (0-1)
        opacity_long: Long shadow opacity
        opacity_ground: Ground shadow opacity
        ui_zero_is_up: Whether UI zero corresponds to up direction
    
    Returns:
        Dict with canonical shadow parameters (resolution-independent base)
    """
    short = min(h, w)
    
    # Convert relative to pixels
    offset = short * float(distance_norm)
    blur_px = max(1, int(round(short * 0.02 * float(softness))))
    
    # Convert UI angle to math angle
    deg = ui_to_math_deg(ui_deg, ui_zero_is_up)
    rad = radians(deg)
    
    # Compute direction vector (math coordinates)
    lx, ly = cos(rad), sin(rad)          # math座標: 0°=右, 90°=上
    # 画像座標は y↓が正。UIの角度は「影の向き」なので、そのまま影ベクトルに使う
    dx =  offset * lx                    # 右が正 → そのまま
    dy = -offset * ly                    # 上(ly>0)は画像では負にする
    
    # Clipping prevention margin
    margin = int(ceil(abs(dx) + abs(dy) + 3 * blur_px))
    
    # Ground contact shadow erode kernel size
    erode_k = max(1, int(round(short * 0.006)))  # 0.6% of short side
    
    # 画像座標の単位ベクトル（影と光）
    eps = max(1e-6, offset)
    shadow_vec_img = (dx/eps, dy/eps)
    light_vec_img  = (-shadow_vec_img[0], -shadow_vec_img[1])
    
    return dict(
        dx=dx, dy=dy, blur_px=blur_px,
        opacity_long=opacity_long, opacity_ground=opacity_ground,
        light_vec=light_vec_img,
        margin=margin, erode_k=erode_k,
        math_deg=deg, ui_deg=ui_deg, offset=offset,
        distance_norm=distance_norm
    )


def scale_for_preview(params, scale_factor):
    """
    Scale shadow parameters for preview processing while maintaining visual consistency
    
    Args:
        params: Original shadow parameters (canonical SSOT)
        scale_factor: Scale factor (e.g., 0.5 for half size)
    
    Returns:
        Scaled parameters dict with float precision preserved for positions
    """
    scaled = dict(params)  # Copy original params
    
    # Position/distance vectors remain FLOAT (critical for visual consistency)
    for key in ["dx", "dy", "offset"]:
        if key in scaled:
            scaled[key] = scaled[key] * scale_factor
    
    # Kernel/radius/margin → integer (for OpenCV functions)
    for key in ["blur_px", "erode_k", "margin", "ao_radius", "midpoint_radius"]:
        if key in scaled:
            scaled[key] = max(1, int(round(scaled[key] * scale_factor)))
    
    # Pixel area based thresholds → scale_factor^2倍
    area_scale_keys = ["min_shadow_area_px", "max_sample_points"]
    for key in area_scale_keys:
        if key in scaled:
            scaled[key] = max(1, int(round(scaled[key] * scale_factor * scale_factor)))
    
    # Ensure Gaussian kernel is odd (required for cv2.GaussianBlur)
    if "blur_px" in scaled:
        scaled["blur_px"] = int(scaled["blur_px"]) | 1  # Make odd (force int first)
    
    sys.stderr.write(f"scale_for_preview: scale={scale_factor:.3f}, dx {params.get('dx', 0):.2f} -> {scaled.get('dx', 0):.2f}, dy {params.get('dy', 0):.2f} -> {scaled.get('dy', 0):.2f}, blur {params.get('blur_px', 0)} -> {scaled.get('blur_px', 0)}\n")
    
    return scaled


def calc_ssim(img1, img2):
    """
    Calculate SSIM between two grayscale images for consistency checking
    
    Returns:
        SSIM value (0-1, where 1 is perfect match)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        if img1.shape != img2.shape:
            return 0.0
        return ssim(img1, img2, data_range=255)
    except ImportError:
        # Fallback: simple MSE-based similarity
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        max_val = 255.0
        return max(0.0, 1.0 - mse / (max_val ** 2))


def render_shadow_stable_from_params(img_rgba, scaled_params, shared_crop_bounds=None):
    """
    Render shadow using pre-calculated and scaled parameters (SSOT approach)
    With margin-based expansion/crop to prevent shadow clipping
    
    Args:
        img_rgba: RGBA image array (at target processing size)
        scaled_params: Pre-scaled shadow parameters
        shared_crop_bounds: Optional shared crop bounds from full-size calculation
        
    Returns:
        Shadow layer (RGBA) with proper bounds, and crop bounds info
    """
    original_h, original_w = img_rgba.shape[:2]
    
    # Extract scaled parameters
    dx = scaled_params['dx']
    dy = scaled_params['dy']
    blur_px = int(round(scaled_params['blur_px'])) | 1  # Force odd for both preview and final
    op_long = scaled_params['opacity_long']
    op_ground = scaled_params['opacity_ground']
    margin = scaled_params.get('margin', int(abs(dx) + abs(dy) + 3 * blur_px))
    erode_k = scaled_params.get('erode_k')
    
    sys.stderr.write(f"render_shadow_stable_from_params: orig=({original_w}x{original_h}), dx={dx:.2f}, dy={dy:.2f}, blur={blur_px}, erode_k={erode_k}, margin={margin}\n")
    
    # Create expanded canvas to prevent clipping
    expanded_w = original_w + 2 * margin
    expanded_h = original_h + 2 * margin
    
    # Place original image in expanded canvas
    expanded_img = np.zeros((expanded_h, expanded_w, 4), dtype=np.uint8)
    expanded_img[margin:margin+original_h, margin:margin+original_w] = img_rgba
    alpha = expanded_img[:, :, 3]
    
    # --- Long shadow (parallel translation + blur in linear space) ---
    # Convert to linear space for proper blur
    lin = np.power(alpha.astype(np.float32) / 255.0, 2.2)
    
    # Apply translation (no margin adjustment - already positioned in expanded canvas)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    lin_shift = cv2.warpAffine(lin, M, (expanded_w, expanded_h), flags=cv2.INTER_LINEAR, borderValue=0.0)
    
    # Blur in linear space with odd kernel (guaranteed by scale_for_preview)
    lin_blur = cv2.GaussianBlur(lin_shift, (blur_px, blur_px), 0)
    
    # Convert back to gamma space
    long_shadow = np.clip(np.power(lin_blur, 1/2.2) * 255.0, 0, 255).astype(np.uint8)
    
    # --- Ground contact shadow (eroded + small blur + slight offset) ---
    # Use scaled erode_k if available, otherwise calculate
    if erode_k is None:
        short = min(expanded_h, expanded_w)
        erode_k = max(1, int(round(short * 0.006)))  # Fallback calculation
    
    erode_k = max(1, erode_k)  # Ensure at least 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    spot_src = cv2.erode(alpha, kernel)
    
    # Small offset toward shadow direction (no margin adjustment)
    M_spot = np.float32([[1, 0, dx * 0.2], [0, 1, dy * 0.2]])
    spot_shift = cv2.warpAffine(spot_src, M_spot, (expanded_w, expanded_h), flags=cv2.INTER_LINEAR, borderValue=0)
    
    # Small blur with odd kernel
    spot_blur = max(1, (int(round(blur_px/2)) | 1))  # Half of long shadow blur, ensure odd
    spot = cv2.GaussianBlur(spot_shift, (spot_blur, spot_blur), 0)
    
    # --- Combine shadows with opacity (Multiply blend) ---
    long_a = (long_shadow.astype(np.float32) / 255.0 * op_long).clip(0, 1)
    ground_a = (spot.astype(np.float32) / 255.0 * op_ground).clip(0, 1)
    shadow_a = np.clip(long_a + ground_a, 0, 1)
    
    # --- Find union bounds of object + shadow for optimal cropping ---
    if shared_crop_bounds is not None:
        # Use shared crop bounds (scaled from full-size calculation)
        y_min, x_min, y_max, x_max = shared_crop_bounds
        # Ensure bounds are within expanded canvas
        y_min = max(0, min(y_min, expanded_h-1))
        x_min = max(0, min(x_min, expanded_w-1))
        y_max = max(y_min, min(y_max, expanded_h-1))
        x_max = max(x_min, min(x_max, expanded_w-1))
    else:
        # Calculate crop bounds from actual content
        object_mask = (alpha > 0)
        shadow_mask = (shadow_a > 0.01)  # Small threshold for shadow detection
        union_mask = object_mask | shadow_mask
        
        # Find bounding box of union
        coords = np.column_stack(np.where(union_mask))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Ensure we include original object area
            y_min = min(y_min, margin)
            x_min = min(x_min, margin)
            y_max = max(y_max, margin + original_h - 1)
            x_max = max(x_max, margin + original_w - 1)
        else:
            # Fallback: use original bounds
            y_min, x_min = margin, margin
            y_max, x_max = margin + original_h - 1, margin + original_w - 1
    
    # Crop to determined bounds
    cropped_shadow = shadow_a[y_min:y_max+1, x_min:x_max+1]
    crop_h, crop_w = cropped_shadow.shape
    
    # Store crop bounds info (relative to expanded canvas)
    crop_bounds_info = {
        'y_min': y_min, 'x_min': x_min, 'y_max': y_max, 'x_max': x_max,
        'expanded_w': expanded_w, 'expanded_h': expanded_h,
        'margin': margin, 'original_w': original_w, 'original_h': original_h
    }
    
    # Create final RGBA shadow layer
    shadow_layer = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
    shadow_layer[:, :, 0:3] = [20, 20, 20]  # Dark shadow color
    shadow_layer[:, :, 3] = (cropped_shadow * 255).astype(np.uint8)
    
    sys.stderr.write(f"render_shadow_stable_from_params complete: expanded=({expanded_w}x{expanded_h}), cropped=({crop_w}x{crop_h}), erode_k={erode_k}\n")
    
    return shadow_layer, crop_bounds_info


def estimate_light_source(image):
    """
    Estimate light source direction from image brightness patterns and gradients
    Returns: (x, y, intensity) where x,y are normalized coordinates
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    h, w = gray.shape
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Method 1: Brightest region
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    brightest_x = max_loc[0] / w
    brightest_y = max_loc[1] / h
    
    # Method 2: Gradient-based light direction estimation
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate dominant gradient direction
    avg_grad_x = np.mean(grad_x)
    avg_grad_y = np.mean(grad_y)
    
    # Convert gradient to light source position (inverted)
    gradient_x = 0.5 - (avg_grad_x / 255.0 * 0.3)
    gradient_y = 0.5 - (avg_grad_y / 255.0 * 0.3)
    
    # Combine both methods (weighted average)
    confidence_brightest = (max_val - np.mean(gray)) / 255.0
    confidence_gradient = min(1.0, (abs(avg_grad_x) + abs(avg_grad_y)) / 100.0)
    
    total_confidence = confidence_brightest + confidence_gradient
    if total_confidence > 0:
        weight_brightest = confidence_brightest / total_confidence
        weight_gradient = confidence_gradient / total_confidence
        
        light_x = weight_brightest * brightest_x + weight_gradient * gradient_x
        light_y = weight_brightest * brightest_y + weight_gradient * gradient_y
    else:
        # Fallback to center-top (common lighting scenario)
        light_x = 0.5
        light_y = 0.3
    
    # Clamp to reasonable bounds
    light_x = max(0.2, min(0.8, light_x))
    light_y = max(0.2, min(0.8, light_y))
    
    # Calculate intensity based on brightness distribution
    intensity = (max_val - np.mean(gray)) / 255.0
    intensity = max(0.3, min(1.0, intensity))
    
    sys.stderr.write(f"Light estimation: brightest=({brightest_x:.2f},{brightest_y:.2f}), gradient=({gradient_x:.2f},{gradient_y:.2f}), final=({light_x:.2f},{light_y:.2f})\n")
    
    return light_x, light_y, intensity


def generate_unified_shadow(image, shadow_params, quality='preview', shared_crop_bounds=None):
    """
    Generate shadow using unified resolution-independent parameters (SSOT approach)
    """
    if STABLE_BASELINE:
        sys.stderr.write(f"STABLE_BASELINE: Using render_shadow_stable_from_params, quality={quality}\n")
        
        # Special case: only calculate bounding box for shared crop system
        if quality == 'full_bbox_only':
            shadow_layer, crop_bounds_info = render_shadow_stable_from_params(image, shadow_params, None)
            # Return empty shadow but with crop bounds info
            return None, crop_bounds_info
        
        # Use SSOT-based stable shadow generation
        shadow_layer, crop_bounds_info = render_shadow_stable_from_params(image, shadow_params, shared_crop_bounds)
        
        # TODO: Add enhanced features here when enabled
        # if ENABLE_AO_SPOTS:
        #     shadow_layer = add_ao_spots_ssot(shadow_layer, image, shadow_params)
        # if ENABLE_SKELETON_SWEEP:
        #     shadow_layer = add_skeleton_sweep_ssot(shadow_layer, image, shadow_params)
        
        return shadow_layer, crop_bounds_info
    else:
        # Future: Enhanced shadow system (not implemented in clean version)
        raise NotImplementedError("Enhanced shadow system removed. Use STABLE_BASELINE=True")


def generate_shadow_layer(cutout_base64, options=None):
    """
    Main entry point: Generate shadow layer only (no object composition)
    
    CURRENT MODE: STABLE_BASELINE=True (minimal, consistent shadow generation)
    
    Args:
        cutout_base64: Base64 encoded cutout image
        options: Dict with optional parameters:
            - quality: 'preview' | 'final' (default: 'preview')
            - directionDeg: float (shadow direction in UI degrees, 0°=up)
            - distancePx: float (shadow distance override in pixels)
            - preset: 'soft' | 'hard' | 'fabric' | 'none' (default: 'soft')
            - blur: float (blur softness override, 0-1)
            - opacity: float (shadow opacity multiplier)
    
    Returns:
        Dict with shadow layer data:
        - shadowLayerBase64: Base64 RGBA shadow image
        - shadowParams: Direction, distance, blur, opacity values used
        - debug: Processing info including STABLE vs enhanced mode
    """
    try:
        if options is None:
            options = {}
        
        # Convert input image
        image = base64_to_cv2(cutout_base64)
        original_h, original_w = image.shape[:2]
        
        # SSOT approach: calculate parameters once from original dimensions
        quality = options.get('quality', 'preview')
        
        # Log active implementation for clarity
        sys.stderr.write(f"ACTIVE_IMPL=preview_fast_ssot, quality={quality}, STABLE_BASELINE={STABLE_BASELINE}, CONSISTENCY_CHECK={CONSISTENCY_CHECK}\n")
        
        # Apply placement transformations if specified
        if 'placement' in options:
            placement = options['placement']
            scale = placement.get('scale', 1.0)
            rotate = placement.get('rotate', 0.0)
            tx = placement.get('tx', 0.0)
            ty = placement.get('ty', 0.0)
            
            # Apply transformations
            center = (original_w // 2, original_h // 2)
            M_rot = cv2.getRotationMatrix2D(center, rotate, scale)
            M_rot[0, 2] += tx
            M_rot[1, 2] += ty
            image = cv2.warpAffine(image, M_rot, (original_w, original_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        
        # SSOT: Calculate canonical parameters from ORIGINAL dimensions ONCE
        if 'directionDeg' in options:
            # Manual direction (in UI coordinates: 0°=up)
            ui_deg = options['directionDeg']
            intensity = 0.7  # Default intensity
        else:
            # Auto estimation from light source (use original image for accurate estimation)
            light_x, light_y, intensity = estimate_light_source(image)
            # Convert light position to UI angle where light at (0.5, 0.3) = up = 0°
            # Light position relative to center
            light_vector_x = light_x - 0.5
            light_vector_y = light_y - 0.5
            # Convert to UI angle (0°=up, clockwise)
            math_rad = np.arctan2(-light_vector_y, -light_vector_x)  # Shadow direction in math coordinates
            math_deg = np.degrees(math_rad)
            # Convert math angle to UI angle (inverse of ui_to_math_deg)
            ui_deg = (90 - math_deg) % 360
            sys.stderr.write(f"Auto light estimation: light=({light_x:.2f},{light_y:.2f}) -> math_deg={math_deg:.1f}° -> ui_deg={ui_deg:.1f}°\n")
        
        # Compute canonical shadow parameters (SSOT) - ALWAYS from original dimensions
        distance_norm = 0.04  # 4% of shorter side (tunable, default)
        softness = options.get('blur', 0.7)  # Default softness
        
        # Handle distancePx override - modify distance_norm instead of dx/dy
        if 'distancePx' in options:
            short_side = min(original_w, original_h)
            distance_norm = options['distancePx'] / short_side
            sys.stderr.write(f"distancePx override: {options['distancePx']}px -> distance_norm={distance_norm:.3f}\n")
        
        canonical_params = compute_shadow_params(
            original_h, original_w, ui_deg, distance_norm, 
            softness, 0.16, 0.20,  # opacity_long, opacity_ground
            ui_zero_is_up=True
        )
        
        sys.stderr.write(f"CANONICAL params: dx={canonical_params['dx']:.1f}, dy={canonical_params['dy']:.1f}, blur={canonical_params['blur_px']}, ui_deg={ui_deg:.1f}°\n")
        
        # Determine processing approach and scale parameters
        if quality == 'preview':
            # preview_fast: process at reduced resolution with scaled parameters
            target_short = 640  # Configurable preview quality
            original_short = min(original_w, original_h)
            scale_factor = min(1.0, target_short / original_short)
            
            new_w = max(1, int(round(original_w * scale_factor)))
            new_h = max(1, int(round(original_h * scale_factor)))
            
            # Scale image for preview processing
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Scale parameters for preview
            shadow_params = scale_for_preview(canonical_params, scale_factor)
            
            sys.stderr.write(f"preview_fast: scale={scale_factor:.3f}, size=({new_w}x{new_h})\n")
        else:
            # final: use original size and canonical parameters
            new_w, new_h = original_w, original_h
            shadow_params = canonical_params
            scale_factor = 1.0
            sys.stderr.write(f"final: original size ({new_w}x{new_h})\n")
        
        # Apply preset modifications to shadow params
        preset = options.get('preset', 'soft')
        if preset == 'hard':
            shadow_params['blur_px'] = max(1, int(shadow_params['blur_px'] * 0.3))
            shadow_params['opacity_long'] *= 1.2
        elif preset == 'fabric':
            shadow_params['blur_px'] = int(shadow_params['blur_px'] * 1.5)
            shadow_params['opacity_long'] *= 0.8
        elif preset == 'none':
            # Return transparent image
            empty_shadow = np.zeros((new_h, new_w, 4), dtype=np.uint8)
            return {
                "ok": True,
                "shadowLayerBase64": cv2_to_base64(empty_shadow),
                "originalSize": {"width": original_w, "height": original_h},
                "processedSize": {"width": new_w, "height": new_h}
            }
        
        # Apply manual overrides
        if 'opacity' in options:
            shadow_params['opacity_long'] *= options['opacity']
            shadow_params['opacity_ground'] *= options['opacity']
        
        # Generate shadow layer using SSOT parameters with shared crop system
        sys.stderr.write(f"Quality setting: {quality}\n")
        sys.stderr.write(f"Shadow generation at target size: ({new_w}x{new_h})\n")
        
        # Shared crop system: Calculate full-size bounding box first
        shared_crop_bounds = None
        if quality == 'preview':
            # Calculate full-size crop bounds for consistency
            try:
                full_image = base64_to_cv2(cutout_base64)
                _, full_crop_info = generate_unified_shadow(full_image, canonical_params, 'full_bbox_only')
                
                # Scale the crop bounds for preview
                scale_factor_used = scale_factor
                shared_crop_bounds = (
                    int(full_crop_info['y_min'] * scale_factor_used),
                    int(full_crop_info['x_min'] * scale_factor_used),
                    int(full_crop_info['y_max'] * scale_factor_used),
                    int(full_crop_info['x_max'] * scale_factor_used)
                )
                sys.stderr.write(f"Shared crop bounds (scaled): {shared_crop_bounds}\n")
            except Exception as e:
                sys.stderr.write(f"Shared crop calculation failed, using dynamic crop: {e}\n")
                shared_crop_bounds = None
        
        shadow_layer, crop_bounds_info = generate_unified_shadow(image, shadow_params, quality, shared_crop_bounds)
        final_w, final_h = shadow_layer.shape[1], shadow_layer.shape[0]
        
        # SSIM consistency check (development mode only)
        if CONSISTENCY_CHECK and quality == 'preview':
            try:
                # Generate final version for comparison
                final_image = cv2.resize(base64_to_cv2(cutout_base64), (original_w, original_h))
                final_shadow, _ = generate_unified_shadow(final_image, canonical_params, 'final')
                
                # Downscale final to preview size for comparison
                final_downscaled = cv2.resize(final_shadow, (final_w, final_h), interpolation=cv2.INTER_AREA)
                
                # Compare alpha channels
                ssim_score = calc_ssim(shadow_layer[:, :, 3], final_downscaled[:, :, 3])
                
                # Detailed parameter comparison logging
                sys.stderr.write(f"PARAM_COMPARISON: preview_dx={shadow_params['dx']:.2f}, final_dx={canonical_params['dx']:.2f}\n")
                sys.stderr.write(f"PARAM_COMPARISON: preview_dy={shadow_params['dy']:.2f}, final_dy={canonical_params['dy']:.2f}\n")
                sys.stderr.write(f"PARAM_COMPARISON: preview_blur={shadow_params['blur_px']}, final_blur={canonical_params['blur_px']}, scale={scale_factor:.3f}\n")
                sys.stderr.write(f"SSIM consistency check: {ssim_score:.3f}\n")
                
                if ssim_score < 0.985:
                    sys.stderr.write(f"WARNING: Preview/final mismatch! SSIM={ssim_score:.3f} < 0.985\n")
                    sys.stderr.write(f"DIFF_ANALYSIS: Potential causes - dx rounding, dy rounding, even kernel, margin mismatch\n")
                    # Additional debugging info
                    preview_erode = shadow_params.get('erode_k', 'calc')
                    final_erode = canonical_params.get('erode_k', 'calc')
                    preview_margin = shadow_params.get('margin', 'calc')
                    final_margin = canonical_params.get('margin', 'calc')
                    sys.stderr.write(f"DIFF_ANALYSIS: preview_erode_k={preview_erode}, final_erode_k={final_erode}\n")
                    sys.stderr.write(f"DIFF_ANALYSIS: preview_margin={preview_margin}, final_margin={final_margin}\n")
                else:
                    sys.stderr.write(f"✓ Preview/final consistency validated: SSIM={ssim_score:.3f}\n")
                    
            except Exception as e:
                sys.stderr.write(f"SSIM check failed: {str(e)}\n")
        
        # Convert result to base64
        result_base64 = cv2_to_base64(shadow_layer)
        
        # Debug info for verification
        shadow_stats = {
            "non_zero_pixels": int(np.sum(shadow_layer[:, :, 3] > 0)),
            "mean_intensity": float(np.mean(shadow_layer[:, :, 3][shadow_layer[:, :, 3] > 0])) if np.sum(shadow_layer[:, :, 3] > 0) > 0 else 0
        }
        
        return {
            "ok": True,
            "shadowLayerBase64": result_base64,
            "originalSize": {"width": original_w, "height": original_h},
            "processedSize": {"width": final_w, "height": final_h},
            "shadowParams": {
                "direction_deg": float(shadow_params['ui_deg']),
                "distance_norm": float(distance_norm),
                "dx": float(shadow_params['dx']),
                "dy": float(shadow_params['dy']),
                "blur_px": int(shadow_params['blur_px']),
                "opacity_long": float(shadow_params['opacity_long']),
                "opacity_ground": float(shadow_params['opacity_ground'])
            },
            "debug": {
                "quality": quality,
                "algorithm": "preview_fast_ssot",
                "shadow_stats": shadow_stats,
                "processing_size": {"w": new_w, "h": new_h},
                "output_size": {"w": final_w, "h": final_h},
                "scale_factor": scale_factor,
                "enhancement_flags": {
                    "ao_spots": ENABLE_AO_SPOTS,
                    "skeleton_sweep": ENABLE_SKELETON_SWEEP,
                    "perspective": ENABLE_PERSPECTIVE
                }
            }
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


# === FUTURE ENHANCEMENT FRAMEWORK ===
# Enhanced features will be added here with SSOT compliance

# Future SSOT-compliant enhancement functions:
# def generate_ao_spots_ssot(img_rgba, scaled_params):
#     """Generate AO spots with SSOT parameter compliance"""
#     pass

# def generate_skeleton_sweep_ssot(img_rgba, scaled_params):
#     """Generate skeleton-sweep enhanced shadows with SSOT compliance"""
#     pass

# def analyze_perspective_ssot(img_rgba, scaled_params):
#     """Analyze object perspective and adjust shadow parameters"""
#     pass


if __name__ == "__main__":
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    
    # Always use unified generate_shadow_layer (SSOT approach)
    result = generate_shadow_layer(
        input_data.get("cutoutImageBase64"),
        input_data.get("options", {})
    )
    
    # Output result
    print(json.dumps(result))