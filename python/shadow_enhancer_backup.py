#!/usr/bin/env python3
"""
OpenCV-based shadow enhancement for OKIBAE
Implements light source estimation, perspective analysis, and natural shadow generation
"""

import cv2
import numpy as np
import base64
import json
import sys
from io import BytesIO
from PIL import Image
from math import cos, sin, radians

"""
STABLE_BASELINE MODE + SSOT SYSTEM:
- When True: Use minimal, consistent shadow generation (render_shadow_stable)
- When False: Use complex enhanced shadow system with AO, skeleton sweep, etc.

SSOT (Single Source of Truth) approach:
1. Parameters calculated ONCE from original image dimensions
2. Scaled versions derived from the canonical parameters
3. Identical pipeline for preview_fast and final
4. SSIM consistency check in development mode

Current issues being fixed:
1. Shadow direction mismatch with UI angle dial
2. Preview vs final version differences  
3. Shadow clipping at image edges
4. Performance optimization with preview_fast
"""
STABLE_BASELINE = True
CONSISTENCY_CHECK = True  # Enable SSIM check in development

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
    
    Args:
        ui_deg: Angle in UI coordinates (0°=up)
        ui_zero_is_up: Whether UI zero corresponds to up direction
    
    Returns:
        Angle in math coordinates (0°=right)
    """
    if ui_zero_is_up:
        # UI: 0°=up, 90°=right, 180°=down, 270°=left (clockwise)
        # Math: 0°=right, 90°=up, 180°=left, 270°=down (counterclockwise)
        return (90 - ui_deg) % 360
    else:
        return ui_deg

def compute_shadow_params(h, w, ui_deg, distance_norm, softness, opacity_long, opacity_ground, ui_zero_is_up=True):
    """
    Compute unified shadow parameters (resolution-independent)
    
    Args:
        h, w: Image dimensions
        ui_deg: Shadow direction in UI coordinates (0°=up by default)
        distance_norm: Shadow distance as fraction of shorter side (e.g., 0.04 = 4%)
        softness: Blur softness factor (0-1)
        opacity_long: Long shadow opacity
        opacity_ground: Ground shadow opacity
        ui_zero_is_up: Whether UI zero corresponds to up direction
    
    Returns:
        Dict with unified shadow parameters
    """
    from math import ceil
    
    short = min(h, w)
    
    # Convert relative to pixels (easy to tune initial values)
    offset = short * float(distance_norm)          # e.g., distance_norm=0.04 for 4% of short side
    blur_px = max(1, int(round(short * 0.02 * float(softness))))  # e.g., softness=1 for 2% of short side
    
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
        light_vec=light_vec_img,  # ←以後は"画像座標の光ベクトル"として使う
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
        Scaled parameters dict
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


def render_shadow_stable_from_params(img_rgba, scaled_params):
    """
    Render shadow using pre-calculated and scaled parameters (SSOT approach)
    With margin-based expansion/crop to prevent shadow clipping
    
    Args:
        img_rgba: RGBA image array (at target processing size)
        scaled_params: Pre-scaled shadow parameters
        
    Returns:
        Shadow layer (RGBA) with proper bounds
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
    
    # Apply translation (adjusted for expanded canvas)
    M = np.float32([[1, 0, dx + margin], [0, 1, dy + margin]])
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
    
    # Small offset toward shadow direction
    M_spot = np.float32([[1, 0, (dx * 0.2) + margin], [0, 1, (dy * 0.2) + margin]])
    spot_shift = cv2.warpAffine(spot_src, M_spot, (expanded_w, expanded_h), flags=cv2.INTER_LINEAR, borderValue=0)
    
    # Small blur with odd kernel
    spot_blur = max(1, (int(round(blur_px/2)) | 1))  # Half of long shadow blur, ensure odd
    spot = cv2.GaussianBlur(spot_shift, (spot_blur, spot_blur), 0)
    
    # --- Combine shadows with opacity (Multiply blend) ---
    long_a = (long_shadow.astype(np.float32) / 255.0 * op_long).clip(0, 1)
    ground_a = (spot.astype(np.float32) / 255.0 * op_ground).clip(0, 1)
    shadow_a = np.clip(long_a + ground_a, 0, 1)
    
    # --- Find union bounds of object + shadow for optimal cropping ---
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
        
        # Crop to union bounds
        cropped_shadow = shadow_a[y_min:y_max+1, x_min:x_max+1]
        crop_h, crop_w = cropped_shadow.shape
    else:
        # Fallback: use original bounds
        y_min, x_min = margin, margin
        cropped_shadow = shadow_a[margin:margin+original_h, margin:margin+original_w]
        crop_h, crop_w = original_h, original_w
    
    # Create final RGBA shadow layer
    shadow_layer = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
    shadow_layer[:, :, 0:3] = [20, 20, 20]  # Dark shadow color
    shadow_layer[:, :, 3] = (cropped_shadow * 255).astype(np.uint8)
    
    sys.stderr.write(f"render_shadow_stable_from_params complete: expanded=({expanded_w}x{expanded_h}), cropped=({crop_w}x{crop_h}), erode_k={erode_k}\n")
    
    return shadow_layer


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
    
    # Method 1: Brightest region (existing approach)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    brightest_x = max_loc[0] / w
    brightest_y = max_loc[1] / h
    
    # Method 2: Gradient-based light direction estimation
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate dominant gradient direction (points toward darker areas)
    # Light source is opposite to dominant shadow direction
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
    
    # Clamp to reasonable bounds (prevent extreme shadow positions)
    light_x = max(0.2, min(0.8, light_x))  # More conservative range
    light_y = max(0.2, min(0.8, light_y))
    
    # Calculate intensity based on brightness distribution
    intensity = (max_val - np.mean(gray)) / 255.0
    intensity = max(0.3, min(1.0, intensity))
    
    sys.stderr.write(f"Light estimation: brightest=({brightest_x:.2f},{brightest_y:.2f}), gradient=({gradient_x:.2f},{gradient_y:.2f}), final=({light_x:.2f},{light_y:.2f})\n")
    
    return light_x, light_y, intensity


def estimate_vanishing_point(lines, image_shape):
    """
    Estimate vanishing point from detected lines using line intersection analysis
    Returns perspective information with confidence score
    """
    if lines is None or len(lines) < 2:
        return None
    
    h, w = image_shape
    intersections = []
    
    # Calculate intersections between line pairs
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]
            
            # Convert to line equations (ax + by + c = 0)
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            # Calculate line intersection
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:  # Lines are parallel
                continue
                
            px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
            
            # Filter reasonable intersection points
            if -w*2 < px < w*3 and -h*2 < py < h*3:
                intersections.append((px, py))
    
    if not intersections:
        return None
    
    # Cluster intersections to find dominant vanishing points
    intersections = np.array(intersections)
    
    # Simple clustering: find the most dense region
    best_vp = None
    best_score = 0
    
    for point in intersections:
        # Count nearby points (within reasonable distance)
        distances = np.linalg.norm(intersections - point, axis=1)
        nearby_count = np.sum(distances < min(w, h) * 0.5)
        
        if nearby_count > best_score:
            best_score = nearby_count
            best_vp = point
    
    if best_vp is None:
        return None
    
    vp_x, vp_y = best_vp
    
    # Calculate confidence based on clustering density
    confidence = min(1.0, best_score / max(3, len(intersections) * 0.3))
    
    # Calculate ground plane angle from vanishing point
    # VP above center suggests upward perspective, below suggests downward
    center_y = h / 2
    vp_relative_y = (center_y - vp_y) / h
    
    # Convert to shadow angle (0° = horizontal, 90° = vertical)
    # Objects with upward perspective have shadows that stretch more horizontally
    if vp_y < center_y:  # VP above center - upward perspective
        shadow_angle = 30 + vp_relative_y * 30  # 30-60°
    else:  # VP below center - downward perspective  
        shadow_angle = 45 + vp_relative_y * 15  # 30-60°
    
    shadow_angle = max(15, min(75, shadow_angle))  # Clamp to reasonable range
    
    sys.stderr.write(f"Vanishing point estimated at ({vp_x:.1f}, {vp_y:.1f}), confidence: {confidence:.2f}\n")
    
    return {
        'vanishing_point': (vp_x, vp_y),
        'confidence': confidence,
        'shadow_angle': shadow_angle,
        'ground_plane_normal': vp_relative_y
    }


def analyze_object_perspective(image):
    """
    Analyze object perspective using line detection and vanishing point estimation
    Returns: shadow angle and softness parameters with perspective information
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    
    # Enhanced edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                           minLineLength=30, maxLineGap=10)
    
    perspective_info = estimate_vanishing_point(lines, gray.shape) if lines is not None else None
    
    # Find contours for basic shape analysis
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Use perspective info if available, otherwise default
        if perspective_info:
            return perspective_info['shadow_angle'], 0.6
        return 45, 0.6
    
    # Get the largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate shadow angle using perspective information
    if perspective_info and perspective_info['confidence'] > 0.3:
        # Use perspective-informed shadow angle
        shadow_angle = perspective_info['shadow_angle']
        sys.stderr.write(f"Using perspective-based shadow angle: {shadow_angle:.1f}°\n")
    else:
        # Fallback to shape-based estimation
        aspect_ratio = w / h
        if aspect_ratio > 1.5:  # Wide object
            shadow_angle = 30
        elif aspect_ratio < 0.7:  # Tall object  
            shadow_angle = 60
        else:  # Square-ish object
            shadow_angle = 45
        sys.stderr.write(f"Using shape-based shadow angle: {shadow_angle:.1f}°\n")
    
    # Estimate softness based on object complexity
    perimeter = cv2.arcLength(largest_contour, True)
    area = cv2.contourArea(largest_contour)
    if area > 0:
        complexity = perimeter * perimeter / area
        softness = min(0.9, max(0.3, complexity / 50.0))
    else:
        softness = 0.6
    
    return shadow_angle, softness


def generate_unified_shadow(image, shadow_params, quality='preview'):
    """
    Generate shadow using unified resolution-independent parameters (SSOT approach)
    """
    # Check if we should use stable baseline
    if STABLE_BASELINE:
        sys.stderr.write(f"STABLE_BASELINE: Using render_shadow_stable_from_params, quality={quality}\n")
        
        # Use SSOT-based stable shadow generation (returns RGBA shadow layer directly)
        shadow_layer = render_shadow_stable_from_params(image, shadow_params)
        
        return shadow_layer
    
    # Original complex shadow generation (when STABLE_BASELINE=False)
    h, w = image.shape[:2]
    alpha_channel = image[:, :, 3]
    
    # Extract parameters
    dx, dy = shadow_params['dx'], shadow_params['dy']
    blur_px = shadow_params['blur_px']
    opacity_long = shadow_params['opacity_long']
    opacity_ground = shadow_params['opacity_ground']
    
    sys.stderr.write(f"Unified shadow generation: dx={dx:.1f}, dy={dy:.1f}, blur={blur_px}, long_op={opacity_long:.2f}\n")
    
    # Create binary mask
    binary_mask = (alpha_channel > 127).astype(np.uint8) * 255
    
    # STEP 1: Base long shadow (guaranteed to work)
    # Simple parallel translation + blur
    M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # Add margin to prevent clipping
    margin = int(abs(dx) + abs(dy) + 3 * blur_px)
    expanded_w = w + 2 * margin
    expanded_h = h + 2 * margin
    
    # Create expanded canvas
    expanded_mask = np.zeros((expanded_h, expanded_w), dtype=np.uint8)
    expanded_mask[margin:margin+h, margin:margin+w] = binary_mask
    
    # Apply translation
    M_translate[0, 2] += margin  # Adjust for margin
    M_translate[1, 2] += margin
    base_shadow = cv2.warpAffine(expanded_mask, M_translate, (expanded_w, expanded_h), 
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Apply blur in linear color space for better quality
    if blur_px > 1:
        kernel_size = blur_px * 2 + 1  # Ensure odd
        # Convert to linear space (gamma correction)
        shadow_linear = (base_shadow.astype(np.float32) / 255.0) ** 2.2
        # Blur in linear space
        shadow_blurred = cv2.GaussianBlur(shadow_linear, (kernel_size, kernel_size), blur_px/3)
        # Convert back to gamma space
        base_shadow = ((shadow_blurred ** (1.0/2.2)) * 255.0).astype(np.uint8)
    
    # Apply opacity
    base_shadow = (base_shadow.astype(np.float32) * opacity_long).astype(np.uint8)
    
    # STEP 2: Enhanced long shadow (optional, additive)
    enhanced_shadow = np.zeros_like(base_shadow, dtype=np.uint8)
    enhanced_nonzero_ratio = 0.0
    
    try:
        # Use existing enhanced shadow generation
        alpha_expanded = np.zeros((expanded_h, expanded_w), dtype=np.uint8)
        alpha_expanded[margin:margin+h, margin:margin+w] = alpha_channel
        
        # Convert unified params back to legacy format for enhanced generation
        light_vec = shadow_params['light_vec']  # 画像座標: x→右+, y→下+
        light_x = 0.5 + light_vec[0] * 0.3
        light_y = 0.5 + light_vec[1] * 0.3      # ←yはそのまま（下が+）
        
        # Call existing enhanced generation with expanded canvas
        height_map, _ = compute_pseudo_height_map(alpha_expanded)
        enhanced_raw = generate_skeleton_sweep_shadow(
            alpha_expanded, light_x, light_y, beta_deg=60, height_map=height_map, scale_factor=1.0
        )
        
        # Check if enhanced shadow is significant
        enhanced_nonzero = np.sum(enhanced_raw > 0)
        total_pixels = expanded_h * expanded_w
        enhanced_nonzero_ratio = enhanced_nonzero / total_pixels
        
        if enhanced_nonzero_ratio > 0.001:  # At least 0.1% coverage
            enhanced_shadow = enhanced_raw
            sys.stderr.write(f"Enhanced shadow added: {enhanced_nonzero_ratio:.3f} coverage\n")
        else:
            sys.stderr.write(f"Enhanced shadow skipped: {enhanced_nonzero_ratio:.3f} coverage too low\n")
            
    except Exception as e:
        sys.stderr.write(f"Enhanced shadow failed, using base only: {str(e)}\n")
    
    # STEP 3: Combine base + enhanced (additive)
    combined_shadow = np.maximum(base_shadow, enhanced_shadow)
    
    # STEP 4: Add ground contact shadow
    ground_shadow = generate_simple_ground_shadow(alpha_expanded, dx/2, dy/2, opacity_ground)
    combined_shadow = np.maximum(combined_shadow, ground_shadow)
    
    # STEP 5: Crop back to original size + object bounds
    # Find the union of object and shadow bounds
    object_mask_expanded = np.zeros((expanded_h, expanded_w), dtype=np.uint8)
    object_mask_expanded[margin:margin+h, margin:margin+w] = binary_mask
    
    union_mask = np.maximum(object_mask_expanded, combined_shadow > 0)
    coords = np.column_stack(np.where(union_mask > 0))
    
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Ensure we include the original object area
        y_min = min(y_min, margin)
        x_min = min(x_min, margin)
        y_max = max(y_max, margin + h - 1)
        x_max = max(x_max, margin + w - 1)
        
        # Crop to bounds
        final_shadow = combined_shadow[y_min:y_max+1, x_min:x_max+1]
        
        # Adjust coordinates to match original image position
        offset_y = y_min - margin
        offset_x = x_min - margin
        
        # Create final RGBA shadow layer matching original dimensions
        shadow_layer = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Place cropped shadow at correct position
        src_h, src_w = final_shadow.shape
        dst_y_start = max(0, -offset_y)
        dst_y_end = min(h, dst_y_start + src_h)
        dst_x_start = max(0, -offset_x)
        dst_x_end = min(w, dst_x_start + src_w)
        
        src_y_start = max(0, offset_y)
        src_y_end = src_y_start + (dst_y_end - dst_y_start)
        src_x_start = max(0, offset_x)
        src_x_end = src_x_start + (dst_x_end - dst_x_start)
        
        if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
            shadow_layer[dst_y_start:dst_y_end, dst_x_start:dst_x_end, 3] = \
                final_shadow[src_y_start:src_y_end, src_x_start:src_x_end]
    else:
        # Fallback: empty shadow
        shadow_layer = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Set shadow color (dark gray)
    shadow_layer[:, :, 0:3] = [20, 20, 20]
    
    sys.stderr.write(f"Shadow generation complete: enhanced_ratio={enhanced_nonzero_ratio:.3f}\n")
    
    return shadow_layer


def generate_simple_ground_shadow(alpha_channel, dx, dy, opacity):
    """
    Generate simple ground contact shadow using Underlap method
    """
    h, w = alpha_channel.shape
    
    # Simple shift for ground contact
    M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # Normalize and shift
    mask_norm = alpha_channel.astype(np.float32) / 255.0
    mask_shifted = cv2.warpAffine(mask_norm, M_shift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Underlap: max(0, shifted - original)
    ground_shadow = np.maximum(0, mask_shifted - mask_norm)
    
    # Apply slight blur and opacity
    if ground_shadow.max() > 0:
        ground_shadow = cv2.GaussianBlur(ground_shadow, (5, 5), 1.0)
        ground_shadow = (ground_shadow * opacity * 255).astype(np.uint8)
    else:
        ground_shadow = np.zeros((h, w), dtype=np.uint8)
    
    return ground_shadow






# === FUTURE ENHANCEMENT FRAMEWORK ===
# Enhanced features will be added here with SSOT compliance

# Feature flags for gradual enhancement rollout
ENABLE_AO_SPOTS = False
ENABLE_SKELETON_SWEEP = False  
ENABLE_PERSPECTIVE = False

# Future SSOT-compliant enhancement functions will be added here:
# def generate_ao_spots_ssot(img_rgba, scaled_params): pass
# def generate_skeleton_sweep_ssot(img_rgba, scaled_params): pass
# def analyze_perspective_ssot(img_rgba, scaled_params): pass


def generate_skeleton_sweep_shadow(alpha_channel, light_x, light_y, beta_deg=60, height_map=None, ao_spots_mask=None, scale_factor=1.0):
    """
    Generate skeleton-sweep based long shadow (NEW ChatGPT approach)
    Implements "骨格スイープ" method with geodesic root zones
    
    Args:
        alpha_channel: Object mask
        light_x, light_y: Light source position (0-1 normalized)
        beta_deg: Light elevation angle in degrees
        height_map: Precomputed height map (optional)
        ao_spots_mask: AO spots for root zone expansion
    
    Returns:
        Skeleton-sweep enhanced long shadow
    """
    h, w = alpha_channel.shape
    
    try:
        # Compute height map if not provided
        if height_map is None:
            height_map, binary_mask = compute_pseudo_height_map(alpha_channel)
        else:
            binary_mask = (alpha_channel > 127).astype(np.uint8) * 255
        
        # Calculate shadow direction vector s (opposite to light direction ℓ)
        light_vector_x = light_x - 0.5
        light_vector_y = light_y - 0.5
        magnitude = np.sqrt(light_vector_x**2 + light_vector_y**2)
        
        if magnitude > 0:
            s_x = -light_vector_x / magnitude  # Shadow direction s = -ℓ
            s_y = -light_vector_y / magnitude
        else:
            s_x, s_y = -1, 1
        
        # Get rim_outer (外周リム)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(binary_mask, kernel_erode, iterations=1)
        rim_outer = binary_mask - eroded
        
        # Simple root zone from rim_outer (simplified version for now)
        # TODO: Implement proper geodesic distance from AO spots
        r_root = max(1, min(2, int(min(w, h) * 0.005)))  # 1-2px as recommended
        kernel_root = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_root*2+1, r_root*2+1))
        root_zone = cv2.dilate(rim_outer, kernel_root, iterations=1)
        
        # Calculate projection parameters
        beta_rad = np.radians(beta_deg)
        cot_beta = max(0.2, min(3.0, 1.0 / np.tan(beta_rad)))  # Safe range
        
        # Enhanced skeleton-sweep: Generate shadows from contact surface
        # Instead of using interior skeleton points, use contact-surface traced points
        long_shadow_sum = np.zeros((h, w), dtype=np.float32)
        
        # ROBUST skeleton point collection with extensive debugging
        skeleton_points = []
        distance_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        sys.stderr.write(f"Skeleton generation debug: image_size=({h},{w}), binary_mask_pixels={np.sum(binary_mask > 0)}, rim_pixels={np.sum(rim_outer > 0)}\n")
        
        # Find object bounding box for ground detection
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x_obj, y_obj, w_obj, h_obj = cv2.boundingRect(largest_contour)
            ground_threshold = y_obj + h_obj * 0.7  # Bottom 30% region
            
            # Create ground contact mask for enhanced contact points
            ground_contact_mask = np.zeros((h, w), dtype=np.uint8)
            ground_contact_mask[int(ground_threshold):, :] = 255
            rim_ground = cv2.bitwise_and(rim_outer, ground_contact_mask)
            
            sys.stderr.write(f"Ground detection: bbox=({x_obj},{y_obj},{w_obj},{h_obj}), threshold={ground_threshold:.1f}, ground_rim_pixels={np.sum(rim_ground > 0)}\n")
            
            # METHOD 1: Enhanced contact surface points (with 2x weight)
            contact_pixels = np.where(rim_ground > 0)
            contact_candidates = 0
            for i in range(0, len(contact_pixels[0]), 1):  # Check every pixel for contact
                y_pos = contact_pixels[0][i]
                x_pos = contact_pixels[1][i]
                
                if 0 < x_pos < w-1 and 0 < y_pos < h-1:  # Bounds check
                    height_val = max(1.5, height_map[y_pos, x_pos] * 0.8)  # Slightly higher for visibility
                    # Use FIXED absolute projection distance for consistency
                    # Base distance independent of image resolution
                    base_proj_distance = height_val * cot_beta * 25.0  # Fixed base distance (increased)
                    # No scaling - use absolute pixel distance
                    proj_distance = base_proj_distance
                    contact_candidates += 1
                    
                    if proj_distance > 0.2:  # Lower threshold
                        # Mark as contact surface point with higher weight
                        skeleton_points.append((x_pos, y_pos, height_val, proj_distance, 'contact'))
            
            contact_count = len([p for p in skeleton_points if p[4] == 'contact'])
            sys.stderr.write(f"Contact surface: {contact_candidates} candidates -> {contact_count} valid points\n")
        
        # METHOD 2: Interior skeleton points (for volume shadow, lower weight)
        # Sample interior skeleton points (every 2-3 pixels for better coverage)
        interior_candidates = 0
        for y in range(1, h-1, 2):  # Denser sampling
            for x in range(1, w-1, 2):
                if binary_mask[y, x] > 0 and distance_transform[y, x] > 0.8:  # Looser interior check
                    height_val = height_map[y, x]
                    # Use FIXED absolute projection distance for consistency
                    base_proj_distance = height_val * cot_beta * 20.0  # Fixed base distance (increased)
                    # No scaling - use absolute pixel distance
                    proj_distance = base_proj_distance
                    interior_candidates += 1
                    
                    if proj_distance > 0.3:
                        # Mark as interior point with normal weight
                        skeleton_points.append((x, y, height_val, proj_distance, 'interior'))
        
        interior_count = len([p for p in skeleton_points if p[4] == 'interior'])
        sys.stderr.write(f"Interior points: {interior_candidates} candidates -> {interior_count} valid points\n")
        
        # METHOD 3: Fallback rim sampling (always available)
        rim_pixels = np.where(rim_outer > 0)
        fallback_candidates = 0
        for i in range(0, len(rim_pixels[0]), 1):  # Check every rim pixel
            y_pos = rim_pixels[0][i]
            x_pos = rim_pixels[1][i]
            if 0 < x_pos < w-1 and 0 < y_pos < h-1:
                height_val = max(2.0, height_map[y_pos, x_pos])  # Ensure minimum height
                # Use FIXED absolute projection distance for consistency
                base_proj_distance = height_val * cot_beta * 15.0  # Fixed base distance (increased)
                # No scaling - use absolute pixel distance
                proj_distance = base_proj_distance
                fallback_candidates += 1
                
                if proj_distance > 0.2:  # Lower threshold for fallback
                    skeleton_points.append((x_pos, y_pos, height_val, proj_distance, 'fallback'))
        
        fallback_count = len([p for p in skeleton_points if p[4] == 'fallback'])
        sys.stderr.write(f"Fallback rim: {fallback_candidates} candidates -> {fallback_count} valid points\n")
        
        # Count by type for comprehensive debugging
        type_counts = {'contact': 0, 'interior': 0, 'fallback': 0, 'other': 0}
        for point in skeleton_points:
            if len(point) > 4:
                point_type = point[4]
                type_counts[point_type] = type_counts.get(point_type, 0) + 1
            else:
                type_counts['other'] += 1
        
        sys.stderr.write(f"Final skeleton summary: {type_counts['contact']} contact + {type_counts['interior']} interior + {type_counts['fallback']} fallback + {type_counts['other']} other = {len(skeleton_points)} total\n")
        
        # Emergency fallback if still no points
        if len(skeleton_points) == 0:
            sys.stderr.write("CRITICAL: No skeleton points found, creating emergency fallback\n")
            # Create at least one point at object center
            if np.sum(binary_mask > 0) > 0:
                center_y, center_x = np.where(binary_mask > 0)
                if len(center_y) > 0:
                    cy = int(np.mean(center_y))
                    cx = int(np.mean(center_x))
                    emergency_height = 3.0
                    emergency_proj = emergency_height * cot_beta * 5.0
                    skeleton_points.append((cx, cy, emergency_height, emergency_proj, 'emergency'))
                    sys.stderr.write(f"Emergency point created at ({cx}, {cy}) with projection {emergency_proj:.1f}\n")
        
        # For each contact-surface skeleton point, generate its shadow contribution
        sigma_low = 2.5   # Sharper for contact shadows
        sigma_high = 7.0  # More defined far shadows
        
        for i, point_data in enumerate(skeleton_points[:100]):  # Increased limit
            if len(point_data) == 5:
                sx, sy, height_val, proj_dist, point_type = point_data
            else:
                sx, sy, height_val, proj_dist = point_data[:4]
                point_type = 'legacy'
            
            # Calculate shadow position from skeleton point
            shadow_x = sx + s_x * proj_dist
            shadow_y = sy + s_y * proj_dist
            
            # Adaptive weight calculation based on point type
            if point_type == 'contact':
                # Contact points: stronger, more defined shadows
                base_weight = 1.3
                height_weight = 0.2 * (height_val / 6.0)
            elif point_type == 'interior':
                # Interior points: moderate shadows for volume
                base_weight = 0.9
                height_weight = 0.4 * (height_val / 6.0)
            else:
                # Fallback/legacy points: standard weight
                base_weight = 1.0
                height_weight = 0.3 * (height_val / 6.0)
            
            weight = base_weight + height_weight
            weight = min(1.5, weight)  # Allow higher weights
            
            # Create Gaussian shadow at this position
            y_coords, x_coords = np.ogrid[:h, :w]
            
            # Modified distance-dependent blur for contact shadows
            # Contact shadows start sharp and get softer with distance
            distance_factor = min(1.0, proj_dist / 15.0)  # Faster blur transition
            sigma = sigma_low * (1 - distance_factor) + sigma_high * distance_factor
            
            # Create Gaussian contribution with enhanced intensity
            distance_sq = (x_coords - shadow_x)**2 + (y_coords - shadow_y)**2
            gaussian = np.exp(-distance_sq / (2 * sigma**2)) * weight
            
            # Add contribution with type-specific intensity
            if point_type == 'contact':
                contribution = 0.12  # Stronger for contact shadows
            elif point_type == 'interior':
                contribution = 0.06  # Moderate for interior volume
            else:
                contribution = 0.08  # Standard for fallback
            
            long_shadow_sum += gaussian * contribution
        
        # Apply final opacity with enhanced visibility
        opacity_long = 0.45  # Increased opacity for better visibility
        long_shadow_final = np.clip(long_shadow_sum * 255 * opacity_long, 0, 255).astype(np.uint8)
        
        # Debug final shadow statistics
        shadow_nonzero = np.sum(long_shadow_final > 0)
        shadow_mean = np.mean(long_shadow_final[long_shadow_final > 0]) if shadow_nonzero > 0 else 0
        sys.stderr.write(f"Final shadow stats: {shadow_nonzero} non-zero pixels, mean_intensity={shadow_mean:.1f}, max={np.max(long_shadow_final)}\n")
        
        sys.stderr.write(f"Skeleton sweep complete: beta={beta_deg}°, total_points={len(skeleton_points)}, opacity={opacity_long}, processed_points={min(100, len(skeleton_points))}\n")
        
        return long_shadow_final
        
    except Exception as e:
        sys.stderr.write(f"Skeleton sweep shadow error: {str(e)}\n")
        # Safe fallback: return zero shadow
        return np.zeros((h, w), dtype=np.uint8)


def generate_ao_spots(alpha_channel, light_x, light_y, K_spots=2):
    """
    Generate AO (Ambient Occlusion) spots at contact points
    Enhanced to detect only true ground-contact points (like fins, jaw, tail)
    
    Args:
        alpha_channel: Object mask
        light_x, light_y: Light source position (0-1 normalized)
        K_spots: Number of contact spots to create
    
    Returns:
        AO spots layer as uint8 array
    """
    h, w = alpha_channel.shape
    
    # Convert to binary mask
    binary_mask = (alpha_channel > 127).astype(np.uint8) * 255
    
    # Compute SDF and its gradient (normal vectors)
    sdf = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Calculate gradient of SDF to get normal vectors
    grad_x = cv2.Sobel(sdf, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(sdf, cv2.CV_64F, 0, 1, ksize=3)
    
    # Normalize gradients to get unit normal vectors
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = np.maximum(grad_magnitude, 1e-6)  # Avoid division by zero
    normal_x = grad_x / grad_magnitude
    normal_y = grad_y / grad_magnitude
    
    # Calculate shadow direction vector
    light_vector_x = light_x - 0.5
    light_vector_y = light_y - 0.5
    magnitude = np.sqrt(light_vector_x**2 + light_vector_y**2)
    
    if magnitude > 0:
        shadow_x = -light_vector_x / magnitude
        shadow_y = -light_vector_y / magnitude
    else:
        shadow_x, shadow_y = -1, 1
    
    # Get object rim
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary_mask, kernel_erode, iterations=1)
    rim = binary_mask - eroded
    
    # ENHANCED: Detect actual ground-contact points
    # Ground contact criteria:
    # 1. Must be at the BOTTOM region of the object (lower 30% of object height)
    # 2. Must have downward-facing normal (contact with ground)
    # 3. Must be protrusion points (convex tips like fins)
    
    # Find object bounding box for ground detection
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8)
    
    largest_contour = max(contours, key=cv2.contourArea)
    x_obj, y_obj, w_obj, h_obj = cv2.boundingRect(largest_contour)
    ground_threshold = y_obj + h_obj * 0.7  # Only consider bottom 30% as potential ground contact
    
    # Create ground contact mask (only bottom region)
    ground_contact_mask = np.zeros((h, w), dtype=np.uint8)
    ground_contact_mask[int(ground_threshold):, :] = 255
    rim_ground = cv2.bitwise_and(rim, ground_contact_mask)
    
    # Calculate enhanced contact score with ground bias
    # 1. Downward normal bias (normal points down = potential ground contact)
    downward_bias = np.clip(-normal_y, 0, 1)  # -normal_y because positive y is down
    
    # 2. Shadow direction alignment (as before)
    dot_product = normal_x * shadow_x + normal_y * shadow_y
    shadow_alignment = np.clip(dot_product, 0, 1)
    
    # 3. Convexity detection (protrusion points like fin tips)
    # Use distance transform to identify thin protrusions
    convexity_score = np.clip(sdf / 5.0, 0, 1)  # Normalize by 5px typical thickness
    
    # Combined contact score: must satisfy all criteria
    contact_score = (rim_ground.astype(np.float32) / 255.0) * \
                   (downward_bias * 0.4 + shadow_alignment * 0.4 + convexity_score * 0.2)
    
    sys.stderr.write(f"Ground contact detection: ground_threshold={ground_threshold:.1f}, rim_ground_pixels={np.sum(rim_ground > 0)}\n")
    
    # Find connected components in enhanced contact score
    contact_score_binary = (contact_score > 0.2).astype(np.uint8)  # Lowered threshold for better detection
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(contact_score_binary, connectivity=8)
    
    # Enhanced component selection with ground contact prioritization
    component_candidates = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        
        # Calculate ground proximity score (closer to bottom = higher score)
        ground_proximity = (cy - ground_threshold) / (h - ground_threshold) if (h - ground_threshold) > 0 else 0
        ground_proximity = np.clip(ground_proximity, 0, 1)
        
        # Calculate average contact score in this component
        component_mask = (labels == i)
        avg_contact_score = np.mean(contact_score[component_mask]) if np.sum(component_mask) > 0 else 0
        
        # Combined priority: small area (tips) + ground proximity + high contact score
        priority_score = avg_contact_score * 0.5 + ground_proximity * 0.3 + (1.0 - min(area / 50.0, 1.0)) * 0.2
        
        component_candidates.append((priority_score, area, i, centroids[i]))
        
        sys.stderr.write(f"Component {i}: area={area}, pos=({cx:.1f},{cy:.1f}), ground_prox={ground_proximity:.2f}, contact={avg_contact_score:.2f}, priority={priority_score:.2f}\n")
    
    # Sort by priority score (highest first) and take top K_spots
    component_candidates.sort(key=lambda x: x[0], reverse=True)
    selected_spots = component_candidates[:min(K_spots, len(component_candidates))]
    
    # Create AO spots layer
    ao_spots = np.zeros((h, w), dtype=np.float32)
    
    for priority_score, area, label_id, centroid in selected_spots:
        cx, cy = int(centroid[0]), int(centroid[1])
        
        # Ensure centroid is within bounds
        if 0 <= cx < w and 0 <= cy < h:
            # Create Gaussian spot with radius based on area and priority
            radius = max(2, min(6, int(np.sqrt(area) * 0.7)))  # Slightly larger for ground contact
            
            # Create coordinate grids
            y_grid, x_grid = np.ogrid[:h, :w]
            
            # Calculate distance from centroid
            distance_sq = (x_grid - cx)**2 + (y_grid - cy)**2
            
            # Create Gaussian spot with priority-based intensity
            intensity_multiplier = 0.8 + priority_score * 0.4  # 0.8-1.2 range
            gaussian_spot = np.exp(-distance_sq / (2 * radius**2)) * intensity_multiplier
            
            # Add to AO spots layer
            ao_spots = np.maximum(ao_spots, gaussian_spot)
            
            sys.stderr.write(f"Added AO spot at ({cx},{cy}) with radius={radius}, intensity={intensity_multiplier:.2f}\n")
    
    # Convert to uint8 and apply intensity
    ao_spots_uint8 = (ao_spots * 255 * 0.18).astype(np.uint8)  # Slightly stronger for ground contact
    
    sys.stderr.write(f"Enhanced AO spots generated: {len(selected_spots)} ground-contact spots, K_spots={K_spots}\n")
    
    return ao_spots_uint8


def create_mid_band_mask(alpha_channel, band_width=3):
    """
    Create mid-band mask for contact shadow attenuation
    
    Args:
        alpha_channel: Object mask
        band_width: Width of the attenuation band in pixels
    
    Returns:
        Mid-band mask (0-1 float) where 1 = full attenuation, 0 = no attenuation
    """
    h, w = alpha_channel.shape
    
    # Convert to binary mask
    binary_mask = (alpha_channel > 127).astype(np.uint8) * 255
    
    # Create distance transform from object boundary
    distance_from_object = cv2.distanceTransform((binary_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    
    # Create band mask: strongest attenuation in the middle distance
    # Distance range: 0 to band_width pixels from object
    mid_distance = band_width / 2.0
    band_mask = np.exp(-0.5 * ((distance_from_object - mid_distance) / (band_width / 4.0))**2)
    
    # Normalize and clamp
    band_mask = np.clip(band_mask, 0, 1)
    
    sys.stderr.write(f"Mid-band mask created: width={band_width}px, peak at {mid_distance:.1f}px\n")
    
    return band_mask


def combine_shadows_multiply(contact_shadow, long_shadow, h, w):
    """
    Combine contact and long shadows using multiply blend
    Shadow = 1 - (1-Contact)*(1-Long)
    """
    # Normalize to 0-1 range
    contact_norm = contact_shadow.astype(np.float32) / 255.0
    long_norm = long_shadow.astype(np.float32) / 255.0
    
    # Multiply blend: 1 - (1-Contact)*(1-Long)
    combined = 1.0 - (1.0 - contact_norm) * (1.0 - long_norm)
    
    # Convert back to uint8
    combined_shadow = (combined * 255).astype(np.uint8)
    
    return combined_shadow


def hex_to_bgr(hex_color):
    """Convert hex color to BGR tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))  # BGR order


def composite_shadow_only(image, shadow_layer):
    """Composite object and shadow without background (keep transparent)"""
    h, w = image.shape[:2]
    
    # Create transparent canvas
    result = np.zeros((h, w, 4), dtype=np.uint8)
    
    # First, composite shadow onto transparent background
    for y in range(h):
        for x in range(w):
            if shadow_layer[y, x, 3] > 0:  # If shadow has alpha
                shadow_alpha = shadow_layer[y, x, 3] / 255.0
                
                # Blend shadow with transparent background
                for c in range(3):
                    result[y, x, c] = shadow_layer[y, x, c] * shadow_alpha
                result[y, x, 3] = shadow_layer[y, x, 3]
    
    # Then, composite object on top
    for y in range(h):
        for x in range(w):
            if image[y, x, 3] > 0:  # If object has alpha
                object_alpha = image[y, x, 3] / 255.0
                
                # Blend object with shadow+background
                for c in range(3):
                    result[y, x, c] = (
                        result[y, x, c] * (1 - object_alpha) +
                        image[y, x, c] * object_alpha
                    )
                # Update alpha channel (combine object and shadow alpha)
                result[y, x, 3] = max(result[y, x, 3], image[y, x, 3])
    
    return result


def composite_with_shadow(image, shadow_layer, background_color="#FFFFFF"):
    """Composite object and shadow onto background"""
    h, w = image.shape[:2]
    
    # Convert background color to BGR
    bg_color = hex_to_bgr(background_color)
    
    # Create background
    background = np.full((h, w, 3), bg_color, dtype=np.uint8)
    background_with_alpha = np.dstack([background, np.full((h, w), 255, dtype=np.uint8)])
    
    # Composite shadow first (behind the object)
    for y in range(h):
        for x in range(w):
            if shadow_layer[y, x, 3] > 0:  # If shadow has alpha
                shadow_alpha = shadow_layer[y, x, 3] / 255.0
                
                # Blend shadow with background
                for c in range(3):
                    background_with_alpha[y, x, c] = (
                        background_with_alpha[y, x, c] * (1 - shadow_alpha) +
                        shadow_layer[y, x, c] * shadow_alpha
                    )
    
    # Composite object on top
    for y in range(h):
        for x in range(w):
            if image[y, x, 3] > 0:  # If object has alpha
                object_alpha = image[y, x, 3] / 255.0
                
                # Blend object with background+shadow
                for c in range(3):
                    background_with_alpha[y, x, c] = (
                        background_with_alpha[y, x, c] * (1 - object_alpha) +
                        image[y, x, c] * object_alpha
                    )
    
    return background_with_alpha


def generate_shadow_layer(cutout_base64, options=None):
    """
    Main entry point: Generate shadow layer only (no object composition)
    
    CURRENT MODE: STABLE_BASELINE=True (minimal, consistent shadow generation)
    
    Args:
        cutout_base64: Base64 encoded cutout image
        options: Dict with optional parameters:
            - quality: 'preview' | 'final' (default: 'preview')
            - directionDeg: float (shadow direction in UI degrees, 0°=up)
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
            center = (new_w // 2, new_h // 2)
            M_rot = cv2.getRotationMatrix2D(center, rotate, scale)
            M_rot[0, 2] += tx
            M_rot[1, 2] += ty
            image = cv2.warpAffine(image, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        
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
            
            new_w = max(256, int(original_w * scale_factor))
            new_h = max(256, int(original_h * scale_factor))
            
            # Scale image for preview processing
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Scale parameters for preview
            shadow_params = scale_for_preview(canonical_params, scale_factor)
            
            sys.stderr.write(f"preview_fast: scale={scale_factor:.3f}, size=({new_w}x{new_h})\n")
        else:
            # final: use original size and canonical parameters
            new_w, new_h = original_w, original_h
            shadow_params = canonical_params
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
        
        
        # Generate shadow layer using SSOT parameters
        sys.stderr.write(f"Quality setting: {quality}\n")
        sys.stderr.write(f"Shadow generation at target size: ({new_w}x{new_h})\n")
        
        shadow_layer = generate_unified_shadow(image, shadow_params, quality)
        final_w, final_h = new_w, new_h
        
        # SSIM consistency check (development mode only)
        if CONSISTENCY_CHECK and quality == 'preview':
            try:
                # Generate final version for comparison
                final_image = cv2.resize(base64_to_cv2(cutout_base64), (original_w, original_h))
                final_shadow = generate_unified_shadow(final_image, canonical_params, 'final')
                
                # Downscale final to preview size for comparison
                final_downscaled = cv2.resize(final_shadow, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
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
                "algorithm": "unified_resolution_independent",
                "shadow_stats": shadow_stats,
                "processing_size": {"w": new_w, "h": new_h},
                "output_size": {"w": final_w, "h": final_h}
            }
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }




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