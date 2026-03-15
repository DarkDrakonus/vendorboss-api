#!/usr/bin/env python3
"""
VendorBoss Card Detector
========================
Automatically detects and crops a trading card from a photo.
Uses OpenCV contour detection to find the card rectangle,
corrects perspective, and returns a clean flat card image
suitable for hashing.

Works best when:
- Card is on a contrasting background (playmat, table, etc.)
- Card fills at least 30% of the frame
- Reasonably even lighting

Usage:
    from card_detector import detect_and_crop_card
    cropped = detect_and_crop_card("my_photo.jpg")
    if cropped:
        cropped.save("cropped_card.jpg")
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import cv2
except ImportError:
    print("OpenCV not installed. Run: pip3 install opencv-python-headless")
    sys.exit(1)


# ── Card detection ────────────────────────────────────────────────────────────

def load_image_cv2(img_path: str) -> np.ndarray:
    """Load image including HEIC support."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass

    # Load via PIL first (handles HEIC), then convert to CV2
    pil_img = Image.open(img_path).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform to get a flat top-down view of the card."""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # Compute output dimensions
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def detect_card_contour(image: np.ndarray, debug: bool = False) -> np.ndarray | None:
    """
    Find the largest rectangular contour in the image (the card).
    Returns 4 corner points or None if no card found.
    """
    # Resize for faster processing while keeping aspect ratio
    h, w = image.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Convert to grayscale and blur
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try multiple edge detection approaches and pick the best result
    candidates = []

    # Approach 1: Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(contours)

    # Approach 2: Adaptive threshold (better for varied lighting)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.dilate(thresh, None, iterations=1)
    contours2, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(contours2)

    # Sort by area descending, look for rectangular contour
    candidates = sorted(candidates, key=cv2.contourArea, reverse=True)

    image_area = small.shape[0] * small.shape[1]
    best_quad = None
    best_area = 0

    for contour in candidates[:20]:  # Check top 20 largest contours
        area = cv2.contourArea(contour)

        # Card must be at least 10% of image area
        if area < image_area * 0.10:
            continue

        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # We want a quadrilateral (4 corners)
        if len(approx) == 4 and area > best_area:
            best_quad = approx
            best_area = area

    if best_quad is None:
        return None

    # Scale points back to original image size
    pts = best_quad.reshape(4, 2).astype("float32")
    pts /= scale

    return pts


def detect_and_crop_card(img_path: str, debug: bool = False,
                          target_size: tuple = (429, 600)) -> Image.Image | None:
    """
    Main function: detect card in photo and return cropped, perspective-corrected image.

    Args:
        img_path: Path to input image (JPEG, PNG, HEIC supported)
        debug: If True, save debug images showing detection steps
        target_size: Output size (width, height) — default is standard card ratio

    Returns:
        PIL Image of the cropped card, or None if no card detected
    """
    try:
        image = load_image_cv2(img_path)
    except Exception as e:
        print(f"  ✗ Could not load image: {e}")
        return None

    pts = detect_card_contour(image, debug=debug)

    if pts is None:
        if debug:
            print("  ✗ No card contour found — trying fallback center crop")
        return _fallback_center_crop(img_path, target_size)

    # Apply perspective transform
    warped = four_point_transform(image, pts)

    # Ensure portrait orientation (cards are taller than wide)
    h, w = warped.shape[:2]
    if w > h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    # Resize to standard card dimensions
    warped_resized = cv2.resize(warped, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Convert back to PIL
    result = Image.fromarray(cv2.cvtColor(warped_resized, cv2.COLOR_BGR2RGB))

    if debug:
        stem = Path(img_path).stem
        debug_path = f"/tmp/{stem}_detected.jpg"
        result.save(debug_path)
        print(f"  📸 Debug crop saved: {debug_path}")

    return result


def _fallback_center_crop(img_path: str,
                           target_size: tuple = (429, 600)) -> Image.Image | None:
    """
    Fallback: if card detection fails, take the center 60% of the image.
    Better than nothing for cards that fill most of the frame.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.10)
        cropped = img.crop((margin_x, margin_y, w - margin_x, h - margin_y))
        return cropped.resize(target_size, Image.LANCZOS)
    except Exception:
        return None


# ── CLI for testing ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect and crop a card from a photo")
    parser.add_argument("image", help="Path to card photo")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    parser.add_argument("--out", help="Output path for cropped card (default: {stem}_cropped.jpg)")
    args = parser.parse_args()

    print(f"🔍 Detecting card in: {args.image}")
    result = detect_and_crop_card(args.image, debug=args.debug)

    if result:
        out_path = args.out or str(Path(args.image).stem) + "_cropped.jpg"
        result.save(out_path)
        print(f"✅ Card detected and saved to: {out_path}")
        print(f"   Size: {result.size}")
    else:
        print("❌ Could not detect card in image")
