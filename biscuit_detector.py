"""
Biscuit Quality Inspection – Edge Detection & Shape Analysis
Assignment 02 – Image Processing

Pipeline
--------
1.  Load image and resize to a working resolution (≤ 900 px on the long side)
2.  Convert to grayscale; apply Gaussian blur
3.  Canny edge detection          ← core "edge detection" step (visualised)
4.  Colour segmentation (R−B Otsu threshold) to obtain clean filled regions
    NOTE: The background in these images is warm-grey (~130-180 in grayscale),
    not pure white, so grayscale thresholding alone cannot separate biscuits
    from the background.  Biscuits are golden/yellow (R >> B) while the
    background is neutral, making R−B the most reliable single feature.
5.  Morphological open/close to remove crumbs and fill holes
6.  External contour extraction (one contour per biscuit)
7.  Shape-metric classification
       Round biscuits  → circularity (4πA/P²) + solidity (A/hull_A)
       Square biscuits → extent (A/bbox_A) + solidity
8.  Annotated result image + 4-panel debug strip (Gray|Edges|Mask|Result)

Usage
-----
  python biscuit_detector.py "images/Round 1.jpg"
  python biscuit_detector.py "images/square_biscuit_1.jpg" --type square
  python biscuit_detector.py --all images --output results
  python biscuit_detector.py --all images --output results --no-show
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MAX_DIM          = 900    # Resize so longest side is at most this many pixels
MIN_AREA_FRAC    = 0.004  # Minimum contour area as fraction of image area
                          # (~3 600 px for a 900×800 image – filters crumbs)

# Classification thresholds (tuned on the provided dataset)
# Round: full circle has circularity ≈ 1.0; a semi-circle ≈ 0.748 (theoretical)
ROUND_CIRC_MIN   = 0.82   # Below → broken arc / partial piece
ROUND_SOL_MIN    = 0.90   # Below → significant concavities → broken

# Square: complete rectangle has extent ≈ 0.93; irregular chunk ≈ 0.50
SQUARE_EXT_MIN   = 0.78   # Below → too much empty space in bbox → broken
SQUARE_SOL_MIN   = 0.88   # Below → jagged / concave edges → broken

# Area guard: an intact biscuit must be at least this fraction of the
# largest biscuit detected in the same image.  Broken pieces that pass
# all shape checks but are much smaller (thin slabs, edge strips) are
# caught here.
AREA_INTACT_FRAC = 0.80   # Within 20 % of the largest biscuit's area


# ══════════════════════════════════════════════════════════════════════════════
# 1 – IMAGE LOADING & RESIZE
# ══════════════════════════════════════════════════════════════════════════════

def load_and_resize(path: str) -> tuple:
    """
    Load BGR image and downscale so the longest dimension equals MAX_DIM.
    Returns (img_resized, scale_factor).
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Cannot open: {path}')
    h, w  = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img, min(scale, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 2 – PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(img: np.ndarray) -> tuple:
    """
    Returns (gray, blurred).
    blurred is passed to Canny; gray is shown in the debug strip.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    return gray, blurred


# ══════════════════════════════════════════════════════════════════════════════
# 3 – CANNY EDGE DETECTION (primary visualised step)
# ══════════════════════════════════════════════════════════════════════════════

def detect_edges(blurred: np.ndarray,
                 low: int  = 25,
                 high: int = 75) -> np.ndarray:
    """
    Canny edge detector.  Morphological closing connects gaps caused by
    biscuit texture and lighting variation.

    The returned edge map is shown in the debug strip and used as an
    overlay on the result to highlight biscuit outlines.
    """
    edges  = cv2.Canny(blurred, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


# ══════════════════════════════════════════════════════════════════════════════
# 4 – COLOUR-BASED BISCUIT MASK  (R − B channel, Otsu threshold)
# ══════════════════════════════════════════════════════════════════════════════

def build_biscuit_mask(img: np.ndarray) -> np.ndarray:
    """
    Segment biscuit regions from the background using the red-minus-blue
    colour difference channel and Otsu's automatic threshold.

    Rationale
    ---------
    The background paper is a warm grey (R ≈ G ≈ B, all ≈ 130-185).
    Biscuits are golden / yellow (R >> B, typically R−B ≈ 40-100).
    Thresholding this difference reliably isolates biscuit pixels even
    though background and biscuit grayscale values overlap significantly.

    Returns binary mask: 255 = biscuit, 0 = background.
    """
    r    = img[:, :, 2].astype(np.int16)
    b    = img[:, :, 0].astype(np.int16)
    diff = np.clip(r - b, 0, 255).astype(np.uint8)

    # Otsu's threshold adapts to each image's contrast automatically
    _, mask = cv2.threshold(diff, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove isolated pixels / crumbs smaller than ~ 3×3
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k,  iterations=2)

    # Fill small holes inside the biscuit body (texture, embossing)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=3)

    return mask


# ══════════════════════════════════════════════════════════════════════════════
# 5 – CONTOUR EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_contours(mask: np.ndarray) -> list:
    """
    Find external contours in the biscuit mask.
    Tiny regions (crumbs) are removed by the MIN_AREA_FRAC threshold.
    Returns contours sorted largest-first.
    """
    h, w      = mask.shape[:2]
    min_area  = MIN_AREA_FRAC * h * w

    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    return sorted(contours, key=cv2.contourArea, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6 – SHAPE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def circularity(c: np.ndarray) -> float:
    """4πA/P²  →  1.0 for perfect circle, lower for any other shape."""
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    return (4 * np.pi * area / peri ** 2) if peri > 0 else 0.0


def solidity(c: np.ndarray) -> float:
    """A / ConvexHullA  →  1.0 when fully convex; drops for concave shapes."""
    area      = cv2.contourArea(c)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    return (area / hull_area) if hull_area > 0 else 0.0


def extent(c: np.ndarray) -> float:
    """A / BoundingBoxA  →  ~1.0 for a filled rectangle."""
    area      = cv2.contourArea(c)
    _, _, w, h = cv2.boundingRect(c)
    return (area / (w * h)) if (w * h) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 7 – CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify(contour: np.ndarray, biscuit_type: str, max_area: float) -> tuple:
    """
    Returns (is_unbroken: bool, metrics: dict).

    Round  → circularity + solidity + area ratio
    Square → extent + solidity + area ratio

    max_area: area of the largest contour in this image (pixels).
    A biscuit is only Unbroken if its area is >= AREA_INTACT_FRAC * max_area.
    This catches thin slabs / edge strips that pass shape checks but are
    clearly too small to be an intact biscuit.
    """
    area    = cv2.contourArea(contour)
    sol     = solidity(contour)
    ar      = area / max_area if max_area > 0 else 0.0
    area_ok = ar >= AREA_INTACT_FRAC

    metrics = {'Solidity': sol, 'AreaR': ar}

    if biscuit_type == 'round':
        circ              = circularity(contour)
        metrics['Circ']   = circ
        is_unbroken       = circ >= ROUND_CIRC_MIN and sol >= ROUND_SOL_MIN and area_ok
    else:
        ext               = extent(contour)
        metrics['Extent'] = ext
        is_unbroken       = ext >= SQUARE_EXT_MIN and sol >= SQUARE_SOL_MIN and area_ok

    return is_unbroken, metrics


def infer_type(filename: str) -> str:
    return 'square' if 'square' in filename.lower() else 'round'


# ══════════════════════════════════════════════════════════════════════════════
# 8 – DRAWING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

GREEN = (34, 139, 34)
RED   = (0,   0, 210)


def _centroid(c: np.ndarray) -> tuple:
    M = cv2.moments(c)
    if M['m00']:
        return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    x, y, w, h = cv2.boundingRect(c)
    return x + w // 2, y + h // 2


def annotate_result(img: np.ndarray,
                    contours: list,
                    labels: list,
                    metrics_list: list,
                    edges: np.ndarray) -> np.ndarray:
    """Draw contours, edge overlay, status labels, and summary banner."""
    out = img.copy()

    # Overlay Canny edges in blue for context
    edge_colour              = np.zeros_like(out)
    edge_colour[edges > 0]   = (180, 60, 0)   # dark-blue where edges are
    cv2.addWeighted(edge_colour, 0.4, out, 0.6, 0, out)

    for contour, is_unbroken, metrics in zip(contours, labels, metrics_list):
        colour = GREEN if is_unbroken else RED
        status = 'Unbroken' if is_unbroken else 'Broken'

        # Semi-transparent fill
        overlay = out.copy()
        cv2.drawContours(overlay, [contour], -1, colour, cv2.FILLED)
        cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

        # Contour border
        cv2.drawContours(out, [contour], -1, colour, 3)

        cx, cy     = _centroid(contour)
        mstr       = '  '.join(f'{k}:{v:.2f}' for k, v in metrics.items())

        # White background pill for readability
        (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (cx - 48, cy - 30), (cx - 48 + tw + 6, cy - 8),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(out, status, (cx - 45, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)
        cv2.putText(out, mstr, (cx - 58, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

    total    = len(labels)
    unbroken = sum(labels)
    broken   = total - unbroken
    banner   = f'Total: {total}   Unbroken: {unbroken}   Broken: {broken}'
    cv2.rectangle(out, (0, 0), (out.shape[1], 48), (20, 20, 20), cv2.FILLED)
    cv2.putText(out, banner, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def build_debug_strip(gray: np.ndarray,
                      edges: np.ndarray,
                      mask: np.ndarray,
                      result: np.ndarray,
                      target_h: int = 640) -> np.ndarray:
    """
    4-panel horizontal strip:
        Grayscale | Canny Edges | Biscuit Mask | Classification Result
    """
    def _fit(im):
        h, w = im.shape[:2]
        scale = target_h / h
        return cv2.resize(im, (int(w * scale), target_h),
                          interpolation=cv2.INTER_LINEAR)

    panels = {
        'Grayscale':    cv2.cvtColor(_fit(gray),  cv2.COLOR_GRAY2BGR),
        'Canny Edges':  cv2.cvtColor(_fit(edges), cv2.COLOR_GRAY2BGR),
        'Biscuit Mask': cv2.cvtColor(_fit(mask),  cv2.COLOR_GRAY2BGR),
        'Classification': _fit(result),
    }
    for label, panel in panels.items():
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 36), (15, 15, 15), cv2.FILLED)
        cv2.putText(panel, label, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 200, 255), 2, cv2.LINE_AA)

    return np.hstack(list(panels.values()))


# ══════════════════════════════════════════════════════════════════════════════
# 9 – MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_image(image_path: str,
                  biscuit_type: str = None,
                  output_path: str  = None,
                  show: bool        = True) -> np.ndarray:
    """
    Run the full detection pipeline on one image.

    Parameters
    ----------
    image_path   : path to input image file
    biscuit_type : 'round' | 'square' | None  (None → inferred from filename)
    output_path  : save annotated result here when provided
    show         : display OpenCV windows

    Returns annotated result as a BGR ndarray.
    """
    img, _  = load_and_resize(image_path)
    btype   = biscuit_type or infer_type(os.path.basename(image_path))

    # ── Pipeline ──────────────────────────────────────────────
    gray, blurred = preprocess(img)
    edges         = detect_edges(blurred)           # Canny
    mask          = build_biscuit_mask(img)         # colour segmentation
    contours      = extract_contours(mask)

    if not contours:
        print(f'[WARN] No biscuits found in: {os.path.basename(image_path)}')
        return None

    max_area = max(cv2.contourArea(c) for c in contours)

    labels, metrics_list = [], []
    for c in contours:
        is_unbroken, met = classify(c, btype, max_area)
        labels.append(is_unbroken)
        metrics_list.append(met)

    result = annotate_result(img, contours, labels, metrics_list, edges)

    # ── Console report ────────────────────────────────────────
    total    = len(labels)
    unbroken = sum(labels)
    print(f'\n  {os.path.basename(image_path):<32} type={btype:<7} '
          f'total={total}  unbroken={unbroken}  broken={total - unbroken}')
    for i, (is_ub, met) in enumerate(zip(labels, metrics_list)):
        status = 'Unbroken' if is_ub else 'Broken  '
        mstr   = '  '.join(f'{k}={v:.3f}' for k, v in met.items())
        print(f'    Biscuit {i + 1:2d}: {status}  [{mstr}]')

    # ── Save ──────────────────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f'  Saved: {output_path}')

    # ── Display ───────────────────────────────────────────────
    if show:
        strip = build_debug_strip(gray, edges, mask, result)
        win   = os.path.basename(image_path)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(1800, strip.shape[1]), strip.shape[0])
        cv2.imshow(win, strip)
        print('  [Press any key to advance]')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def process_directory(images_dir: str,
                      output_dir: str = None,
                      show: bool      = True):
    """Process every image in images_dir."""
    images_dir = Path(images_dir)
    out_dir    = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    exts  = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in exts)

    if not files:
        print(f'No image files found in: {images_dir}')
        return

    print(f'Processing {len(files)} image(s) in "{images_dir}"')

    for f in files:
        out_path = str(out_dir / f'result_{f.name}') if out_dir else None
        process_image(str(f), output_path=out_path, show=show)


# ══════════════════════════════════════════════════════════════════════════════
# 10 – CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Biscuit Broken/Unbroken Inspector using Edge Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python biscuit_detector.py "images/Round 1.jpg"\n'
               '  python biscuit_detector.py --all images --output results --no-show')
    p.add_argument('image',    nargs='?',
                   help='Path to a single image')
    p.add_argument('--all',    metavar='DIR',
                   help='Process every image inside DIR')
    p.add_argument('--type',   choices=['round', 'square'],
                   help='Override biscuit type (default: inferred from filename)')
    p.add_argument('--output', metavar='PATH',
                   help='Output file (single image) or directory (--all mode)')
    p.add_argument('--no-show', action='store_true',
                   help='Suppress display windows')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    show = not args.no_show

    if args.all:
        process_directory(args.all, output_dir=args.output, show=show)
    elif args.image:
        process_image(args.image, biscuit_type=args.type,
                      output_path=args.output, show=show)
    else:
        # Demo: run on Round 1.jpg
        demo = 'images/Round 1.jpg'
        print(f'No arguments supplied. Running demo on: {demo}')
        process_image(demo, show=show)
