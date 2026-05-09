<<<<<<< HEAD
# Assignment_02_Biscuit_Detection
=======
# Biscuit Quality Inspection – Edge Detection & Shape Analysis

**Assignment 02 – Image Processing**

## Problem Description

In biscuit manufacturing, broken or damaged biscuits must be identified before packaging. This project builds a classical image processing pipeline that inspects photographs of biscuits and classifies each one as **Unbroken** or **Broken** based on its shape.

The system handles two biscuit types — round and square — using separate shape metrics tuned for each geometry.

## Tools and Libraries

| Library | Purpose |
|---|---|
| OpenCV (`opencv-python`) | Image I/O, colour conversion, edge detection, morphology, contour analysis, drawing |
| NumPy | Array operations and pixel-level arithmetic |
| Python standard library (`os`, `argparse`, `pathlib`) | File handling and CLI |

## Image Processing Methods

This project uses **classical image processing only** — no machine learning or deep learning.

1. **Grayscale conversion** – reduces colour image to single-channel intensity
2. **Gaussian blur** – smooths noise before edge detection (7x7 kernel)
3. **Canny edge detection** – detects biscuit outlines using gradient thresholds (low=25, high=75), followed by morphological closing to bridge texture gaps
4. **Colour segmentation** – R-B (red minus blue) channel difference with Otsu's automatic threshold isolates golden/yellow biscuit pixels from the warm-grey background
5. **Morphological operations** – opening removes crumbs/noise; closing fills holes inside biscuit bodies
6. **Contour extraction** – `cv2.findContours` with `RETR_EXTERNAL` finds one contour per biscuit; small contours are filtered out
7. **Shape metric classification**
   - Round biscuits: circularity (4piA/P^2) + solidity (A/ConvexHullA) + area ratio
   - Square biscuits: extent (A/BoundingBoxA) + solidity + area ratio

## Dataset

The `images/` folder contains 10 photographs:

- `Round 1.jpg` to `Round 5.jpg` – round biscuits (mix of broken and unbroken)
- `square_biscuit_1.jpg` to `square_biscuit_5.jpg` – square biscuits (mix of broken and unbroken)

Each image shows multiple biscuits on a warm-grey paper background.

## Project Structure

```
image processing/
├── biscuit_detector.py   # Main detection script
├── images/               # Input biscuit images
├── results/              # Output annotated images
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run on a single image

```bash
python biscuit_detector.py "images/Round 1.jpg"
```

Override biscuit type if needed:

```bash
python biscuit_detector.py "images/square_biscuit_1.jpg" --type square
```

### 3. Run on all images and save results

```bash
python biscuit_detector.py --all images --output results
```

To skip the display windows (batch mode):

```bash
python biscuit_detector.py --all images --output results --no-show
```

### 4. Run with no arguments (demo mode)

```bash
python biscuit_detector.py
```

This runs on `images/Round 1.jpg` by default.

## Output

- **Annotated result image** – each biscuit outlined in green (unbroken) or red (broken) with shape metrics displayed
- **4-panel debug strip** – Grayscale | Canny Edges | Biscuit Mask | Classification Result
- **Console report** – per-biscuit classification with metric values
>>>>>>> d102b29 (Initial commit)
