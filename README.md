# Comic Book Panel Detector (OpenCV)

*Made with AI generated code.*

A highly optimized, CPU-friendly, and lightweight classical Computer Vision pipeline built in Python using OpenCV to detect comic book panels (and sequence them in reading order) from page scans. 

It runs in milliseconds, requires no training data, and automatically adapts to both light gutters (white backgrounds) and dark gutters (black backgrounds).

---

## Key Features

1. Dual Segmentation Engines:
   * XY-Cut Projection (xycut) [Default]: Segment pages by projecting rows and columns to find gutter peaks. Robust against speech bubbles and artwork crossing panel borders.
   * Contour Detection (contour): Segment pages using adaptive thresholding and contour hierarchy filtering. Best for complex, non-grid, or floating layouts.
2. Automatic Gutter Theme Detection: Samples outer borders inward by 1.5% to automatically determine whether the page features light (white) or dark (black) gutters.
3. Hierarchy and Nesting Filters: Automatically discards outer page scan wrappers and merges/ignores internal elements (such as text boxes or details inside panels) that form false sub-contours.
4. Non-Maximum Suppression (NMS): Deduplicates overlapping bounding box candidates, prioritizing the tightest fit.
5. Reading Order Layout Sort: Groups extracted panels into horizontal rows and sequences them according to the language reading flow:
   * ltr (Left-to-Right) for Western comics and Webtoons.
   * rtl (Right-to-Left) for Japanese Manga.

---

## Installation and Setup

We recommend using a Python 3.9+ virtual environment:

```bash
# Clone the repository and navigate into it
cd panel-detection

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## CLI Usage (detect_panels_cv.py)

Run the detector on any comic page scan to retrieve normalized coordinates as JSON and save a visual debug overlay showing the identified panels and reading sequence.

```bash
python3 detect_panels_cv.py <image_path> \
  --mode xycut \
  --output-json output.json \
  --output-img visual_overlay.png \
  --dir ltr
```

### Command Line Arguments:
* image_path: Path to the input comic book page image.
* --mode: Segmentation method to use: xycut or contour (default: xycut).
* --output-json: Path to save the extracted panel coordinates as a JSON file.
* --output-img: Path to save a visual overlay representing the detected boundaries and reading order numbers.
* --dir: Reading direction of the page: ltr (default) or rtl.
* --min-area: Minimum area a panel must occupy as a percentage of the total page area (default: 1.5).
* --max-area: Maximum area a panel can occupy as a percentage of the total page area (default: 95.0).
* --xycut-thr: Sensitivity threshold (0-255) for white gutter pixels in xycut mode (default: 215).
* --block-size: Local neighborhood size for adaptive thresholding in contour mode (default: 15).
* --adaptive-c: Constant subtracted from the mean in adaptive thresholding (default: 10).
* --morph-size: Morphological rectangular kernel size to close gaps in borders (default: 5).
* --overlap-thr: Non-Maximum Suppression (NMS) intersection-over-minimum threshold (default: 0.8).

---

## Live Parameter Tuner (tune_parameters.py)

If you are dealing with a vintage comic scan featuring textured paper, uneven scanning gutters, or custom artwork, you can visually fine-tune the parameters in real time using trackbars.

```bash
python3 tune_parameters.py <image_path>
```

* Controls: Adjust the sliders in the window to see the green panel boundaries update instantly.
* Exit: Press 'q' or 'ESC' in the tuner window. The script will print the exact detect_panels_cv.py CLI command with your tuned values, ready to copy-paste.

---

## Real-World Case Studies and Parameter Tuning

Comic pages frequently deviate from simple rectangular grids. Here is how to configure the tool for common layouts:

### Case 1: Speech Bubbles Overlapping Borders
In professional comics, speech bubbles often break through panel lines to save space. 
* The Problem: Contour-based algorithms will fail here because the panel's white interior leaks out into the white gutter, preventing a closed shape.
* The Solution: Use --mode xycut. Since it projects row/column averages across the entire page, it splits the layout cleanly despite minor line breaks.

### Case 2: Out-of-Panel Artwork (e.g. Hatching Dinosaur)
Sometimes, characters or sound effects physically cross horizontal or vertical gutters (e.g., a dinosaur hatching upward).
* The Problem: The gutter line is broken, and the average row/column intensity drops, causing two panels to merge.
* The Solution: Lower the gutter threshold parameter using --xycut-thr. For example, lowering --xycut-thr from 215 down to 190 enables the algorithm to tolerate up to 25% non-white gutter blockage and successfully segment the panels.

---

## Output Schema (JSON)

The output JSON file contains raw coordinates in pixels, responsive normalized coordinates (from 0.0 to 1.0), and page metadata:

```json
{
  "panels": [
    {
      "x": 33,
      "y": 42,
      "width": 363,
      "height": 284
    },
    ...
  ],
  "normalized_panels": [
    {
      "x": 0.0495,
      "y": 0.0410,
      "width": 0.5450,
      "height": 0.2773
    },
    ...
  ],
  "gutter_is_dark": false,
  "metadata": {
    "image_width": 666,
    "image_height": 1024
  }
}
```
Normalizing coordinates is ideal for responsive web readers, viewport canvas scaling, or cross-platform native mobile frameworks (like UIKit, SwiftUI, or Android Jetpack Compose).
