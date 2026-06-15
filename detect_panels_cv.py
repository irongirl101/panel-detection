#!/usr/bin/env python3
"""
Comic Book Panel Detector (OpenCV Implementation)

This script detects comic book panels by identifying gutter regions using classical
image processing techniques. It supports two modes:
1. "contour": Contour-based segmentation (good for floating/irregular layouts).
2. "xycut": Recursive XY-Cut projection splitting (extremely robust to border-crossing speech bubbles).
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

class PanelDetectorCV:
    def __init__(
        self,
        mode="xycut",
        min_area_pct=1.5,
        max_area_pct=95.0,
        aspect_ratio_range=(0.1, 10.0),
        bilateral_d=9,
        bilateral_sigma_color=75,
        bilateral_sigma_space=75,
        adaptive_block_size=15,
        adaptive_c=10,
        morph_kernel_size=5,
        overlap_threshold=0.8,
        reading_direction="ltr",
        xycut_threshold=215
    ):
        """
        Initialize the Panel Detector with tunable parameters.
        
        Args:
            mode (str): Segmentation mode, either 'contour' or 'xycut' (default: 'xycut').
            min_area_pct (float): Minimum area of a panel as a percentage of total page area (default: 1.5).
            max_area_pct (float): Maximum area of a panel as a percentage of total page area (default: 95.0).
            aspect_ratio_range (tuple): Allowed range for panel width / height (default: 0.1 to 10.0).
            bilateral_d (int): Bilateral filter pixel neighborhood diameter (default: 9).
            bilateral_sigma_color (int): Filter sigma in the color space (default: 75).
            bilateral_sigma_space (int): Filter sigma in the coordinate space (default: 75).
            adaptive_block_size (int): Block size for adaptive thresholding; must be odd (default: 15).
            adaptive_c (int): Constant subtracted from mean in adaptive thresholding (default: 10).
            morph_kernel_size (int): Size of structuring element for dilation/closing (default: 5).
            overlap_threshold (float): Intersection-over-minimum-area threshold for removing duplicates (default: 0.8).
            reading_direction (str): Comic reading direction, either 'ltr' (left-to-right) or 'rtl' (right-to-left).
            xycut_threshold (int): Gutter sensitivity threshold (0-255) for XY-Cut (default: 215).
        """
        self.mode = mode.lower()
        self.min_area_pct = min_area_pct
        self.max_area_pct = max_area_pct
        self.aspect_ratio_range = aspect_ratio_range
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        
        # Ensure block size is odd and >= 3
        self.adaptive_block_size = max(3, adaptive_block_size | 1)
        self.adaptive_c = adaptive_c
        self.morph_kernel_size = morph_kernel_size
        self.overlap_threshold = overlap_threshold
        self.reading_direction = reading_direction.lower()
        self.xycut_threshold = xycut_threshold

    def detect_gutter_properties(self, image):
        """
        Analyze the outer borders of the image to determine the median background
        (gutter) color and whether it is a dark or light theme page.
        """
        h, w = image.shape[:2]
        
        # Sample margins: 1.5% inward from boundaries to bypass scanning edge artifacts
        margin_x = max(5, int(w * 0.015))
        margin_y = max(5, int(h * 0.015))
        
        # Get coordinates of edge pixels
        border_pixels = []
        
        # Top and bottom rows
        for d in range(3):
            border_pixels.append(image[margin_y + d, margin_x:-margin_x])
            border_pixels.append(image[-(margin_y + d + 1), margin_x:-margin_x])
            
        # Left and right columns
        for d in range(3):
            border_pixels.append(image[margin_y:-margin_y, margin_x + d])
            border_pixels.append(image[margin_y:-margin_y, -(margin_x + d + 1)])
            
        border_pixels = np.concatenate([p.reshape(-1, p.shape[-1] if len(p.shape) > 1 else 1) for p in border_pixels])
        
        # Compute median color of border samples
        median_color = np.median(border_pixels, axis=0)
        
        # Estimate overall brightness
        if len(median_color) >= 3: # Color image
            r, g, b = median_color[:3]
            brightness = (r + g + b) / 3.0
        else: # Grayscale
            brightness = float(median_color[0])
            
        # If average brightness is low, assume a dark gutter theme (black pages)
        is_dark = bool(brightness < 90.0)
        
        return median_color, is_dark

    def detect_panels(self, image_path):
        """
        Detect panels in the given image.
        
        Returns:
            dict: {
                "panels": [list of CGRect-like rects: {"x": x, "y": y, "width": w, "height": h}],
                "normalized_panels": [normalized [0..1] coordinates],
                "gutter_is_dark": bool,
                "metadata": { "image_width": W, "image_height": H }
            }
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at path: {image_path}")
            
        h, w = img.shape[:2]
        
        # Bilateral filter to smooth print grain while keeping panel boundaries sharp
        smoothed = cv2.bilateralFilter(
            img, 
            d=self.bilateral_d, 
            sigmaColor=self.bilateral_sigma_color, 
            sigmaSpace=self.bilateral_sigma_space
        )
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        
        # Automatically detect gutter properties
        _, is_dark_gutter = self.detect_gutter_properties(img)
        
        if self.mode == "xycut":
            final_rects = self._detect_panels_xycut(gray, is_dark_gutter)
        else:
            final_rects = self._detect_panels_contour(gray, is_dark_gutter, w, h)
            
        # Sort panels by reading order
        sorted_rects = self.sort_by_reading_order(final_rects)
        
        # Format results
        output_panels = [{"x": int(r[0]), "y": int(r[1]), "width": int(r[2]), "height": int(r[3])} for r in sorted_rects]
        normalized_panels = [
            {
                "x": float(r[0]) / w,
                "y": float(r[1]) / h,
                "width": float(r[2]) / w,
                "height": float(r[3]) / h
            }
            for r in sorted_rects
        ]
        
        return {
            "panels": output_panels,
            "normalized_panels": normalized_panels,
            "gutter_is_dark": is_dark_gutter,
            "metadata": {
                "image_width": w,
                "image_height": h
            }
        }

    def _detect_panels_xycut(self, gray, is_dark_gutter):
        """
        Recursive XY-Cut algorithm to segment the page by projecting columns and rows.
        """
        h, w = gray.shape[:2]
        
        # Threshold to isolate gutters as white (255) and artwork as black (0)
        if is_dark_gutter:
            # Dark background: gutters are black, panels are bright
            _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        else:
            # Light background: gutters are white, panels are dark
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
        def find_splits(rect, axis):
            x1, y1, x2, y2 = rect
            sub_mask = thresh[y1:y2, x1:x2]
            sh, sw = sub_mask.shape
            
            if sh <= 10 or sw <= 10:
                return []
                
            if axis == 0:
                # Horizontal projection
                row_means = np.mean(sub_mask, axis=1)
                is_gutter = row_means > self.xycut_threshold
                
                splits = []
                in_gutter = False
                start_y = 0
                for y in range(sh):
                    if is_gutter[y] and not in_gutter:
                        in_gutter = True
                        start_y = y
                    elif not is_gutter[y] and in_gutter:
                        in_gutter = False
                        end_y = y
                        if end_y - start_y > 4:
                            splits.append((y1 + start_y, y1 + end_y))
                return splits
            else:
                # Vertical projection
                col_means = np.mean(sub_mask, axis=0)
                is_gutter = col_means > self.xycut_threshold
                
                splits = []
                in_gutter = False
                start_x = 0
                for x in range(sw):
                    if is_gutter[x] and not in_gutter:
                        in_gutter = True
                        start_x = x
                    elif not is_gutter[x] and in_gutter:
                        in_gutter = False
                        end_x = x
                        if end_x - start_x > 4:
                            splits.append((x1 + start_x, x1 + end_x))
                return splits

        def recursive_split(rect):
            x1, y1, x2, y2 = rect
            
            # Try horizontal splits (rows)
            h_splits = find_splits(rect, axis=0)
            h_splits = [s for s in h_splits if s[0] - y1 > 20 and y2 - s[1] > 20]
            
            if h_splits:
                y_coords = [y1]
                for s in h_splits:
                    y_coords.append((s[0] + s[1]) // 2)
                y_coords.append(y2)
                
                panels = []
                for i in range(len(y_coords) - 1):
                    sub_rect = (x1, y_coords[i], x2, y_coords[i+1])
                    panels.extend(recursive_split(sub_rect))
                return panels
                
            # Try vertical splits (columns)
            v_splits = find_splits(rect, axis=1)
            v_splits = [s for s in v_splits if s[0] - x1 > 20 and x2 - s[1] > 20]
            
            if v_splits:
                x_coords = [x1]
                for s in v_splits:
                    x_coords.append((s[0] + s[1]) // 2)
                x_coords.append(x2)
                
                panels = []
                for i in range(len(x_coords) - 1):
                    sub_rect = (x_coords[i], y1, x_coords[i+1], y2)
                    panels.extend(recursive_split(sub_rect))
                return panels
                
            # Crop to fit the black border outlines tightly
            sub_img = thresh[y1:y2, x1:x2]
            non_white = np.argwhere(sub_img < self.xycut_threshold)
            if len(non_white) > 0:
                min_y, min_x = non_white.min(axis=0)
                max_y, max_x = non_white.max(axis=0)
                return [(x1 + min_x, y1 + min_y, max_x - min_x, max_y - min_y)]
                
            return [(x1, y1, x2 - x1, y2 - y1)]

        # Initial crop to remove margins
        row_means = np.mean(thresh, axis=1)
        col_means = np.mean(thresh, axis=0)
        content_y = np.argwhere(row_means < 254)
        content_x = np.argwhere(col_means < 254)
        
        if len(content_y) == 0 or len(content_x) == 0:
            return []
            
        y_min, y_max = int(content_y.min()), int(content_y.max())
        x_min, x_max = int(content_x.min()), int(content_x.max())
        
        initial_rect = (x_min, y_min, x_max, y_max)
        raw_panels = recursive_split(initial_rect)
        
        # Filter panels based on minimum and maximum area
        total_area = w * h
        min_area = total_area * (self.min_area_pct / 100.0)
        max_area = total_area * (self.max_area_pct / 100.0)
        
        filtered = []
        for x, y, pw, ph in raw_panels:
            area = pw * ph
            if area < min_area or area > max_area:
                continue
            aspect = float(pw) / float(ph)
            if aspect < self.aspect_ratio_range[0] or aspect > self.aspect_ratio_range[1]:
                continue
            filtered.append((x, y, pw, ph))
            
        return filtered

    def _detect_panels_contour(self, gray, is_dark_gutter, w, h):
        """
        Contour-based detection using thresholding, morphology, and hierarchical filtering.
        """
        total_area = w * h
        
        if is_dark_gutter:
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                self.adaptive_block_size, 
                -self.adaptive_c
            )
        else:
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                self.adaptive_block_size, 
                self.adaptive_c
            )
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size))
        processed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
        
        contours, hierarchy = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        min_area = total_area * (self.min_area_pct / 100.0)
        max_area = total_area * (self.max_area_pct / 100.0)
        has_hierarchy = hierarchy is not None and len(hierarchy) > 0
        
        for idx, cnt in enumerate(contours):
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            
            if area < min_area or area > max_area:
                continue
                
            aspect = float(cw) / float(ch)
            if aspect < self.aspect_ratio_range[0] or aspect > self.aspect_ratio_range[1]:
                continue
                
            parent_idx = hierarchy[0][idx][3] if has_hierarchy else -1
            candidates.append({
                "rect": (x, y, cw, ch),
                "area": area,
                "index": idx,
                "parent": parent_idx
            })
            
        to_reject = set()
        for i, cand_a in enumerate(candidates):
            xa, ya, wa, ha = cand_a["rect"]
            area_a = cand_a["area"]
            
            children = []
            for j, cand_b in enumerate(candidates):
                if i == j:
                    continue
                xb, yb, wb, hb = cand_b["rect"]
                
                is_inside = (xb >= xa - 2 and 
                             yb >= ya - 2 and 
                             xb + wb <= xa + wa + 2 and 
                             yb + hb <= ya + ha + 2)
                
                if is_inside and cand_b["area"] < area_a * 0.9:
                    children.append(j)
                    
            if children:
                if area_a > (total_area * 0.6) and len(children) >= 2:
                    to_reject.add(i)
                else:
                    for child_idx in children:
                        to_reject.add(child_idx)
                        
        filtered_candidates = [c for idx, c in enumerate(candidates) if idx not in to_reject]
        filtered_candidates.sort(key=lambda x: x["area"])
        
        final_rects = []
        for c in filtered_candidates:
            x, y, cw, ch = c["rect"]
            keep = True
            
            for fx, fy, fw, fh in final_rects:
                ix = max(x, fx)
                iy = max(y, fy)
                iw = min(x + cw, fx + fw) - ix
                ih = min(y + ch, fy + fh) - iy
                
                if iw > 0 and ih > 0:
                    inter_area = iw * ih
                    min_box_area = min(cw * ch, fw * fh)
                    if (inter_area / min_box_area) > self.overlap_threshold:
                        keep = False
                        break
            if keep:
                final_rects.append((x, y, cw, ch))
                
        return final_rects

    def sort_by_reading_order(self, rects):
        """
        Sort panel rects into a natural comic book reading layout.
        Groups panels into horizontal rows and sorts each row based on direction.
        """
        if len(rects) <= 1:
            return rects
            
        avg_height = sum(r[3] for r in rects) / len(rects)
        row_threshold = max(20.0, avg_height * 0.5)
        
        sorted_y = sorted(rects, key=lambda r: r[1])
        
        rows = []
        remaining = list(sorted_y)
        
        while len(remaining) > 0:
            pivot = remaining.pop(0)
            current_row = [pivot]
            
            temp_remaining = []
            for r in remaining:
                pivot_mid_y = pivot[1] + pivot[3]/2.0
                r_mid_y = r[1] + r[3]/2.0
                
                if abs(r_mid_y - pivot_mid_y) < row_threshold:
                    current_row.append(r)
                else:
                    temp_remaining.append(r)
            remaining = temp_remaining
            
            if self.reading_direction == "rtl":
                current_row.sort(key=lambda r: r[0], reverse=True)
            else:
                current_row.sort(key=lambda r: r[0])
                
            rows.append(current_row)
            
        return [rect for row in rows for rect in row]

def draw_visualizations(image_path, panels, gutter_is_dark, output_path):
    """
    Draw bounding boxes and reading order labels on the image and save it.
    """
    img = cv2.imread(image_path)
    if img is None:
        return
        
    h, w = img.shape[:2]
    box_color = (0, 255, 0)
    text_color = (0, 0, 255)
    if gutter_is_dark:
        box_color = (0, 255, 255)
        text_color = (255, 255, 0)
        
    overlay = img.copy()
    
    for i, p in enumerate(panels):
        x, y, pw, ph = p["x"], p["y"], p["width"], p["height"]
        
        cv2.rectangle(overlay, (x, y), (x + pw, y + ph), box_color, -1)
        cv2.rectangle(img, (x, y), (x + pw, y + ph), box_color, 2)
        
        label = str(i + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(pw, ph) / 150.0)
        thickness = max(1, int(font_scale * 2))
        
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        center_x = x + 25
        center_y = y + 25
        cv2.circle(img, (center_x, center_y), 18, (255, 255, 255), -1)
        cv2.circle(img, (center_x, center_y), 18, text_color, 2)
        
        cv2.putText(
            img, label, 
            (center_x - label_w // 2, center_y + label_h // 2), 
            font, font_scale, text_color, thickness, cv2.LINE_AA
        )
        
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.imwrite(output_path, img)

def main():
    parser = argparse.ArgumentParser(description="Detect panels in a comic book page.")
    parser.add_argument("image_path", help="Path to input comic page image.")
    parser.add_argument("--mode", choices=["contour", "xycut"], default="xycut", help="Segmentation mode: contour or xycut (default: xycut).")
    parser.add_argument("--output-json", help="Path to save output coordinates as JSON.")
    parser.add_argument("--output-img", help="Path to save output debug visualization image.")
    parser.add_argument("--dir", choices=["ltr", "rtl"], default="ltr", help="Reading direction: ltr (left-to-right) or rtl (right-to-left).")
    parser.add_argument("--min-area", type=float, default=1.5, help="Minimum panel area as percent of page (default: 1.5).")
    parser.add_argument("--max-area", type=float, default=95.0, help="Maximum panel area as percent of page (default: 95.0).")
    parser.add_argument("--block-size", type=int, default=15, help="Adaptive threshold block size (default: 15).")
    parser.add_argument("--adaptive-c", type=int, default=10, help="Adaptive threshold C constant (default: 10).")
    parser.add_argument("--morph-size", type=int, default=5, help="Morphological kernel size (default: 5).")
    parser.add_argument("--overlap-thr", type=float, default=0.8, help="Overlap suppression threshold (default: 0.8).")
    parser.add_argument("--xycut-thr", type=int, default=215, help="Gutter sensitivity threshold for xycut (default: 215).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}", file=sys.stderr)
        sys.exit(1)
        
    detector = PanelDetectorCV(
        mode=args.mode,
        min_area_pct=args.min_area,
        max_area_pct=args.max_area,
        adaptive_block_size=args.block_size,
        adaptive_c=args.adaptive_c,
        morph_kernel_size=args.morph_size,
        overlap_threshold=args.overlap_thr,
        reading_direction=args.dir,
        xycut_threshold=args.xycut_thr
    )
    
    try:
        results = detector.detect_panels(args.image_path)
    except Exception as e:
        print(f"Error running detector: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(json.dumps(results, indent=2))
    
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
            
    if args.output_img:
        draw_visualizations(
            args.image_path, 
            results["panels"], 
            results["gutter_is_dark"], 
            args.output_img
        )
        print(f"Saved debug visual to: {args.output_img}", file=sys.stderr)

if __name__ == "__main__":
    main()
