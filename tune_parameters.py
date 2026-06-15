#!/usr/bin/env python3
"""
Comic Book Panel Parameter Tuner (OpenCV GUI)

This script loads a comic book page and provides interactive trackbars to tune
the parameters of the classical OpenCV panel detector in real-time.
Press 'q' or 'ESC' to exit and print the final tuned arguments.
"""

import argparse
import os
import sys
import cv2
import numpy as np
from detect_panels_cv import PanelDetectorCV

# Global variables to handle callback refresh
detector_args = {}
img_path = ""
original_image = None

def refresh_visualization(*args):
    global original_image, img_path
    
    if original_image is None:
        return
        
    # Read values from trackbars
    min_area_val = cv2.getTrackbarPos("Min Area % (x10)", "Comic Panel Parameter Tuner") / 10.0
    max_area_val = cv2.getTrackbarPos("Max Area %", "Comic Panel Parameter Tuner")
    block_size_val = cv2.getTrackbarPos("Block Size", "Comic Panel Parameter Tuner")
    adaptive_c_val = cv2.getTrackbarPos("Adaptive C (-30 to 30)", "Comic Panel Parameter Tuner") - 30
    morph_size_val = cv2.getTrackbarPos("Morph Kernel Size", "Comic Panel Parameter Tuner")
    overlap_val = cv2.getTrackbarPos("Overlap Thr %", "Comic Panel Parameter Tuner") / 100.0
    direction_val = cv2.getTrackbarPos("Direction (0:LTR, 1:RTL)", "Comic Panel Parameter Tuner")
    
    # Enforce constraints
    if block_size_val < 3:
        block_size_val = 3
    if block_size_val % 2 == 0:
        block_size_val += 1
        
    if morph_size_val < 1:
        morph_size_val = 1
        
    reading_dir = "rtl" if direction_val == 1 else "ltr"
    
    # Initialize detector with trackbar values
    detector = PanelDetectorCV(
        min_area_pct=min_area_val,
        max_area_pct=max_area_val,
        adaptive_block_size=block_size_val,
        adaptive_c=adaptive_c_val,
        morph_kernel_size=morph_size_val,
        overlap_threshold=overlap_val,
        reading_direction=reading_dir
    )
    
    try:
        results = detector.detect_panels(img_path)
    except Exception as e:
        print(f"Error executing detector: {e}")
        return
        
    # Create copy of image to draw on
    h, w = original_image.shape[:2]
    disp_img = original_image.copy()
    
    # Select colors
    box_color = (0, 255, 0)
    text_color = (0, 0, 255)
    if results["gutter_is_dark"]:
        box_color = (0, 255, 255)
        text_color = (255, 255, 0)
        
    # Draw transparent fills
    overlay = disp_img.copy()
    for idx, p in enumerate(results["panels"]):
        x, y, pw, ph = p["x"], p["y"], p["width"], p["height"]
        cv2.rectangle(overlay, (x, y), (x + pw, y + ph), box_color, -1)
        cv2.rectangle(disp_img, (x, y), (x + pw, y + ph), box_color, 2)
        
        # Draw text label inside
        label = str(idx + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(pw, ph) / 180.0)
        thickness = max(1, int(font_scale * 2))
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        cx, cy = x + 20, y + 20
        cv2.circle(disp_img, (cx, cy), 14, (255, 255, 255), -1)
        cv2.circle(disp_img, (cx, cy), 14, text_color, 1)
        cv2.putText(
            disp_img, label, 
            (cx - label_w // 2, cy + label_h // 2), 
            font, font_scale, text_color, thickness, cv2.LINE_AA
        )
        
    cv2.addWeighted(overlay, 0.2, disp_img, 0.8, 0, disp_img)
    
    # Scale for displaying if image is too large
    max_disp_h = 800
    if h > max_disp_h:
        scale = max_disp_h / h
        disp_img = cv2.resize(disp_img, (int(w * scale), int(h * scale)))
        
    # Draw instructions overlay on the screen
    cv2.putText(
        disp_img, "Press 'q' or 'ESC' to exit and save",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        disp_img, "Press 'q' or 'ESC' to exit and save",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
    )
    
    # Show
    cv2.imshow("Comic Panel Parameter Tuner", disp_img)
    
    # Save parameters to global for printout on exit
    global detector_args
    detector_args = {
        "min_area": min_area_val,
        "max_area": max_area_val,
        "block_size": block_size_val,
        "adaptive_c": adaptive_c_val,
        "morph_size": morph_size_val,
        "overlap_thr": overlap_val,
        "dir": reading_dir
    }

def main():
    global img_path, original_image
    
    parser = argparse.ArgumentParser(description="Tune parameters for panel detection interactively.")
    parser.add_argument("image_path", help="Path to sample comic page image.")
    args = parser.parse_args()
    
    img_path = args.image_path
    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        sys.exit(1)
        
    original_image = cv2.imread(img_path)
    if original_image is None:
        print(f"Error: Could not load image from {img_path}")
        sys.exit(1)
        
    # Check if display is available (will fail on headless environments)
    if os.environ.get('DISPLAY') is None and sys.platform != 'darwin':
        print("Warning: No graphical display environment found. Running interactive tuner requires a GUI environment.")
        print("Please run this script directly on your local machine with a screen.")
        sys.exit(1)
        
    cv2.namedWindow("Comic Panel Parameter Tuner", cv2.WINDOW_AUTOSIZE)
    
    # Create trackbars
    # 1. Min Area (mapped 0 to 100 representing 0.0 to 10.0%)
    cv2.createTrackbar("Min Area % (x10)", "Comic Panel Parameter Tuner", 15, 100, refresh_visualization)
    # 2. Max Area (mapped 50 to 100%)
    cv2.createTrackbar("Max Area %", "Comic Panel Parameter Tuner", 95, 100, refresh_visualization)
    # 3. Block Size (mapped 3 to 99, default 15)
    cv2.createTrackbar("Block Size", "Comic Panel Parameter Tuner", 15, 99, refresh_visualization)
    # 4. Adaptive C (mapped 0 to 60 representing -30 to +30, default 40 -> +10)
    cv2.createTrackbar("Adaptive C (-30 to 30)", "Comic Panel Parameter Tuner", 40, 60, refresh_visualization)
    # 5. Morph Kernel Size (mapped 1 to 30, default 5)
    cv2.createTrackbar("Morph Kernel Size", "Comic Panel Parameter Tuner", 5, 30, refresh_visualization)
    # 6. Overlap threshold (mapped 0 to 100 representing 0% to 100%, default 80 -> 0.8)
    cv2.createTrackbar("Overlap Thr %", "Comic Panel Parameter Tuner", 80, 100, refresh_visualization)
    # 7. Direction (0: LTR, 1: RTL)
    cv2.createTrackbar("Direction (0:LTR, 1:RTL)", "Comic Panel Parameter Tuner", 0, 1, refresh_visualization)
    
    # Run first frame render
    refresh_visualization()
    
    print("\n--- Comic Panel Parameter Tuner ---")
    print("Adjust the trackbars to tune detection in real-time.")
    print("Press 'q' or 'ESC' in the window when you are satisfied with the results.")
    
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC
            break
            
        # Check if window was closed
        if cv2.getWindowProperty("Comic Panel Parameter Tuner", cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cv2.destroyAllWindows()
    
    # Print final arguments
    print("\n--- Tuning Complete! ---")
    print("Recommended parameters based on your tuning:")
    print(f"  --min-area {detector_args['min_area']}")
    print(f"  --max-area {detector_args['max_area']}")
    print(f"  --block-size {detector_args['block_size']}")
    print(f"  --adaptive-c {detector_args['adaptive_c']}")
    print(f"  --morph-size {detector_args['morph_size']}")
    print(f"  --overlap-thr {detector_args['overlap_thr']}")
    print(f"  --dir {detector_args['dir']}")
    print("\nYou can run the detector script with these parameters:")
    print(f"python3 detect_panels_cv.py \"{img_path}\" \\")
    print(f"  --min-area {detector_args['min_area']} \\")
    print(f"  --max-area {detector_args['max_area']} \\")
    print(f"  --block-size {detector_args['block_size']} \\")
    print(f"  --adaptive-c {detector_args['adaptive_c']} \\")
    print(f"  --morph-size {detector_args['morph_size']} \\")
    print(f"  --overlap-thr {detector_args['overlap_thr']} \\")
    print(f"  --dir {detector_args['dir']} --output-img output_tuned.png")

if __name__ == "__main__":
    main()
