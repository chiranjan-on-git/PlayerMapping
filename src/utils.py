# src/utils.py

import cv2
import numpy as np

def get_mouse_click_coords(video_path, frame_number=0, max_display_width=1280):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []

    # Jump to the requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()

    if not success:
        print(f"Error: Cannot read frame {frame_number}")
        return []

    # Resize logic
    orig_height, orig_width = frame.shape[:2]
    scale = 1.0
    if orig_width > max_display_width:
        scale = max_display_width / orig_width
        resized_width = int(orig_width * scale)
        resized_height = int(orig_height * scale)
        frame_resized = cv2.resize(frame, (resized_width, resized_height))
    else:
        frame_resized = frame.copy()

    clicked_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            clicked_points.append((x_orig, y_orig))
            print(f"Point {len(clicked_points)}: ({x_orig}, {y_orig})")
            cv2.circle(frame_resized, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, frame_resized)

    window_name = f"Click Points - {video_path.split('/')[-1]}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame_resized)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\nClick on points in the video frame (Frame {frame_number}).")
    print("Press ESC when done.")

    while True:
        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    print(f"\nCollected {len(clicked_points)} point(s):")
    print(clicked_points)
    return clicked_points


def calculate_and_save_homography(src_points, dst_points, save_path="homography_matrix.npy"):

    src_pts = np.array(src_points, dtype="float32")
    dst_pts = np.array(dst_points, dtype="float32")

    if len(src_pts) < 4 or len(dst_pts) < 4:
        print("Error: Need at least 4 points in each set to calculate homography.")
        return

    H, status = cv2.findHomography(src_pts, dst_pts)

    if H is not None:
        print("\nHomography matrix:")
        print(H)
        np.save(save_path, H)
        print(f"Saved homography matrix to '{save_path}'")
    else:
        print("Error: Failed to compute homography. Check point quality.")

def resize_and_pad(frame, target_width, target_height, background_color=(0, 0, 0)):

    original_h, original_w, _ = frame.shape
    
    # Calculate the ratio to fit inside the target dimensions
    ratio = min(target_width / original_w, target_height / original_h)
    
    # Calculate new dimensions
    new_w = int(original_w * ratio)
    new_h = int(original_h * ratio)
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_w, new_h))
    
    # Create a black canvas of the target size
    padded_frame = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)
    
    # Calculate top-left corner coordinates to paste the resized frame
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    
    # Paste the resized frame onto the center of the canvas
    padded_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
    
    return padded_frame

def get_color_histogram(frame, bbox):
    
    # Calculates a 3D color histogram for the region defined by the bounding box.
    
    x1, y1, x2, y2 = map(int, bbox)
    # Get the player region of interest (ROI)
    roi = frame[y1:y2, x1:x2]
    
    # Calculate histogram in the HSV color space (often better for color matching)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Use 32 bins for hue, 32 for saturation, 32 for value
    hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    
    # Normalize the histogram to be comparable
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist.flatten()

def filter_overlapping_detections(detections, iou_threshold=0.3):

    if not detections:
        return []
    
    # Sort by confidence (highest first)
    detections_sorted = sorted(detections, key=lambda x: x[4], reverse=True)
    
    def calculate_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    filtered_detections = []
    
    for current_detection in detections_sorted:
        # Check if this detection overlaps significantly with any already selected detection
        should_keep = True
        for kept_detection in filtered_detections:
            iou = calculate_iou(current_detection, kept_detection)
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            filtered_detections.append(current_detection)
    
    return filtered_detections