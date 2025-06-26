# src/utils.py

import cv2
import numpy as np

def get_mouse_click_coords(video_path, frame_number=0, max_display_width=1280):
    """
    Opens a specific frame of a video, resizes if needed, and prints mouse click coordinates.
    
    Args:
        video_path (str): Path to video file.
        frame_number (int): Frame to extract and display.
        max_display_width (int): Max display width to ensure full visibility.
    
    Returns:
        List of (x, y) coordinates clicked by user (in original frame resolution).
    """
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
    """
    Calculates the homography matrix and saves it to a .npy file.

    Args:
        src_points (list of tuples): Points from the source view (e.g., broadcast).
        dst_points (list of tuples): Corresponding points in the destination view (e.g., tacticam).
        save_path (str): Path to save the matrix.
    """
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
    """
    Resizes a frame to fit within target dimensions while maintaining aspect ratio,
    and pads the remaining area with a background color.
    
    Args:
        frame: The image frame to resize and pad.
        target_width (int): The width of the final output panel.
        target_height (int): The height of the final output panel.
        background_color (tuple): The (B, G, R) color for the padding.
        
    Returns:
        The resized and padded frame.
    """
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
