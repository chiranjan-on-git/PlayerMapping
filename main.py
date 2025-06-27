import cv2
import numpy as np
from src.detector import PlayerDetector
from src.tracker import Tracker
from src.mapper import PlayerMapper
from src.utils import resize_and_pad

#1. Configuration
MODEL_PATH = 'models/best.pt'
TACTICAM_VIDEO_PATH = 'data/tacticam.mp4'
BROADCAST_VIDEO_PATH = 'data/broadcast.mp4'
HOMOGRAPHY_PATH = 'homography_matrix.npy'
CONFIDENCE_THRESHOLD = 0.7

FRAME_OFFSET = 59

def filter_overlapping_detections(detections, iou_threshold=0.3):
    #Remove redundant detections that overlap too much
    if not detections:
        return []
    
    detections_sorted = sorted(detections, key=lambda x: x[4], reverse=True)
    
    def calculate_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    filtered_detections = []
    for current_detection in detections_sorted:
        should_keep = True
        for kept_detection in filtered_detections:
            iou = calculate_iou(current_detection, kept_detection)
            if iou > iou_threshold:
                should_keep = False
                break
        if should_keep:
            filtered_detections.append(current_detection)
    
    return filtered_detections

def draw_players_simple(frame, tracked_players, id_map, color=(0, 0, 255)):
    # Simple non-overlapping label drawing
    valid_players = []
    for track_id, bbox in tracked_players:
        final_id = id_map.get(track_id)
        if final_id is not None:
            x1, y1, x2, y2 = map(int, bbox)
            valid_players.append({
                'bbox': (x1, y1, x2, y2),
                'final_id': final_id,
                'label_x': x1,
                'label_y': y1 - 10
            })
    
    # Draw bounding boxes
    for player in valid_players:
        x1, y1, x2, y2 = player['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw labels with simple vertical offset for overlaps
    valid_players.sort(key=lambda p: (p['label_x'], p['label_y']))
    
    label_height = 25
    used_positions = []
    
    for player in valid_players:
        label_x = player['label_x']
        label_y = player['label_y']
        
        # Simple overlap avoidance
        while any(abs(label_x - used_x) < 50 and abs(label_y - used_y) < label_height 
                 for used_x, used_y in used_positions):
            label_y -= label_height
        
        if label_y < 20:
            label_y = player['bbox'][3] + 20
        
        # Draw label with background
        label_text = f"P{player['final_id']}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Background rectangle
        cv2.rectangle(frame, (label_x-2, label_y-text_height-2), 
                     (label_x+text_width+2, label_y+2), (0, 0, 0), -1)
        
        # Text
        cv2.putText(frame, label_text, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        used_positions.append((label_x, label_y))

def main():
    # --- 2. INITIALIZE ---
    detector = PlayerDetector(model_path=MODEL_PATH)
    
    # Enhanced trackers with better parameters
    tacticam_tracker = Tracker(max_disappeared=15, max_distance=80, min_box_area=800)
    broadcast_tracker = Tracker(max_disappeared=15, max_distance=80, min_box_area=800)
    
    # Simpler mapper with focus on consistency
    mapper = PlayerMapper(
        distance_threshold=70,
        weight_spatial=0.6,
        weight_color=0.4,
        id_persistence_frames=20,
        color_history_size=3
    )

    try:
        homography_matrix = np.load(HOMOGRAPHY_PATH)
    except FileNotFoundError:
        print(f"Error: Homography matrix not found at {HOMOGRAPHY_PATH}")
        return

    cap_tacticam = cv2.VideoCapture(TACTICAM_VIDEO_PATH)
    cap_broadcast = cv2.VideoCapture(BROADCAST_VIDEO_PATH)

    if not cap_tacticam.isOpened() or not cap_broadcast.isOpened():
        print("Error: Could not open one or both video files.")
        return

    # --- 3. CALCULATE SYNC & DURATION ---
    total_frames_t = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_b = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_COUNT))

    cap_tacticam.set(cv2.CAP_PROP_POS_FRAMES, FRAME_OFFSET)
    cap_broadcast.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frames_to_process = min(total_frames_t - FRAME_OFFSET, total_frames_b)
    print(f"Videos synchronized. Processing {int(frames_to_process)} overlapping frames.")

    frame_number = 0
    
    # --- 4. PROCESS VIDEOS ---
    for _ in range(int(frames_to_process)):
        success_t, frame_t = cap_tacticam.read()
        success_b, frame_b = cap_broadcast.read()

        if not success_t or not success_b:
            break

        frame_number += 1

        # --- 5. DETECT & TRACK WITH IMPROVED FILTERING ---
        # Tacticam
        t_detections = detector.detect_players(frame_t)
        t_detections_filtered = filter_overlapping_detections(t_detections, iou_threshold=0.25)
        t_bboxes = [bbox for *bbox, conf, _ in t_detections_filtered if conf > CONFIDENCE_THRESHOLD]
        # Pass frame to tracker for color information
        t_tracked_players = tacticam_tracker.update(t_bboxes, frame_t)

        # Broadcast
        b_detections = detector.detect_players(frame_b)
        b_detections_filtered = filter_overlapping_detections(b_detections, iou_threshold=0.25)
        b_bboxes = [bbox for *bbox, conf, _ in b_detections_filtered if conf > CONFIDENCE_THRESHOLD]
        # Pass frame to tracker for color information
        b_tracked_players = broadcast_tracker.update(b_bboxes, frame_b)

        # --- 6. MAP PLAYERS ---
        t_id_map, b_id_map = mapper.map_players(t_tracked_players, frame_t, b_tracked_players, frame_b, homography_matrix)

        # --- 7. VISUALIZE ---
        frame_t_display = frame_t.copy()
        frame_b_display = frame_b.copy()
        
        draw_players_simple(frame_t_display, t_tracked_players, t_id_map)
        draw_players_simple(frame_b_display, b_tracked_players, b_id_map)
        
        # --- 8. DISPLAY ---
        PANEL_HEIGHT = 720
        PANEL_WIDTH = 640

        frame_t_resized = resize_and_pad(frame_t_display, PANEL_WIDTH, PANEL_HEIGHT)
        frame_b_resized = resize_and_pad(frame_b_display, PANEL_WIDTH, PANEL_HEIGHT)

        combined_frame = np.hstack((frame_t_resized, frame_b_resized))
        
        # Add info
        cv2.putText(combined_frame, f"Frame: {frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_frame, f"T: {len(t_tracked_players)} | B: {len(b_tracked_players)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Player Mapping (press q to quit)", combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)

    # --- 9. CLEANUP ---
    cap_tacticam.release()
    cap_broadcast.release()
    cv2.destroyAllWindows()
    print("Script finished.")
    
    print(f"\nFinal Statistics:")
    print(f"Total unique players: {mapper.next_final_id}")
    print(f"Tacticam tracker next ID: {tacticam_tracker.next_id}")
    print(f"Broadcast tracker next ID: {broadcast_tracker.next_id}")

if __name__ == "__main__":
    main()