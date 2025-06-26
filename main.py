# main.py

import cv2
import numpy as np
from src.detector import PlayerDetector
from src.tracker import Tracker
from src.mapper import PlayerMapper
from src.utils import resize_and_pad

# --- 1. CONFIGURATION ---
MODEL_PATH = 'models/best.pt'
TACTICAM_VIDEO_PATH = 'data/tacticam.mp4'
BROADCAST_VIDEO_PATH = 'data/broadcast.mp4'
HOMOGRAPHY_PATH = 'homography_matrix.npy'
CONFIDENCE_THRESHOLD = 0.5

# Based on your finding: tacticam video starts ~59 frames before the broadcast video's content.
# This means tacticam_frame_59 corresponds to broadcast_frame_0.
FRAME_OFFSET = 59

def main():
    # --- 2. INITIALIZE ---
    detector = PlayerDetector(model_path=MODEL_PATH)
    tacticam_tracker = Tracker()
    broadcast_tracker = Tracker()
    mapper = PlayerMapper()

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
    # Get total frames for each video
    total_frames_t = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_b = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_COUNT))

    # We start tacticam at the offset and broadcast at the beginning
    cap_tacticam.set(cv2.CAP_PROP_POS_FRAMES, FRAME_OFFSET)
    cap_broadcast.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # The number of frames we can process is limited by the shorter of the two clips
    # after accounting for the offset.
    frames_to_process = min(total_frames_t - FRAME_OFFSET, total_frames_b)
    print(f"Videos synchronized. Processing {int(frames_to_process)} overlapping frames.")


    # --- 4. PROCESS VIDEOS for the overlapping duration ---
    for _ in range(int(frames_to_process)):
        success_t, frame_t = cap_tacticam.read()
        success_b, frame_b = cap_broadcast.read()

        # This check is a safety net in case of reading errors
        if not success_t or not success_b:
            break

        # --- 5. DETECT & TRACK ---
        t_detections = detector.detect_players(frame_t)
        t_bboxes = [bbox for *bbox, conf, _ in t_detections if conf > CONFIDENCE_THRESHOLD]
        t_tracked_players = tacticam_tracker.update(t_bboxes)

        b_detections = detector.detect_players(frame_b)
        b_bboxes = [bbox for *bbox, conf, _ in b_detections if conf > CONFIDENCE_THRESHOLD]
        b_tracked_players = broadcast_tracker.update(b_bboxes)

        # --- 6. MAP PLAYERS ---
        t_id_map, b_id_map = mapper.map_players(t_tracked_players, b_tracked_players, homography_matrix)

        # --- 7. VISUALIZE RESULTS ---
        for t_track_id, bbox in t_tracked_players:
            final_id = t_id_map.get(t_track_id)
            if final_id is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame_t, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_t, f"{final_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        for b_track_id, bbox in b_tracked_players:
            final_id = b_id_map.get(b_track_id)
            if final_id is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame_b, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_b, f"{final_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        # --- 8. RESIZE & DISPLAY ---
        # Sticking with your preferred display settings
        PANEL_HEIGHT = 720
        PANEL_WIDTH = 640

        frame_t_resized = resize_and_pad(frame_t, PANEL_WIDTH, PANEL_HEIGHT)
        frame_b_resized = resize_and_pad(frame_b, PANEL_WIDTH, PANEL_HEIGHT)

        combined_frame = np.hstack((frame_t_resized, frame_b_resized))
        cv2.imshow("Player Mapping", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 9. CLEANUP ---
    cap_tacticam.release()
    cap_broadcast.release()
    cv2.destroyAllWindows()
    print("Script finished.")


if __name__ == "__main__":
    main()