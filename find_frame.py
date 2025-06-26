# find_frame.py
import cv2

# --- CONFIGURATION ---
VIDEO_PATH = 'data/broadcast.mp4'  # Change path as needed
INITIAL_SPEED = 1.0                # 1.0 = normal speed

# --- INITIALIZE ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

cv2.namedWindow('Frame Finder - ←/→: step | +/-: speed | p: pause | q: quit', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame Finder - ←/→: step | +/-: speed | p: pause | q: quit', 1280, 720)

frame_count = 0
speed = INITIAL_SPEED
paused = False

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

while cap.isOpened():
    if not paused:
        success, frame = cap.read()
        if not success:
            print("\nReached end of video.")
            break

        # Overlay frame number
        text = f"Frame: {frame_count}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Frame Finder - ←/→: step | +/-: speed | p: pause | q: quit', frame)

        frame_count += 1

    # Wait for key based on speed
    key = cv2.waitKey(0 if paused else int(25 / speed)) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('p'):
        paused = not paused
        state = "Paused" if paused else "Playing"
        print(f"\n{state} at frame {frame_count}")

    elif key == ord('+'):
        speed = min(speed + 0.1, 5.0)
        print(f"\nSpeed increased to {speed:.1f}x")

    elif key == ord('-'):
        speed = max(speed - 0.1, 0.1)
        print(f"\nSpeed decreased to {speed:.1f}x")

    elif key == 81 or key == ord('a'):  # ← or 'a'
        frame_count = max(0, frame_count - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        print(f"\nMoved back to frame {frame_count}")
        paused = True

    elif key == 83 or key == ord('d'):  # → or 'd'
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        print(f"\nStep forward to frame {frame_count}")
        paused = True


cap.release()
cv2.destroyAllWindows()
print("\nScript finished.")
