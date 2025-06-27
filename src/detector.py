from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path):
        """
        Initializes the PlayerDetector with a YOLO model.
        Args:
            model_path (str): The path to the YOLO model file (e.g., 'models/best.pt').
        """
        print("Loading player detection model...")
        self.model = YOLO(model_path)
        print("Model loaded successfully.")

    def detect_players(self, frame):
        """
        Detects players in a single video frame.
        Args:
            frame: A single frame from a video (e.g., from OpenCV).
        Returns:
            A list of bounding boxes for detected players.
            Each bounding box is a tuple: (x1, y1, x2, y2, confidence, class_name)
        """
        results = self.model(frame, verbose=False) # verbose=False makes it less chatty
        
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            # We only care about 'player' detections
            if class_name == 'player':
                detections.append((x1, y1, x2, y2, confidence, class_name))
        
        return detections
