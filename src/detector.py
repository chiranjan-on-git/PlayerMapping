from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path):

        print("Loading player detection model...")
        self.model = YOLO(model_path)
        print("Model loaded successfully.")

    def detect_players(self, frame):

        results = self.model(frame, verbose=False) 
        
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            if class_name == 'player':
                detections.append((x1, y1, x2, y2, confidence, class_name))
        
        return detections
