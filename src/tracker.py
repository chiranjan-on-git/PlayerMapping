import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

class Tracker:
    def __init__(self, max_disappeared=10, max_distance=100, min_box_area=500):
        self.next_id = 0
        self.objects = {}  # id -> {'bbox': (x1,y1,x2,y2), 'disappeared': count, 'color_hist': hist}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_box_area = min_box_area

    def _get_color_histogram(self, frame, bbox):
        # Get color histogram for a bounding box region
        try:
            x1, y1, x2, y2 = map(int, bbox)
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(64)  # Return empty histogram
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return np.zeros(64)
            
            # Convert to HSV and calculate histogram
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_roi], [0, 1], None, [8, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            return hist.flatten()
        except:
            return np.zeros(64)

    def _calculate_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _calculate_bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _calculate_combined_distance(self, bbox1, hist1, bbox2, hist2):
        # Calculate combined spatial and color distance
        # Spatial distance between centers
        center1 = self._calculate_center(bbox1)
        center2 = self._calculate_center(bbox2)
        spatial_dist = np.linalg.norm(np.array(center1) - np.array(center2))
        
        # Color similarity (using correlation)
        if hist1 is not None and hist2 is not None and hist1.size > 0 and hist2.size > 0:
            color_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            color_dist = 1 - max(0, color_corr)  # Convert to distance
        else:
            color_dist = 1.0  # Maximum distance if no color info
        
        # Combine distances (70% spatial, 30% color)
        combined_dist = 0.7 * spatial_dist + 0.3 * color_dist * 100
        return combined_dist

    def update(self, detections, frame=None):
        # Filter out very small detections
        valid_detections = []
        for detection in detections:
            if self._calculate_bbox_area(detection) >= self.min_box_area:
                valid_detections.append(detection)
        
        if len(valid_detections) == 0:
            # Mark all existing objects as disappeared
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]['disappeared'] += 1
                if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.objects[obj_id]
            return []

        # Calculate color histograms for new detections
        detection_histograms = []
        if frame is not None:
            for detection in valid_detections:
                hist = self._get_color_histogram(frame, detection)
                detection_histograms.append(hist)
        else:
            detection_histograms = [None] * len(valid_detections)

        # If no existing objects, register all detections as new
        if len(self.objects) == 0:
            for i, detection in enumerate(valid_detections):
                self.objects[self.next_id] = {
                    'bbox': detection,
                    'disappeared': 0,
                    'color_hist': detection_histograms[i]
                }
                self.next_id += 1
        else:
            # Create cost matrix for assignment
            object_ids = list(self.objects.keys())
            object_bboxes = [self.objects[obj_id]['bbox'] for obj_id in object_ids]
            object_hists = [self.objects[obj_id]['color_hist'] for obj_id in object_ids]
            
            # Calculate cost matrix
            cost_matrix = np.zeros((len(object_ids), len(valid_detections)))
            
            for i, (obj_bbox, obj_hist) in enumerate(zip(object_bboxes, object_hists)):
                for j, (det_bbox, det_hist) in enumerate(zip(valid_detections, detection_histograms)):
                    cost = self._calculate_combined_distance(obj_bbox, obj_hist, det_bbox, det_hist)
                    cost_matrix[i, j] = cost
            
            # Solve assignment problem
            if cost_matrix.size > 0:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                # Track which detections and objects are assigned
                assigned_detections = set()
                assigned_objects = set()
                
                # Process assignments
                for row_idx, col_idx in zip(row_indices, col_indices):
                    obj_id = object_ids[row_idx]
                    cost = cost_matrix[row_idx, col_idx]
                    
                    # Only accept assignment if cost is reasonable
                    if cost < self.max_distance:
                        # Update existing object
                        self.objects[obj_id]['bbox'] = valid_detections[col_idx]
                        self.objects[obj_id]['disappeared'] = 0
                        self.objects[obj_id]['color_hist'] = detection_histograms[col_idx]
                        
                        assigned_detections.add(col_idx)
                        assigned_objects.add(obj_id)
                
                # Handle unassigned detections (new objects)
                for col_idx in range(len(valid_detections)):
                    if col_idx not in assigned_detections:
                        self.objects[self.next_id] = {
                            'bbox': valid_detections[col_idx],
                            'disappeared': 0,
                            'color_hist': detection_histograms[col_idx]
                        }
                        self.next_id += 1
                
                # Handle unassigned objects (mark as disappeared)
                for obj_id in object_ids:
                    if obj_id not in assigned_objects:
                        self.objects[obj_id]['disappeared'] += 1
            
            # Remove objects that have been disappeared for too long
            for obj_id in list(self.objects.keys()):
                if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.objects[obj_id]

        # Return current tracked objects
        result = []
        for obj_id, obj_info in self.objects.items():
            if obj_info['disappeared'] == 0:  # Only return currently visible objects
                result.append((obj_id, obj_info['bbox']))
        
        return result