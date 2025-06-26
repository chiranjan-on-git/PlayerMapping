# src/tracker.py

import numpy as np
from scipy.optimize import linear_sum_assignment

class Tracker:
    def __init__(self, distance_threshold=50, max_frames_to_skip=10):
        """
        Initializes the tracker.
        
        Args:
            distance_threshold (int): The maximum distance (in pixels) for a track to be matched.
            max_frames_to_skip (int): The maximum number of frames a track can be "lost" before it's deleted.
        """
        self.tracks = []
        self.next_track_id = 0
        self.distance_threshold = distance_threshold
        self.max_frames_to_skip = max_frames_to_skip

    def _get_center(self, bbox):
        """Calculates the center of a bounding box."""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def update(self, detections):
        """
        Updates the tracker with new detections for a frame.
        
        Args:
            detections (list): A list of bounding boxes from the detector.
                               Each bbox is (x1, y1, x2, y2).
        
        Returns:
            A list of tracked bounding boxes, each with its ID.
            Format: [(track_id, bbox), ...]
        """
        
        # If there are no detections in this frame, just update skipped frames and return
        if not detections:
                    for i in range(len(self.tracks)):
                        self.tracks[i]['frames_skipped'] += 1
                    # Remove stale tracks and return
                    self.tracks = [t for t in self.tracks if t['frames_skipped'] < self.max_frames_to_skip]
                    return [(track['id'], track['bbox']) for track in self.tracks]

        # If there are no tracks yet, create one for each detection
        if not self.tracks:
            for bbox in detections:
                self.tracks.append({
                    'id': self.next_track_id,
                    'bbox': bbox,
                    'center': self._get_center(bbox),
                    'frames_skipped': 0
                })
                self.next_track_id += 1
            return [(track['id'], track['bbox']) for track in self.tracks]
        

        # Get centers of existing tracks and new detections
        track_centers = np.array([track['center'] for track in self.tracks])
        detection_centers = np.array([self._get_center(bbox) for bbox in detections])

        # --- Matching ---
        # Calculate the distance between every track and every new detection
        cost_matrix = np.linalg.norm(track_centers[:, np.newaxis] - detection_centers, axis=2)

        # Use the Hungarian algorithm to find the optimal assignment
        # This matches tracks to detections by minimizing the total distance
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        # Filter out matches that are too far apart
        matched_track_indices = []
        matched_detection_indices = []
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            if cost_matrix[track_idx, detection_idx] < self.distance_threshold:
                matched_track_indices.append(track_idx)
                matched_detection_indices.append(detection_idx)
        
        # --- Update, Create, and Delete Tracks ---
        
        # 1. Update matched tracks
        for track_idx, detection_idx in zip(matched_track_indices, matched_detection_indices):
            self.tracks[track_idx]['bbox'] = detections[detection_idx]
            self.tracks[track_idx]['center'] = self._get_center(detections[detection_idx])
            self.tracks[track_idx]['frames_skipped'] = 0

        # 2. Create new tracks for unmatched detections
        unmatched_detections_indices = set(range(len(detections))) - set(matched_detection_indices)
        for detection_idx in unmatched_detections_indices:
            self.tracks.append({
                'id': self.next_track_id,
                'bbox': detections[detection_idx],
                'center': self._get_center(detections[detection_idx]),
                'frames_skipped': 0
            })
            self.next_track_id += 1

        # 3. Handle unmatched tracks (they might be temporarily lost)
        unmatched_track_indices = set(range(len(self.tracks))) - set(matched_track_indices)
        for track_idx in unmatched_track_indices:
            # This check is needed because the list size can change
            if track_idx < len(self.tracks):
                self.tracks[track_idx]['frames_skipped'] += 1

        # 4. Remove stale tracks that have been lost for too long
        self.tracks = [track for track in self.tracks if track['frames_skipped'] < self.max_frames_to_skip]

        # Return the current list of active tracks
        return [(track['id'], track['bbox']) for track in self.tracks]