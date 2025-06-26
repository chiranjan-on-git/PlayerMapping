# src/mapper.py

import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

class PlayerMapper:
    def __init__(self, distance_threshold=50):
        """
        Initializes the PlayerMapper.
        Args:
            distance_threshold (int): The maximum distance (in pixels) in the tacticam view
                                      for a player to be considered a match.
        """
        self.distance_threshold = distance_threshold
        # This dictionary will store the final mapping: {tacticam_id: final_player_id}
        self.tacticam_to_final_id = {}
        # This will store the reverse mapping for convenience: {broadcast_id: final_player_id}
        self.broadcast_to_final_id = {}
        self.next_final_id = 0

    def _get_center(self, bbox):
        """Calculates the center of a bounding box."""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def map_players(self, tacticam_players, broadcast_players, homography_matrix):
        """
        Matches players from tacticam and broadcast views and assigns a consistent final ID.
        
        Args:
            tacticam_players (list): List of (id, bbox) for tacticam view.
            broadcast_players (list): List of (id, bbox) for broadcast view.
            homography_matrix (np.array): The 3x3 homography matrix.
        
        Returns:
            A tuple of two dictionaries:
            1. {tacticam_id: final_id, ...}
            2. {broadcast_id: final_id, ...}
        """
        if not tacticam_players or not broadcast_players:
            return self.tacticam_to_final_id, self.broadcast_to_final_id

        # --- 1. Get Positions and Transform ---
        t_ids = [p[0] for p in tacticam_players]
        t_centers = np.array([self._get_center(p[1]) for p in tacticam_players])

        b_ids = [p[0] for p in broadcast_players]
        # Get bottom-center of broadcast players for better ground-plane accuracy
        b_positions = np.array([((p[1][0] + p[1][2]) / 2, p[1][3]) for p in broadcast_players], dtype="float32").reshape(-1, 1, 2)
        
        # Transform broadcast points to tacticam view
        transformed_b_centers = cv2.perspectiveTransform(b_positions, homography_matrix).reshape(-1, 2)
        
        # --- 2. Match Players ---
        cost_matrix = np.linalg.norm(t_centers[:, np.newaxis] - transformed_b_centers, axis=2)
        t_indices, b_indices = linear_sum_assignment(cost_matrix)
        
        # --- 3. Assign Final IDs ---
        for t_idx, b_idx in zip(t_indices, b_indices):
            # Check if the match is within our distance threshold
            if cost_matrix[t_idx, b_idx] < self.distance_threshold:
                t_id = t_ids[t_idx]
                b_id = b_ids[b_idx]
                
                # If this pair is already mapped, do nothing.
                # If only one of them is mapped, it's a re-detection, so link them.
                # If neither is mapped, assign a new final ID.
                
                if t_id in self.tacticam_to_final_id:
                    final_id = self.tacticam_to_final_id[t_id]
                    self.broadcast_to_final_id[b_id] = final_id
                elif b_id in self.broadcast_to_final_id:
                    final_id = self.broadcast_to_final_id[b_id]
                    self.tacticam_to_final_id[t_id] = final_id
                else:
                    self.tacticam_to_final_id[t_id] = self.next_final_id
                    self.broadcast_to_final_id[b_id] = self.next_final_id
                    self.next_final_id += 1
        
        return self.tacticam_to_final_id, self.broadcast_to_final_id