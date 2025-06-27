import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
from src.utils import get_color_histogram

class PlayerMapper:
    def __init__(self, distance_threshold=75, weight_spatial=0.4, weight_color=0.6, 
                 id_persistence_frames=30, color_history_size=5):
        self.distance_threshold = distance_threshold
        self.weight_spatial = weight_spatial
        self.weight_color = weight_color
        self.id_persistence_frames = id_persistence_frames  # remembering lost players
        self.color_history_size = color_history_size

        # linking temporary tracker IDs to a final, permanent ID
        self.tacticam_to_final_id = {}
        self.broadcast_to_final_id = {}
        self.next_final_id = 0
        
        # making ID consistent with better tracking
        self.final_id_last_seen = {}  # final_id -> frame_count
        self.final_id_positions = {}  # final_id -> {'tacticam' and 'broadcast': (x,y)}
        self.final_id_color_history = {}  # final_id -> {'tacticam': [hist1...], 'broadcast': [...]}
        self.frame_count = 0
        
        # for lost players
        self.lost_tacticam_ids = {}  # temp_id -> {'final_id': x, 'frames_lost': y, 'last_pos': (x,y)}
        self.lost_broadcast_ids = {}

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _get_average_histogram(self, hist_list):
        # calculating average histogram
        if not hist_list:
            return None
        return np.mean(hist_list, axis=0)

    def _update_color_history(self, final_id, view, histogram):
        # updating color history for player
        if final_id not in self.final_id_color_history:
            self.final_id_color_history[final_id] = {'tacticam': [], 'broadcast': []}
        
        history = self.final_id_color_history[final_id][view]
        history.append(histogram)
        
        # for keeping recent history
        if len(history) > self.color_history_size:
            history.pop(0)

    def _calculate_enhanced_cost(self, t_centers, t_histograms, transformed_b_centers, b_histograms):
        #Calculating cost matrix with enchanced features
        num_t, num_b = len(t_centers), len(transformed_b_centers)
        
        # spatial dist
        spatial_dist = np.linalg.norm(t_centers[:, np.newaxis] - transformed_b_centers, axis=2)
        
        # color dist
        color_dist = np.zeros((num_t, num_b))
        for i in range(num_t):
            for j in range(num_b):
                # Use correlation for color comparison (higher is better)
                correlation = cv2.compareHist(t_histograms[i], b_histograms[j], cv2.HISTCMP_CORREL)
                color_dist[i, j] = 1 - max(0, correlation)  # Convert to distance (lower is better)
        
        # normalizing spatial dist
        norm_spatial_dist = np.clip(spatial_dist / self.distance_threshold, 0, 2)
        
        # combining costs
        cost_matrix = (self.weight_spatial * norm_spatial_dist) + (self.weight_color * color_dist)
        
        return cost_matrix, spatial_dist

    def _try_reassign_lost_players(self, t_ids, t_centers, t_histograms, b_ids, transformed_b_centers, b_histograms):
        #reassigning players that were temporarily lost
        reassignments = {'tacticam': {}, 'broadcast': {}}
        
        # check lost tacticam players
        for temp_id, lost_info in list(self.lost_tacticam_ids.items()):
            if temp_id in t_ids:
                idx = t_ids.index(temp_id)
                current_pos = t_centers[idx]
                
                # check if position is reasonable compared to last known position
                pos_diff = np.linalg.norm(np.array(current_pos) - np.array(lost_info['last_pos']))
                if pos_diff < self.distance_threshold * 2:  # Allow some movement
                    # check color similarity with history
                    final_id = lost_info['final_id']
                    if final_id in self.final_id_color_history:
                        avg_hist = self._get_average_histogram(
                            self.final_id_color_history[final_id]['tacticam']
                        )
                        if avg_hist is not None:
                            color_sim = cv2.compareHist(t_histograms[idx], avg_hist, cv2.HISTCMP_CORREL)
                            if color_sim > 0.7:  # Good color match
                                reassignments['tacticam'][temp_id] = final_id
                                del self.lost_tacticam_ids[temp_id]
        
        # similar logic for broadcast players
        for temp_id, lost_info in list(self.lost_broadcast_ids.items()):
            if temp_id in b_ids:
                idx = b_ids.index(temp_id)
                current_pos = transformed_b_centers[idx]
                
                pos_diff = np.linalg.norm(np.array(current_pos) - np.array(lost_info['last_pos']))
                if pos_diff < self.distance_threshold * 2:
                    final_id = lost_info['final_id']
                    if final_id in self.final_id_color_history:
                        avg_hist = self._get_average_histogram(
                            self.final_id_color_history[final_id]['broadcast']
                        )
                        if avg_hist is not None:
                            color_sim = cv2.compareHist(b_histograms[idx], avg_hist, cv2.HISTCMP_CORREL)
                            if color_sim > 0.7:
                                reassignments['broadcast'][temp_id] = final_id
                                del self.lost_broadcast_ids[temp_id]
        
        return reassignments

    def _cleanup_old_data(self):
        # clean up data for players not seen for a long time
        # remove old lost players
        for temp_id in list(self.lost_tacticam_ids.keys()):
            if self.lost_tacticam_ids[temp_id]['frames_lost'] > self.id_persistence_frames:
                del self.lost_tacticam_ids[temp_id]
        
        for temp_id in list(self.lost_broadcast_ids.keys()):
            if self.lost_broadcast_ids[temp_id]['frames_lost'] > self.id_persistence_frames:
                del self.lost_broadcast_ids[temp_id]
        
        # update frames lost counter
        for lost_info in self.lost_tacticam_ids.values():
            lost_info['frames_lost'] += 1
        for lost_info in self.lost_broadcast_ids.values():
            lost_info['frames_lost'] += 1

    def map_players(self, tacticam_players, frame_t, broadcast_players, frame_b, homography_matrix):
        self.frame_count += 1
        
        if not tacticam_players or not broadcast_players:
            self._cleanup_old_data()
            return self.tacticam_to_final_id, self.broadcast_to_final_id

        # extract data
        t_ids = [p[0] for p in tacticam_players]
        t_bboxes = [p[1] for p in tacticam_players]
        t_centers = np.array([self._get_center(bbox) for bbox in t_bboxes])
        t_histograms = [get_color_histogram(frame_t, bbox) for bbox in t_bboxes]

        b_ids = [p[0] for p in broadcast_players]
        b_bboxes = [p[1] for p in broadcast_players]
        b_positions = np.array([((bbox[0] + bbox[2]) / 2, bbox[3]) for bbox in b_bboxes], 
                              dtype="float32").reshape(-1, 1, 2)
        b_histograms = [get_color_histogram(frame_b, bbox) for bbox in b_bboxes]
        
        transformed_b_centers = cv2.perspectiveTransform(b_positions, homography_matrix).reshape(-1, 2)
        
        # try to reassign lost players first
        reassignments = self._try_reassign_lost_players(
            t_ids, t_centers, t_histograms, b_ids, transformed_b_centers, b_histograms
        )
        
        # apply reassignments
        for temp_id, final_id in reassignments['tacticam'].items():
            self.tacticam_to_final_id[temp_id] = final_id
            self.final_id_last_seen[final_id] = self.frame_count
        
        for temp_id, final_id in reassignments['broadcast'].items():
            self.broadcast_to_final_id[temp_id] = final_id
            self.final_id_last_seen[final_id] = self.frame_count

        # calculate cost matrix for remaining assignments
        cost_matrix, spatial_dist = self._calculate_enhanced_cost(
            t_centers, t_histograms, transformed_b_centers, b_histograms
        )
        
        # hungarian algorithm for optimal assignment
        t_indices, b_indices = linear_sum_assignment(cost_matrix)
        
        # process assignments
        current_assignments = set()
        
        for t_idx, b_idx in zip(t_indices, b_indices):
            if spatial_dist[t_idx, b_idx] > self.distance_threshold:
                continue
                
            t_id = t_ids[t_idx]
            b_id = b_ids[b_idx]
            
            # skip if already assigned in this frame
            if (t_id, b_id) in current_assignments:
                continue
            
            # check for existing final IDs
            t_final_id = self.tacticam_to_final_id.get(t_id)
            b_final_id = self.broadcast_to_final_id.get(b_id)
            
            if t_final_id is not None and b_final_id is not None:
                if t_final_id == b_final_id:
                    #perfect match - same final ID
                    final_id = t_final_id
                else:
                    # conflict - choose the more recently seen ID
                    t_last_seen = self.final_id_last_seen.get(t_final_id, 0)
                    b_last_seen = self.final_id_last_seen.get(b_final_id, 0)
                    
                    if t_last_seen >= b_last_seen:
                        final_id = t_final_id
                        self.broadcast_to_final_id[b_id] = final_id
                    else:
                        final_id = b_final_id
                        self.tacticam_to_final_id[t_id] = final_id
            elif t_final_id is not None:
                final_id = t_final_id
                self.broadcast_to_final_id[b_id] = final_id
            elif b_final_id is not None:
                final_id = b_final_id
                self.tacticam_to_final_id[t_id] = final_id
            else:
                # new player pair
                final_id = self.next_final_id
                self.next_final_id += 1
                self.tacticam_to_final_id[t_id] = final_id
                self.broadcast_to_final_id[b_id] = final_id
            
            # ipdate tracking information
            self.final_id_last_seen[final_id] = self.frame_count
            self.final_id_positions[final_id] = {
                'tacticam': tuple(t_centers[t_idx]),
                'broadcast': tuple(transformed_b_centers[b_idx])
            }
            
            # update color history
            self._update_color_history(final_id, 'tacticam', t_histograms[t_idx])
            self._update_color_history(final_id, 'broadcast', b_histograms[b_idx])
            
            current_assignments.add((t_id, b_id))
        
        # handle players that disappeared (for potential reassignment later)
        for temp_id in list(self.tacticam_to_final_id.keys()):
            if temp_id not in t_ids:
                final_id = self.tacticam_to_final_id[temp_id]
                if final_id in self.final_id_positions:
                    self.lost_tacticam_ids[temp_id] = {
                        'final_id': final_id,
                        'frames_lost': 0,
                        'last_pos': self.final_id_positions[final_id]['tacticam']
                    }
                del self.tacticam_to_final_id[temp_id]
        
        for temp_id in list(self.broadcast_to_final_id.keys()):
            if temp_id not in b_ids:
                final_id = self.broadcast_to_final_id[temp_id]
                if final_id in self.final_id_positions:
                    self.lost_broadcast_ids[temp_id] = {
                        'final_id': final_id,
                        'frames_lost': 0,
                        'last_pos': self.final_id_positions[final_id]['broadcast']
                    }
                del self.broadcast_to_final_id[temp_id]
        
        self._cleanup_old_data()
        
        return self.tacticam_to_final_id, self.broadcast_to_final_id