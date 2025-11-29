import numpy as np
from collections import defaultdict
import math

class GOR:
    def __init__(self, window_size=17):
        self.window = window_size 
        self.half_win = window_size // 2
        # matrix[offset][amino_acid][structure]
        self.model_stats = None 
        self.total_s = {'H': 0, 'E': 0, 'C': 0} # Total count of each structure
        self.p_s = {'H': 0, 'E': 0, 'C': 0}     # Probability of each structure

    def train(self, sequences, labels):
        # Initialize counters
        # Position relative to center: -8 to +8
        counts = {pos: defaultdict(lambda: {'H': 0, 'E': 0, 'C': 0}) 
                  for pos in range(-self.half_win, self.half_win + 1)}
        
        total_residues = 0

        for seq, lab in zip(sequences, labels):
            length = len(seq)
            for i in range(length):
                struct_center = lab[i]
                self.total_s[struct_center] += 1
                total_residues += 1
                
                # Check neighbors
                for offset in range(-self.half_win, self.half_win + 1):
                    idx = i + offset
                    if 0 <= idx < length:
                        aa = seq[idx]
                        counts[offset][aa][struct_center] += 1

        self.model_stats = counts
        
        # Calculate background probabilities
        for s in ['H', 'E', 'C']:
            if total_residues > 0:
                self.p_s[s] = self.total_s[s] / total_residues

    def predict(self, sequence):
        length = len(sequence)
        raw_prediction = []
        
        # 1. Raw GOR Prediction
        for i in range(length):
            scores = {'H': 0.0, 'E': 0.0, 'C': 0.0}
            for s in ['H', 'E', 'C']:
                if self.p_s[s] > 0:
                    scores[s] = math.log(self.p_s[s])
                for offset in range(-self.half_win, self.half_win + 1):
                    idx = i + offset
                    if 0 <= idx < length:
                        aa = sequence[idx]
                        count_s_r = self.model_stats[offset][aa][s]
                        denom = self.total_s[s]
                        if count_s_r > 0 and denom > 0:
                            propensity = count_s_r / denom
                            scores[s] += math.log(propensity)
                        else:
                            scores[s] += -10.0 
            raw_prediction.append(max(scores, key=scores.get))
            
        # 2. Post-Processing (Smoothing)
        # Rule: If a residue is X, but neighbors are Y, switch to Y.
        # This fixes "speckled" predictions like H-C-H -> H-H-H
        smoothed_prediction = list(raw_prediction)
        
        # Simple majority filter window of 3
        for i in range(1, length - 1):
            prev_s = raw_prediction[i-1]
            curr_s = raw_prediction[i]
            next_s = raw_prediction[i+1]
            
            if prev_s == next_s and curr_s != prev_s:
                smoothed_prediction[i] = prev_s
                
        return "".join(smoothed_prediction)
        length = len(sequence)
        prediction = []
        
        for i in range(length):
            scores = {'H': 0.0, 'E': 0.0, 'C': 0.0}
            
            for s in ['H', 'E', 'C']:
                # Start with the log probability of the structure (Prior)
                if self.p_s[s] > 0:
                    scores[s] = math.log(self.p_s[s])
                
                # Add information from neighbors (Likelihood)
                for offset in range(-self.half_win, self.half_win + 1):
                    idx = i + offset
                    if 0 <= idx < length:
                        aa = sequence[idx]
                        
                        # Get count of (AA at offset | Structure)
                        count_s_r = self.model_stats[offset][aa][s]
                        
                        # Get total count of that Structure (Normalization factor)
                        denom = self.total_s[s]
                        
                        # FIX: Normalize by the total count of the structure
                        # This calculates P(AA | Structure) instead of just Raw Count
                        if count_s_r > 0 and denom > 0:
                            propensity = count_s_r / denom
                            scores[s] += math.log(propensity)
                        else:
                            # Penalty for zero occurrence
                            scores[s] += -10.0 
            
            prediction.append(max(scores, key=scores.get))
            
        return "".join(prediction)