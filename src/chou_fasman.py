import numpy as np

class ChouFasman:
    def __init__(self):
        # ESTABLISHED PROPENSITY VALUES (Chou & Fasman, 1974/1978)
        # These are hardcoded as per the project proposal.
        
        self.P_ALPHA = {
            "A": 1.45, "C": 0.77, "D": 0.98, "E": 1.54, "F": 1.12,
            "G": 0.53, "H": 1.24, "I": 1.00, "K": 1.07, "L": 1.34,
            "M": 1.20, "N": 0.73, "P": 0.59, "Q": 1.17, "R": 0.79,
            "S": 0.79, "T": 0.82, "V": 1.06, "W": 1.14, "Y": 0.61,
            # Default for unknown amino acids
            "X": 1.0, "B": 1.0, "Z": 1.0, "J": 1.0
        }
        
        self.P_BETA = {
            "A": 0.97, "C": 1.30, "D": 0.80, "E": 0.26, "F": 1.28,
            "G": 0.81, "H": 0.71, "I": 1.60, "K": 0.74, "L": 1.22,
            "M": 1.67, "N": 0.65, "P": 0.62, "Q": 1.23, "R": 0.90,
            "S": 0.72, "T": 1.20, "V": 1.70, "W": 1.19, "Y": 1.29,
            # Default for unknown amino acids
            "X": 1.0, "B": 1.0, "Z": 1.0, "J": 1.0
        }

    def train(self, sequences, labels):
        """
        Since we are using established values, 'training' is not required.
        This function exists to maintain consistency with the other classes (GOR, BiLSTM).
        """
        print("Chou-Fasman: Using established propensity values (No training needed).")
        pass

    def predict(self, sequence):
        """
        Implements the classic Rule-Based prediction:
        1. Identify Helix Nucleation Sites (4 out of 6 residues have P_alpha > 1.0)
        2. Identify Sheet Nucleation Sites (3 out of 5 residues have P_beta > 1.0)
        3. Extension
        4. Conflict Resolution
        """
        length = len(sequence)
        
        # 1. Map sequence to propensity scores
        # Use .get(aa, 1.0) to handle any weird characters in the dataset
        score_h = [self.P_ALPHA.get(aa, 1.0) for aa in sequence]
        score_e = [self.P_BETA.get(aa, 1.0) for aa in sequence]
        
        # Arrays to mark potential structures (1=Yes, 0=No)
        is_helix = [0] * length
        is_sheet = [0] * length
        
        # --- STEP 1: Helix Nucleation & Extension ---
        # Rule: Find 6 residues where at least 4 have P_alpha > 1.00
        for i in range(length - 5):
            window = score_h[i:i+6]
            if sum(1 for s in window if s > 1.0) >= 4:
                # Nucleation found: Extend forward
                # (Simplified extension: extend while average > 1.0 or individual > 1.0)
                for k in range(i, i+6): is_helix[k] = 1
                
                # Extend Left
                left = i - 1
                while left >= 0 and score_h[left] > 1.0:
                    is_helix[left] = 1
                    left -= 1
                
                # Extend Right
                right = i + 6
                while right < length and score_h[right] > 1.0:
                    is_helix[right] = 1
                    right += 1

        # --- STEP 2: Sheet Nucleation & Extension ---
        # Rule: Find 5 residues where at least 3 have P_beta > 1.00
        for i in range(length - 4):
            window = score_e[i:i+5]
            if sum(1 for s in window if s > 1.0) >= 3:
                # Nucleation found
                for k in range(i, i+5): is_sheet[k] = 1
                
                # Extend Left
                left = i - 1
                while left >= 0 and score_e[left] > 1.0:
                    is_sheet[left] = 1
                    left -= 1
                
                # Extend Right
                right = i + 5
                while right < length and score_e[right] > 1.0:
                    is_sheet[right] = 1
                    right += 1

        # --- STEP 3: Conflict Resolution ---
        # If a region is both Helix and Sheet, the one with higher average score wins.
        final_pred = []
        for i in range(length):
            h = is_helix[i]
            e = is_sheet[i]
            
            if h and not e:
                final_pred.append('H')
            elif e and not h:
                final_pred.append('E')
            elif h and e:
                # Conflict! Compare scores.
                if score_h[i] >= score_e[i]:
                    final_pred.append('H')
                else:
                    final_pred.append('E')
            else:
                final_pred.append('C') # Default to Coil
                
        return "".join(final_pred)