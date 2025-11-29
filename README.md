# Study and Comparison of Algorithms for Protein Secondary Structure Prediction

**Course:** CAP 5510 – BIOINFORMATICS (Fall 2025)  
**Team Members:**
*   **Shahid Shareef Mohammad** (UF-ID: 8438-1774)
*   **Dhanush Kumar Reddy Gujjula** (UF-ID: 8451-5185)
*   **Pranay Reddy Pullaiahgari** (UF-ID: 6238-1134)

---

## 1. Abstract
Proteins are the fundamental machinery of life, and their function is dictated by their 3D structure. A critical intermediate step in determining this structure is **Protein Secondary Structure Prediction (PSSP)**, which involves mapping a primary amino acid sequence to a sequence of structural states: **Helix (H)**, **Sheet (E)**, and **Coil (C)**.

This project implements and compares three distinct generations of prediction algorithms to demonstrate the evolution of the field:
1.  **Chou-Fasman (1974):** A classical empirical method based on single-residue propensities.
2.  **GOR (Garnier-Osguthorpe-Robson):** An information-theoretic method utilizing sliding windows to capture local context.
3.  **BiLSTM (Deep Learning):** A modern Bidirectional Long Short-Term Memory network capable of learning complex, non-linear patterns and long-range dependencies.

---

## 2. Dataset Details
We utilize the **CB513** dataset, a widely accepted benchmark for secondary structure prediction.

*   **Source:** [Kaggle / Princeton CB513](https://www.kaggle.com/datasets/moklesur/cb513-dataset-for-protein-structure-prediction)
*   **Data Size:** 513 protein sequences (after cleaning/filtering).
*   **Preprocessing Pipeline:**
    1.  **Extraction:** Sequences are extracted from the `input` column.
    2.  **Label Mapping (Q8 $\to$ Q3):** The original DSSP 8-state labels (`dssp8`) are reduced to 3 states using the standard reduction method:
        *   **Helix (H):** `H` (Alpha helix), `G` (3-10 helix), `I` (Pi helix).
        *   **Sheet (E):** `E` (Beta bridge), `B` (Beta ladder).
        *   **Coil (C):** `T` (Turn), `S` (Bend), `L` (Loop), and others.
    3.  **Splitting:** 80% Training, 20% Testing (stratified by sequence availability).

---

## 3. Implementation Methodology

### A. Chou-Fasman Algorithm (Rule-Based)
This method relies on the hypothesis that certain amino acids are "helix-formers" or "sheet-formers."

*   **Propensity Parameters:** Instead of calculating propensities from the small dataset, we implemented the **established historical values** defined by Chou & Fasman (1974).
    *   *Example:* Glutamic Acid (E) is a strong helix former ($P_{\alpha} = 1.54$).
    *   *Example:* Valine (V) is a strong sheet former ($P_{\beta} = 1.70$).
*   **Algorithm Logic:**
    1.  **Nucleation (Helix):** Scan for a 6-residue window where at least 4 residues have $P_{\alpha} > 1.0$.
    2.  **Nucleation (Sheet):** Scan for a 5-residue window where at least 3 residues have $P_{\beta} > 1.0$.
    3.  **Extension:** Extend the nucleation sites in both directions until the average propensity drops below 1.0.
    4.  **Conflict Resolution:** If a region is predicted as both Helix and Sheet, the structure with the higher average propensity score wins.

### B. GOR Algorithm (Information Theory)
The GOR method improves upon Chou-Fasman by acknowledging that the folding of an amino acid is influenced by its neighbors.

*   **Logic:** We calculate the information difference (log-odds) favoring a structure $S$ based on a window of residues $R$.
    $$ \text{Score}(S) = \log(P(S)) + \sum_{j=-8}^{+8} \log \left( \frac{P(R_j | S)}{P(R_j)} \right) $$
*   **Window Size:** We utilize a sliding window of **17 residues** (central residue $\pm 8$ neighbors).
*   **Normalization:** Raw counts are normalized by the total frequency of each structure to prevent majority-class bias (where the model blindly predicts "Coil" because it is the most common).
*   **Smoothing (Post-Processing):** A cleanup phase is applied to eliminate physically unrealistic "singlets" (e.g., a sequence `H-C-H` is smoothed to `H-H-H`).

### C. BiLSTM (Deep Learning)
This model treats protein folding as a sequence-to-sequence classification problem.

*   **Architecture:**
    1.  **Input Layer:** Integer-encoded amino acid sequences.
    2.  **Embedding Layer (128 dim):** Converts integers to dense vectors. **Masking (`mask_zero=True`)** is enabled to force the model to ignore zero-padding, which is critical for accurate variable-length prediction.
    3.  **Bidirectional LSTM (2 Layers, 128 units):** Processes the sequence from N-terminus to C-terminus and vice-versa simultaneously. This captures dependencies from both "past" and "future" residues.
    4.  **Dense Layer (64 units):** ReLU activation for feature extraction.
    5.  **Output Layer:** Softmax activation returning probabilities for classes [H, E, C].
*   **Training Strategy:**
    *   **Optimizer:** Adam (Learning Rate = 0.001).
    *   **Callbacks:** `EarlyStopping` (prevents overfitting) and `ReduceLROnPlateau` (fine-tunes weights when loss stagnates).
*   **Inference Logic:** To handle padding bias, predictions are **truncated** to the exact length of the original protein sequence before evaluation.

---




## 4. Project Directory Structure
```text
Protein_Secondary_Structure_Prediction/
│
├── data/
│   └── cb513.csv            # Raw dataset
│
├── models/
│   └── bilstm_model.keras   # Saved trained model (Auto-generated)
│
├── reports/
│   ├── bilstm_loss_curve.png # Training convergence graph
│   ├── results_BiLSTM.png    # Confusion Matrix
│   ├── results_GOR.png       # Confusion Matrix
│   └── results_Chou-Fasman.png
│
├── src/
│   ├── preprocessing.py     # Data loading, Q8->Q3 mapping, One-Hot Encoding
│   ├── chou_fasman.py       # Implementation of C-F Rules
│   ├── gor.py               # Implementation of GOR Information Theory
│   ├── bilstm_model.py      # TensorFlow Model Definition & Training
│   └── evaluation.py        # Metrics (Accuracy, F1, Heatmaps)
│
├── main.py                  # Entry point for the application
└── requirements.txt         # Dependencies


### 5. Results and Analysis
We evaluated all models on the same test split of the CB513 dataset.

| Metric | Chou-Fasman | GOR (Smoothed) | BiLSTM (Enhanced) |
| :--- | :---: | :---: | :---: |
| **Q3 Accuracy** | **~53.2%** | **~59.4%** | **~64.5%** |
| **Helix F1** | 0.50 | 0.61 | 0.64 |
| **Sheet F1** | 0.47 | 0.47 | 0.54 |
| **Coil F1** | 0.60 | 0.64 | 0.70 |

### Key Observations
*   **Baseline Performance:** Chou-Fasman provides a baseline of ~53%. Its lower accuracy highlights the limitation of analyzing residues in isolation.
*   **Impact of Context:** GOR improves accuracy by ~6% simply by looking at neighbors, proving that local sequence context is a strong determinant of structure.
*   **Deep Learning Superiority:** The BiLSTM achieves the highest accuracy (~64.5%). The significant improvement in **Sheet (E)** detection (Recall increased from 0.13 to 0.50 after masking) demonstrates the model's ability to learn long-range dependencies that statistical methods miss.
*   **Theoretical Limits:** While modern state-of-the-art tools achieve >80%, they rely on **PSSMs (Evolutionary Profiles)**. Using *only* the primary sequence (as done in this project), the theoretical limit is often cited around 65-70%, which our BiLSTM successfully approaches.
