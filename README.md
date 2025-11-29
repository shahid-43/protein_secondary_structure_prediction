# Study and Comparison of Algorithms for Protein’s Secondary Structure Detection

**Course:** CAP 5510 – BIOINFORMATICS (Fall 2025)  
**Team Members:** Shahid Shareef Mohammad, Dhanush Kumar Reddy Gujjula, Pranay Reddy Pullaiahgari

## Abstract
This project compares three distinct approaches to Protein Secondary Structure Prediction (PSSP) using the CB513 dataset. We map amino acid sequences to secondary structures (Helix, Sheet, Coil). The methods evaluated are:
1.  **Chou-Fasman:** A classical rule-based empirical method.
2.  **GOR (Garnier-Osguthorpe-Robson):** An information-theoretic method.
3.  **BiLSTM:** A modern Deep Learning approach using Bidirectional LSTMs.

## Dataset
We use the **CB513** dataset.
*   **Preprocessing:** The dataset's original DSSP Q8 labels (H, G, I, E, B, T, S, L) are reduced to Q3 labels:
    *   **Helix (H):** H, G, I
    *   **Sheet (E):** E, B
    *   **Coil (C):** T, S, L, (and others)

## Requirements
Install dependencies using:
`pip install -r requirements.txt`

## Usage
1.  Place the `cb513.csv` file in the `data/` folder.
2.  Run the main comparison script:
    `python main.py`

## Methodology
*   **Chou-Fasman:** Implemented from scratch. Calculates conformational parameters (P_alpha, P_beta) from the training data and applies heuristic rules for nucleation and extension.
*   **GOR III:** Implemented from scratch. Uses a sliding window (typically +/- 8 residues) to calculate the information difference (log-odds) for each state based on neighbors.
*   **BiLSTM:** Implemented using TensorFlow/Keras. Uses an Embedding layer, Bidirectional LSTM layers to capture long-range dependencies, and a Dense output layer.