import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Mapping Q8 to Q3 based on standard DSSP definitions
# H = Helix (H, G, I)
# E = Sheet (E, B)
# C = Coil (T, S, L, and others)
Q8_TO_Q3 = {
    'H': 'H', 'G': 'H', 'I': 'H',  # Helix variants
    'E': 'E', 'B': 'E',            # Sheet variants
    'T': 'C', 'S': 'C', 'L': 'C',  # Coil variants
    '_': 'C', ' ': 'C', 'X': 'C'   # Handling placeholders/unknowns
}

def load_and_process_data(filepath):
    """
    Loads CB513 dataset and converts Q8 labels to Q3.
    Columns used: 'input' (sequence) and 'dssp8' (structure).
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return [], [], [], []

    processed_seqs = []
    processed_labels = []

    print(f"Raw dataset size: {len(df)}")

    for idx, row in df.iterrows():
        # 1. Get the Amino Acid Sequence
        seq = row['input']
        
        # 2. Get the Q8 Structure Label
        # We use dssp8 to fulfill the proposal requirement of converting Q8 -> Q3 manually.
        raw_label = row['dssp8'] 
        
        # 3. Convert Q8 to Q3
        # We iterate through the raw label string and map every character.
        # If a character isn't in our dict, default to 'C' (Coil).
        mapped_label = "".join([Q8_TO_Q3.get(x, 'C') for x in raw_label])
        
        # 4. Validation
        # Ensure sequence and label lengths match before adding to dataset
        if len(seq) == len(mapped_label):
            processed_seqs.append(seq)
            processed_labels.append(mapped_label)
        else:
            print(f"Skipping index {idx}: Length mismatch (Seq: {len(seq)}, Label: {len(mapped_label)})")

    print(f"Processed {len(processed_seqs)} valid sequences.")

    # 5. Split into Train and Test sets (80% Train, 20% Test)
    return train_test_split(processed_seqs, processed_labels, test_size=0.2, random_state=42)

def one_hot_encode(sequences, max_len=None):
    """
    Helper function to One-Hot Encode sequences for the BiLSTM model.
    """
    # Standard Amino Acid Vocabulary
    aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
    # Create a mapping: A->1, C->2, ... (0 is reserved for padding)
    char_to_int = {c: i+1 for i, c in enumerate(aa_vocab)}
    
    encoded_seqs = []
    for seq in sequences:
        # Convert chars to integers. Unknown chars get 0.
        encoded = [char_to_int.get(aa, 0) for aa in seq]
        encoded_seqs.append(encoded)
    
    # Determine max length for padding if not provided
    if not max_len:
        max_len = max(len(s) for s in sequences)
        
    # Pad sequences so they are all the same length for the Neural Network
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    # padding='post' adds zeros at the end
    padded_seqs = pad_sequences(encoded_seqs, maxlen=max_len, padding='post')
    
    return padded_seqs, max_len, len(aa_vocab)+1