import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

class BiLSTM_PSSP:
    def __init__(self, vocab_size, max_len):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # CRITICAL FIX: mask_zero=True tells the LSTM to ignore padding (0s)
        model.add(Embedding(input_dim=self.vocab_size, output_dim=128, input_length=self.max_len, mask_zero=True))
        
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0)))
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0)))
        
        model.add(TimeDistributed(Dense(64, activation='relu')))
        model.add(TimeDistributed(Dense(3, activation='softmax'))) 
        
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=40):
        y_encoded = self._encode_labels(y_train)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
        
        print("Training with Masking enabled...")
        self.model.fit(
            X_train, y_encoded, 
            epochs=epochs, 
            batch_size=32, # Smaller batch size often helps generalization
            validation_split=0.15,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

    def predict(self, X_test, original_lengths):
        """
        Predicts and TRUNCATES the output to the original sequence length.
        """
        probs = self.model.predict(X_test)
        preds = []
        map_inv = {0: 'H', 1: 'E', 2: 'C'}
        
        for i in range(len(X_test)):
            # Get the raw probabilities
            p_indices = np.argmax(probs[i], axis=1)
            
            # CRITICAL FIX: Only take the first N characters, where N is the real length
            real_len = original_lengths[i]
            valid_indices = p_indices[:real_len]
            
            pred_str = "".join([map_inv[x] for x in valid_indices])
            preds.append(pred_str)
            
        return preds

    def _encode_labels(self, labels):
        map_dict = {'H': 0, 'E': 1, 'C': 2}
        encoded = []
        for lab in labels:
            nums = [map_dict.get(c, 2) for c in lab]
            if len(nums) < self.max_len:
                nums += [2] * (self.max_len - len(nums))
            encoded.append(nums)
        return to_categorical(np.array(encoded), num_classes=3)