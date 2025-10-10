import os
import time
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import edit_distance as ed

import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

from model.configs import SR, UNQ_CHARS, INPUT_DIM, MODEL_NAME, NUM_UNQ_CHARS
from model.utils import CER_from_mfccs, batchify, clean_single_wav, gen_mfcc, indices_from_texts, load_model
from model.model import get_model

# ===================== GPU SETUP =====================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" GPU detected: {gpus[0].name}")
    except RuntimeError as e:
        print(f" GPU setup error: {e}")
else:
    print(" No GPU detected. Using CPU.")

# ===================== TRAINING FUNCTION =====================
def train_model(model, optimizer, train_wavs, train_texts, test_wavs, test_texts, epochs=100, batch_size=50):
    train_losses, test_losses, test_CERs = [], [], []

    for e in range(epochs):
        start_time = time.time()
        len_train, len_test = len(train_wavs), len(test_wavs)
        train_loss = test_loss = test_CER = 0
        train_batch_count = test_batch_count = 0

        print(f"\n==================== Epoch {e+1}/{epochs} ====================")
        print("Training...")

        # ---------------- TRAINING LOOP ----------------
        for start in tqdm(range(0, len_train, batch_size)):
            end = min(start + batch_size, len_train)
            x, target, target_lengths, output_lengths = batchify(train_wavs[start:end], train_texts[start:end], UNQ_CHARS)

            with tf.GradientTape() as tape:
                output = model(x, training=True)
                loss = K.ctc_batch_cost(target, output, output_lengths, target_lengths)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_loss += np.mean(loss.numpy())
            train_batch_count += 1

        print("Validating...")
        for start in tqdm(range(0, len_test, batch_size)):
            end = min(start + batch_size, len_test)
            x, target, target_lengths, output_lengths = batchify(test_wavs[start:end], test_texts[start:end], UNQ_CHARS)

            output = model(x, training=False)
            loss = K.ctc_batch_cost(target, output, output_lengths, target_lengths)
            test_loss += np.mean(loss.numpy())
            test_batch_count += 1

            # Compute CER
            input_len = np.ones(output.shape[0]) * output.shape[1]
            decoded_indices = K.ctc_decode(output, input_length=input_len, greedy=False, beam_width=100)[0][0]
            target_indices = [sent[sent != 0].tolist() for sent in target]
            predicted_indices = [sent[sent > 1].numpy().tolist() for sent in decoded_indices]

            batch_cer = 0
            for pred, truth in zip(predicted_indices, target_indices):
                sm = ed.SequenceMatcher(pred, truth)
                batch_cer += sm.distance() / max(len(truth), 1)
            test_CER += batch_cer / (end - start)

        # ---------------- END OF EPOCH ----------------
        train_loss /= train_batch_count
        test_loss /= test_batch_count
        test_CER /= test_batch_count

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_CERs.append(test_CER)

        print(f"Epoch {e+1}: Train Loss={train_loss:.4f}, Val Loss={test_loss:.4f}, CER={test_CER*100:.2f}%, Time={time.time()-start_time:.2f}s")

    # =================== PLOTTING ===================
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(test_losses, label="Validation Loss", linewidth=2)
    plt.title("Training vs Validation Loss Curve", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("CTC Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(np.array(test_CERs) * 100, label="Validation CER (%)", color='orange', linewidth=2)
    plt.title("Validation CER over Epochs", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Character Error Rate (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cer_curve.png", dpi=200)
    plt.show()

# ===================== LOAD DATA FUNCTION =====================
def load_data(wavs_dir, texts_dir):
    texts_df = pd.read_csv(texts_dir)
    train_wavs = []
    for f_name in texts_df["file"]:
        wav_path = os.path.join(wavs_dir, f_name + ".wav")
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        wav, _ = librosa.load(wav_path, sr=SR)
        train_wavs.append(wav)
    train_texts = texts_df["text"].tolist()
    return train_wavs, train_texts

# ===================== MAIN SCRIPT =====================
if __name__ == "__main__":
    # Define model
    model = get_model(INPUT_DIM, NUM_UNQ_CHARS, num_res_blocks=5, num_cnn_layers=2,
                      cnn_filters=50, cnn_kernel_size=15, rnn_dim=170, rnn_dropout=0.15, num_rnn_layers=2,
                      num_dense_layers=1, dense_dim=340, model_name=MODEL_NAME, rnn_type="lstm",
                      use_birnn=True)
    print("✅ Model defined\n")

    optimizer = tf.keras.optimizers.Adam()

    # Load dataset
    print("Loading data.....")
    train_wavs, train_texts = load_data(
        wavs_dir="ne_np_female/wavs",
        texts_dir="ne_np_female_dataset.csv"
    )
    print("✅ Data loaded\n")

    # Clean audio files
    print("Cleaning audio files.....")
    train_wavs = [clean_single_wav(wav) for wav in train_wavs]
    print("✅ Audio cleaned\n")

    # Generate MFCC features
    print("Generating MFCC features.....")
    train_wavs = [gen_mfcc(wav) for wav in train_wavs]
    print("✅ MFCC features generated\n")

    # Train-test split
    train_wavs, test_wavs, train_texts, test_texts = train_test_split(train_wavs, train_texts, test_size=0.2)

    # Train model
    train_model(model, optimizer, train_wavs, train_texts, test_wavs, test_texts, epochs=60, batch_size=2)

    # Save model
    os.makedirs("model", exist_ok=True)
    model.save("model/trained_model_with_curves.h5")
    print("✅ Model saved")
