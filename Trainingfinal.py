import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import librosa
import time
from tqdm import tqdm
import edit_distance as ed
import matplotlib.pyplot as plt

from model.configs import SR, device_name, UNQ_CHARS, INPUT_DIM, MODEL_NAME, NUM_UNQ_CHARS
from model.utils import batchify, clean_single_wav, gen_mfcc
from model.final_model import get_model


# ------------------ TRAIN FUNCTION ------------------
def train_model(
    model,
    optimizer,
    train_wavs,
    train_texts,
    test_wavs,
    test_texts,
    epochs=50,
    batch_size=4
):

    train_losses = []
    val_losses = []
    val_cers = []

    with tf.device(device_name):

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = 0.0
            test_loss = 0.0
            test_CER = 0.0

            train_steps = 0
            test_steps = 0
            total_chars = 0

            print(f"\nEpoch {epoch + 1}/{epochs}")

            # -------- TRAINING --------
            for start in tqdm(range(0, len(train_wavs), batch_size)):
                end = min(start + batch_size, len(train_wavs))

                x, target, target_lengths, output_lengths = batchify(
                    train_wavs[start:end],
                    train_texts[start:end],
                    UNQ_CHARS
                )

                with tf.GradientTape() as tape:
                    output = model(x, training=True)
                    loss = K.ctc_batch_cost(
                        target,
                        output,
                        output_lengths,
                        target_lengths
                    )

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_weights)
                )

                train_loss += tf.reduce_mean(loss).numpy()
                train_steps += 1

            # -------- VALIDATION --------
            for start in tqdm(range(0, len(test_wavs), batch_size)):
                end = min(start + batch_size, len(test_wavs))

                x, target, target_lengths, output_lengths = batchify(
                    test_wavs[start:end],
                    test_texts[start:end],
                    UNQ_CHARS
                )

                output = model(x, training=False)

                loss = K.ctc_batch_cost(
                    target,
                    output,
                    output_lengths,
                    target_lengths
                )

                test_loss += tf.reduce_mean(loss).numpy()
                test_steps += 1

                # -------- CER --------
                input_len = np.ones(output.shape[0]) * output.shape[1]
                decoded = K.ctc_decode(
                    output,
                    input_length=input_len,
                    greedy=False,
                    beam_width=100
                )[0][0]

                target_indices = [
                    sent[sent != 0].tolist() for sent in target
                ]

                predicted_indices = [
                    sent[sent > 1].numpy().tolist()
                    for sent in decoded
                ]

                for pred, truth in zip(predicted_indices, target_indices):
                    sm = ed.SequenceMatcher(pred, truth)
                    test_CER += sm.distance()
                    total_chars += len(truth)

            # -------- METRICS --------
            train_loss /= train_steps
            test_loss /= test_steps
            test_CER /= total_chars

            train_losses.append(train_loss)
            val_losses.append(test_loss)
            val_cers.append(test_CER)

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {test_loss:.4f} | "
                f"Val CER: {test_CER * 100:.2f}% | "
                f"Time: {time.time() - start_time:.1f}s"
            )

    return train_losses, val_losses, val_cers


# ------------------ DATA LOADING ------------------
def load_data(wavs_dir, texts_dir):
    texts_df = pd.read_csv(texts_dir)

    wavs = []
    for fname in texts_df["file"]:
        wav, _ = librosa.load(
            f"{wavs_dir}/{fname}.flac",
            sr=SR
        )
        wavs.append(wav)

    texts = texts_df["text"].tolist()
    return wavs, texts


# ------------------ MAIN ------------------
if __name__ == "__main__":

    # -------- MODEL --------
    model = get_model(
        INPUT_DIM,
        NUM_UNQ_CHARS,
        model_name=MODEL_NAME
    )
    print("Model defined")

    # -------- OPTIMIZER --------
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        clipnorm=5.0
    )

    # -------- LOAD DATA --------
    print("Loading data...")
    wavs, texts = load_data(
        wavs_dir="dataset/wav_files(sampled)",
        texts_dir="dataset/transcriptions(sampled)/file_speaker_text(sampled).csv"
    )

    # -------- CLEAN AUDIO --------
    print("Cleaning audio...")
    wavs = [clean_single_wav(w) for w in wavs]

    print("Generating MFCCs...")
    wavs = [gen_mfcc(w) for w in wavs]

    # -------- SPLIT --------
    train_wavs, test_wavs, train_texts, test_texts = train_test_split(
        wavs,
        texts,
        test_size=0.2,
        random_state=42
    )

    # -------- TRAIN --------
    train_losses, val_losses, val_cers = train_model(
        model,
        optimizer,
        train_wavs,
        train_texts,
        test_wavs,
        test_texts,
        epochs=60,
        batch_size=4
    )

    # -------- PLOTS --------
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CTC Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs_range, val_cers)
    plt.xlabel("Epoch")
    plt.ylabel("CER")
    plt.title("Validation CER")
    plt.show()

    # -------- SAVE --------
    model.save("model/New_trained_model.h5")
    print("Model saved successfully")
