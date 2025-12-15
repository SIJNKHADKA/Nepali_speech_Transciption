"""
Improved ASR Architecture
Based on: CNN + Residual Separable Conv + BiLSTM (CTC)
Compatible with existing training pipeline

Original Author: Manish Dhakal (2022)
Architecture Improved: 2025
"""

from model.configs import MODEL_NAME, INPUT_DIM, NUM_UNQ_CHARS

from tensorflow.keras import layers, Model, Input
import tensorflow.keras.backend as K
import numpy as np


# ------------------ Residual Separable CNN Block ------------------
def sep_res_block(x, filters, kernel_size, dropout):
    shortcut = x

    x = layers.SeparableConv1D(
        filters,
        kernel_size,
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.SeparableConv1D(
        filters,
        kernel_size,
        padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)

    return x


# ------------------ Main ASR Model ------------------
def get_model(
    ip_channel,
    num_classes,
    cnn_filters=128,
    cnn_kernel_size=11,
    num_res_blocks=6,
    num_rnn_layers=3,
    rnn_dim=256,
    dense_dim=256,
    dropout=0.2,
    model_name=None
):

    inputs = Input(shape=(None, ip_channel), name="input_features")

    # -------- CNN Frontend (Temporal Downsampling) --------
    x = layers.Conv1D(
        cnn_filters,
        cnn_kernel_size,
        strides=2,
        padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(
        cnn_filters,
        cnn_kernel_size,
        strides=2,
        padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # -------- Residual Separable CNN Blocks --------
    for _ in range(num_res_blocks):
        x = sep_res_block(
            x,
            filters=cnn_filters,
            kernel_size=cnn_kernel_size,
            dropout=dropout
        )

    # -------- BiLSTM Encoder --------
    for _ in range(num_rnn_layers):
        x = layers.Bidirectional(
            layers.LSTM(
                rnn_dim,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=0.0
            )
        )(x)
        x = layers.LayerNormalization()(x)

    # -------- Dense Projection --------
    x = layers.Dense(dense_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    logits = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax", name="softmax")(logits)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


# ------------------ Sanity Check ------------------
if __name__ == "__main__":

    model = get_model(
        INPUT_DIM,
        NUM_UNQ_CHARS,
        model_name=MODEL_NAME
    )

    x = np.random.rand(2, 100, INPUT_DIM)
    y = model(x)

    print("Output shape:", y.shape)
    model.summary()
