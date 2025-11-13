from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers


def _residual_conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dropout: float = 0.0,
    name: Optional[str] = None,
) -> tf.Tensor:
    shortcut = x
    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{name}_conv1" if name else None,
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1" if name else None)(x)
    x = layers.Activation("relu", name=f"{name}_relu1" if name else None)(x)

    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"{name}_conv2" if name else None,
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2" if name else None)(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{name}_shortcut" if name else None,
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn" if name else None)(shortcut)

    x = layers.Add(name=f"{name}_add" if name else None)([x, shortcut])
    x = layers.Activation("relu", name=f"{name}_relu2" if name else None)(x)

    if dropout > 0.0:
        x = layers.Dropout(dropout, name=f"{name}_drop" if name else None)(x)
    return x


def _temporal_attention(x: tf.Tensor, name: Optional[str] = None) -> tf.Tensor:
    score = layers.Dense(128, activation="tanh", name=f"{name}_dense" if name else None)(x)
    score = layers.Dense(1, activation=None, name=f"{name}_score" if name else None)(score)
    weights = layers.Softmax(axis=1, name=f"{name}_weights" if name else None)(score)
    weighted = layers.Multiply(name=f"{name}_weighted" if name else None)([x, weights])
    context = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1),
        name=f"{name}_context" if name else None,
    )(weighted)
    return context


def build_squat_encoder(
    input_shape: tuple[int, int],
    embedding_dropout: float = 0.0,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="squat_input")
    x = layers.Masking(mask_value=0.0, name="masking")(inputs)
    x = layers.LayerNormalization(name="input_ln")(x)

    x = _residual_conv_block(x, filters=128, kernel_size=7, dropout=0.1, name="resblock1")
    x = layers.MaxPooling1D(pool_size=2, name="maxpool1")(x)
    x = layers.SpatialDropout1D(0.1, name="spatial_drop1")(x)
    x = _residual_conv_block(x, filters=192, kernel_size=5, dropout=0.15, name="resblock2")
    x = layers.MaxPooling1D(pool_size=2, name="maxpool2")(x)
    x = layers.SpatialDropout1D(0.1, name="spatial_drop2")(x)

    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.1),
        name="bigru",
    )(x)
    x = _temporal_attention(x, name="attention")

    x = layers.Dense(192, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="embedding")(x)
    if embedding_dropout > 0.0:
        x = layers.Dropout(embedding_dropout, name="embedding_dropout")(x)

    return models.Model(inputs=inputs, outputs=x, name="squat_encoder_v8")


def build_squat_classifier(
    input_shape: tuple[int, int],
    num_classes: int,
    dropout: float = 0.3,
) -> tf.keras.Model:
    encoder = build_squat_encoder(input_shape, embedding_dropout=dropout)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(encoder.output)
    model = models.Model(inputs=encoder.input, outputs=outputs, name="squat_classifier_v8")
    return model
