from __future__ import annotations

from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from .modeling import build_squat_encoder


def build_adapter_wrapped_classifier(
	input_shape: tuple[int, int],
	num_classes: int,
	pretrained_encoder_path: str | Path,
	dropout: float = 0.5,
	adapter_filters: int = 6,
) -> tf.keras.Model:
	"""
	Build classifier for 18-ch input by inserting a 1x1 Conv adapter (18->6)
	in front of a pretrained 6-ch encoder.
	"""
	# Load 6-ch encoder
	pretrained = tf.keras.models.load_model(pretrained_encoder_path)
	# Freeze names alignment not required; we wrap as a subgraph

	inputs = layers.Input(shape=input_shape, name="squat_input_18")
	# 1x1 Conv adapter to map 18 -> 6
	x = layers.Conv1D(
		adapter_filters,
		kernel_size=1,
		padding="same",
		use_bias=False,
		kernel_regularizer=regularizers.l2(1e-4),
		name="channel_adapter",
	)(inputs)
	x = layers.BatchNormalization(name="channel_adapter_bn")(x)
	# Pass through pretrained 6ch encoder (expects shape (T, 6))
	x = pretrained(x)
	# Classifier head
	if dropout and dropout > 0.0:
		x = layers.Dropout(dropout)(x)
	outputs = layers.Dense(
		num_classes,
		activation="softmax",
		kernel_regularizer=regularizers.l2(1e-4),
		name="classifier",
	)(x)
	return models.Model(inputs=inputs, outputs=outputs, name="squat_classifier_adapter_v1")


