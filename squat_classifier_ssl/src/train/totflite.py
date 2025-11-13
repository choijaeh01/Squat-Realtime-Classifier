from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf

from .constants import NUM_CLASSES, NUM_FEATURES, FEATURE_OFFSET
from .modeling import build_squat_classifier
from .data_utils import resample_sequence


def _load_config(config_path: Optional[Path]) -> dict:
	if not config_path:
		return {}
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	return json.loads(config_path.read_text(encoding="utf-8"))


def _make_representative_dataset(
	rep_dir: Optional[Path],
	scaler_path: Optional[Path],
	target_len: int,
	num_features: int,
	per_window_zscore: bool,
) -> Optional[tf.lite.RepresentativeDataset]:
	if rep_dir is None or not rep_dir.exists():
		return None
	if scaler_path is None or not scaler_path.exists():
		print("[INT8] scaler_path is required for representative dataset. Skipping INT8 calibration.")
		return None
	try:
		import joblib  # noqa: WPS433
		import pandas as pd  # noqa: WPS433
	except Exception:
		print("[INT8] joblib/pandas not available. Skipping INT8 calibration.")
		return None

	scaler = joblib.load(scaler_path)
	csv_files = sorted(rep_dir.rglob("*.csv"))[:500]
	if not csv_files:
		print("[INT8] No CSV found under rep_data_dir. Skipping INT8 calibration.")
		return None

	def gen() -> Iterator[Tuple[tf.Tensor]]:
		for csv_path in csv_files:
			try:
				df = pd.read_csv(csv_path, header=0)
				# Apply FEATURE_OFFSET to skip first columns (consistent with training)
				values = df.iloc[:, FEATURE_OFFSET:].to_numpy(dtype=np.float32)
				window = resample_sequence(values, target_len)
				window = scaler.transform(window)
				if per_window_zscore:
					# CRITICAL: Match training pipeline normalization
					# Training: features.mean(axis=1) where features shape is (batch, time, features)
					#           -> normalizes each feature across time dimension per sample
					# Here: window shape is (time, features), so axis=0 is time dimension
					#       -> normalizes each feature across time dimension (same effect)
					mean = window.mean(axis=0, keepdims=True)  # (1, features) - mean over time
					std = window.std(axis=0, keepdims=True) + 1e-6  # (1, features) - std over time
					window = (window - mean) / std
				batch = np.expand_dims(window.astype(np.float32, copy=False), axis=0)
				yield (tf.convert_to_tensor(batch),)
			except Exception as e:
				print(f"[INT8] Skipping {csv_path.name}: {e}")
				continue

	return tf.lite.RepresentativeDataset(gen)


def convert_to_tflite(
	weights_path: Path,
	output_path: Path,
	target_len: int,
	dropout: float,
	fp16: bool,
	int8: bool,
	rep_data_dir: Optional[Path],
	scaler_path: Optional[Path],
	per_window_zscore: bool,
) -> None:
	if not weights_path.exists():
		raise FileNotFoundError(f"Weights file not found: {weights_path}")

	model = build_squat_classifier(
		input_shape=(target_len, NUM_FEATURES),
		num_classes=NUM_CLASSES,
		dropout=dropout,
	)
	# Build variables for Keras 3 weight loading compatibility
	model.build((None, target_len, NUM_FEATURES))
	model.load_weights(weights_path)
	
	# Set model to inference mode (disable dropout, batch norm in training mode)
	# This is critical for TFLite conversion to match inference behavior
	model.trainable = False
	for layer in model.layers:
		layer.trainable = False

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter._experimental_lower_tensor_list_ops = False  # noqa: SLF001
	converter.target_spec.supported_ops = [
		tf.lite.OpsSet.TFLITE_BUILTINS,
		tf.lite.OpsSet.SELECT_TF_OPS,
	]

	if fp16:
		converter.target_spec.supported_types = [tf.float16]

	if int8:
		repr_ds = _make_representative_dataset(rep_data_dir, scaler_path, target_len, NUM_FEATURES, per_window_zscore)
		if repr_ds is not None:
			converter.representative_dataset = repr_ds
			converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
			converter.inference_input_type = tf.uint8
			converter.inference_output_type = tf.uint8
		else:
			print("[INT8] Representative dataset unavailable. Falling back to default (no INT8).")

	tflite_model = converter.convert()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_bytes(tflite_model)
	print(f"[TFLite] Saved model to {output_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert trained squat classifier to TFLite.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Basic conversion (uses defaults from config)
  python -m src.train.totflite --weights checkpoints/squat_classifier_best.weights.h5 --output exports/model.tflite
  
  # FP16 conversion
  python -m src.train.totflite --weights checkpoints/squat_classifier_best.weights.h5 --output exports/model_fp16.tflite --fp16
  
  # INT8 conversion (requires representative data)
  python -m src.train.totflite --weights checkpoints/squat_classifier_best.weights.h5 --output exports/model_int8.tflite --int8 --rep-data-dir data/manually_labeled/class0
		"""
	)
	parser.add_argument(
		"--weights", 
		type=Path, 
		default=Path("checkpoints/squat_classifier_best.weights.h5"),
		help="Path to .weights.h5 file (default: checkpoints/squat_classifier_best.weights.h5)."
	)
	parser.add_argument(
		"--output", 
		type=Path, 
		default=Path("exports/squat_classifier.tflite"),
		help="Destination .tflite file (default: exports/squat_classifier.tflite)."
	)
	parser.add_argument("--config", type=Path, default=Path("config/training_config.json"))
	parser.add_argument("--target-len", type=int, default=None, help="Override model input length.")
	parser.add_argument("--dropout", type=float, default=None, help="Override dropout rate.")
	parser.add_argument("--fp16", action="store_true", help="Enable FP16 conversion.")
	parser.add_argument("--int8", action="store_true", help="Enable INT8 conversion (requires rep data).")
	parser.add_argument("--rep-data-dir", type=Path, default=None, help="Directory with CSV windows for INT8 calibration.")
	parser.add_argument("--scaler", type=Path, default=Path("squat_scaler_18axis.pkl"))
	parser.add_argument("--per-window-zscore", action="store_true", help="Apply per-window z-score in representative dataset.")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	
	# Validate required files exist
	if not args.weights.exists():
		raise FileNotFoundError(f"Weights file not found: {args.weights}")
	
	cfg = _load_config(args.config)
	
	# Get target_len with proper fallback and warning
	if args.target_len:
		target_len = args.target_len
		print(f"[TFLite] Using target_len from command line: {target_len}")
	elif "target_len" in cfg:
		target_len = int(cfg["target_len"])
		print(f"[TFLite] Using target_len from config: {target_len}")
	else:
		target_len = 320  # Fallback to constants.TARGET_LEN
		print(f"[WARN] target_len not found in config, using default: {target_len}")
		print(f"[WARN] Make sure this matches the training target_len! Check training_config.json")
	
	dropout = args.dropout if args.dropout is not None else float(cfg.get("dropout", 0.3))
	per_window_zscore = bool(cfg.get("per_window_zscore", False)) or bool(args.per_window_zscore)

	convert_to_tflite(
		weights_path=args.weights,
		output_path=args.output,
		target_len=target_len,
		dropout=dropout,
		fp16=bool(args.fp16),
		int8=bool(args.int8),
		rep_data_dir=args.rep_data_dir,
		scaler_path=args.scaler,
		per_window_zscore=per_window_zscore,
	)
