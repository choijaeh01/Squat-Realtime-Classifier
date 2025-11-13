from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .data_utils import resample_sequence


NUM_SENSORS = 3
FEATURES_PER_SENSOR = 6  # ax, ay, az, gx, gy, gz


@dataclass
class AugmentationConfig:
    """Configuration controlling per-window augmentations."""

    jitter_sigma: float = 0.015
    jitter_prob: float = 0.9
    drift_max: float = 0.02
    drift_prob: float = 0.5
    global_scale_range: Tuple[float, float] = (0.93, 1.07)
    global_scale_prob: float = 0.6
    time_scale_range: Optional[Tuple[float, float]] = (0.9, 1.1)
    time_scale_prob: float = 0.5
    strong_time_scale_range: Tuple[float, float] = (0.82, 1.18)
    rotation_max_deg: float = 8.0
    rotation_prob: float = 0.6
    strong_rotation_max_deg: float = 12.0
    strong_prob: float = 0.2
    # New: random circular time shift
    time_shift_max_ratio: float = 0.15
    time_shift_prob: float = 0.6
    time_mask_ratio: float = 0.05
    time_mask_count: int = 2
    time_mask_prob: float = 0.4
    sensor_dropout_prob: float = 0.05
    sensor_dropout_group_prob: float = 0.3


def _random_rotation_matrix(max_degrees: float) -> np.ndarray:
    angles = np.deg2rad(np.random.uniform(-max_degrees, max_degrees, size=3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def _apply_sensor_rotations(window: np.ndarray, max_degrees: float) -> None:
    if max_degrees <= 0.0:
        return
    for sensor_idx in range(NUM_SENSORS):
        start = sensor_idx * FEATURES_PER_SENSOR
        acc_slice = slice(start, start + 3)
        gyro_slice = slice(start + 3, start + 6)
        rotation = _random_rotation_matrix(max_degrees)
        window[:, acc_slice] = window[:, acc_slice] @ rotation.T
        window[:, gyro_slice] = window[:, gyro_slice] @ rotation.T


def _apply_time_masks(window: np.ndarray, mask_ratio: float, mask_count: int) -> None:
    seq_len = window.shape[0]
    mask_len = max(1, int(seq_len * mask_ratio))
    for _ in range(mask_count):
        start = np.random.randint(0, max(1, seq_len - mask_len + 1))
        end = start + mask_len
        window[start:end, :] = 0.0


def _random_time_scale(window: np.ndarray, target_len: int, scale_range: Tuple[float, float]) -> np.ndarray:
    min_scale, max_scale = scale_range
    if max_scale <= min_scale:
        return resample_sequence(window, target_len)

    scale = np.random.uniform(min_scale, max_scale)
    new_len = max(8, int(round(window.shape[0] * scale)))
    scaled = resample_sequence(window, new_len)
    return resample_sequence(scaled, target_len)


def _apply_random_drift(window: np.ndarray, max_drift: float) -> None:
    if max_drift <= 0.0:
        return
    seq_len, num_features = window.shape
    end_values = np.random.uniform(-max_drift, max_drift, size=num_features).astype(np.float32)
    drift = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)[:, None] * end_values[None, :]
    window += drift


def _apply_sensor_dropout(window: np.ndarray, dropout_prob: float) -> None:
    if dropout_prob <= 0.0:
        return
    mask = np.ones(window.shape[1], dtype=np.float32)
    for sensor_idx in range(NUM_SENSORS):
        if np.random.rand() < dropout_prob:
            start = sensor_idx * FEATURES_PER_SENSOR
            mask[start : start + FEATURES_PER_SENSOR] = 0.0
    window *= mask


def augment_sequence(
    window: np.ndarray,
    target_len: int,
    config: AugmentationConfig,
) -> np.ndarray:
    augmented = np.array(window, dtype=np.float32, copy=True)

    # Tempo variations
    if (
        config.time_scale_range is not None
        and config.time_scale_prob > 0.0
        and np.random.rand() < config.time_scale_prob
    ):
        augmented = _random_time_scale(augmented, target_len, config.time_scale_range)
    else:
        augmented = resample_sequence(augmented, target_len)

    # Occasionally apply stronger tempo/rotation variants
    rotation_max = config.rotation_max_deg
    if config.strong_prob > 0.0 and np.random.rand() < config.strong_prob:
        augmented = _random_time_scale(augmented, target_len, config.strong_time_scale_range)
        rotation_max = max(rotation_max, config.strong_rotation_max_deg)

    # Noise and drift
    if config.jitter_sigma > 0 and np.random.rand() < config.jitter_prob:
        noise = np.random.normal(0.0, config.jitter_sigma, size=augmented.shape).astype(np.float32)
        augmented += noise

    if config.drift_max > 0 and np.random.rand() < config.drift_prob:
        _apply_random_drift(augmented, config.drift_max)

    # Random circular time shift (robustness to phase misalignment)
    if (
        config.time_shift_max_ratio > 0.0
        and np.random.rand() < config.time_shift_prob
    ):
        max_shift = int(target_len * config.time_shift_max_ratio)
        if max_shift > 0:
            shift = np.random.randint(-max_shift, max_shift + 1)
            augmented = np.roll(augmented, shift=shift, axis=0)

    # Scaling and rotations
    if (
        config.global_scale_range
        and np.random.rand() < config.global_scale_prob
    ):
        low, high = config.global_scale_range
        scale = np.random.uniform(low, high)
        augmented *= np.float32(scale)

    if config.rotation_prob > 0.0 and np.random.rand() < config.rotation_prob:
        _apply_sensor_rotations(augmented, rotation_max)

    # Channel dropout
    if (
        config.sensor_dropout_prob > 0.0
        and np.random.rand() < config.sensor_dropout_group_prob
    ):
        _apply_sensor_dropout(augmented, config.sensor_dropout_prob)

    # Time masking
    if (
        config.time_mask_ratio > 0
        and config.time_mask_count > 0
        and np.random.rand() < config.time_mask_prob
    ):
        _apply_time_masks(augmented, config.time_mask_ratio, config.time_mask_count)

    return augmented.astype(np.float32)
