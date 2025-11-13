from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from .data_utils import load_unlabeled_windows, load_labeled_windows_for_ssl
from .modeling import build_squat_encoder

class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup to base_lr, then cosine decay."""
    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int = 0):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        ws = tf.cast(self.warmup_steps, tf.float32)
        ts = tf.cast(self.total_steps, tf.float32)
        # Warmup
        lr_warm = self.base_lr * (step / tf.maximum(1.0, ws))
        # Cosine after warmup
        progress = (step - ws) / tf.maximum(1.0, (ts - ws))
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        lr_cos = 0.5 * self.base_lr * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < ws, lr_warm, lr_cos)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }

def tf_add_jitter(x, sigma=0.02):
    noise = tf.random.normal(tf.shape(x), stddev=sigma)
    return x + noise


def tf_random_scaling(x, min_scale=0.85, max_scale=1.15):
    scale = tf.random.uniform([], min_scale, max_scale)
    return x * scale


def tf_time_shift(x, max_shift=25):
    shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
    return tf.roll(x, shift=shift, axis=0)


def tf_time_dropout(x, drop_rate=0.05):
    mask = tf.cast(tf.random.uniform(tf.shape(x)[:1], 0, 1) > drop_rate, x.dtype)
    mask = tf.expand_dims(mask, -1)
    return x * mask


def tf_time_warp(x, sigma=0.1):
    seq_len = tf.shape(x)[0]
    noise = tf.random.normal([seq_len], mean=1.0, stddev=sigma)
    cum = tf.math.cumsum(noise)
    cum = (cum - tf.reduce_min(cum)) / (tf.reduce_max(cum) - tf.reduce_min(cum) + 1e-8)
    idx = tf.cast(cum * tf.cast(seq_len - 1, tf.float32), tf.int32)
    return tf.gather(x, idx)


def tf_random_rotation(x, num_features: int, max_degrees: float = 12.0):
    sensors = num_features // 6
    radians = tf.random.uniform([sensors, 3], minval=-max_degrees, maxval=max_degrees)
    radians = radians * (np.pi / 180.0)

    cx, cy, cz = tf.unstack(tf.cos(radians), axis=1)
    sx, sy, sz = tf.unstack(tf.sin(radians), axis=1)

    ones = tf.ones_like(cx)
    zeros = tf.zeros_like(cx)

    rx = tf.stack(
        [
            tf.stack([ones, zeros, zeros], axis=-1),
            tf.stack([zeros, cx, -sx], axis=-1),
            tf.stack([zeros, sx, cx], axis=-1),
        ],
        axis=1,
    )
    ry = tf.stack(
        [
            tf.stack([cy, zeros, sy], axis=-1),
            tf.stack([zeros, ones, zeros], axis=-1),
            tf.stack([-sy, zeros, cy], axis=-1),
        ],
        axis=1,
    )
    rz = tf.stack(
        [
            tf.stack([cz, -sz, zeros], axis=-1),
            tf.stack([sz, cz, zeros], axis=-1),
            tf.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=1,
    )

    rotation = tf.matmul(rz, tf.matmul(ry, rx))

    seq_len = tf.shape(x)[0]
    reshaped = tf.reshape(x, (seq_len, sensors, 6))
    acc = reshaped[:, :, :3]
    gyro = reshaped[:, :, 3:]

    rotated_acc = tf.einsum("tsc,scd->tsd", acc, rotation)
    rotated_gyro = tf.einsum("tsc,scd->tsd", gyro, rotation)
    rotated = tf.concat([rotated_acc, rotated_gyro], axis=-1)
    return tf.reshape(rotated, (seq_len, num_features))


def tf_add_drift(x, max_drift=0.02):
    seq_len = tf.shape(x)[0]
    feat_dim = tf.shape(x)[1]
    end_values = tf.random.uniform([feat_dim], -max_drift, max_drift)
    ramp = tf.linspace(0.0, 1.0, seq_len)
    drift = tf.expand_dims(ramp, -1) * end_values
    return x + drift


def tf_time_mask(x, mask_ratio=0.05, mask_count=1):
    seq_len = tf.shape(x)[0]
    mask = tf.ones([seq_len], dtype=x.dtype)
    mask_len = tf.maximum(
        1, tf.cast(tf.math.round(tf.cast(seq_len, tf.float32) * mask_ratio), tf.int32)
    )
    for _ in range(mask_count):
        max_start = tf.maximum(1, seq_len - mask_len + 1)
        start = tf.random.uniform([], 0, max_start, dtype=tf.int32)
        indices = tf.range(start, start + mask_len)
        updates = tf.zeros([mask_len], dtype=x.dtype)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, 1), updates)
    return x * tf.expand_dims(mask, -1)


def apply_ssl_augmentations(x, max_len, num_features):
    # Calmer augmentations for better subject generalization
    x = tf_add_jitter(x, sigma=0.02)
    if tf.random.uniform([]) < 0.6:
        x = tf_random_scaling(x, 0.92, 1.08)
    if tf.random.uniform([]) < 0.5:
        x = tf_time_shift(x, max_shift=20)
    if tf.random.uniform([]) < 0.5:
        x = tf_time_dropout(x, drop_rate=0.04)
    if tf.random.uniform([]) < 0.4:
        x = tf_time_warp(x, sigma=0.08)
    if tf.random.uniform([]) < 0.6:
        x = tf_random_rotation(x, num_features=num_features, max_degrees=10.0)
    if tf.random.uniform([]) < 0.4:
        x = tf_time_mask(x, mask_ratio=0.06, mask_count=2)
    if tf.random.uniform([]) < 0.5:
        x = tf_add_drift(x, max_drift=0.015)
    x = tf.ensure_shape(x, (max_len, num_features))
    return x


def make_contrastive_pair(x, max_len, num_features):
    v1 = apply_ssl_augmentations(x, max_len, num_features)
    v2 = apply_ssl_augmentations(x, max_len, num_features)
    return v1, v2


def create_ssl_dataset(windows_np, batch_size, max_len, num_features):
    dataset = tf.data.Dataset.from_tensor_slices(windows_np.astype("float32"))
    dataset = dataset.shuffle(len(windows_np))

    def _augment(x):
        v1, v2 = make_contrastive_pair(x, max_len, num_features)
        return v1, v2

    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_balanced_ssl_dataset(
    unlabeled_windows_np, labeled_windows_np, batch_size, max_len, num_features, labeled_ratio=0.2
):
    """
    Create SSL dataset with balanced ratio per batch.
    Each batch contains (1-labeled_ratio) unlabeled and labeled_ratio labeled samples.
    
    Parameters
    ----------
    unlabeled_windows_np: np.ndarray
        Unlabeled windows array
    labeled_windows_np: np.ndarray
        Labeled windows array (labels ignored)
    batch_size: int
        Batch size
    max_len: int
        Sequence length
    num_features: int
        Number of features
    labeled_ratio: float
        Ratio of labeled samples in each batch (default: 0.2 = 20%)
    """
    unlabeled_count = len(unlabeled_windows_np)
    labeled_count = len(labeled_windows_np)
    
    if unlabeled_count == 0:
        raise ValueError("Unlabeled windows cannot be empty")
    
    # Calculate samples per batch
    unlabeled_per_batch = int(batch_size * (1 - labeled_ratio))
    labeled_per_batch = batch_size - unlabeled_per_batch
    
    if labeled_per_batch == 0:
        # Fallback to unlabeled only if batch_size is too small
        dataset = create_ssl_dataset(unlabeled_windows_np, batch_size, max_len, num_features)
        steps_per_epoch = int(np.ceil(unlabeled_count / float(batch_size)))
        return dataset, steps_per_epoch
    
    # Create separate datasets
    unlabeled_ds = tf.data.Dataset.from_tensor_slices(unlabeled_windows_np.astype("float32"))
    unlabeled_ds = unlabeled_ds.shuffle(unlabeled_count, reshuffle_each_iteration=True)
    unlabeled_ds = unlabeled_ds.repeat()  # Repeat to ensure we have enough samples
    
    labeled_ds = tf.data.Dataset.from_tensor_slices(labeled_windows_np.astype("float32"))
    labeled_ds = labeled_ds.shuffle(labeled_count, reshuffle_each_iteration=True)
    labeled_ds = labeled_ds.repeat()  # Repeat to ensure we have enough samples
    
    # Augment both datasets
    def _augment(x):
        v1, v2 = make_contrastive_pair(x, max_len, num_features)
        return v1, v2
    
    unlabeled_ds = unlabeled_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    labeled_ds = labeled_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch separately
    unlabeled_ds = unlabeled_ds.batch(unlabeled_per_batch)
    labeled_ds = labeled_ds.batch(labeled_per_batch)
    
    # Interleave to create balanced batches
    def _combine_batches(unlabeled_batch, labeled_batch):
        # unlabeled_batch: tuple of (v1_unlabeled, v2_unlabeled)
        #   v1_unlabeled: (unlabeled_per_batch, max_len, num_features)
        #   v2_unlabeled: (unlabeled_per_batch, max_len, num_features)
        # labeled_batch: tuple of (v1_labeled, v2_labeled)
        #   v1_labeled: (labeled_per_batch, max_len, num_features)
        #   v2_labeled: (labeled_per_batch, max_len, num_features)
        v1_unlabeled, v2_unlabeled = unlabeled_batch
        v1_labeled, v2_labeled = labeled_batch
        
        # Combine along batch dimension
        v1_combined = tf.concat([v1_unlabeled, v1_labeled], axis=0)  # (batch_size, max_len, num_features)
        v2_combined = tf.concat([v2_unlabeled, v2_labeled], axis=0)  # (batch_size, max_len, num_features)
        
        # Shuffle within batch to mix unlabeled and labeled
        indices = tf.range(batch_size)
        indices = tf.random.shuffle(indices)
        
        v1_combined = tf.gather(v1_combined, indices)
        v2_combined = tf.gather(v2_combined, indices)
        
        return (v1_combined, v2_combined)
    
    dataset = tf.data.Dataset.zip((unlabeled_ds, labeled_ds))
    dataset = dataset.map(_combine_batches, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Calculate steps per epoch based on unlabeled data (since it's the majority)
    steps_per_epoch = int(np.ceil(unlabeled_count / float(unlabeled_per_batch)))
    dataset = dataset.take(steps_per_epoch)
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, steps_per_epoch


def build_projection_head(embedding_dim=192, projection_dim=128):
    inputs = tf.keras.layers.Input(shape=(embedding_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(projection_dim)(x)
    return tf.keras.Model(inputs, x, name="projection_head")

def build_predictor(projection_dim=128, hidden_dim=256):
    inputs = tf.keras.layers.Input(shape=(projection_dim,))
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(projection_dim)(x)
    return tf.keras.Model(inputs, x, name="predictor_head")

class NTXentLoss(tf.keras.losses.Loss):
    """SimCLR InfoNCE with explicit one-hot targets and diagonal masking."""
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, zis, zjs):
        # Normalize
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)
        reps = tf.concat([zis, zjs], axis=0)  # [2N, D]
        n = tf.shape(zis)[0]
        # Similarity logits
        logits = tf.matmul(reps, reps, transpose_b=True) / self.temperature  # [2N,2N]
        # Mask self-similarity
        logits = logits - 1e9 * tf.eye(2 * n)
        # Positive indices for each row
        pos_idx = tf.concat([tf.range(n, 2 * n), tf.range(0, n)], axis=0)  # [2N]
        labels = tf.one_hot(pos_idx, depth=2 * n, dtype=logits.dtype)  # [2N,2N]
        # Cross-entropy over columns
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(loss)


class SSLModel(tf.keras.Model):
    def __init__(self, encoder, projector, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.loss_fn = NTXentLoss(temperature=temperature)

    def call(self, inputs, training=False):
        v1, v2 = inputs
        e1 = self.encoder(v1, training=training)
        e2 = self.encoder(v2, training=training)
        z1 = self.projector(e1, training=training)
        z2 = self.projector(e2, training=training)
        return z1, z2

    def train_step(self, data):
        v1, v2 = data
        with tf.GradientTape() as tape:
            z1, z2 = self((v1, v2), training=True)
            loss = self.loss_fn(z1, z2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

class SimSiamModel(tf.keras.Model):
    """SimSiam without negatives; robust to batch size."""
    def __init__(self, encoder, projector, predictor):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor

    @staticmethod
    def neg_cosine_similarity(p, z):
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)
        return -tf.reduce_mean(tf.reduce_sum(p * z, axis=1))

    def call(self, inputs, training=False):
        v1, v2 = inputs
        e1 = self.encoder(v1, training=training)
        e2 = self.encoder(v2, training=training)
        z1 = self.projector(e1, training=training)
        z2 = self.projector(e2, training=training)
        p1 = self.predictor(z1, training=training)
        p2 = self.predictor(z2, training=training)
        return (p1, tf.stop_gradient(z2)), (p2, tf.stop_gradient(z1))

    def train_step(self, data):
        v1, v2 = data
        with tf.GradientTape() as tape:
            (p1, z2), (p2, z1) = self((v1, v2), training=True)
            loss = self.neg_cosine_similarity(p1, z2) / 2.0 + self.neg_cosine_similarity(p2, z1) / 2.0
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}


def pretrain_ssl_encoder(
    ssl_dir: str,
    scaler_path: str,
    max_len: int,
    num_features: int,
    batch_size: int = 64,
    epochs: int = 120,
    temperature: float = 0.4,
    method: str = "simclr",
    fit_scaler_on_ssl: bool = False,
    debug_eval: bool = True,
    save_path: Optional[str] = None,
    excluded_subjects: Optional[Set[str]] = None,
    labeled_data_dir: Optional[str] = None,
    num_classes: int = 5,
    train_windows: Optional[List[np.ndarray]] = None,
):
    """
    Pretrain SSL encoder with optional labeled data augmentation.
    
    Parameters
    ----------
    excluded_subjects:
        Subject IDs to exclude from SSL data (e.g., validation subject to prevent LOSO leakage).
    labeled_data_dir:
        Optional directory containing labeled data (class0, class1, ...) to augment SSL training.
        Labels are ignored; only windows are used.
    num_classes:
        Number of classes in labeled_data_dir (used when labeled_data_dir is provided).
    train_windows:
        Optional list of training windows to use as unlabeled data for SSL.
        These are treated as unlabeled even though they come from labeled data.
    """
    ssl_dir_path = Path(ssl_dir)
    excluded_subjects_set = set(excluded_subjects or [])
    
    # Load unlabeled SSL windows from SSL directory
    ssl_unlabeled_windows = load_unlabeled_windows(ssl_dir_path, max_len, excluded_subjects=excluded_subjects_set)
    
    # Add train split windows as unlabeled data (they are treated as unlabeled for SSL)
    unlabeled_windows = list(ssl_unlabeled_windows)
    if train_windows:
        unlabeled_windows.extend(train_windows)
        print(f"[SSL] Added {len(train_windows)} train windows as unlabeled data")
    
    unlabeled_count = len(unlabeled_windows)
    
    if unlabeled_count == 0:
        raise ValueError("SSL 사전학습에 사용할 무라벨 윈도우가 없습니다.")
    
    print(f"[SSL] Total unlabeled: {unlabeled_count} (SSL dir: {len(ssl_unlabeled_windows)}, train: {len(train_windows) if train_windows else 0})")
    
    # Optionally load labeled windows (ignoring labels) to augment SSL data
    # Sample 20% of unlabeled count with subject and class balance
    labeled_windows = []
    if labeled_data_dir:
        labeled_windows = load_labeled_windows_for_ssl(
            labeled_data_dir, max_len, num_classes, 
            excluded_subjects=excluded_subjects_set,
            target_ratio=0.2,
            unlabeled_count=unlabeled_count,
        )
        if labeled_windows:
            print(f"[SSL] Using {len(labeled_windows)} labeled windows (labels ignored) for 20% per-batch ratio")
        else:
            print(f"[SSL] No labeled windows available, using unlabeled only")
    
    # Combine all windows for scaler fitting
    all_windows = unlabeled_windows + labeled_windows
    
    if Path(scaler_path).exists() and not fit_scaler_on_ssl:
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        concat_train = np.concatenate(all_windows, axis=0)
        scaler.fit(concat_train)
        joblib.dump(scaler, scaler_path)

    # Scale separately to maintain distinction
    scaled_unlabeled = np.stack([scaler.transform(w) for w in unlabeled_windows]).astype("float32")
    scaled_labeled = np.stack([scaler.transform(w) for w in labeled_windows]).astype("float32") if labeled_windows else np.array([]).reshape(0, max_len, num_features).astype("float32")
    
    # Create balanced dataset with 80:20 ratio per batch
    if len(labeled_windows) > 0:
        dataset, steps_per_epoch = create_balanced_ssl_dataset(
            scaled_unlabeled, scaled_labeled, batch_size, max_len, num_features, labeled_ratio=0.2
        )
    else:
        dataset = create_ssl_dataset(scaled_unlabeled, batch_size, max_len, num_features)
        steps_per_epoch = int(np.ceil(len(scaled_unlabeled) / float(batch_size)))
    encoder = build_squat_encoder((max_len, num_features), embedding_dropout=0.0)
    embedding_dim = encoder.output_shape[-1]
    projector = build_projection_head(embedding_dim=embedding_dim)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(10, int(0.1 * total_steps))
    lr_schedule = WarmupCosine(base_lr=1e-3, total_steps=total_steps, warmup_steps=warmup_steps)
    if method.lower() == "simsiam":
        predictor = build_predictor(projection_dim=projector.output_shape[-1])
        ssl_model = SimSiamModel(encoder, projector, predictor)
    else:
        ssl_model = SSLModel(encoder, projector, temperature=temperature)
    ssl_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), jit_compile=False)
    ssl_model.fit(dataset, epochs=epochs, verbose=1)

    if debug_eval:
        for v1, v2 in dataset.take(1):
            if method.lower() == "simsiam":
                (p1, z2), (p2, z1) = ssl_model((v1, v2), training=False)
                loss_value = float(SimSiamModel.neg_cosine_similarity(p1, z2).numpy())
                emb_std = float(ssl_model.encoder(v1, training=False).numpy().std())
                print(f"[SSL Debug SimSiam] Loss: {loss_value:.6f} | Embedding std: {emb_std:.6f}")
            else:
                z1, z2 = ssl_model((v1, v2), training=False)
                loss_value = ssl_model.loss_fn(z1, z2).numpy()
                emb_std = float(ssl_model.encoder(v1, training=False).numpy().std())
                print(f"[SSL Debug SimCLR] Loss: {loss_value:.6f} | Embedding std: {emb_std:.6f}")
            break

    if save_path:
        save_path_p = Path(save_path)
        save_path_p.parent.mkdir(parents=True, exist_ok=True)
        ssl_model.encoder.save(save_path_p)
        print(f"[SSL] Encoder saved to: {save_path_p}")

    return ssl_model.encoder


def load_ssl_encoder(model_path: str) -> tf.keras.Model:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"SSL encoder not found: {model_path}")
    try:
        model = load_model(p)
        print(f"[SSL] Loaded encoder from: {p}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load SSL encoder: {e}") from e


def transfer_encoder_weights(ssl_encoder: tf.keras.Model, classifier_model: tf.keras.Model):
    classifier_layers = {layer.name: layer for layer in classifier_model.layers}
    transferred = 0
    for layer in ssl_encoder.layers:
        target = classifier_layers.get(layer.name)
        weights = layer.get_weights()
        if target is not None and weights:
            target.set_weights(weights)
            transferred += 1
    print(f"[SSL] Transferred weights for {transferred} layers.")
    return classifier_model
