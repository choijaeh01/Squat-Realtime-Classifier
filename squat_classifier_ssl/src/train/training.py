from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

from .augmentations import AugmentationConfig, augment_sequence
from .constants import NUM_CLASSES, NUM_FEATURES, TARGET_LEN
from .data_utils import DatasetSplit, load_labeled_splits, resample_sequence, list_all_subjects
from .modeling import build_squat_classifier
from .ssl_pretrain import pretrain_ssl_encoder, transfer_encoder_weights

# Ensure XLA JIT is disabled (additional guard)
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

@dataclass
class TrainingConfig:
    data_dir: Path
    ssl_dir: Path
    output_dir: Path
    scaler_path: Path
    target_len: int = TARGET_LEN
    batch_size: int = 32
    epochs: int = 120
    validation_subject: Optional[str] = None
    learning_rate: float = 3e-4
    use_ssl: bool = False
    ssl_epochs: int = 150
    patience: int = 25
    plot_path: Optional[Path] = None
    excluded_subjects: Optional[tuple[str, ...]] = None
    label_smoothing: float = 0.05
    dropout: float = 0.4
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.25
    # Time CutMix across samples (temporal splice)
    time_cutmix_prob: float = 0.15
    time_cutmix_alpha: float = 0.5
    # CutMix safety: restrict splice to same-class pairs only
    time_cutmix_same_class: bool = True
    # Freeze-then-unfreeze fine-tuning
    freeze_epochs: int = 8
    # Class reweighting and focal loss
    use_class_weight: bool = False
    use_focal_loss: bool = False
    focal_gamma: float = 1.5
    # Per-window standardization to reduce subject variability
    per_window_zscore: bool = True
    # Optimizer weight decay
    weight_decay: float = 1e-4
    # Optimizer choice: "adam" or "adamw"
    optimizer: str = "adam"
    # Balanced sampling and conservative augmentation
    balanced_sampling: bool = False
    conservative_augment: bool = False
    # Diagnostics
    diag_label_shuffle: bool = False
    diag_overfit_small: bool = False
    diag_overfit_n_per_class: int = 30


DEFAULT_TRAIN_AUG = AugmentationConfig(
    jitter_sigma=0.022,
    jitter_prob=0.9,
    drift_max=0.02,
    drift_prob=0.5,
    global_scale_range=(0.93, 1.07),
    global_scale_prob=0.65,
    time_scale_range=(0.88, 1.12),
    time_scale_prob=0.7,
    strong_time_scale_range=(0.86, 1.14),
    rotation_max_deg=12.0,
    rotation_prob=0.7,
    strong_rotation_max_deg=14.0,
    strong_prob=0.15,
    time_shift_max_ratio=0.15,
    time_shift_prob=0.6,
    time_mask_ratio=0.05,
    time_mask_count=1,
    time_mask_prob=0.35,
    sensor_dropout_prob=0.1,
    sensor_dropout_group_prob=0.3,
)


class SquatSequence(Sequence):
    def __init__(
        self,
        split: DatasetSplit,
        scaler: StandardScaler,
        batch_size: int,
        target_len: int,
        num_classes: int,
        augment: bool = False,
        augment_config: Optional[AugmentationConfig] = None,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
        time_cutmix_prob: float = 0.0,
        time_cutmix_alpha: float = 0.5,
        time_cutmix_same_class: bool = False,
        per_window_zscore: bool = False,
        balanced_sampling: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.target_len = target_len
        self.augment = augment
        self.augment_config = augment_config or AugmentationConfig()
        self.mixup_alpha = float(mixup_alpha or 0.0)
        self.mixup_prob = float(mixup_prob or 0.0)
        self.time_cutmix_prob = float(time_cutmix_prob or 0.0)
        self.time_cutmix_alpha = float(time_cutmix_alpha or 0.5)
        self.time_cutmix_same_class = bool(time_cutmix_same_class)
        self.per_window_zscore = bool(per_window_zscore)
        self.balanced_sampling = bool(balanced_sampling)

        scaled_windows = []
        for window in split.windows:
            scaled = scaler.transform(resample_sequence(window, target_len))
            scaled_windows.append(scaled.astype(np.float32))
        self.windows = np.stack(scaled_windows).astype(np.float32)
        self.labels = to_categorical(split.labels, num_classes=num_classes)
        self.y_int = np.argmax(self.labels, axis=1)
        self.indices = np.arange(len(self.windows))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.windows) / self.batch_size))

    def __getitem__(self, index: int):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_idx = self.indices[start:end]
        batch_windows = self.windows[batch_idx]

        if self.augment:
            augmented = [
                augment_sequence(window, self.target_len, self.augment_config)
                for window in batch_windows
            ]
            features = np.stack(augmented).astype(np.float32)
        else:
            features = batch_windows

        labels = self.labels[batch_idx]
        # Temporal CutMix (splice along time axis)
        if (
            self.augment
            and self.time_cutmix_prob > 0.0
            and np.random.rand() < self.time_cutmix_prob
            and len(features) > 1
        ):
            lam = np.random.beta(self.time_cutmix_alpha, self.time_cutmix_alpha)
            split = int(self.target_len * lam)
            a = features.copy()
            mixed = a.copy()
            if self.time_cutmix_same_class:
                y_int = np.argmax(labels, axis=1)
                for c in np.unique(y_int):
                    idx = np.where(y_int == c)[0]
                    if len(idx) < 2:
                        continue
                    perm_local = np.random.permutation(idx)
                    b = features[perm_local]
                    mixed[idx, :split, :] = a[idx, :split, :]
                    mixed[idx, split:, :] = b[:, split:, :]
                    lam_eff = split / float(self.target_len)
                    labels[idx] = lam_eff * labels[idx] + (1.0 - lam_eff) * labels[perm_local]
            else:
                perm = np.random.permutation(len(features))
                b = features[perm]
                mixed[:, :split, :] = a[:, :split, :]
                mixed[:, split:, :] = b[:, split:, :]
                lam_eff = split / float(self.target_len)
                labels = lam_eff * labels + (1.0 - lam_eff) * labels[perm]
            features = mixed

        # Batch-wise MixUp for additional regularization
        if (
            self.augment
            and self.mixup_alpha > 0.0
            and self.mixup_prob > 0.0
            and np.random.rand() < self.mixup_prob
            and len(features) > 1
        ):
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            perm = np.random.permutation(len(features))
            features = lam * features + (1.0 - lam) * features[perm]
            labels = lam * labels + (1.0 - lam) * labels[perm]

        # Per-window Z-score normalization (mitigate subject-specific amplitude/offset)
        if self.per_window_zscore:
            mean = features.mean(axis=1, keepdims=True)
            std = features.std(axis=1, keepdims=True) + 1e-6
            features = (features - mean) / std
        return features, labels

    def on_epoch_end(self) -> None:
        if self.balanced_sampling:
            # Upsample each class to match the majority, then shuffle
            class_to_idx = {c: np.where(self.y_int == c)[0].tolist() for c in np.unique(self.y_int)}
            if class_to_idx:
                max_len = max(len(v) for v in class_to_idx.values())
                balanced_indices = []
                for c, idxs in class_to_idx.items():
                    if not idxs:
                        continue
                    repeat = max_len - len(idxs)
                    if repeat > 0:
                        extra = np.random.choice(idxs, size=repeat, replace=True).tolist()
                        idxs = idxs + extra
                    balanced_indices.extend(idxs)
                self.indices = np.array(balanced_indices, dtype=np.int32)
                np.random.shuffle(self.indices)
            else:
                self.indices = np.arange(len(self.windows))
        else:
            if self.augment:
                np.random.shuffle(self.indices)


def _fit_scaler(train_split: DatasetSplit, target_len: int, scaler_path: Path) -> StandardScaler:
    scaler = StandardScaler()
    stacked = np.concatenate(
        [resample_sequence(window, target_len) for window in train_split.windows],
        axis=0,
    )
    scaler.fit(stacked)
    joblib.dump(scaler, scaler_path)
    print(f"[Scaler] Saved to {scaler_path}")
    return scaler


def _plot_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    if not acc or not val_acc:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(acc, label="Train Accuracy")
    axes[0].plot(val_acc, label="Validation Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(loss, label="Train Loss")
    axes[1].plot(val_loss, label="Validation Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[Plot] Training curves saved to {output_path}")


def _log_split_info(subject_counts: dict[str, dict[str, int]]) -> None:
    print("\n[Split] Subject distribution:")
    for split_name, counts in subject_counts.items():
        if not counts:
            continue
        total = sum(counts.values())
        details = ", ".join(f"{subj}: {count}" for subj, count in counts.items())
        print(f"  {split_name} ({total} windows): {details}")
    print("[Classes] Using fixed mapping: " + ", ".join([f"class{i}->{i}" for i in range(NUM_CLASSES)]))


def _make_loss(config: TrainingConfig):
    """Return the appropriate loss function per config."""
    if config.use_focal_loss:
        gamma = float(config.focal_gamma)
        # Multi-class focal loss for one-hot labels
        def focal_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
            pt = tf.reduce_sum(y_true * y_pred, axis=-1)
            modulating = tf.pow(1.0 - pt, gamma)
            return tf.reduce_mean(modulating * cross_entropy)

        return focal_loss
    # Default: categorical crossentropy with label smoothing
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.label_smoothing)

def _make_optimizer(config: TrainingConfig):
    """Create an AdamW optimizer if available across TF versions; otherwise fallback to Adam."""
    try:
        from tensorflow.keras.optimizers import AdamW as KAdamW  # type: ignore
        return KAdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    except Exception:
        pass
    try:
        from tensorflow.keras.optimizers.experimental import AdamW as ExpAdamW  # type: ignore
        return ExpAdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    except Exception:
        pass
    try:
        import tensorflow_addons as tfa  # type: ignore
        return tfa.optimizers.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    except Exception:
        pass
    print("[Warn] AdamW not available in this environment. Falling back to Adam without weight decay.")
    return tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

def _select_optimizer(config: TrainingConfig):
    opt = (config.optimizer or "adam").lower()
    if opt == "adamw":
        return _make_optimizer(config)
    return tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
def run_training(config: TrainingConfig) -> tf.keras.Model:
    if config.excluded_subjects:
        print(f"[Data] Excluding subjects: {', '.join(config.excluded_subjects)}")

    train_split, val_split, subject_counts, max_len = load_labeled_splits(
        config.data_dir,
        num_classes=NUM_CLASSES,
        target_len=config.target_len,
        validation_subject=config.validation_subject,
        excluded_subjects=set(config.excluded_subjects or []),
    )
    print(f"[Data] Max original window length: {max_len}")
    _log_split_info(subject_counts)
    
    # Log training mode
    if config.validation_subject is None:
        print(f"[Training-Mode] FULL TRAINING: Using all {len(train_split.windows)} samples for training (no validation split)")
    else:
        print(f"[Training-Mode] VALIDATION SPLIT: Train={len(train_split.windows)}, Val={len(val_split.windows) if val_split.windows else 0}")

    # Diagnostics: tiny overfit mode (same train/val, few samples per class)
    if config.diag_overfit_small:
        print(f"[Diag] Tiny overfit mode: {config.diag_overfit_n_per_class} per class, train=val identical")
        sel_train_windows, sel_train_labels = [], []
        for c in range(NUM_CLASSES):
            idxs = [i for i, y in enumerate(train_split.labels) if y == c]
            if idxs:
                chosen = np.random.choice(idxs, size=min(config.diag_overfit_n_per_class, len(idxs)), replace=len(idxs) < config.diag_overfit_n_per_class)
                for i in chosen:
                    sel_train_windows.append(train_split.windows[i])
                    sel_train_labels.append(train_split.labels[i])
        train_split = DatasetSplit(sel_train_windows, np.array(sel_train_labels, dtype=np.int32))
        val_split = DatasetSplit(sel_train_windows, np.array(sel_train_labels, dtype=np.int32))

    # Diagnostics: label shuffle test
    if config.diag_label_shuffle:
        print("[Diag] Label shuffle test ENABLED (train labels randomly permuted)")
        perm = np.random.permutation(len(train_split.labels))
        train_split = DatasetSplit(train_split.windows, train_split.labels[perm])

    scaler = _fit_scaler(train_split, config.target_len, config.scaler_path)
    # Build augmentation config (optionally conservative)
    train_aug_cfg = DEFAULT_TRAIN_AUG
    if config.conservative_augment:
        train_aug_cfg = dataclass_replace(
            DEFAULT_TRAIN_AUG,
            time_mask_prob=0.0,
            time_mask_count=0,
        )

    train_gen = SquatSequence(
        train_split,
        scaler,
        batch_size=config.batch_size,
        target_len=config.target_len,
        num_classes=NUM_CLASSES,
        augment=True,
        augment_config=train_aug_cfg,
        mixup_alpha=config.mixup_alpha,
        mixup_prob=config.mixup_prob,
        time_cutmix_prob=config.time_cutmix_prob,
        time_cutmix_alpha=config.time_cutmix_alpha,
        time_cutmix_same_class=config.time_cutmix_same_class,
        per_window_zscore=config.per_window_zscore,
        balanced_sampling=config.balanced_sampling,
    )

    val_gen = (
        SquatSequence(
            val_split,
            scaler,
            batch_size=config.batch_size,
            target_len=config.target_len,
            num_classes=NUM_CLASSES,
            augment=False,
            time_cutmix_same_class=config.time_cutmix_same_class,
            per_window_zscore=config.per_window_zscore,
            balanced_sampling=False,
        )
        if val_split.windows
        else None
    )

    model = build_squat_classifier(
        input_shape=(config.target_len, NUM_FEATURES),
        num_classes=NUM_CLASSES,
        dropout=config.dropout,
    )

    if config.use_ssl:
        print("[SSL] Starting encoder pretraining...")
        # Exclude validation subject from SSL to prevent LOSO leakage
        excluded_for_ssl = set()
        if config.validation_subject:
            excluded_for_ssl.add(config.validation_subject)
        if config.excluded_subjects:
            excluded_for_ssl.update(config.excluded_subjects)
        
        encoder = pretrain_ssl_encoder(
            ssl_dir=str(config.ssl_dir),
            scaler_path=str(config.scaler_path),
            max_len=config.target_len,
            num_features=NUM_FEATURES,
            epochs=config.ssl_epochs,
            batch_size=config.batch_size,
            save_path=str(config.output_dir / "ssl_encoder.keras"),
            method="simsiam",
            excluded_subjects=excluded_for_ssl,
            labeled_data_dir=str(config.data_dir),
            num_classes=NUM_CLASSES,
            train_windows=train_split.windows,  # Use train split windows as unlabeled for SSL
        )
        transfer_encoder_weights(encoder, model)
        print("[SSL] Encoder weights transferred to classifier.")

    optimizer = _select_optimizer(config)
    model.compile(optimizer=optimizer, loss=_make_loss(config), metrics=["accuracy"], jit_compile=False)

    # Optional warmup: freeze encoder for a few epochs
    if config.freeze_epochs and config.freeze_epochs > 0:
        for layer in model.layers:
            if layer.name != "classifier":
                layer.trainable = False
        model.compile(optimizer=_select_optimizer(config), loss=_make_loss(config), metrics=["accuracy"], jit_compile=False)
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.freeze_epochs,
            verbose=1,
        )
        # unfreeze all
        for layer in model.layers:
            layer.trainable = True
        model.compile(optimizer=_select_optimizer(config), loss=_make_loss(config), metrics=["accuracy"], jit_compile=False)

    # Configure callbacks based on whether validation data exists
    # For full training (validation_subject=None), use all data for training
    # and disable validation-based callbacks
    callbacks = []
    if val_gen is not None:
        # Validation-based callbacks (for LOSO or validation split)
        callbacks = [
            ModelCheckpoint(
                filepath=config.output_dir / "squat_classifier_best.weights.h5",
                save_weights_only=True,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_accuracy",
                mode="max",
                factor=0.5,
                patience=6,
                min_lr=1e-6,
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=config.patience,
                restore_best_weights=True,
                verbose=1,
            ),
        ]
        print("[Training] Using validation-based callbacks (early stopping, model checkpoint)")
    else:
        # Full training mode: no validation, save final model
        callbacks = [
            ModelCheckpoint(
                filepath=config.output_dir / "squat_classifier_best.weights.h5",
                save_weights_only=True,
                monitor="loss",  # Monitor training loss instead
                mode="min",
                save_best_only=False,  # Save every epoch (will keep last)
                verbose=1,
            ),
            # No early stopping or LR reduction without validation
        ]
        print("[Training] Full training mode: using all data, no validation, saving final model")
        print(f"[Training] Will train for {config.epochs} epochs (no early stopping)")

    # Optional class weights (to mitigate class imbalance)
    class_weights = None
    if config.use_class_weight:
        classes = np.arange(NUM_CLASSES, dtype=np.int32)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=train_split.labels
        )
        class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    # Fit model with or without validation
    fit_kwargs = {
        "epochs": config.epochs,
        "callbacks": callbacks,
        "verbose": 1,
        "class_weight": class_weights,
    }
    if val_gen is not None:
        fit_kwargs["validation_data"] = val_gen
        print(f"[Training] Training with validation (early stopping enabled)")
    else:
        print(f"[Training] Training without validation (full data, {config.epochs} epochs)")
    
    history = model.fit(train_gen, **fit_kwargs)

    history_path = config.output_dir / "training_history.json"

    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    serializable_history = {k: [_to_serializable(v) for v in values] for k, values in history.history.items()}
    history_path.write_text(json.dumps(serializable_history, indent=2))
    print(f"[History] Saved to {history_path}")

    if config.plot_path:
        _plot_history(history, config.plot_path)

    metrics = model.evaluate(val_gen, verbose=0) if val_gen is not None else None
    if metrics:
        print(f"[Eval] Validation loss: {metrics[0]:.4f}, accuracy: {metrics[1]:.4f}")

    # Manual accuracy with identical preprocessing (to detect metric path mismatch)
    if val_gen is not None and val_split.windows:
        Xv = []
        for w in val_split.windows:
            Xv.append(scaler.transform(resample_sequence(w, config.target_len)))
        Xv = np.stack(Xv).astype(np.float32)
        if config.per_window_zscore:
            mean = Xv.mean(axis=1, keepdims=True)
            std = Xv.std(axis=1, keepdims=True) + 1e-6
            Xv = (Xv - mean) / std
        y_true = val_split.labels
        y_prob = model.predict(Xv, batch_size=config.batch_size, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        manual_acc = float(accuracy_score(y_true, y_pred))
        if metrics:
            print(f"[Eval-Manual] accuracy={manual_acc:.4f} | keras_eval={metrics[1]:.4f}")
            if abs(manual_acc - float(metrics[1])) > 0.05:
                print("[Warn] Metric mismatch detected between manual and keras evaluate. Check label mapping/aggregation path.")
        # Save sample predictions CSV for first 200 items
        out_csv = config.output_dir / "sample_predictions.csv"
        try:
            paths = val_split.file_paths or ["" for _ in range(len(y_true))]
            n = min(200, len(y_true))
            with out_csv.open("w", encoding="utf-8") as f:
                f.write("filepath,true_idx,pred_idx,prob_max\n")
                for i in range(n):
                    f.write(f"{paths[i]},{int(y_true[i])},{int(y_pred[i])},{float(np.max(y_prob[i])):.6f}\n")
            print(f"[Eval] Sample predictions saved to {out_csv}")
        except Exception as e:
            print(f"[Eval] Failed to write sample_predictions.csv: {e}")

    return model


def run_loso_cv(config: TrainingConfig, config_path: Optional[Path] = None) -> dict[str, float]:
    """
    Run Leave-One-Subject-Out cross-validation.
    Returns a mapping subject -> val_accuracy.
    
    Parameters
    ----------
    config : TrainingConfig
        Training configuration
    config_path : Optional[Path]
        Path to the config JSON file to copy into loso directory
    """
    from statistics import mean, stdev
    import json
    import shutil

    subjects = list_all_subjects(
        base_dir=config.data_dir,
        num_classes=NUM_CLASSES,
        excluded_subjects=set(config.excluded_subjects or []),
    )
    print(f"[LOSO] Subjects: {subjects}")
    results: dict[str, float] = {}
    for subj in subjects:
        print(f"\n[LOSO] Fold: val={subj}")
        fold_out = (config.output_dir / "loso" / subj)
        fold_out.mkdir(parents=True, exist_ok=True)
        fold_scaler = fold_out / "squat_scaler_18axis.pkl"
        fold_plot = fold_out / "training_curves.png"

        fold_cfg = TrainingConfig(
            data_dir=config.data_dir,
            ssl_dir=config.ssl_dir,
            output_dir=fold_out,
            scaler_path=fold_scaler,
            target_len=config.target_len,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_subject=subj,
            learning_rate=config.learning_rate,
            use_ssl=config.use_ssl,
            ssl_epochs=config.ssl_epochs,
            patience=config.patience,
            plot_path=fold_plot,
            excluded_subjects=config.excluded_subjects,
            label_smoothing=config.label_smoothing,
            dropout=config.dropout,
            mixup_alpha=config.mixup_alpha,
            mixup_prob=config.mixup_prob,
            time_cutmix_prob=config.time_cutmix_prob,
            time_cutmix_alpha=config.time_cutmix_alpha,
            time_cutmix_same_class=config.time_cutmix_same_class,
            freeze_epochs=config.freeze_epochs,
            use_class_weight=config.use_class_weight,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            per_window_zscore=config.per_window_zscore,
            weight_decay=config.weight_decay,
            optimizer=config.optimizer,
            balanced_sampling=config.balanced_sampling,
            conservative_augment=config.conservative_augment,
            diag_label_shuffle=config.diag_label_shuffle,
            diag_overfit_small=config.diag_overfit_small,
            diag_overfit_n_per_class=config.diag_overfit_n_per_class,
        )
        run_training(fold_cfg)
        # Evaluate using freshly trained best weights
        train_split, val_split, _, _ = load_labeled_splits(
            config.data_dir,
            num_classes=NUM_CLASSES,
            target_len=config.target_len,
            validation_subject=subj,
            excluded_subjects=set(config.excluded_subjects or []),
        )
        scaler = joblib.load(fold_scaler)
        val_gen = SquatSequence(
            val_split,
            scaler,
            batch_size=config.batch_size,
            target_len=config.target_len,
            num_classes=NUM_CLASSES,
            augment=False,
            per_window_zscore=config.per_window_zscore,
        )
        model = build_squat_classifier(
            input_shape=(config.target_len, NUM_FEATURES),
            num_classes=NUM_CLASSES,
            dropout=config.dropout,
        )
        # Compile is required before evaluate to attach metrics
        model.compile(optimizer=_make_optimizer(config), loss=_make_loss(config), metrics=["accuracy"])
        best_weights = fold_out / "squat_classifier_best.weights.h5"
        model.load_weights(best_weights)
        metrics = model.evaluate(val_gen, verbose=0)
        acc = float(metrics[1])
        # Manual accuracy
        Xv = []
        for w in val_split.windows:
            Xv.append(scaler.transform(resample_sequence(w, config.target_len)))
        Xv = np.stack(Xv).astype(np.float32)
        if config.per_window_zscore:
            Xv_mean = Xv.mean(axis=1, keepdims=True)
            Xv_std = Xv.std(axis=1, keepdims=True) + 1e-6
            Xv = (Xv - Xv_mean) / Xv_std
        y_true = val_split.labels
        y_prob = model.predict(Xv, batch_size=config.batch_size, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        manual_acc = float(accuracy_score(y_true, y_pred))
        print(f"[LOSO] {subj} -> val_acc(keras)={acc:.4f} | val_acc(manual)={manual_acc:.4f}")
        if abs(manual_acc - acc) > 0.05:
            print("[Warn] LOSO metric mismatch detected. Investigate label mapping/aggregation consistency.")
        results[subj] = manual_acc
        # Confusion matrix & classification report
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        report = classification_report(
            y_true, y_pred, labels=list(range(NUM_CLASSES)), output_dict=True
        )
        # Save confusion matrix plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8)
        fig.tight_layout()
        (fold_out / "confusion_matrix.png").parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fold_out / "confusion_matrix.png")
        plt.close(fig)
        # Save classification report
        with (fold_out / "classification_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    if len(results) >= 2:
        vals = list(results.values())
        print(f"\n[LOSO] Mean acc={mean(vals):.4f} | std={stdev(vals):.4f}")
    else:
        only = next(iter(results.values()))
        print(f"\n[LOSO] Acc={only:.4f}")
    # Save summary
    loso_dir = config.output_dir / "loso"
    loso_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config file to loso directory if provided
    if config_path is not None and config_path.exists():
        config_copy_path = loso_dir / config_path.name
        shutil.copy2(config_path, config_copy_path)
        print(f"[LOSO] Config file copied to {config_copy_path}")
    
    summary_path = loso_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "mean_accuracy": float(mean(list(results.values()))),
                "std_accuracy": float(stdev(list(results.values()))) if len(results) >= 2 else 0.0,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[LOSO] Summary saved to {summary_path}")
    return results


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train squat classifier with LOSO split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/manually_labeled"))
    parser.add_argument("--ssl-dir", type=Path, default=Path("data/manually_labeled/ssl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--scaler-path", type=Path, default=Path("squat_scaler_18axis.pkl"))
    parser.add_argument("--validation-subject", type=str, default=None)
    parser.add_argument("--target-len", type=int, default=TARGET_LEN)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--use-ssl", action="store_true")
    parser.add_argument("--ssl-epochs", type=int, default=150)
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--loso", action="store_true", help="Run LOSO cross-validation instead of a single train/val split.")
    parser.add_argument(
        "--exclude-subjects",
        nargs="+",
        default=None,
        help="Subject IDs to exclude entirely from training/validation.",
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    return TrainingConfig(
        data_dir=args.data_dir,
        ssl_dir=args.ssl_dir,
        output_dir=output_dir,
        scaler_path=args.scaler_path,
        target_len=args.target_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_subject=args.validation_subject,
        learning_rate=args.learning_rate,
        patience=args.patience,
        use_ssl=args.use_ssl,
        ssl_epochs=args.ssl_epochs,
        plot_path=args.plot_path,
        excluded_subjects=tuple(args.exclude_subjects) if args.exclude_subjects else None,
    )


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    config = parse_args()
    if hasattr(config, "loso") and getattr(config, "loso"):
        run_loso_cv(config)
    else:
        run_training(config)
