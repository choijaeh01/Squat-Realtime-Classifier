from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Disable XLA to avoid ptxas compilation issues on some CUDA setups (must be set before TF import)
# Force auto_jit off and disable XLA devices explicitly
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.training import TrainingConfig, run_training, run_loso_cv


def load_config(config_path: Path) -> TrainingConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))

    def to_path(value):
        return Path(value) if value else None

    output_dir = to_path(data.get("output_dir", "checkpoints"))
    if output_dir is None:
        output_dir = Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path_value = data.get("plot_path")
    plot_path = to_path(plot_path_value) if plot_path_value else None

    excluded_subjects_value = data.get("excluded_subjects")
    if excluded_subjects_value is None:
        excluded_subjects = None
    elif isinstance(excluded_subjects_value, str):
        excluded_subjects = (excluded_subjects_value,)
    else:
        excluded_subjects = tuple(excluded_subjects_value)

    return TrainingConfig(
        data_dir=to_path(data.get("data_dir", "data/manually_labeled")),
        ssl_dir=to_path(data.get("ssl_dir", "data/manually_labeled/ssl")),
        output_dir=output_dir,
        scaler_path=to_path(data.get("scaler_path", "squat_scaler_18axis.pkl")),
        target_len=int(data.get("target_len", 384)),
        batch_size=int(data.get("batch_size", 32)),
        epochs=int(data.get("epochs", 120)),
        validation_subject=data.get("validation_subject"),
        learning_rate=float(data.get("learning_rate", 1e-4)),
        patience=int(data.get("patience", 25)),
        use_ssl=bool(data.get("use_ssl", False)),
        ssl_epochs=int(data.get("ssl_epochs", 80)),
        plot_path=plot_path,
        excluded_subjects=excluded_subjects,
        label_smoothing=float(data.get("label_smoothing", 0.05)),
        dropout=float(data.get("dropout", 0.3)),
        mixup_alpha=float(data.get("mixup_alpha", 0.2)),
        mixup_prob=float(data.get("mixup_prob", 0.25)),
        time_cutmix_prob=float(data.get("time_cutmix_prob", 0.15)),
        time_cutmix_alpha=float(data.get("time_cutmix_alpha", 0.5)),
        freeze_epochs=int(data.get("freeze_epochs", 8)),
        use_class_weight=bool(data.get("use_class_weight", True)),
        use_focal_loss=bool(data.get("use_focal_loss", False)),
        focal_gamma=float(data.get("focal_gamma", 1.5)),
        per_window_zscore=bool(data.get("per_window_zscore", True)),
        weight_decay=float(data.get("weight_decay", 1e-4)),
        optimizer=str(data.get("optimizer", "adam")).lower(),
        balanced_sampling=bool(data.get("balanced_sampling", True)),
        conservative_augment=bool(data.get("conservative_augment", True)),
        diag_label_shuffle=bool(data.get("diag_label_shuffle", False)),
        diag_overfit_small=bool(data.get("diag_overfit_small", False)),
        diag_overfit_n_per_class=int(data.get("diag_overfit_n_per_class", 30)),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train or evaluate with LOSO using training_config.json")
    parser.add_argument("--config", type=Path, default=Path("config/training_config.json"))
    parser.add_argument("--loso", action="store_true", help="Run LOSO cross-validation")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.loso:
        run_loso_cv(config, config_path=args.config)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
