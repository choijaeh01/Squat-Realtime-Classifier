from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import random

import numpy as np
import pandas as pd

from .constants import FEATURE_OFFSET


@dataclass(frozen=True)
class DatasetSplit:
    """In-memory representation of a labeled dataset split."""

    windows: List[np.ndarray]
    labels: np.ndarray
    file_paths: Optional[List[str]] = None


def extract_subject_name(filename: str) -> str:
    """Return the subject identifier prefix (e.g., 'subject3') from a file name."""
    return filename.split("_")[0]


def resample_sequence(window: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample an IMU window to a fixed length using linear interpolation.

    Parameters
    ----------
    window:
        Array shaped (T, F) where T is the original number of timesteps.
    target_len:
        Desired output sequence length.
    """
    current_len, num_features = window.shape
    if current_len == target_len:
        return window.astype(np.float32, copy=False)

    base_idx = np.linspace(0.0, 1.0, num=current_len, dtype=np.float32)
    target_idx = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    resampled = np.empty((target_len, num_features), dtype=np.float32)
    for feature_idx in range(num_features):
        resampled[:, feature_idx] = np.interp(
            target_idx, base_idx, window[:, feature_idx]
        )
    return resampled


def _load_window_from_csv(
    csv_path: Path, target_len: int, feature_offset: int
) -> Tuple[np.ndarray, int]:
    df = pd.read_csv(csv_path, header=0)
    values = df.iloc[:, feature_offset:].to_numpy(dtype=np.float32)
    return resample_sequence(values, target_len), len(df)


def load_labeled_splits(
    base_dir: Union[str, Path],
    num_classes: int,
    target_len: int,
    validation_subject: Optional[str],
    excluded_subjects: Optional[Set[str]] = None,
    feature_offset: int = FEATURE_OFFSET,
) -> Tuple[DatasetSplit, DatasetSplit, Dict[str, Dict[str, int]], int]:
    """
    Load labeled squat windows and perform LOSO splitting.

    Parameters
    ----------
    base_dir:
        Directory containing class subfolders (class0, class1, ...).
    num_classes:
        Total number of classes.
    target_len:
        Desired sequence length after resampling.
    validation_subject:
        Subject ID reserved for validation (e.g., 'subject6'). If None, all
        data are returned as training split.
    excluded_subjects:
        Iterable of subject IDs to ignore entirely (neither train nor validation).
    feature_offset:
        Index of the first sensor feature column in the CSV.

    Returns
    -------
    train_split, val_split, subject_counts, max_original_len
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Labeled data directory not found: {base_path}")

    train_windows: List[np.ndarray] = []
    train_labels: List[int] = []
    train_paths: List[str] = []
    val_windows: List[np.ndarray] = []
    val_labels: List[int] = []
    val_paths: List[str] = []
    subject_counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}}
    max_original_len = 0

    excluded_subjects_set: Set[str] = set(excluded_subjects or [])
    if validation_subject and validation_subject in excluded_subjects_set:
        raise ValueError("Validation subject cannot be listed in excluded_subjects.")
    for class_idx in range(num_classes):
        class_dir = base_path / f"class{class_idx}"
        if not class_dir.exists():
            continue

        for csv_path in sorted(class_dir.glob("*.csv")):
            if csv_path.suffix.lower() != ".csv":
                continue
            try:
                window, original_len = _load_window_from_csv(
                    csv_path, target_len, feature_offset
                )
            except Exception as exc:
                print(f"[load_labeled_splits] Failed to load {csv_path}: {exc}")
                continue

            subject_name = extract_subject_name(csv_path.name)
            if subject_name in excluded_subjects_set:
                continue
            max_original_len = max(max_original_len, original_len)

            if validation_subject and subject_name == validation_subject:
                val_windows.append(window)
                val_labels.append(class_idx)
                val_paths.append(csv_path.as_posix())
                subject_counts["val"][subject_name] = subject_counts["val"].get(subject_name, 0) + 1
            else:
                train_windows.append(window)
                train_labels.append(class_idx)
                train_paths.append(csv_path.as_posix())
                subject_counts["train"][subject_name] = subject_counts["train"].get(subject_name, 0) + 1

    if validation_subject and not val_windows:
        raise ValueError(
            f"No validation samples found for subject '{validation_subject}'."
        )
    if not train_windows:
        raise ValueError("Training split is empty. Check the dataset directory or split.")

    train_split = DatasetSplit(train_windows, np.array(train_labels, dtype=np.int32), train_paths)
    val_split = DatasetSplit(val_windows, np.array(val_labels, dtype=np.int32), val_paths)
    return train_split, val_split, subject_counts, max_original_len


def load_labeled_windows_for_ssl(
    labeled_data_dir: Union[str, Path],
    target_len: int,
    num_classes: int,
    excluded_subjects: Optional[Set[str]] = None,
    feature_offset: int = FEATURE_OFFSET,
    target_ratio: float = 0.2,
    unlabeled_count: int = 0,
) -> List[np.ndarray]:
    """
    Load labeled windows from class folders, ignoring labels.
    Used to augment SSL training data while preventing LOSO leakage.
    Samples with balanced subject and class distribution.
    
    Parameters
    ----------
    labeled_data_dir:
        Directory containing class subfolders (class0, class1, ...).
    target_len:
        Desired sequence length after resampling.
    num_classes:
        Total number of classes.
    excluded_subjects:
        Subject IDs to exclude (e.g., validation subject to prevent LOSO leakage).
    feature_offset:
        Index of the first sensor feature column in the CSV.
    target_ratio:
        Target ratio of labeled data relative to unlabeled data (default: 0.2 = 20%).
    unlabeled_count:
        Number of unlabeled windows. Used to calculate target labeled count.
    
    Returns
    -------
    List of windows (labels are ignored), sampled with subject and class balance.
    """
    base_path = Path(labeled_data_dir)
    if not base_path.exists():
        return []
    
    excluded_subjects_set: Set[str] = set(excluded_subjects or [])
    
    # Group by (subject, class) for balanced sampling
    subject_class_windows: Dict[Tuple[str, int], List[np.ndarray]] = {}
    
    for class_idx in range(num_classes):
        class_dir = base_path / f"class{class_idx}"
        if not class_dir.exists():
            continue
        
        for csv_path in sorted(class_dir.glob("*.csv")):
            if csv_path.suffix.lower() != ".csv":
                continue
            try:
                subject_name = extract_subject_name(csv_path.name)
                if subject_name in excluded_subjects_set:
                    continue
                window, _ = _load_window_from_csv(csv_path, target_len, feature_offset)
                key = (subject_name, class_idx)
                if key not in subject_class_windows:
                    subject_class_windows[key] = []
                subject_class_windows[key].append(window)
            except Exception as exc:
                print(f"[load_labeled_windows_for_ssl] Failed to load {csv_path}: {exc}")
    
    if not subject_class_windows:
        return []
    
    # Calculate target count: unlabeled_count * target_ratio
    target_labeled_count = int(unlabeled_count * target_ratio) if unlabeled_count > 0 else 0
    
    # Get all available windows
    all_windows: List[np.ndarray] = []
    for windows_list in subject_class_windows.values():
        all_windows.extend(windows_list)
    
    print(f"[SSL-Labeled] Available: {len(all_windows)} windows in {len(subject_class_windows)} (subject,class) combinations")
    print(f"[SSL-Labeled] Target: {target_labeled_count} windows (20% of {unlabeled_count} unlabeled)")
    
    if target_labeled_count <= 0:
        # If no target specified, return all (for backward compatibility)
        return all_windows
    
    # If we have fewer or equal to target, return all
    if len(all_windows) <= target_labeled_count:
        print(f"[SSL-Labeled] Using all {len(all_windows)} available windows (less than target {target_labeled_count})")
        return all_windows
    
    # Sample with balance: equal samples per (subject, class) combination
    # First, determine how many samples per (subject, class) pair
    num_combinations = len(subject_class_windows)
    samples_per_combination = max(1, target_labeled_count // num_combinations)
    
    print(f"[SSL-Labeled] Sampling {samples_per_combination} per combination from {num_combinations} combinations")
    
    sampled_windows: List[np.ndarray] = []
    for (subject, class_idx), windows_list in subject_class_windows.items():
        if len(windows_list) <= samples_per_combination:
            # Take all if we have fewer than target per combination
            sampled_windows.extend(windows_list)
        else:
            # Randomly sample
            sampled = random.sample(windows_list, samples_per_combination)
            sampled_windows.extend(sampled)
    
    print(f"[SSL-Labeled] After initial sampling: {len(sampled_windows)} windows")
    
    # If we still need more samples to reach target, randomly sample from remaining
    if len(sampled_windows) < target_labeled_count:
        remaining_windows: List[np.ndarray] = []
        for (subject, class_idx), windows_list in subject_class_windows.items():
            if len(windows_list) > samples_per_combination:
                remaining_windows.extend(windows_list[samples_per_combination:])
        
        if remaining_windows:
            needed = target_labeled_count - len(sampled_windows)
            additional = random.sample(remaining_windows, min(needed, len(remaining_windows)))
            sampled_windows.extend(additional)
            print(f"[SSL-Labeled] Added {len(additional)} more windows from remaining pool")
    
    print(f"[SSL-Labeled] Final sampled: {len(sampled_windows)} windows (target: {target_labeled_count})")
    return sampled_windows


def load_unlabeled_windows(
    ssl_dir: Union[str, Path],
    target_len: int,
    feature_offset: int = FEATURE_OFFSET,
    excluded_subjects: Optional[Set[str]] = None,
) -> List[np.ndarray]:
    """
    Load and resample unlabeled SSL windows.
    
    Parameters
    ----------
    ssl_dir:
        Directory containing unlabeled SSL CSV files.
    target_len:
        Desired sequence length after resampling.
    feature_offset:
        Index of the first sensor feature column in the CSV.
    excluded_subjects:
        Subject IDs to exclude (e.g., validation subject to prevent LOSO leakage).
    """
    ssl_path = Path(ssl_dir)
    if not ssl_path.exists():
        return []

    excluded_subjects_set: Set[str] = set(excluded_subjects or [])
    windows: List[np.ndarray] = []
    for csv_path in sorted(ssl_path.glob("*.csv")):
        if csv_path.suffix.lower() != ".csv":
            continue
        try:
            subject_name = extract_subject_name(csv_path.name)
            if subject_name in excluded_subjects_set:
                continue
            window, _ = _load_window_from_csv(csv_path, target_len, feature_offset)
            windows.append(window)
        except Exception as exc:
            print(f"[load_unlabeled_windows] Failed to load {csv_path}: {exc}")
    return windows


def list_all_subjects(
    base_dir: Union[str, Path],
    num_classes: int,
    excluded_subjects: Optional[Set[str]] = None,
) -> List[str]:
    """
    Scan labeled folders and return a sorted list of unique subject IDs.
    Respects excluded_subjects (subjects to ignore entirely).
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Labeled data directory not found: {base_path}")

    excluded_subjects_set: Set[str] = set(excluded_subjects or [])
    subjects: Set[str] = set()
    for class_idx in range(num_classes):
        class_dir = base_path / f"class{class_idx}"
        if not class_dir.exists():
            continue
        for csv_path in sorted(class_dir.glob("*.csv")):
            name = extract_subject_name(csv_path.name)
            if name in excluded_subjects_set:
                continue
            subjects.add(name)
    return sorted(subjects)
