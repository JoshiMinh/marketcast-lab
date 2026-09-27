from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalPartitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    final_test: pd.DataFrame

    def boundaries(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "rows": len(partition),
                "start": partition["timestamp"].iloc[0].isoformat(),
                "end": partition["timestamp"].iloc[-1].isoformat(),
            }
            for name, partition in (
                ("train", self.train),
                ("validation", self.validation),
                ("final_test", self.final_test),
            )
        }


def chronological_partitions(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> ChronologicalPartitions:
    if frame["asset_id"].nunique() != 1:
        raise ValueError("Split input must contain exactly one preselected asset")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1")
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    train_end = int(len(ordered) * (1 - validation_fraction - test_fraction))
    validation_end = int(len(ordered) * (1 - test_fraction))
    if train_end < 2 or validation_end <= train_end or validation_end >= len(ordered):
        raise ValueError("Not enough rows for non-empty chronological partitions")
    parts = ChronologicalPartitions(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        final_test=ordered.iloc[validation_end:].copy(),
    )
    if not parts.train["timestamp"].max() < parts.validation["timestamp"].min():
        raise ValueError("Training and validation periods overlap")
    if not parts.validation["timestamp"].max() < parts.final_test["timestamp"].min():
        raise ValueError("Validation and final-test periods overlap")
    return parts


@dataclass(frozen=True)
class BacktestFold:
    fold: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def expanding_window_folds(
    row_count: int,
    *,
    folds: int,
    test_size: int,
    final_test_size: int,
    min_train_size: int,
) -> tuple[BacktestFold, ...]:
    if min(folds, test_size, final_test_size, min_train_size) < 1:
        raise ValueError("Fold parameters must be positive")
    selection_end = row_count - final_test_size
    first_test_start = selection_end - folds * test_size
    if first_test_start < min_train_size:
        raise ValueError("Not enough rows for requested folds and locked final test")
    result = []
    for fold in range(folds):
        test_start = first_test_start + fold * test_size
        result.append(
            BacktestFold(
                fold=fold + 1,
                train_indices=tuple(range(test_start)),
                test_indices=tuple(range(test_start, test_start + test_size)),
            )
        )
    return tuple(result)

