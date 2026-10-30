"""Map-level cross-validation splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossValidationFold:
    """One outer test fold with an inner checkpoint-validation subset."""

    fold: int
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "fold": self.fold,
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }


def create_nested_folds(
    sample_keys: list[str],
    *,
    folds: int = 5,
    inner_validation_fraction: float = 0.10,
    seed: int = 42,
) -> list[CrossValidationFold]:
    """Create outer folds and deterministic inner validation partitions.

    Every map appears in exactly one outer test set.  The inner validation set
    is selected only from the corresponding outer-training maps and is used for
    checkpoint selection; it is never included in that fold's model fitting.
    """

    keys = np.asarray(sorted(set(sample_keys)), dtype=object)
    if len(keys) != len(sample_keys):
        raise ValueError("sample_keys must be unique")
    if not 2 <= folds <= len(keys):
        raise ValueError("folds must be between 2 and the number of samples")
    if not 0.0 < inner_validation_fraction < 0.5:
        raise ValueError("inner_validation_fraction must be in (0, 0.5)")

    rng = np.random.default_rng(seed)
    shuffled = keys.copy()
    rng.shuffle(shuffled)
    outer_parts = np.array_split(shuffled, folds)
    records: list[CrossValidationFold] = []
    for fold_index, test_values in enumerate(outer_parts, start=1):
        test = {str(value) for value in test_values}
        outer_train = np.asarray(
            [str(value) for value in shuffled if str(value) not in test],
            dtype=object,
        )
        inner_rng = np.random.default_rng(seed + 10_000 + fold_index)
        inner_rng.shuffle(outer_train)
        val_count = max(1, int(round(len(outer_train) * inner_validation_fraction)))
        val = tuple(sorted(str(value) for value in outer_train[:val_count]))
        train = tuple(sorted(str(value) for value in outer_train[val_count:]))
        records.append(
            CrossValidationFold(
                fold=fold_index,
                train=train,
                val=val,
                test=tuple(sorted(test)),
            )
        )
    validate_nested_folds(records, expected_keys=set(str(key) for key in keys))
    return records


def validate_nested_folds(
    records: list[CrossValidationFold], *, expected_keys: set[str]
) -> None:
    """Reject overlap, missing maps, and repeated outer-test assignments."""

    observed_test: list[str] = []
    for record in records:
        train, val, test = map(set, (record.train, record.val, record.test))
        if train & val or train & test or val & test:
            raise ValueError(f"Fold {record.fold} contains overlapping partitions")
        if train | val | test != expected_keys:
            raise ValueError(f"Fold {record.fold} does not cover the complete sample set")
        observed_test.extend(record.test)
    if len(observed_test) != len(set(observed_test)):
        raise ValueError("A sample appears in more than one outer test fold")
    if set(observed_test) != expected_keys:
        raise ValueError("Outer test folds do not cover every sample exactly once")
