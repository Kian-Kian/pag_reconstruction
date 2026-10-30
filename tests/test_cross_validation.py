from __future__ import annotations

from pag_recon.support.cross_validation import create_nested_folds


def test_five_fold_partition_is_complete_disjoint_and_deterministic() -> None:
    keys = [f"train/{index:03d}" for index in range(100)]
    first = create_nested_folds(keys, folds=5, seed=42)
    second = create_nested_folds(keys, folds=5, seed=42)
    assert first == second
    assert [len(record.train) for record in first] == [72] * 5
    assert [len(record.val) for record in first] == [8] * 5
    assert [len(record.test) for record in first] == [20] * 5
    test_keys = [key for record in first for key in record.test]
    assert len(test_keys) == len(set(test_keys)) == 100
    for record in first:
        assert not (set(record.train) & set(record.val))
        assert not (set(record.train) & set(record.test))
        assert not (set(record.val) & set(record.test))


def test_duplicate_cross_validation_keys_are_rejected() -> None:
    try:
        create_nested_folds(["train/001", "train/001"], folds=2)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate keys should be rejected")
