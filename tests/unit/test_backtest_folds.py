import pytest

from market_forecast.evaluation import expanding_window_folds


def test_expanding_folds_share_test_size_and_leave_final_test_locked() -> None:
    folds = expanding_window_folds(
        200, folds=3, test_size=20, final_test_size=20, min_train_size=100
    )
    assert [len(fold.train_indices) for fold in folds] == [120, 140, 160]
    assert all(len(fold.test_indices) == 20 for fold in folds)
    assert max(index for fold in folds for index in fold.test_indices) == 179
    assert set(range(180, 200)).isdisjoint(
        index for fold in folds for index in (*fold.train_indices, *fold.test_indices)
    )


def test_impossible_fold_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="Not enough rows"):
        expanding_window_folds(100, folds=3, test_size=20, final_test_size=20, min_train_size=30)

