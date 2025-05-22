import pandas as pd
import numpy as np
import pytest
from src.ml.time_split import split_fixed, walk_forward_splits
from datetime import datetime

def test_split_fixed() -> None:
    """Testet den festen Split von Zeitreihen."""
    idx = pd.date_range("2020-01-01", periods=10)
    df = pd.DataFrame({"Close": np.arange(10)}, index=idx)
    train, sim = split_fixed(df, idx[5], idx[6], idx[8])
    assert len(train) == 6
    assert len(sim) == 3

def test_walk_forward_splits() -> None:
    """Testet Walk-Forward-Splits."""
    idx = pd.date_range("2020-01-01", periods=20)
    df = pd.DataFrame({"Close": np.arange(20)}, index=idx)
    splits = walk_forward_splits(df, window=5, step=3)
    assert len(splits) > 0
    for train, test in splits:
        assert len(train) == 5
        assert len(test) <= 3 