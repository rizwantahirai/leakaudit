import numpy as np
import pandas as pd
import pytest

from leakaudit import audit, leakage_rate, leak_free_split, assert_no_leakage


def _toy(seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for lid in range(200):
        for j in range(rng.randint(1, 4)):
            rows.append({"image_id": f"L{lid}_{j}", "lesion_id": f"L{lid}",
                         "cls": rng.choice(["a", "b", "c"])})
    return pd.DataFrame(rows)


def test_leak_free_split_has_zero_overlap():
    df = leak_free_split(_toy(), group="lesion_id", label="cls", seed=1)
    # every lesion sits in exactly one partition
    assert (df.groupby("lesion_id")["split"].nunique() == 1).all()
    assert set(df["split"]) == {"train", "val", "test"}
    assert leakage_rate(df, group="lesion_id") == 0.0
    assert_no_leakage(df, group="lesion_id")  # must not raise


def test_audit_detects_naive_leakage():
    df = _toy()
    rng = np.random.RandomState(2)
    df["split"] = rng.choice(["train", "test"], size=len(df))
    rep = audit(df, group="lesion_id", split="split")
    assert rep.leakage_rate > 0.0          # naive split leaks
    assert rep.is_leak_free is False
    with pytest.raises(AssertionError):
        assert_no_leakage(df, group="lesion_id")


def test_split_without_label_works():
    df = leak_free_split(_toy(), group="lesion_id", label=None, seed=3)
    assert leakage_rate(df, group="lesion_id") == 0.0


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        leak_free_split(_toy(), group="lesion_id", ratios=(0.6, 0.2, 0.3))
