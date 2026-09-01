"""
Leak-free, group-aware train/val/test splitting for image datasets.

A leak-free split keeps *all* images of a given group (lesion/patient/scan) inside a
single partition, so the test set is genuinely unseen. When a class label is available
the split is also stratified, so class proportions are preserved across partitions.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .audit import assert_no_leakage


def leak_free_split(
    df: pd.DataFrame,
    group: str,
    label: Optional[str] = None,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    split_col: str = "split",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``split`` column (train/val/test).

    Every group is assigned to exactly one partition (verified). If ``label`` is given,
    the split is class-stratified via ``StratifiedGroupKFold``; otherwise it is a plain
    grouped split via ``GroupShuffleSplit``.

    Parameters
    ----------
    df : one row per image.
    group : column naming the shared entity that must not cross partitions.
    label : optional class column for stratification.
    ratios : (train, val, test) fractions; must sum to 1.
    seed : random seed for reproducibility.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    if group not in df.columns:
        raise KeyError(f"group column {group!r} not in dataframe")
    tr_frac, va_frac, te_frac = ratios
    out = df.copy().reset_index(drop=True)

    if label is not None:
        from sklearn.model_selection import StratifiedGroupKFold
        # first carve off the training set, then halve the remainder into val/test
        n_splits = max(2, round(1.0 / (va_frac + te_frac)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr_idx, hold_idx = next(iter(sgkf.split(out, out[label], groups=out[group])))
        hold = out.iloc[hold_idx]
        # split the holdout groups into val/test, keeping groups intact
        rng = np.random.RandomState(seed)
        hold_groups = hold[group].drop_duplicates().sample(frac=1.0, random_state=seed)
        cut = int(round(len(hold_groups) * va_frac / (va_frac + te_frac)))
        val_groups = set(hold_groups.iloc[:cut])
        out[split_col] = "train"
        out.loc[out.index.isin(hold_idx), split_col] = "test"
        out.loc[out[group].isin(val_groups), split_col] = "val"
    else:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=va_frac + te_frac, random_state=seed)
        tr_idx, hold_idx = next(iter(gss.split(out, groups=out[group])))
        out[split_col] = "train"
        out.loc[out.index.isin(hold_idx), split_col] = "test"
        hold = out.iloc[hold_idx]
        gss2 = GroupShuffleSplit(n_splits=1,
                                 test_size=te_frac / (va_frac + te_frac),
                                 random_state=seed)
        va_rel, _ = next(iter(gss2.split(hold, groups=hold[group])))
        val_pos = hold.index[va_rel]
        out.loc[out.index.isin(val_pos), split_col] = "val"

    assert_no_leakage(out, group, split_col)   # checked invariant, not an assumption
    return out
