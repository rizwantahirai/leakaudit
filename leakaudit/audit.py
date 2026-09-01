"""
Leakage auditing for grouped image datasets.

The core idea: many image datasets contain several images of the *same* underlying
entity (a skin lesion, a patient, a scan). If a random *image-level* train/test split
places different images of one entity on both sides, the model can memorise the entity
and be re-tested on it, inflating reported performance. This module quantifies that
contamination and verifies that a split is leak-free.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class LeakageReport:
    n_rows: int
    n_groups: int
    max_images_per_group: int
    mean_images_per_group: float
    duplication_rate: float          # 1 - n_groups / n_rows
    leakage_rate: Optional[float]    # fraction of TEST rows whose group is also in TRAIN
    groups_spanning_partitions: int  # groups that appear in >1 split
    is_leak_free: Optional[bool]
    per_split_counts: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Leakage audit",
            "-------------",
            f"images                 : {self.n_rows:,}",
            f"groups                 : {self.n_groups:,}",
            f"images/group (max, mean): {self.max_images_per_group}, "
            f"{self.mean_images_per_group:.2f}",
            f"duplication rate       : {self.duplication_rate:.1%}  "
            f"(share of images that are extra views of a group)",
        ]
        if self.per_split_counts:
            lines.append(f"split sizes            : {self.per_split_counts}")
        if self.leakage_rate is not None:
            verdict = "LEAK-FREE" if self.is_leak_free else "LEAKAGE DETECTED"
            lines += [
                f"groups spanning splits : {self.groups_spanning_partitions}",
                f"naive test leakage     : {self.leakage_rate:.1%}  "
                f"(fraction of test images sharing a group with training)",
                f"verdict                : {verdict}",
            ]
        return "\n".join(lines)


def leakage_rate(df: pd.DataFrame, group: str, split: str = "split",
                 train_value: str = "train", test_value: str = "test") -> float:
    """Fraction of test-set rows whose group also occurs in the training set.

    This is the quantity to report when arguing that an image-level split contaminates
    the test set. 0.0 means leak-free; higher means more contamination.
    """
    train_groups = set(df.loc[df[split] == train_value, group])
    test = df[df[split] == test_value]
    if len(test) == 0:
        raise ValueError(f"no rows with {split}=={test_value!r}")
    return float(test[group].isin(train_groups).sum()) / len(test)


def groups_spanning(df: pd.DataFrame, group: str, split: str = "split") -> pd.Index:
    """Group ids that appear in more than one split partition (i.e. leak)."""
    g = df.groupby(group)[split].nunique()
    return g[g > 1].index


def assert_no_leakage(df: pd.DataFrame, group: str, split: str = "split") -> None:
    """Raise AssertionError if any group appears in more than one partition.

    Use this as a checked invariant right after creating a split, so leakage is a
    guaranteed-absent property rather than an assumption.
    """
    bad = groups_spanning(df, group, split)
    assert len(bad) == 0, (
        f"leakage: {len(bad)} group(s) appear in >1 split partition, e.g. "
        f"{list(bad[:5])}"
    )


def audit(df: pd.DataFrame, group: str, split: Optional[str] = "split",
          train_value: str = "train", test_value: str = "test") -> LeakageReport:
    """Audit a dataset for group-level leakage.

    Parameters
    ----------
    df : DataFrame with one row per image.
    group : column naming the shared entity (e.g. ``lesion_id`` or ``patient_id``).
    split : column with the partition label (``train``/``val``/``test``); pass ``None``
            to report only the duplication structure without a leakage rate.
    """
    if group not in df.columns:
        raise KeyError(f"group column {group!r} not in dataframe")
    n_rows = len(df)
    sizes = df.groupby(group).size()
    n_groups = int(sizes.shape[0])
    dup_rate = 1.0 - n_groups / n_rows if n_rows else 0.0

    lr = span = None
    is_free = None
    counts: dict = {}
    if split is not None and split in df.columns:
        counts = df[split].value_counts().to_dict()
        span = int(len(groups_spanning(df, group, split)))
        is_free = span == 0
        if (df[split] == test_value).any() and (df[split] == train_value).any():
            lr = leakage_rate(df, group, split, train_value, test_value)

    return LeakageReport(
        n_rows=n_rows, n_groups=n_groups,
        max_images_per_group=int(sizes.max()) if n_rows else 0,
        mean_images_per_group=float(sizes.mean()) if n_rows else 0.0,
        duplication_rate=dup_rate, leakage_rate=lr,
        groups_spanning_partitions=span if span is not None else 0,
        is_leak_free=is_free, per_split_counts=counts,
    )
