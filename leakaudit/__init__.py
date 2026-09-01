"""leakaudit: detect data leakage and build leak-free splits for grouped image datasets.

Quick start
-----------
    import pandas as pd
    from leakaudit import audit, leak_free_split

    df = pd.read_csv("my_dataset.csv")          # one row per image
    print(audit(df, group="lesion_id").summary()) # inspect duplication/leakage
    df = leak_free_split(df, group="lesion_id", label="diagnosis")
    df.to_csv("my_dataset_leakfree.csv", index=False)
"""
from .audit import (
    audit, leakage_rate, assert_no_leakage, groups_spanning, LeakageReport,
)
from .split import leak_free_split

__version__ = "0.1.0"
__all__ = [
    "audit", "leakage_rate", "assert_no_leakage", "groups_spanning",
    "LeakageReport", "leak_free_split",
]
