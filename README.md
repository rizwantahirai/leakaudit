# leakaudit

**Detect data leakage and build leak-free, group-aware train/val/test splits for image datasets.**

Many image datasets contain several images of the *same* underlying entity — a skin
lesion, a patient, a scan. If a naive *image-level* split scatters those images across
train and test, a model can memorise the entity and be re-tested on it, silently
**inflating reported performance** (and, in fairness studies, the subgroup metrics used to
claim equity). `leakaudit` quantifies this contamination and produces splits that are
verified leak-free.

This tool accompanies the paper *"Rethinking skin-tone fairness evaluation in dermatology
AI: data leakage, luminance proxies, and a tone-adaptive pipeline"* — but it is
dataset-agnostic and works on any grouped image dataset.

## Install

```bash
pip install leakaudit          # from PyPI (once published)
# or, from a checkout:
pip install .
```

## Use it in Python

```python
import pandas as pd
from leakaudit import audit, leak_free_split

df = pd.read_csv("my_dataset.csv")     # one row per image; a column identifies the group

# 1. Audit an existing split for leakage
print(audit(df, group="lesion_id", split="split").summary())

# 2. Build a verified leak-free, class-stratified split
df = leak_free_split(df, group="lesion_id", label="diagnosis",
                     ratios=(0.70, 0.15, 0.15), seed=42)
df.to_csv("my_dataset_leakfree.csv", index=False)
```

## Use it from the command line

```bash
# audit an existing split
leakaudit audit data.csv --group lesion_id --split split

# create a leak-free split and write it out
leakaudit split data.csv --group lesion_id --label diagnosis \
    --ratios 0.7 0.15 0.15 --seed 42 --out data_leakfree.csv
```

Example output on a naive image-level split:

```
Leakage audit
-------------
images                 : 10,015
groups                 : 7,470
images/group (max, mean): 6, 1.34
duplication rate       : 25.4%
naive test leakage     : 36.1%   (fraction of test images sharing a group with training)
verdict                : LEAKAGE DETECTED
```

## What it does

- **`audit(df, group, split=...)`** — reports the number of images vs. groups, the
  duplication rate, how many groups span partitions, and the **leakage rate** (fraction of
  test images whose group also appears in training). Returns a `LeakageReport`.
- **`leak_free_split(df, group, label=..., ratios=..., seed=...)`** — assigns every group
  to exactly one partition (class-stratified when a label is given), and **verifies zero
  group overlap** as a checked invariant.
- **`assert_no_leakage(df, group, split)`** — drop into any pipeline to guarantee a split
  is leak-free (raises if not).

## Reporting checklist for grouped-data evaluation

When reporting results (especially subgroup / fairness metrics) on datasets with multiple
images per entity, we recommend:

1. **Split by group, not by image** — keep all images of a lesion/patient in one
   partition, and verify zero group overlap.
2. **Report the leakage rate** of the naive image-level split, so readers can judge the
   inflation risk.
3. **Report subgroup test counts and consider power** — small subgroups support only weak
   claims.
4. **Report confidence intervals and significance over multiple seeds.**

## Citation

If you use this tool, please cite the accompanying paper (details to be added on
publication).

## License

MIT — see [LICENSE](LICENSE).
