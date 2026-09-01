"""Minimal end-to-end demo on a synthetic dataset with known leakage.

Run:  python examples/demo.py
"""
import numpy as np
import pandas as pd
from leakaudit import audit, leak_free_split

# --- build a synthetic dataset: 400 lesions, 1-4 images each, 3 classes ---
rng = np.random.RandomState(0)
rows = []
for lid in range(400):
    k = rng.randint(1, 5)                      # images of this lesion
    cls = rng.choice(["benign", "malignant", "other"], p=[0.6, 0.25, 0.15])
    for j in range(k):
        rows.append({"image_id": f"L{lid}_{j}", "lesion_id": f"L{lid}", "diagnosis": cls})
df = pd.DataFrame(rows)

# --- a NAIVE image-level split (this leaks: images of one lesion span partitions) ---
naive = df.copy()
naive["split"] = rng.choice(["train", "val", "test"], size=len(df), p=[0.7, 0.15, 0.15])
print("=== NAIVE image-level split ===")
print(audit(naive, group="lesion_id", split="split").summary())

# --- the leak-free grouped, class-stratified split ---
print("\n=== leak-free lesion-grouped split ===")
clean = leak_free_split(df, group="lesion_id", label="diagnosis", seed=42)
print(audit(clean, group="lesion_id", split="split").summary())
