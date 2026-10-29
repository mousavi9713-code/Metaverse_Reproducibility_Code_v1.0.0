#!/usr/bin/env python3
"""Independent data-origin, scoring and preliminary sensitivity audit.

Usage:
    python metaverse_initial_audit.py --data-dir /path/to/folder

NOTE: The exploratory HC3 OLS sensitivity regression is NOT a replacement
for the authors' 25-item PLS-SEM, bootstrap, MICOM, or PLSpredict code.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ITEMS = (
    [f"PU{i}" for i in range(1, 5)]
    + [f"PEU{i}" for i in range(1, 5)]
    + [f"SE{i}" for i in range(1, 4)]
    + [f"PE{i}" for i in range(1, 5)]
    + [f"PCR{i}" for i in range(1, 5)]
    + [f"PIIT{i}" for i in range(1, 5)]
    + [f"BI{i}" for i in range(1, 4)]
)
KEY = {"خیلی مخالفم": 1, "مخالفم": 2, "تا حدودی": 3, "موافقم": 4, "خیلی موافقم": 5}
REVERSE = {"PCR1", "PCR2", "PIIT3"}


def alpha(frame):
    x = frame.to_numpy(dtype=float)
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1))


def load_raw(path):
    # Row 1 contains the Persian question texts; actual responses start on row 2.
    frame = pd.read_csv(path, low_memory=False)
    if len(frame.columns) < 33:
        raise ValueError(f"Missing columns: {path}")
    raw_items = frame.iloc[1:, 7:33].copy().reset_index(drop=True)
    raw_items.columns = ITEMS
    observed = set(raw_items.to_numpy().ravel())
    if not observed.issubset(set(KEY)):
        raise ValueError(f"Unexpected response categories in {path}: {observed - set(KEY)}")
    numeric = raw_items.replace(KEY).astype(int)
    scored = numeric.copy()
    for item in REVERSE:
        scored[item] = 6 - scored[item]
    return scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.data_dir
    primary_path = root / "Metaverse_25Item_Primary_Scored_Data.csv"
    full_path = root / "Metaverse_Cleaned_Scored_Data.csv"
    primary = pd.read_csv(primary_path, low_memory=False)
    full = pd.read_csv(full_path, low_memory=False)
    par = load_raw(root / "Paramedical Metaverse.csv")
    med = load_raw(root / "MedicalMetaverse.csv")
    expected = pd.concat([par, med], ignore_index=True)
    if primary.shape[0] != 185 or full.shape[0] != 185:
        raise AssertionError("Sample size is not 185.")
    if not np.array_equal(expected[ITEMS].to_numpy(), primary[ITEMS].to_numpy()):
        raise AssertionError("Raw-to-scored mismatch")
    if not np.array_equal(full[ITEMS].to_numpy(), primary[ITEMS].to_numpy()):
        raise AssertionError("Primary vs full raw item mismatch")
    assert primary["ID"].is_unique and full["ID"].is_unique
    assert not primary[ITEMS].isna().any().any()
    assert primary[ITEMS].apply(lambda c: c.between(1, 5).all()).all()
    assert np.allclose(primary["PU_primary"], primary[["PU1", "PU2", "PU3"]].mean(axis=1))
    assert np.allclose(full["PU_mean"], full[["PU1", "PU2", "PU3", "PU4"]].mean(axis=1))
    for file in [primary_path, full_path]:
        print("FILE SHA256", file.name, hashlib.sha256(file.read_bytes()).hexdigest())
    print("RAW VERIFIED N", len(expected), "Paramedical", len(par), "Medical", len(med))
    print("ALL 26 ITEM RESPONSES MATCH: TRUE")
    print("PRIMARY BI mean, sd", round(primary.BI_primary.mean(), 6),
          round(primary.BI_primary.std(ddof=1), 6))
    for group in ["Paramedical sciences", "Medicine"]:
        x = primary.loc[primary.Faculty_EN.eq(group), "BI_primary"]
        print("BI", group, len(x), round(x.mean(), 6), round(x.std(ddof=1), 6))
    print("alpha PIIT", round(alpha(primary[[f"PIIT{i}" for i in range(1, 5)]]), 6))
    print("alpha PCR", round(alpha(primary[[f"PCR{i}" for i in range(1, 5)]]), 6))
    print("alpha PU 3-item", round(alpha(primary[["PU1", "PU2", "PU3"]]), 6))
    print("alpha PU 4-item", round(alpha(primary[["PU1", "PU2", "PU3", "PU4"]]), 6))
    print("PU mean primary", round(primary.PU_primary.mean(), 6),
          "PU mean full", round(full.PU_mean.mean(), 6))

    # Clearly labeled exploratory observed-score sensitivity, not PLS-SEM.
    selected = ["PE_primary", "PU_primary", "PEU_primary", "SE_primary"]
    z = primary[["BI_primary"] + selected].astype(float)
    z = (z - z.mean()) / z.std(ddof=0)
    result = sm.OLS(z.BI_primary, sm.add_constant(z[selected])).fit(cov_type="HC3")
    print("\nEXPLORATORY SENSITIVITY: standardized OLS, HC3; PIIT and PCR excluded")
    print("n", int(result.nobs), "R2", round(result.rsquared, 6))
    lohi = result.conf_int()
    for c in selected:
        print(c, "standardized beta", round(result.params[c], 6),
              "95% HC3 CI", tuple(np.round(lohi.loc[c].to_numpy(), 6)),
              "p", round(result.pvalues[c], 6))
    print("\nNOTICE: This is NOT PLS-SEM, PLSpredict, or independent verification of manuscript's beta/R2.")


if __name__ == "__main__":
    main()
