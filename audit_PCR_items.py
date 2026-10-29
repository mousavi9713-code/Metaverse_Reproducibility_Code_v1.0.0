#!/usr/bin/env python3
"""Reproducible item-level scoring audit of perceived cyber risk (PCR).

Run:
  python audit_PCR_items.py --data-dir /path/to/uploads

Uses the *original Persian raw survey exports*, not just the pre-scored file.
The first row in each raw CSV contains the original questionnaire item wording.
Reversal 6 - raw is applied ONLY to PCR1 and PCR2; PCR3/4 remain direct.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RESPONSE = {
    "خیلی مخالفم": 1, "مخالفم": 2,
    "تا حدودی": 3, "موافقم": 4, "خیلی موافقم": 5,
}
ALL_ITEMS = (
    [f"PU{i}" for i in range(1, 5)]
    + [f"PEU{i}" for i in range(1, 5)]
    + [f"SE{i}" for i in range(1, 4)]
    + [f"PE{i}" for i in range(1, 5)]
    + [f"PCR{i}" for i in range(1, 5)]
    + [f"PIIT{i}" for i in range(1, 5)]
    + [f"BI{i}" for i in range(1, 4)]
)
PCR = [f"PCR{i}" for i in range(1, 5)]

def load_source(path):
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    assert len(df.columns) >= 33
    texts = {key: str(df.iloc[0, j]) for j, key in enumerate(ALL_ITEMS, start=7)}
    source = df.iloc[1:, 7:33].copy()
    source.columns = ALL_ITEMS
    source = source.reset_index(drop=True)
    observed = set(source.to_numpy().ravel())
    assert observed.issubset(set(RESPONSE)), f"Unexpected responses: {observed - set(RESPONSE)}"
    numeric = source.map(lambda x: RESPONSE[x]).astype(int)
    return numeric, texts

def cronbach_alpha(x):
    a = np.asarray(x, dtype=float)
    return a.shape[1]/(a.shape[1]-1) * (1 - np.sum(a.var(axis=0,ddof=1)) / a.sum(axis=1).var(ddof=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    args = parser.parse_args()
    root = Path(args.data_dir)
    p, pword = load_source(root/"Paramedical Metaverse.csv")
    m, mword = load_source(root/"MedicalMetaverse.csv")
    assert all(pword[k] == mword[k] for k in PCR), "Question wording differs between faculties"
    raw = pd.concat([p, m], ignore_index=True)
    assert len(raw) == 185 and len(p) == 87 and len(m) == 98
    score = raw[PCR].copy()
    for key in ("PCR1", "PCR2"):
        score[key] = 6 - score[key]
    published = pd.read_csv(root/"Metaverse_25Item_Primary_Scored_Data.csv")
    for key in PCR:
        assert np.array_equal(score[key].to_numpy(), published[key].to_numpy()), key
        assert raw[key].between(1, 5).all()
    print("N=185 (87 paramedical, 98 medical); 740/740 PCR score values verified.")
    print("Missing responses:",score.isna().sum().sum())
    print("Coding: PCR1=6-raw, PCR2=6-raw, PCR3=raw, PCR4=raw.")
    print("Cronbach alpha scored:", round(cronbach_alpha(score), 6))
    print("Cronbach alpha all unreversed (NOT conceptually interpretable as cyber risk):",
          round(cronbach_alpha(raw[PCR]), 6))
    print("Original Persian item wording:")
    for key in PCR: print(key, pword[key])
    print("Item means, SD and corrected item-total:")
    for key in PCR:
        c = np.corrcoef(score[key],score.drop(columns=[key]).sum(axis=1))[0,1]
        print(key, "raw_mean",round(raw[key].mean(),4),
              "scored_mean",round(score[key].mean(),4),
              "scored_SD",round(score[key].std(ddof=1),4),
              "corrected_item_total",round(c,4))
    print("Pearson correlation of scored PCR indicators:")
    print(score.corr().round(4).to_string())
    print("Pooled result only; the source Appendix A does not document an unequivocal PCR scoring key.")
    print("No causal explanation for poor internal consistency can be inferred.")

if __name__ == "__main__":
    main()
