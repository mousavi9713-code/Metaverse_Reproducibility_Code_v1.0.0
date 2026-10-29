#!/usr/bin/env python3
"""Reproduce Holm adjusted p-values in Table 4 from values printed in Manuscript(4).

Six p-values are published only as <0.001, so the output for these paths
reports a conservative adjusted upper bound, not a fabricated exact value.
For exact adjusted p-values, replace 0.001 with the original unrounded p-values.
No participant data are required.
"""
import csv
from pathlib import Path

tests = [
    ("H1","PIIT→SE", "<0.001"),
    ("H2","PIIT→PU", "<0.001"),
    ("H3","PIIT→PEU", "<0.001"),
    ("H4","PIIT→BI", "<0.001"),
    ("H5","SE→PU", "0.035"),
    ("H6","SE→PEU", "0.009"),
    ("H7","PCR→PU", "0.102"),
    ("H8","PCR→BI", "0.747"),
    ("H9","PEU→PU", "0.271"),
    ("H10","PEU→PE", "<0.001"),
    ("H11","PEU→BI", "0.898"),
    ("H12","PU→BI", "0.057"),
    ("H13","PE→BI", "<0.001"),
]
m=len(tests)
# A placeholder 0.001 is an upper bound for every '<0.001' p-value.
p=[0.001 if x[2].startswith("<") else float(x[2]) for x in tests]
order=sorted(range(m),key=lambda k:p[k])
adj=[None]*m
running=0.0
for rank,k in enumerate(order):
    running=max(running,min(1.0,(m-rank)*p[k]))
    adj[k]=running
expected={"H5":.210,"H6":.063,"H7":.408,"H8":1.0,"H9":.813,"H11":1.0,"H12":.285}
for k,v in expected.items():
    idx=next(i for i,t in enumerate(tests) if t[0]==k)
    assert abs(adj[idx]-v)<1e-10,(k,adj[idx])
assert sum(x<0.05 for x in adj)==6
rows=[]
for (hyp,path,raw),a in zip(tests,adj):
    adjusted=("<0.013" if raw=="<0.001" else f"{a:.3f}")
    rows.append((hyp,path,raw,adjusted,"Supported" if a<0.05 else "Not supported"))
dest=Path(__file__).resolve().with_name("Table4_Holm_recomputed.csv")
with dest.open("w",newline="",encoding="utf-8-sig") as fh:
    w=csv.writer(fh)
    w.writerow(["Hypothesis","Path","Unadjusted p as printed","Holm-adjusted p / upper bound","Decision"])
    w.writerows(rows)
print("Holm-adjusted Table 4 saved:",dest)
for row in rows:
    print(" | ".join(row))
print("Note: '<0.013' represents a conservative upper bound, not an exact adjusted p.")
