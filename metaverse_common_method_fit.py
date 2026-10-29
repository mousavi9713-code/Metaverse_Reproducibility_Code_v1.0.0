#!/usr/bin/env python3
"""Reproduce common-method screens and approximate fit from participant-level data.

Important: full collinearity VIF in manuscript S2 is based on arithmetic-mean
OBSERVED construct scores, not PLS latent-composite scores. Both are reported
to disclose the scoring choice. Neither rules out common-method bias.
Approximate SRMR here is a reconstruction, not standard software fit.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PLS_PIIT_PCR_sensitivity_reproducible import FULL_BLOCKS,PATHS
from metaverse_composite_diagnostics import (fit_weights,harman_pca_share,
    full_collinearity_vifs,approximate_srmr)

def run(d,blocks=FULL_BLOCKS,paths=PATHS):
    fit=fit_weights(d,blocks,paths)
    observed={k:d[v].mean(axis=1).to_numpy(float) for k,v in blocks.items()}
    latent=fit["scores"]
    vobs=full_collinearity_vifs(observed)
    vlatent=full_collinearity_vifs(latent)
    items=sum(blocks.values(),[])
    return (harman_pca_share(d,items),vobs,vlatent,
            approximate_srmr(d,blocks,fit))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("output"))
    a=ap.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    d=pd.read_csv(a.input)
    common,vobs,vlatent,srmr=run(d)
    rows=[{"diagnostic":"Harman first PCA eigenvalue / p","value":common*100,
           "interpretation":"first component %; not proof of no common method bias"},
          {"diagnostic":"approximate overall SRMR","value":srmr,
           "interpretation":"custom approximation, not package-validated fit"}]
    for k in FULL_BLOCKS:
        rows.append({"diagnostic":f"observed-score full-collinearity VIF {k}",
                     "value":vobs[k],"interpretation":"arithmetic-mean construct scores"})
        rows.append({"diagnostic":f"PLS-composite full-collinearity VIF {k}",
                     "value":vlatent[k],"interpretation":"PLS composite scores, different definition"})
    for label,fac in [("Paramedical sciences","paramedical"),("Medicine","medical")]:
        sub=d[d.Faculty_EN==label]
        _,_,_,ss=run(sub)
        rows.append({"diagnostic":f"approximate {fac} SRMR","value":ss,
                     "interpretation":"custom approximation"})
    pd.DataFrame(rows).to_csv(a.out/"common_method_and_approx_fit.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__":main()
