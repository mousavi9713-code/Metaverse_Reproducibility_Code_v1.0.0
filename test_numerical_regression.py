#!/usr/bin/env python3
"""Numerical regression tests for scored 185-person data.

Running this checks internal self-consistency of the custom reconstruction.
It is NOT independent validation against SmartPLS/cSEM.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PLS_PIIT_PCR_sensitivity_reproducible import FULL_BLOCKS,PATHS,pls_mode_a,alpha
from metaverse_full_structural_reanalysis import blocks_for
from metaverse_composite_diagnostics import (fit_weights,harman_pca_share,
 full_collinearity_vifs,approximate_srmr,pooled_project)

def near(actual,expected,atol,label):
    if not np.isclose(actual,expected,atol=atol,rtol=0):
        raise AssertionError(f"{label}: {actual:.6f} expected {expected:.6f}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    args=p.parse_args()
    df=pd.read_csv(args.input)
    assert len(df)==185 and df.ID.is_unique
    assert (df.Faculty_EN=="Medicine").sum()==98
    assert (df.Faculty_EN=="Paramedical sciences").sum()==87
    m25=pls_mode_a(df,*blocks_for("25item"))
    m26=pls_mode_a(df,*blocks_for("26item"))
    m17=pls_mode_a(df,*blocks_for("reduced17"))
    near(m25["r2"]["BI"],.7006368183683143,1e-8,"25 item R² BI")
    near(m26["r2"]["BI"],.697508099608165,1e-8,"26 item R² BI")
    near(m17["r2"]["BI"],.629549834558299,1e-8,"17 item R² BI")
    near(m25["coefficients"][("PIIT","BI")],.419378,1e-5,"PIIT→BI")
    near(m25["coefficients"][("PE","BI")],.387747,1e-5,"PE→BI")
    near(m17["coefficients"][("PE","BI")],.558545,1e-5,"reduced PE→BI")
    near(alpha(df[["PIIT1","PIIT2","PIIT3","PIIT4"]]),.493,0.001,"alpha PIIT")
    near(alpha(df[["PCR1","PCR2","PCR3","PCR4"]]),.035,0.001,"alpha PCR")
    near(harman_pca_share(df,sum(FULL_BLOCKS.values(),[])),.4603558,1e-5,"Harman PCA")
    f=fit_weights(df,FULL_BLOCKS,PATHS)
    near(approximate_srmr(df,FULL_BLOCKS,f),.104684,1e-5,"custom SRMR")
    observed={k:df[v].mean(1).to_numpy(float) for k,v in FULL_BLOCKS.items()}
    v=full_collinearity_vifs(observed)
    near(v["PIIT"],1.923,.001,"VIF PIIT observed")
    near(v["PCR"],1.215,.001,"VIF PCR observed")
    mean={k:df[v].to_numpy(float).mean(0) for k,v in FULL_BLOCKS.items()}
    sd={k:df[v].to_numpy(float).std(0,ddof=0) for k,v in FULL_BLOCKS.items()}
    a=fit_weights(df[df.Faculty_EN=="Paramedical sciences"],FULL_BLOCKS,PATHS)
    b=fit_weights(df[df.Faculty_EN=="Medicine"],FULL_BLOCKS,PATHS)
    pa=pooled_project(df,FULL_BLOCKS,a,mean,sd)
    pb=pooled_project(df,FULL_BLOCKS,b,mean,sd)
    for k,exp in {"PIIT":.942,"PCR":.761,"SE":.997,"PEU":.992,
                  "PU":1.000,"PE":.998,"BI":1.000}.items():
        near(np.corrcoef(pa[k],pb[k])[0,1],exp,.0011,f"MICOM {k}")
    print("PASS: sample size, raw coefficient point estimates, three BI R²,")
    print("      alpha, Harman, approx SRMR, observed-score VIF, MICOM step2 composition.")
    print("NOTE: does not validate 5,000-resample inference or standard-package equivalence.")
if __name__=="__main__":main()
