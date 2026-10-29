#!/usr/bin/env python3
"""Redraw structural/predictive/IPMA figures from reproducible output tables.

Figures 2,3,4 are independent renditions; visual design is not guaranteed to
match the initially submitted figures. Axes/reference-line rules are explicit.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("outputs"))
    a=p.parse_args()
    output=a.output
    src=pd.read_csv(output/"25item_paths_bootstrap.csv")
    fig,ax=plt.subplots(figsize=(9,7))
    ys=np.arange(len(src))
    ax.errorbar(src.beta,ys,xerr=[src.beta-src.ci_low_95,
                                  src.ci_high_95-src.beta],fmt="o",capsize=3)
    ax.axvline(0,linestyle="--",linewidth=1)
    ax.set_yticks(ys,src.path)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized path coefficient β (95% percentile bootstrap CI)")
    fig.tight_layout()
    fig.savefig(output/"Figure2_path_coefficients_recreated.png",dpi=300)
    plt.close(fig)

    cv=pd.read_csv(output/"primary_25item_CV_metrics.csv")
    cv=cv[cv.Target.eq("BI composite")].copy()
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(cv.Model,cv.RMSE)
    ax.set_ylabel("Repeated cross-validated RMSE (BI composite)")
    ax.tick_params(axis="x",labelrotation=10)
    fig.tight_layout()
    fig.savefig(output/"Figure3_predictive_comparison_recreated.png",dpi=300)
    plt.close(fig)

    ipma=pd.read_csv(output/"25item_IPMA.csv")
    meanperf=float(ipma.performance_0_100_observed_mean.mean())
    meanimp=float(ipma.total_importance.mean())
    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(ipma.performance_0_100_observed_mean,ipma.total_importance)
    for row in ipma.itertuples():
        ax.annotate(row.construct,(row.performance_0_100_observed_mean,row.total_importance),
                    xytext=(5,5),textcoords="offset points")
    ax.axvline(meanperf,linestyle="--",label="Mean observed performance")
    ax.axhline(meanimp,linestyle=":",label="Mean estimated total importance")
    ax.set_xlabel("Observed construct performance (0–100)")
    ax.set_ylabel("Total association with BI (importance)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output/"Figure4_IPMA_recreated.png",dpi=300)
    plt.close(fig)

if __name__=="__main__":main()
