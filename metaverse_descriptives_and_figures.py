#!/usr/bin/env python3
"""Observed-score descriptive outputs, seven faculty contrasts, and figures.
Do not interpret faculty comparisons as causal or measurement invariant.
Figures are independently generated and must be compared to manuscript artwork.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PLS_PIIT_PCR_sensitivity_reproducible import FULL_BLOCKS, holm

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--out",default=Path("output"),type=Path)
    a=ap.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    d=pd.read_csv(a.input)
    scores=pd.DataFrame({k:d[v].mean(axis=1) for k,v in FULL_BLOCKS.items()})
    scores["Faculty_EN"]=d.Faculty_EN
    scores["Study_Category_EN"]=d.Study_Category_EN
    listrows=[]
    for f,group in d.groupby("Faculty_EN"):
        for c in ["Gender_EN","Degree_EN","Study_Category_EN","Entry_Year_Clean","Entry_Semester_EN"]:
            for value,n in group[c].value_counts(dropna=False).items():
                listrows.append({"faculty":f,"characteristic":c,"category":value,"n":int(n),
                                 "percentage_within_faculty":100*n/len(group)})
    pd.DataFrame(listrows).to_csv(a.out/"sample_characteristics.csv",index=False)
    rows=[]
    for k in FULL_BLOCKS:
        pa=scores.loc[scores.Faculty_EN.eq("Paramedical sciences"),k].to_numpy()
        me=scores.loc[scores.Faculty_EN.eq("Medicine"),k].to_numpy()
        t,p=stats.ttest_ind(pa,me,equal_var=False)
        v1=np.var(pa,ddof=1)/len(pa);v2=np.var(me,ddof=1)/len(me)
        se=np.sqrt(v1+v2)
        df=(v1+v2)**2/(v1*v1/(len(pa)-1)+v2*v2/(len(me)-1))
        diff=float(np.mean(pa)-np.mean(me))
        low,high=stats.t.interval(.95,df,loc=diff,scale=se)
        pool=np.sqrt(((len(pa)-1)*np.var(pa,ddof=1)+(len(me)-1)*np.var(me,ddof=1))/
                     (len(pa)+len(me)-2))
        hedges=(diff/pool)*(1-3/(4*(len(pa)+len(me))-9))
        rows.append({"construct":k,"overall_mean":scores[k].mean(),
                     "overall_sd":scores[k].std(ddof=1),
                     "paramedical_mean":np.mean(pa),"medical_mean":np.mean(me),
                     "difference_para_minus_med":diff,"welch_ci_low":low,"welch_ci_high":high,
                     "welch_p":p,"hedges_g":hedges})
    comparison=pd.DataFrame(rows)
    comparison["holm_p_7"]=holm(comparison.welch_p.to_numpy())
    comparison.to_csv(a.out/"faculty_construct_descriptives.csv",index=False)

    fig,ax=plt.subplots(figsize=(10,5.3))
    vals=[scores[k].values for k in FULL_BLOCKS]
    ax.boxplot(vals,tick_labels=list(FULL_BLOCKS),patch_artist=False)
    ax.set_ylabel("Construct score (1–5)")
    ax.set_title("Construct-score distributions, n=185")
    fig.tight_layout()
    fig.savefig(a.out/"Figure1_construct_scores_recreated.png",dpi=300)
    plt.close(fig)

    # Recreate descriptive, NOT inferential, program/stage heatmaps.
    for faculty,fn in [("Paramedical sciences","Figure5_paramedical_program_descriptive.png"),
                       ("Medicine","Figure6_medical_stage_descriptive.png")]:
        sub=scores[scores.Faculty_EN==faculty]
        means=sub.groupby("Study_Category_EN")[list(FULL_BLOCKS)].mean()
        ns=sub.groupby("Study_Category_EN").size()
        fig,ax=plt.subplots(figsize=(10,max(4,len(means)*0.55)))
        im=ax.imshow(means.values,vmin=1,vmax=5,cmap="viridis",aspect="auto")
        ax.set_xticks(np.arange(len(means.columns)),means.columns)
        ax.set_yticks(np.arange(len(means)),
                      [f"{k} (n={ns[k]})" for k in means.index])
        fig.colorbar(im,ax=ax,label="Observed mean construct score (1–5)")
        ax.set_title(f"{faculty} — descriptive only (small cells)")
        fig.tight_layout()
        fig.savefig(a.out/fn,dpi=300)
        plt.close(fig)
    print(comparison.round(4).to_string(index=False))
if __name__=="__main__":main()
