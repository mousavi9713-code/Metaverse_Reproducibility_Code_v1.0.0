#!/usr/bin/env python3
"""Permutation-based MICOM and PLS multi-group reanalysis (primary 25 items).

MICOM step 2: correlation between composites weighted using the separately
estimated faculty weights, evaluated on the same pooled indicators under a
common pooled standardization; compares observed corr with left 5th percentile
of faculty-label-permutation distribution. Step 3 uses comparisons of
composite score means and variances with weights reestimated per permutation.

MGA: reestimate PLS separately after random faculty-label reassignment,
preserving group sizes; empirical two-sided p from absolute path differences.
Holm adjustment across all 13 path comparisons.

This is a *custom reconstruction*, not an independently verified implementation
from SmartPLS/cSEM. User must review permutation conventions and seed.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PLS_PIIT_PCR_sensitivity_reproducible import FULL_BLOCKS,PATHS,pls_mode_a,holm
from metaverse_composite_diagnostics import fit_weights,pooled_project

F1="Paramedical sciences"
F2="Medicine"

def project(frame,blocks,fit,common_mean,common_sd):
    return pooled_project(frame,blocks,fit,common_mean,common_sd)

def run(df,B=5000,BB=5000,seed=20261029,out=Path("output")):
    out.mkdir(parents=True,exist_ok=True)
    blocks=FULL_BLOCKS;paths=PATHS
    faculty=df.Faculty_EN.to_numpy()
    ix1=np.flatnonzero(faculty==F1);ix2=np.flatnonzero(faculty==F2)
    if len(ix1)!=87 or len(ix2)!=98:raise ValueError("Faculty counts mismatch")
    pooledmean={k:df[v].to_numpy(float).mean(axis=0) for k,v in blocks.items()}
    pooledsd={k:df[v].to_numpy(float).std(axis=0,ddof=0) for k,v in blocks.items()}
    def fit_pair(a,b):
        A=df.iloc[a];C=df.iloc[b]
        wa=fit_weights(A,blocks,paths)
        wb=fit_weights(C,blocks,paths)
        x=project(df,blocks,wa,pooledmean,pooledsd)
        y=project(df,blocks,wb,pooledmean,pooledsd)
        corrs=[np.corrcoef(x[k],y[k])[0,1] for k in blocks]
        # use weights fitted on each group; apply to group data with same pooled scale
        v1=project(A,blocks,wa,pooledmean,pooledsd)
        v2=project(C,blocks,wb,pooledmean,pooledsd)
        mdelta=np.array([np.mean(v1[k])-np.mean(v2[k]) for k in blocks])
        lvar=np.array([np.log(np.var(v1[k],ddof=1)/np.var(v2[k],ddof=1)) for k in blocks])
        return corrs,mdelta,lvar
    obcorr,obmean,obvar=fit_pair(ix1,ix2)
    first=pls_mode_a(df.iloc[ix1],blocks,paths)
    second=pls_mode_a(df.iloc[ix2],blocks,paths)
    obdiff=np.array([first["coefficients"][path]-second["coefficients"][path] for path in paths])
    rng=np.random.default_rng(seed)
    cperm=np.full((B,len(blocks)),np.nan)
    mperm=np.full_like(cperm,np.nan)
    vperm=np.full_like(cperm,np.nan)
    pathperm=np.full((B,len(paths)),np.nan)
    for z in range(B):
        inds=rng.permutation(len(df));a,b=inds[:len(ix1)],inds[len(ix1):]
        try:
            cor,mu,var=fit_pair(a,b)
            cperm[z]=cor; mperm[z]=mu; vperm[z]=var
            f=pls_mode_a(df.iloc[a],blocks,paths)
            s=pls_mode_a(df.iloc[b],blocks,paths)
            pathperm[z]=[f["coefficients"][x]-s["coefficients"][x] for x in paths]
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            continue
        if (z+1)%1000==0:print(f"permutations {z+1}/{B}",flush=True)
    mic=[]
    for j,k in enumerate(blocks):
        z=cperm[:,j];z=z[np.isfinite(z)]
        m=mperm[:,j];m=m[np.isfinite(m)]
        v=vperm[:,j];v=v[np.isfinite(v)]
        lo=float(np.quantile(z,.05))
        pmean=float((np.sum(np.abs(m)>=abs(obmean[j]))+1)/(len(m)+1))
        pvar=float((np.sum(np.abs(v)>=abs(obvar[j]))+1)/(len(v)+1))
        mic.append({"construct":k,"composition_corr":obcorr[j],
                    "permutation_left_5pct":lo,"composition_invariant":bool(obcorr[j]>=lo),
                    "group_mean_difference_in_custom_score":obmean[j],
                    "permutation_p_mean":pmean,
                    "log_group_variance_ratio":obvar[j],"permutation_p_variance":pvar,
                    "valid_permutations":len(z)})
    pd.DataFrame(mic).to_csv(out/"MICOM_25item_reconstructed.csv",index=False)
    mg=[]
    for j,(a,b) in enumerate(paths):
        perm=pathperm[:,j];perm=perm[np.isfinite(perm)]
        pv=float((np.sum(np.abs(perm)>=abs(obdiff[j]))+1)/(len(perm)+1))
        mg.append({"path":f"{a}→{b}","paramedical_beta":first["coefficients"][(a,b)],
                   "medical_beta":second["coefficients"][(a,b)],
                   "difference":obdiff[j],"perm_p_two_sided":pv,
                   "valid_permutations":len(perm)})
    mg=pd.DataFrame(mg)
    mg["holm_p_13_path_comparisons"]=holm(mg.perm_p_two_sided)
    mg.to_csv(out/"MGA_25item_reconstructed.csv",index=False)
    # Separate nonparametric within-faculty bootstrap for group path estimates.
    rng=np.random.default_rng(seed+101)
    bgr=np.full((BB,2,len(paths)),np.nan)
    for i in range(BB):
        for g,idx in enumerate((ix1,ix2)):
            sub=df.iloc[rng.choice(idx,size=len(idx),replace=True)]
            try:
                fit=pls_mode_a(sub,blocks,paths)
                bgr[i,g,:]=[fit["coefficients"][p] for p in paths]
            except (ValueError,RuntimeError,np.linalg.LinAlgError):
                pass
        if (i+1)%1000==0:print(f"subgroup bootstraps {i+1}/{BB}",flush=True)
    gro=[]
    for j,(a,b) in enumerate(paths):
        row={"path":f"{a}→{b}"}
        for i,key in enumerate(("paramedical","medical")):
            arr=bgr[:,i,j];arr=arr[np.isfinite(arr)]
            row.update({f"{key}_ci_low":np.quantile(arr,.025),
                        f"{key}_ci_high":np.quantile(arr,.975),
                        f"{key}_valid_boot":len(arr)})
        gro.append(row)
    pd.DataFrame(gro).to_csv(out/"MGA_25item_subgroup_bootstrap.csv",index=False)
    print("MICOM\n",pd.DataFrame(mic)[["construct","composition_corr","permutation_left_5pct","composition_invariant"]].to_string(index=False))
    print("MGA\n",mg.to_string(index=False))
    return mic,mg

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--out",default=Path("output"),type=Path)
    p.add_argument("--permutations",type=int,default=5000)
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20261029)
    a=p.parse_args()
    df=pd.read_csv(a.input)
    run(df,a.permutations,a.bootstrap,a.seed,a.out)

if __name__=="__main__":main()
