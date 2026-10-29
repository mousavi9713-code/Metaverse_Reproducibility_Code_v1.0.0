#!/usr/bin/env python3
"""Post-hoc measurement-validity sensitivity analysis for the metaverse study.
Reproduces the original 25-item reflective Mode-A path-weighting PLS-PM
point estimates, then excludes entire PIIT and PCR blocks and estimates the
seven pre-existing structural paths among SE, PEU, PU, PE and BI.

Input: Metaverse_25Item_Primary_Scored_Data.csv (scored responses; n=185)
Run: python PLS_PIIT_PCR_sensitivity_reproducible.py --data-dir /path/to/files
Defaults: nonparametric participant bootstrap n=2000; seed=20261028.
This independent Python reconstruction is not a third-party package cross-check.
No new primary/preregistered hypotheses or causal interpretations are claimed.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

FULL_BLOCKS = {
    "PIIT": [f"PIIT{i}" for i in range(1, 5)],
    "PCR": [f"PCR{i}" for i in range(1, 5)],
    "SE": [f"SE{i}" for i in range(1, 4)],
    "PEU": [f"PEU{i}" for i in range(1, 5)],
    "PU": [f"PU{i}" for i in range(1, 4)],  # 25-item primary: exclude PU4
    "PE": [f"PE{i}" for i in range(1, 5)],
    "BI": [f"BI{i}" for i in range(1, 4)]
}
PATHS = [
    ("PIIT", "SE"), ("PIIT", "PU"), ("PIIT", "PEU"), ("PIIT", "BI"),
    ("SE", "PU"), ("SE", "PEU"),
    ("PCR", "PU"), ("PCR", "BI"),
    ("PEU", "PU"), ("PEU", "PE"), ("PEU", "BI"),
    ("PU", "BI"), ("PE", "BI")
]
BLOCKS5 = {name: cols for name, cols in FULL_BLOCKS.items() if name not in {"PIIT","PCR"}}
PATHS5 = [(src,dst) for src,dst in PATHS if src in BLOCKS5 and dst in BLOCKS5]

def zscore(x):
    x=np.asarray(x,dtype=float)
    sd=x.std(axis=0,ddof=0)
    if np.any(sd<=1e-12):
        raise ValueError("A variable is constant; scores cannot be standardized.")
    return (x-x.mean(axis=0))/sd

def pls_mode_a(frame, blocks, paths, tol=1e-9, max_iter=500):
    """Mode A reflective PLS, path inner-weighting, standardized latent scores.
    Endogenous inner estimates use OLS coefficients of latent predecessors.
    Exogenous/successor inner weights use correlations with successor latent
    scores. Outer weights are covariances between standardized indicators and
    the normalized inner estimate, normalized to unit L2 length.
    """
    names=list(blocks)
    xx={k:zscore(frame[v].to_numpy(float)) for k,v in blocks.items()}
    weights={k:np.ones(len(v))/np.sqrt(len(v)) for k,v in blocks.items()}
    parent={k:[a for a,b in paths if b==k] for k in names}
    child={k:[b for a,b in paths if a==k] for k in names}
    def scores(w):
        return {k:zscore(xx[k]@w[k]).reshape(-1) for k in names}
    for it in range(max_iter):
        yy=scores(weights)
        new={}
        for k in names:
            inner=np.zeros(len(frame))
            if parent[k]:
                par=np.column_stack([yy[p] for p in parent[k]])
                coef=np.linalg.lstsq(par,yy[k],rcond=None)[0]
                inner+=par@coef
            for c in child[k]:
                inner+=np.mean(yy[c]*yy[k])*yy[c]
            if np.std(inner)<1e-12:
                raise ValueError("Degenerate inner estimate for "+k)
            w=np.mean(xx[k]*zscore(inner)[:,None],axis=0)
            w=w/np.linalg.norm(w)
            if w.mean()<0:
                w=-w
            new[k]=w
        change=max(np.linalg.norm(new[k]-weights[k]) for k in names)
        weights=new
        if change<tol:
            break
    else:
        raise RuntimeError("PLS failed to converge")
    yy=scores(weights)
    coefficients={}
    r2={}
    for target in names:
        pred=parent[target]
        if not pred:
            continue
        M=np.column_stack([yy[k] for k in pred])
        beta=np.linalg.lstsq(M,yy[target],rcond=None)[0]
        r2[target]=1-np.mean((yy[target]-M@beta)**2)/np.var(yy[target])
        coefficients.update({(k,target):float(b) for k,b in zip(pred,beta)})
    loadings={k:dict(zip(blocks[k],np.mean(xx[k]*yy[k][:,None],axis=0))) for k in names}
    return {"coefficients":coefficients,"r2":r2,"loadings":loadings,
            "iterations":it+1}

def alpha(frame):
    x=frame.to_numpy(dtype=float); k=x.shape[1]
    return float(k/(k-1)*(1-x.var(axis=0,ddof=1).sum()/x.sum(axis=1).var(ddof=1)))

def measurement(frame,model,blocks):
    arr=[]
    for c,items in blocks.items():
        ld=np.array(list(model["loadings"][c].values()))
        cr=(ld.sum()**2)/(ld.sum()**2+np.sum(1-ld**2))
        arr.append({"construct":c,"items":len(items),"alpha":alpha(frame[items]),
                    "composite_reliability":float(cr),"AVE":float(np.mean(ld**2)),
                    "min_loading":float(ld.min())})
    return pd.DataFrame(arr)

def htmt_matrix(frame,blocks):
    columns=sum(blocks.values(),[])
    corr=frame[columns].corr().abs()
    out=[]
    for a,A in blocks.items():
        for b,B in blocks.items():
            if a>=b: continue
            sameA=corr.loc[A,A].to_numpy()
            sameB=corr.loc[B,B].to_numpy()
            num=corr.loc[A,B].to_numpy().mean()
            den=np.sqrt(sameA[~np.eye(len(A),dtype=bool)].mean()*
                        sameB[~np.eye(len(B),dtype=bool)].mean())
            out.append({"construct_A":a,"construct_B":b,"HTMT":float(num/den)})
    return pd.DataFrame(out)

def holm(ps):
    x=np.asarray(ps,float)
    idx=np.argsort(x)
    adj=np.maximum.accumulate((len(x)-np.arange(len(x)))*x[idx])
    y=np.empty(len(x))
    y[idx]=np.minimum(adj,1)
    return y

def bootstrap_reduced(frame,boot,seed):
    rng=np.random.default_rng(seed)
    estimates=np.empty((boot,len(PATHS5)))
    r2=np.empty(boot)
    indirect=np.empty(boot)
    for i in range(boot):
        sample=frame.iloc[rng.integers(0,len(frame),len(frame))]
        m=pls_mode_a(sample,BLOCKS5,PATHS5)
        estimates[i]=[m["coefficients"][p] for p in PATHS5]
        r2[i]=m["r2"]["BI"]
        indirect[i]=m["coefficients"][("PEU","PE")]*m["coefficients"][("PE","BI")]
    return estimates,r2,indirect

def summarize_bootstrap(estimate,boot):
    lo,hi=np.quantile(boot,[.025,.975])
    # Empirical two-sided bootstrap sign-tail estimate with +1 correction.
    n=len(boot)
    p=min(1.,2.*min((np.sum(boot<=0)+1)/(n+1),
                    (np.sum(boot>=0)+1)/(n+1)))
    return lo,hi,p

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=Path("."))
    parser.add_argument("--boot",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=20261028)
    args=parser.parse_args()
    path=args.data_dir/"Metaverse_25Item_Primary_Scored_Data.csv"
    d=pd.read_csv(path)
    if len(d)!=185: raise ValueError("Unexpected sample size")
    col=sum(FULL_BLOCKS.values(),[])
    if not d["ID"].is_unique or d[col].isnull().any().any():
        raise ValueError("Duplicate ID or missing item")
    if not d[col].apply(lambda s:s.between(1,5).all()).all():
        raise ValueError("Scores outside 1..5")
    orig=pls_mode_a(d,FULL_BLOCKS,PATHS)
    red=pls_mode_a(d,BLOCKS5,PATHS5)
    boots,boot_r2,boot_indirect=bootstrap_reduced(d,args.boot,args.seed)
    data=[]
    for j,p in enumerate(PATHS5):
        estimate=red["coefficients"][p]
        low,high,pval=summarize_bootstrap(estimate,boots[:,j])
        data.append(dict(path=f"{p[0]}->{p[1]}",beta=estimate,
                         ci_lower=low,ci_upper=high,p_boot_two_sided=pval))
    effects=pd.DataFrame(data)
    effects["holm_adjusted_p_7_paths"]=holm(effects["p_boot_two_sided"])
    measure=measurement(d,red,BLOCKS5)
    htmt=htmt_matrix(d,BLOCKS5)
    orig_measure=measurement(d,orig,FULL_BLOCKS)
    ind_est=red["coefficients"][("PEU","PE")]*red["coefficients"][("PE","BI")]
    ind_ci=summarize_bootstrap(ind_est,boot_indirect)
    pref=args.data_dir/"PIIT_PCR_posthoc_"
    effects.to_csv(str(pref)+"reduced_PLS_paths.csv",index=False)
    measure.to_csv(str(pref)+"reduced_measurement.csv",index=False)
    htmt.to_csv(str(pref)+"reduced_HTMT.csv",index=False)
    orig_measure.to_csv(str(pref)+"original_measurement.csv",index=False)
    original=pd.DataFrame([{"path":f"{x}->{y}","beta":v} for (x,y),v in orig["coefficients"].items()])
    original.to_csv(str(pref)+"reproduced_original_paths.csv",index=False)
    lines=[
        "# Post hoc sensitivity excluding PIIT and PCR",
        f"n = {len(d)}; bootstrap B = {args.boot}; seed = {args.seed}; "
        "PLS Mode A (path weighting); PU4 excluded from both models.",
        f"Original 25-item PLS: R² BI = {orig['r2']['BI']:.6f}",
        f"Reduced five-construct PLS: R² BI = {red['r2']['BI']:.6f}",
        f"Reduced R² BI 95% percentile bootstrap interval = "
            f"[{np.quantile(boot_r2,.025):.6f}, {np.quantile(boot_r2,.975):.6f}]",
        "",
        "## Seven retained paths (bootstrap sign-tail p; Holm correction of 7 exploratory paths)",
        effects.to_string(index=False,float_format=lambda x:f"{x:.6f}"),
        "",
        "## Measurement",
        measure.to_string(index=False,float_format=lambda x:f"{x:.6f}"),
        "",
        "## HTMT",
        htmt.to_string(index=False,float_format=lambda x:f"{x:.6f}"),
        "",
        "## Predefined indirect PEU -> PE -> BI",
        f"product={ind_est:.6f}; bootstrap 95% CI [{ind_ci[0]:.6f}, {ind_ci[1]:.6f}]; "
            f"p={ind_ci[2]:.6f}",
        "",
        "LIMITATIONS: exploratory, post hoc, not a prospectively specified confirmatory analysis. "
        "Nonparametric resampling assesses sampling variability, not measurement equivalence, "
        "causal identification or external generalizability. Original 25-item PLS estimates "
        "were recreated from the scored file; original full bootstrap and predictive "
        "cross-validation are not reproduced in this script. "
        "Cross-check with an established third-party PLS implementation is pending.",
    ]
    outfile=str(pref)+"summary.md"
    Path(outfile).write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))
    print("OUTPUTS:")
    for path in sorted(args.data_dir.glob("PIIT_PCR_posthoc_*")):
        print(path.name)

if __name__=="__main__":
    main()
