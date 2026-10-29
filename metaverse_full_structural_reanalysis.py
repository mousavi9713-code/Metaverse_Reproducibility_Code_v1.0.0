#!/usr/bin/env python3
"""Reconstruct full 25-item and 26-item reflective Mode-A path-weighted PLS models.

Uses primary participant-level scored CSV; no hidden upstream preprocessing.
Generates path estimates, nonparametric bootstrap percentile CIs, indirect and
total effects, R2, f2, alpha, CR, AVE, HTMT, IPMA and approximate diagnostics.

NOTICE: independent reconstruction, not a verification by an established
third-party PLS implementation. Bootstrap seed / CI convention may differ
from the original submission's exact unpublished settings.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from PLS_PIIT_PCR_sensitivity_reproducible import (FULL_BLOCKS, PATHS, BLOCKS5,
    PATHS5, pls_mode_a, measurement, htmt_matrix, holm, alpha)

def blocks_for(name):
    if name=="25item": return FULL_BLOCKS,PATHS
    if name=="26item":
        b={k:list(v) for k,v in FULL_BLOCKS.items()}
        b["PU"]=["PU1","PU2","PU3","PU4"]
        return b,PATHS
    if name=="reduced17": return BLOCKS5,PATHS5
    raise ValueError(name)

def construct_means(df, blocks):
    return {k:df[v].mean(axis=1).to_numpy(float) for k,v in blocks.items()}

def totals(paths, co):
    """Sum all directed path products in this finite DAG (direct and indirect)."""
    order=sorted(set(x for p in paths for x in p))
    A=np.array([[co.get((a,b),0.0) for b in order] for a in order])
    ans=np.zeros_like(A)
    power=A.copy()
    for _ in range(len(order)-1):
        ans+=power
        power=power@A
    return {(a,b):float(ans[i,j]) for i,a in enumerate(order) for j,b in enumerate(order)}

def boot_model(df, blocks, paths, B, seed):
    rng=np.random.default_rng(seed)
    path_array=np.full((B,len(paths)),np.nan)
    r2_array=np.full((B,len([k for k in blocks if any(b==k for a,b in paths)])),np.nan)
    endo=[k for k in blocks if any(b==k for a,b in paths)]
    alltot=np.full((B,len(blocks)),np.nan)
    keys=list(blocks)
    valid=0
    for i in range(B):
        sample=df.iloc[rng.integers(0,len(df),len(df))]
        try:
            m=pls_mode_a(sample,blocks,paths)
            path_array[i]=[m["coefficients"][p] for p in paths]
            r2_array[i]=[m["r2"][k] for k in endo]
            t=totals(paths,m["coefficients"])
            alltot[i]=[t.get((k,"BI"),0.) for k in keys]
            valid+=1
        except (RuntimeError,ValueError,np.linalg.LinAlgError,FloatingPointError):
            pass
        if (i+1)%1000==0: print(f"  bootstrap {i+1}/{B}, valid {valid}",flush=True)
    return path_array,r2_array,alltot,endo,valid

def summary_stats(est, arr):
    v=arr[np.isfinite(arr)]
    if len(v)==0: return (np.nan,np.nan,np.nan,0)
    ci=np.quantile(v,[.025,.975])
    # two-sided bootstrap sign tail, +1 continuity correction; not t test.
    p=min(1.,2*min((np.sum(v<=0)+1)/(len(v)+1),
                   (np.sum(v>=0)+1)/(len(v)+1)))
    return float(ci[0]),float(ci[1]),float(p),len(v)

def fixed_score_f2(frame,blocks,paths,target):
    """Conventional incremental f² from the SAME fitted latent composite scores.

    No re-fitting of the measurement model in reduced inner OLS models:
      f²=(R² included - R² excluding predecessor)/(1-R² included)
    """
    from metaverse_composite_diagnostics import fit_weights
    fit=fit_weights(frame,blocks,paths)
    yy=fit["scores"]
    parents=[a for a,b in paths if b==target]
    y=yy[target]
    def r2(selected):
        if not selected: return 0.
        mat=np.column_stack([yy[k] for k in selected])
        resid=y-mat@np.linalg.lstsq(mat,y,rcond=None)[0]
        return 1-np.mean(resid**2)/np.var(y)
    full=r2(parents)
    return {name:(full-r2([q for q in parents if q!=name]))/(1-full)
            for name in parents}

def run_one(df,name,B,seed,out):
    blocks,paths=blocks_for(name)
    m=pls_mode_a(df,blocks,paths)
    print(name,"BI R2:",m["r2"]["BI"],"iters:",m["iterations"])
    bpaths,br2,btot,endo,valid=boot_model(df,blocks,paths,B,seed)
    pathrows=[]
    for j,p in enumerate(paths):
        ci_l,ci_u,pv,nvalid=summary_stats(m["coefficients"][p],bpaths[:,j])
        pathrows.append({"hypothesis":f"H{j+1}" if name!="reduced17" else "post hoc",
            "path":f"{p[0]}→{p[1]}","beta":m["coefficients"][p],
            "ci_low_95":ci_l,"ci_high_95":ci_u,"p_boot_sign_tail":pv,
            "valid_bootstrap":nvalid})
    pdf=pd.DataFrame(pathrows)
    pdf["holm_adjusted_p_13_or_7"]=holm(pdf["p_boot_sign_tail"].to_numpy())
    pdf.to_csv(out/f"{name}_paths_bootstrap.csv",index=False)
    measurement(df,m,blocks).to_csv(out/f"{name}_measurement.csv",index=False)
    htmt_matrix(df,blocks).to_csv(out/f"{name}_HTMT.csv",index=False)
    r2rows=[]
    for j,k in enumerate(endo):
        ci=summary_stats(m["r2"][k],br2[:,j])
        r2rows.append({"construct":k,"r2":m["r2"][k],
                       "adj_r2":1-(1-m["r2"][k])*(len(df)-1)/
                       (len(df)-sum(v==k for u,v in paths)-1),
                       "bootstrap_ci_low":ci[0],"bootstrap_ci_high":ci[1]})
    pd.DataFrame(r2rows).to_csv(out/f"{name}_R2.csv",index=False)
    T=totals(paths,m["coefficients"])
    totalrows=[]
    for i,k in enumerate(blocks):
        if k=="BI":continue
        te=T.get((k,"BI"),0.)
        direct=m["coefficients"].get((k,"BI"),0.)
        lo,hi,pv,nvalid=summary_stats(te,btot[:,i])
        direct_ix=paths.index((k,"BI")) if (k,"BI") in paths else None
        boot_direct=bpaths[:,direct_ix] if direct_ix is not None else np.zeros(B)
        boot_ind=btot[:,i]-boot_direct
        il,iu,ip,iv=summary_stats(te-direct,boot_ind)
        totalrows.append({"source":k,"target":"BI","direct":direct,
                          "indirect":te-direct,"indirect_ci_low":il,
                          "indirect_ci_high":iu,"p_indirect":ip,
                          "total":te,"ci_low":lo,"ci_high":hi,"p_total":pv})
    pd.DataFrame(totalrows).to_csv(out/f"{name}_total_effects.csv",index=False)
    ipma=[]
    for k in blocks:
        if k=="BI":continue
        score=df[blocks[k]].mean(axis=1)
        ipma.append({"construct":k,"total_importance":T.get((k,"BI"),0.),
                     "performance_0_100_observed_mean":float((score.mean()-1)/4*100)})
    pd.DataFrame(ipma).to_csv(out/f"{name}_IPMA.csv",index=False)
    f2=fixed_score_f2(df,blocks,paths,"BI")
    pd.DataFrame([{"predictor":k,"f2_fixed_composite_scores":v}
                  for k,v in f2.items()]).to_csv(out/f"{name}_f2.csv",index=False)
    if ("PEU","PE") in paths and ("PE","BI") in paths:
        j1=paths.index(("PEU","PE"));j2=paths.index(("PE","BI"))
        est=m["coefficients"][("PEU","PE")]*m["coefficients"][("PE","BI")]
        ci=summary_stats(est,bpaths[:,j1]*bpaths[:,j2])
        pd.DataFrame([{"effect":"PEU→PE→BI","product":est,
                       "ci_low":ci[0],"ci_high":ci[1],"p_boot":ci[2]}]).to_csv(
                        out/f"{name}_specific_indirect.csv",index=False)
    print(" paths",pdf[["path","beta","ci_low_95","ci_high_95","p_boot_sign_tail"]].to_string(index=False))
    return m

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True,
                   help="Private source: Metaverse_25Item_Primary_Scored_Data.csv")
    p.add_argument("--out",type=Path,default=Path("output"))
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20261029)
    p.add_argument("--models",nargs="+",default=["25item","26item","reduced17"])
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    df=pd.read_csv(a.input)
    if len(df)!=185 or df["ID"].duplicated().any():
        raise ValueError("Expected 185 unique participants")
    for model in a.models:
        blocks,paths=blocks_for(model)
        items=sum(blocks.values(),[])
        if df[items].isna().any().any(): raise ValueError("Missing indicators")
        if not df[items].apply(lambda x:x.between(1,5).all()).all():
            raise ValueError("Items outside 1..5")
        run_one(df,model,a.bootstrap,a.seed,a.out)
    print("Output saved in",a.out)

if __name__=="__main__": main()
