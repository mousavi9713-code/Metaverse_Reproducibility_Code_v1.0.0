#!/usr/bin/env python3
"""Faculty-calibrated (poststratified) PLS-SEM sensitivity analysis.
Uses only stdlib and NumPy; no raw patient data are modified.
The faculty weights reflect target enrollment counts (Medicine 2191, Paramedical 913),
and achieved respondent counts (98, 87). These are faculty calibration
weights, *not* verified inverse inclusion probability weights by individual or
stage. Reflective Mode-A PLS using path inner weighting.
Compare unweighted and faculty-calibrated estimates for 25-item original
and exploratory reduced 17-item PLS models.
Stratified within-faculty bootstrap fixes respondent counts by faculty.
"""
import numpy as np
import csv, argparse, pathlib, sys

BLOCKS={
 "PIIT":[f"PIIT{i}" for i in range(1,5)],
 "PCR":[f"PCR{i}" for i in range(1,5)],
 "SE":[f"SE{i}" for i in range(1,4)],
 "PEU":[f"PEU{i}" for i in range(1,5)],
 "PU":[f"PU{i}" for i in range(1,4)], # source-domain PU4 excluded
 "PE":[f"PE{i}" for i in range(1,5)],
 "BI":[f"BI{i}" for i in range(1,4)]
}
PATHS=[
 ("PIIT","SE"),("PIIT","PU"),("PIIT","PEU"),("PIIT","BI"),
 ("SE","PU"),("SE","PEU"),("PCR","PU"),("PCR","BI"),
 ("PEU","PU"),("PEU","PE"),("PEU","BI"),("PU","BI"),("PE","BI")
]
RED_BLOCKS={k:v for k,v in BLOCKS.items() if k not in {"PIIT","PCR"}}
RED_PATHS=[(a,b) for a,b in PATHS if a in RED_BLOCKS and b in RED_BLOCKS]
def wz(x,w):
    x=np.asarray(x,float)
    mean=np.average(x,axis=0,weights=w)
    var=np.average((x-mean)**2,axis=0,weights=w)
    if np.any(var<1e-15): raise ValueError("Degenerate variable")
    return (x-mean)/np.sqrt(var)
def wls(a,y,w):
    sw=np.sqrt(w)
    return np.linalg.lstsq(a*sw[:,None],y*sw,rcond=None)[0]
def pls(x,keys,blocks,paths,w):
    w=np.asarray(w,float);w=w/w.sum()
    dat={k:wz(x[:,[keys.index(c) for c in v]],w) for k,v in blocks.items()}
    weights={k:np.ones(len(v))/np.sqrt(len(v)) for k,v in blocks.items()}
    parent={k:[a for a,b in paths if b==k] for k in blocks}
    child={k:[b for a,b in paths if a==k] for k in blocks}
    def scores(ws):
        return {k:wz(dat[k]@ws[k],w).ravel() for k in blocks}
    for i in range(400):
        yy=scores(weights)
        new={}
        for k in blocks:
            inner=np.zeros(len(w))
            if parent[k]:
                pa=np.column_stack([yy[p] for p in parent[k]])
                inner+=pa@wls(pa,yy[k],w)
            for c in child[k]:
                inner+=np.average(yy[k]*yy[c],weights=w)*yy[c]
            norm=wz(inner,w)
            b=np.sum(dat[k]*norm[:,None]*w[:,None],axis=0)
            b=b/np.linalg.norm(b)
            if b.mean()<0:b=-b
            new[k]=b
        err=max(np.linalg.norm(new[k]-weights[k]) for k in blocks)
        weights=new
        if err<1e-9:break
    else:raise RuntimeError("PLS did not converge")
    yy=scores(weights)
    betas={}
    r2={}
    for dst in blocks:
        pr=parent[dst]
        if not pr:continue
        M=np.column_stack([yy[z] for z in pr])
        co=wls(M,yy[dst],w)
        betas.update({(src,dst):float(bb) for src,bb in zip(pr,co)})
        resid=yy[dst]-M@co
        r2[dst]=1-np.average(resid**2,weights=w)/np.average((yy[dst]-np.average(yy[dst],weights=w))**2,weights=w)
    return betas,r2,i+1

def load_csv(fn):
    with open(fn,encoding='utf-8-sig',newline='') as f: data=list(csv.DictReader(f))
    keys=[c for b in BLOCKS.values() for c in b]
    x=np.asarray([[float(r[c]) for c in keys] for r in data])
    fac=np.asarray([r["Faculty_EN"] for r in data])
    if len(data)!=185 or (fac=="Medicine").sum()!=98 or (fac=="Paramedical sciences").sum()!=87:
        raise ValueError("Source totals not as expected")
    return keys,x,fac,data

def boot(keys,x,fac,modelblocks,modelpaths,B,seed):
    rng=np.random.default_rng(seed)
    a=np.where(fac=="Medicine")[0];b=np.where(fac=="Paramedical sciences")[0]
    paths=list(modelpaths)
    beta_out=np.full((B,len(paths)),np.nan)
    r2_out=np.full(B,np.nan)
    for j in range(B):
        ids=np.r_[rng.choice(a,len(a),replace=True),rng.choice(b,len(b),replace=True)]
        f=fac[ids]
        w=np.where(f=="Medicine",2191/98,913/87)
        try:
            co,r2,_=pls(x[ids],keys,modelblocks,modelpaths,w)
            beta_out[j]=[co[p] for p in paths]
            r2_out[j]=r2["BI"]
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            continue
    return beta_out,r2_out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default='Metaverse_25Item_Primary_Scored_Data.csv')
    ap.add_argument('--bootstrap',type=int,default=500)
    ap.add_argument('--seed',type=int,default=20261029)
    ap.add_argument('--out',default='output')
    a=ap.parse_args()
    keys,x,fac,records=load_csv(a.csv)
    wt=np.where(fac=="Medicine",2191/98,913/87)
    results=[]
    for label,bl,pa in [("original",BLOCKS,PATHS),("reduced",RED_BLOCKS,RED_PATHS)]:
        u,u2,_=pls(x,keys,bl,pa,np.ones(len(fac)))
        w,w2,_=pls(x,keys,bl,pa,wt)
        bootbeta,bootr=boot(keys,x,fac,bl,pa,a.bootstrap,a.seed)
        for j,p in enumerate(pa):
            valid=bootbeta[:,j][np.isfinite(bootbeta[:,j])]
            lo,hi=np.percentile(valid,[2.5,97.5]) if len(valid)>0 else (np.nan,np.nan)
            results.append({"model":label,"path":f"{p[0]}->{p[1]}","unweighted_beta":u[p],
             "faculty_calibrated_beta":w[p],"difference_calibrated_minus_unweighted":w[p]-u[p],
             "calibrated_boot_ci_lo":lo,"calibrated_boot_ci_hi":hi,"valid_boot":len(valid)})
        print(label, "R2 BI",u2["BI"],w2["BI"],"weighted r2 boot CI",np.nanpercentile(bootr,[2.5,97.5]))
        for p in pa:
            if p[1]=="BI":print(p,round(u[p],4),round(w[p],4))
    out=pathlib.Path(a.out)/'Sampling_faculty_calibration_PLS_sensitivity.csv'
    out.parent.mkdir(exist_ok=True,parents=True)
    with open(out,'w',encoding='utf-8',newline='') as f:
        ww=csv.DictWriter(f,fieldnames=list(results[0]));ww.writeheader();ww.writerows(results)
    print("Saved:",out)
    return results
if __name__=="__main__": main()
