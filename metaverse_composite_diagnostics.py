#!/usr/bin/env python3
"""Latent composite weights and approximate model diagnostics.
Same path-weighting rules as the independent PLS reconstruction.
Caution: approximate SRMR is not a standard package-specific global fit.
"""
import numpy as np
from PLS_PIIT_PCR_sensitivity_reproducible import pls_mode_a

def fit_weights(frame, blocks, paths, tol=1e-9, max_iter=500):
    names=list(blocks)
    means={k:frame[v].to_numpy(float).mean(axis=0) for k,v in blocks.items()}
    sds={k:frame[v].to_numpy(float).std(axis=0,ddof=0) for k,v in blocks.items()}
    x={k:(frame[v].to_numpy(float)-means[k])/sds[k] for k,v in blocks.items()}
    w={k:np.ones(len(v))/np.sqrt(len(v)) for k,v in blocks.items()}
    parents={k:[a for a,b in paths if b==k] for k in names}
    children={k:[b for a,b in paths if a==k] for k in names}
    def standard(z):
        z=np.asarray(z,float)
        return (z-z.mean())/z.std(ddof=0)
    def scores(ww):
        return {k:standard(x[k]@ww[k]) for k in names}
    for it in range(max_iter):
        yy=scores(w)
        nw={}
        for k in names:
            inner=np.zeros(len(frame))
            if parents[k]:
                A=np.column_stack([yy[q] for q in parents[k]])
                inner+=A@np.linalg.lstsq(A,yy[k],rcond=None)[0]
            for nxt in children[k]:
                inner+=np.mean(yy[k]*yy[nxt])*yy[nxt]
            raw=np.mean(x[k]*standard(inner)[:,None],axis=0)
            raw=raw/np.linalg.norm(raw)
            if raw.mean()<0:raw=-raw
            nw[k]=raw
        delta=max(np.linalg.norm(w[k]-nw[k]) for k in names)
        w=nw
        if delta<tol:break
    else:raise RuntimeError("No convergence")
    yy=scores(w)
    return {"weights":w,"means":means,"sds":sds,"scores":yy,
            "iterations":it+1}

def pooled_project(frame, blocks, fit, means, sds):
    # Uses common, pooled scale; score variance is not forced equal in groups.
    out={}
    for k,items in blocks.items():
        a=(frame[items].to_numpy(float)-means[k])/sds[k]
        out[k]=a@fit["weights"][k]
    return out

def full_collinearity_vifs(scores):
    n=len(next(iter(scores.values())))
    zs={k:(np.asarray(a,float)-np.mean(a))/np.std(a,ddof=0)
        for k,a in scores.items()}
    v={}
    for name,y in zs.items():
        keys=[k for k in zs if k!=name]
        M=np.column_stack([zs[k] for k in keys])
        beta=np.linalg.lstsq(M,y,rcond=None)[0]
        resid=y-M@beta
        r2=1-np.mean(resid**2)/np.var(y)
        v[name]=1/(1-r2)
    return v

def harman_pca_share(frame,items):
    x=frame[items].to_numpy(float)
    corr=np.corrcoef(x,rowvar=False)
    return np.linalg.eigvalsh(corr).max()/len(items)

def approximate_srmr(frame,blocks,fit):
    # Outer loadings * construct correlation (within block one-factor model).
    # Explicitly an exploratory approximation: NOT SmartPLS global-fit SRMR.
    cols=sum(blocks.values(),[])
    x=frame[cols].to_numpy(float)
    x=(x-x.mean(axis=0))/x.std(axis=0,ddof=0)
    empirical=np.corrcoef(x,rowvar=False)
    score=np.column_stack([fit["scores"][k] for k in blocks])
    construct_cor=np.corrcoef(score,rowvar=False)
    loadings=np.array([np.mean(x[:,i]*fit["scores"][k])
                       for i,(k,it) in enumerate((k,it) for k,its in blocks.items() for it in its)])
    group=np.array([list(blocks).index(k) for k,its in blocks.items() for it in its])
    implied=np.outer(loadings,loadings)*construct_cor[np.ix_(group,group)]
    np.fill_diagonal(implied,1.0)
    errors=empirical-implied
    upper=errors[np.triu_indices(len(cols),1)]
    return float(np.sqrt(np.mean(upper**2)))
