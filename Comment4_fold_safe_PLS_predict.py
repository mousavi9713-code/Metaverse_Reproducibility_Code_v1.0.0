#!/usr/bin/env python3
"""Leakage-controlled predictive verification of the metaverse manuscript (25 items).

- RepeatedKFold 10 splits x 10 repeats, seed 20261029.
- Both models refit Mode-A reflective PLS *inside every training fold*.
- Train-fold centering/SDs, outer weights, latent scaling, inner path coefficients
  and endogenous indicator loadings are *not* learned from validation participants.
- BI1, BI2, BI3 predicted from the held-out predictors' fitted latent composite
  and trained indicator loading; BI composite = mean of predicted BI indicators.
- Baseline: fold-training mean of each target, separately per target.
- Alternative linear-model (LM) benchmark: independent OLS on exogenous
  predictor indicators fitted only on the fold-training rows.
- Empirical Q2_CV = 1 - pooled model SSE / pooled train-fold mean benchmark SSE.
  This is a PLSpredict-style indicator-level evaluation, not a validated
  numerical equivalence to a commercial PLSpredict implementation.

The study's originally published Q2predict might be computed by another
method, so do not equate these estimates without reviewing original scripts.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression

FULL_BLOCKS = {
"PIIT": ["PIIT1","PIIT2","PIIT3","PIIT4"],
"PCR": ["PCR1","PCR2","PCR3","PCR4"],
"SE": ["SE1","SE2","SE3"],
"PEU": ["PEU1","PEU2","PEU3","PEU4"],
"PU": ["PU1","PU2","PU3"],
"PE": ["PE1","PE2","PE3","PE4"],
"BI": ["BI1","BI2","BI3"],
}
FULL_PATHS = [
("PIIT","SE"),("PIIT","PU"),("PIIT","PEU"),("PIIT","BI"),
("SE","PU"),("SE","PEU"),("PCR","PU"),("PCR","BI"),
("PEU","PU"),("PEU","PE"),("PEU","BI"),("PU","BI"),("PE","BI")]
BASE_BLOCKS={k:FULL_BLOCKS[k] for k in ["PEU","PU","BI"]}
BASE_PATHS=[("PEU","PU"),("PEU","BI"),("PU","BI")]
REDUCED_BLOCKS={k:FULL_BLOCKS[k] for k in ["SE","PEU","PU","PE","BI"]}
REDUCED_PATHS=[p for p in FULL_PATHS if p[0] in REDUCED_BLOCKS and p[1] in REDUCED_BLOCKS]
MODELS={"Base TAM":(BASE_BLOCKS,BASE_PATHS),
        "Reduced model (no PIIT/PCR)":(REDUCED_BLOCKS,REDUCED_PATHS),
        "Extended TAM":(FULL_BLOCKS,FULL_PATHS)}

def pls_fit(train, blocks, paths, tol=1e-9, max_iter=500):
    names=list(blocks)
    norms={}
    xx={}
    for k,items in blocks.items():
        x=train[items].to_numpy(float)
        mean=x.mean(0)
        sd=x.std(0,ddof=0)
        if np.any(sd<1e-10):raise ValueError(f"Constant indicator in {k}")
        norms[k]=(mean,sd)
        xx[k]=(x-mean)/sd
    weights={k:np.ones(len(v))/np.sqrt(len(v)) for k,v in blocks.items()}
    parent={k:[a for a,b in paths if b==k] for k in names}
    child={k:[b for a,b in paths if a==k] for k in names}
    def get_scores(w):
        nonstd={k:xx[k]@w[k] for k in names}
        standard={k:nonstd[k]/nonstd[k].std(ddof=0) for k in names}
        return standard
    for it in range(max_iter):
        yy=get_scores(weights)
        new={}
        for k in names:
            inner=np.zeros(len(train))
            if parent[k]:
                par=np.column_stack([yy[p] for p in parent[k]])
                coef=np.linalg.lstsq(par,yy[k],rcond=None)[0]
                inner+=par@coef
            for c in child[k]:
                inner+=np.mean(yy[c]*yy[k])*yy[c]
            inner=inner/inner.std(ddof=0)
            w=np.mean(xx[k]*inner[:,None],axis=0)
            w=w/np.linalg.norm(w)
            if w.mean()<0:w=-w
            new[k]=w
        delta=max(np.linalg.norm(new[k]-weights[k]) for k in names)
        weights=new
        if delta<tol:break
    else:
        raise RuntimeError("PLS nonconvergence")
    yy=get_scores(weights)
    params={}
    for dst in names:
        p=parent[dst]
        if p:
            X=np.column_stack([yy[k] for k in p])
            params[dst]=(p,np.linalg.lstsq(X, yy[dst], rcond=None)[0])
    loading_bi=np.mean(xx['BI']*yy['BI'][:,None],axis=0)
    score_sd={k:float(np.std(xx[k]@weights[k],ddof=0)) for k in names}
    return dict(norms=norms,weights=weights,score_sd=score_sd,
                params=params,loading_bi=loading_bi,iterations=it+1)

def predict_bi(test, fit, blocks):
    norm=fit["norms"]
    comp={}
    for k in fit['weights']:
        if k=="BI":continue
        mean,sd=norm[k]
        X=(test[blocks[k]].to_numpy(float)-mean)/sd
        comp[k]=(X@fit["weights"][k])/fit["score_sd"][k]
    preds,coef=fit["params"]["BI"]
    zb=np.column_stack([comp[k] for k in preds])@coef
    mean,sd=norm["BI"]
    bi=mean+sd*(zb[:,None]*fit["loading_bi"][None,:])
    return bi

def sign_flip_test(deltas, n_perm=20000, seed=20261029):
    rng=np.random.default_rng(seed)
    observed=abs(np.mean(deltas))
    cnt=0
    for batch in range((n_perm+499)//500):
        m=min(500,n_perm-batch*500)
        sign=rng.choice([-1.0,1.0],size=(m,len(deltas)))
        stat=np.abs(np.mean(sign*deltas[None,:],axis=1))
        cnt+=int(np.sum(stat>=observed))
    return (cnt+1)/(n_perm+1)

def run(data, repeats=10, seed=20261029):
    n=len(data)
    assert n==185
    target_items=["BI1","BI2","BI3"]
    targets=target_items+["BI composite"]
    actual=np.column_stack([data[target_items].to_numpy(float),
            data[target_items].to_numpy(float).mean(1)])
    predictions={name:np.zeros((repeats,n,4)) for name in MODELS}
    lm_predictions={name:np.zeros((repeats,n,4)) for name in MODELS}
    benchmark=np.zeros((repeats,n,4))
    split=RepeatedKFold(n_splits=10,n_repeats=repeats,random_state=seed)
    iters=[]
    for ii,(tr,te) in enumerate(split.split(data)):
        rep=ii//10
        train=data.iloc[tr]
        test=data.iloc[te]
        benchmark[rep,te,:]=actual[tr].mean(0)
        for name,(blocks,paths) in MODELS.items():
            fit=pls_fit(train,blocks,paths)
            iters.append((name,fit['iterations']))
            pred=predict_bi(test,fit,blocks)
            predictions[name][rep,te,:3]=pred
            predictions[name][rep,te,3]=pred.mean(1)
            direct=[src for src,dst in paths if dst=="BI"]
            cols=sum((blocks[c] for c in direct),[])
            reg=LinearRegression().fit(train[cols].to_numpy(float),
                                        actual[tr,:3])
            lm=reg.predict(test[cols].to_numpy(float))
            lm_predictions[name][rep,te,:3]=lm
            lm_predictions[name][rep,te,3]=lm.mean(1)
    errors={}
    baserr=actual[None,:,:]-benchmark
    rows=[]
    for name in MODELS:
        errors[name]=(actual[None,:,:]-predictions[name])
        for j,t in enumerate(targets):
            e=errors[name][:,:,j]
            eb=baserr[:,:,j]
            elm=actual[None,:,j]-lm_predictions[name][:,:,j]
            mse=np.mean(e**2)
            rows.append({"Model":name,"Target":t,
                         "RMSE":np.sqrt(mse),"MAE":np.mean(np.abs(e)),
                         "Q2_CV_train_mean":1-mse/np.mean(eb**2),
                         "RMSE_naive_train_mean":np.sqrt(np.mean(eb**2)),
                         "RMSE_LM_benchmark":np.sqrt(np.mean(elm**2)),
                         "delta_MSE_vs_LM":float(np.mean(elm**2)-mse)})
    result=pd.DataFrame(rows)
    comparison=[]
    for group_i,(base,alt) in enumerate([
      ("Base TAM","Extended TAM"),
      ("Base TAM","Reduced model (no PIIT/PCR)"),
      ("Reduced model (no PIIT/PCR)","Extended TAM")]):
      diffs=(errors[base]**2-errors[alt]**2)
      for j,t in enumerate(targets):
        per_person=diffs[:,:,j].mean(0)
        p=sign_flip_test(per_person, n_perm=20000, seed=seed+j+10*group_i)
        comparison.append({"Target":t,"Compared_to":base,"Preferred_model":alt,
                           "Mean_MSE_improvement":float(per_person.mean()),
                           "Paired_sign_flip_p":p,
                           "Alternate_better_n_of_185":int(np.sum(per_person>0)),
                           "mean_absolute_loss_difference":float(np.mean(np.abs(per_person)))})
    comp=pd.DataFrame(comparison)
    return result,comp,iters,predictions,benchmark,actual

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='Metaverse_25Item_Primary_Scored_Data.csv')
    p.add_argument('--reps',type=int,default=10)
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--prefix',default='output/Comment4_CV_verified')
    args=p.parse_args()
    dat=pd.read_csv(args.input)
    result,comp,iters,pred,bench,actual=run(dat,args.reps,args.seed)
    Path(args.prefix).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.prefix+"_metrics.csv",index=False)
    comp.to_csv(args.prefix+"_paired_comparison.csv",index=False)
    print("METRICS\n",result.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("PAIRED COMPARISON\n",comp.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("Convergence",max(x[1] for x in iters),np.median([x[1] for x in iters]))
