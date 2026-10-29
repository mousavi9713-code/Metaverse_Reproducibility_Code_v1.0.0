#!/usr/bin/env python3
"""26-item PU4 prediction sensitivity using identical fold-wise pipeline.

Both models are trained within each of 100 folds; 26-item model adds PU4
only to PU measurement block. This script does not reuse full-data latent scores.
"""
import argparse
from pathlib import Path
import pandas as pd
import Comment4_fold_safe_PLS_predict as cv

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--out",default=Path("output"),type=Path)
    p.add_argument("--repetitions",type=int,default=10)
    p.add_argument("--seed",type=int,default=20261029)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    d=pd.read_csv(args.input)
    saved=cv.MODELS
    records=[]
    try:
        for n in (25,26):
            models={}
            for label,(blocks,paths) in saved.items():
                b={k:list(v) for k,v in blocks.items()}
                if n==26:b["PU"]=["PU1","PU2","PU3","PU4"]
                models[label]=(b,paths)
            cv.MODELS=models
            metrics,comp,_,_,_,_=cv.run(d,repeats=args.repetitions,seed=args.seed)
            metrics.insert(0,"instrument_items",n)
            comp.insert(0,"instrument_items",n)
            records.append((metrics,comp))
            print(n,"BI composite results\n",
                  metrics.loc[metrics.Target.eq("BI composite")].to_string(index=False))
    finally:
        cv.MODELS=saved
    pd.concat([x[0] for x in records],ignore_index=True).to_csv(
        args.out/"PU4_25v26_CV_metrics.csv",index=False)
    pd.concat([x[1] for x in records],ignore_index=True).to_csv(
        args.out/"PU4_25v26_CV_paired.csv",index=False)
if __name__=="__main__":main()
