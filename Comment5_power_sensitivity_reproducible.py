#!/usr/bin/env python3
"""Reproduce sample-size sensitivity calculations for the metaverse revision.
Uses only sample sizes/model structure, not observed p-values or effect estimates.
This is an approximate linear-regression / correlation benchmark, not an exact
power calculation for PLS-SEM, MICOM or PLS-MGA.
Requires SciPy, prints table and optionally saves CSV.
"""
import csv
import os
from scipy import stats, optimize
from math import sqrt, tanh

N, K = 185, 5
N_PARA, N_MED = 87, 98
POWER = 0.80

def partial_f_power(n, predictors, f2, alpha):
    denominator_df = n-predictors-1
    fcrit = stats.f.isf(alpha, 1, denominator_df)
    return stats.ncf.sf(fcrit, 1, denominator_df, n*f2)

def min_f2(n, predictors, alpha, target=0.80):
    return optimize.brentq(
        lambda f2: partial_f_power(n, predictors, f2, alpha)-target,
        1e-8, 5.0,
    )

def required_n(predictors, f2, alpha, target=0.80, limit=10000):
    for n in range(predictors+4, limit):
        if partial_f_power(n, predictors, f2, alpha) >= target:
            return n
    raise RuntimeError("N over limit")

def diff_fisher_power(delta_z, n1, n2, alpha):
    se=sqrt(1/(n1-3)+1/(n2-3))
    zcrit=stats.norm.isf(alpha/2)
    zsignal=abs(delta_z)/se
    return stats.norm.sf(zcrit-zsignal)+stats.norm.cdf(-zcrit-zsignal)

def fisher_min_delta(n1,n2,alpha,target=.80):
    return optimize.brentq(
        lambda dz: diff_fisher_power(dz,n1,n2,alpha)-target,
        1e-8, 5.0
    )

rows=[]
for alpha_label,alpha in [('two-sided 0.05',0.05),('Bonferroni illustration 0.05/13',0.05/13)]:
    f2=min_f2(N,K,alpha)
    rows.append(dict(analysis='single incremental predictor, five covariates total',
        alpha_scenario=alpha_label, n=N,
        power_target=POWER, minimum_f2=round(f2,6),
        minimum_partial_r2=round(f2/(1+f2),6),
        power_for_f2_0_02=round(partial_f_power(N,K,0.02,alpha),6),
        required_n_for_f2_0_02=required_n(K,.02,alpha)))
    dz=fisher_min_delta(N_PARA,N_MED,alpha)
    rows.append(dict(analysis='illustrative independent-group Fisher-z comparison',
        alpha_scenario=alpha_label,n=f'{N_PARA} vs {N_MED}',
        power_target=POWER,minimum_delta_fisher_z=round(dz,6),
        correlation_if_reference_r0=round(tanh(dz),6),
        power_for_delta_z_0_20=round(diff_fisher_power(.2,N_PARA,N_MED,alpha),6)))
print('Power-analysis benchmark | five direct BI predictors | n=185')
for row in rows:
    print(row)
out=os.environ.get('METAVERSE_POWER_OUTPUT','Comment5_sample_size_sensitivity.csv')
headers=list(dict.fromkeys(key for r in rows for key in r))
with open(out,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=headers)
    w.writeheader()
    w.writerows(rows)
print('Saved to',out)
