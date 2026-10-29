# Metaverse acceptance study — reproducibility code package
**Version:** 1.0.0 (independent reconstruction; prepared 2026-10-29)
**Study:** Behavioral Intention to Use Metaverse-Based Learning Among Medical and Paramedical Students in Iran: A Cross-Sectional Study
**Primary dataset:** 185 participants, 25-item primary PLS model (PU4 excluded after collection).
**Secondary datasets:** same 185 respondents, complete administered 26-item model, post-hoc 17-item model excluding PIIT/PCR.

## What this release actually is
A code-only, versioned reconstruction of the statistical analyses described in the revision. Existing independent reconstruction scripts were gathered together and new source files were written to address gaps in structural bootstrap, MICOM, PLS-MGA, common-method diagnostics, indirect effects, and figures. **This package is NOT evidence that the original historical scripts have been retrieved**, nor that a separate established PLS-SEM program such as SmartPLS/cSEM has reproduced the findings.

The default 5,000 bootstrap samples, 5,000 MICOM/MGA permutations, and 10×10 repeated cross-validation reproduce methods at the specified counts. Individual interval endpoints and p-values can differ from those in the originally submitted tables because the original seeds and implementation details are unavailable. Compare outputs and update the manuscript; never assume exact equivalence.

### Coverage and files

| Manuscript output | Source file(s) | Scope |
|---|---|---|
| Original response and scored-data provenance; S7 | `metaverse_initial_audit.py`, `audit_PCR_items.py` | Exact raw survey scoring audit, PIIT3 and PCR1/2 reversals |
| Table 1 / Table 2 | `metaverse_descriptives_and_figures.py` | Faculty counts, observed-score summaries, Welch tests, Hedges g, Holm adjustment |
| Table 3, Table 4 | `metaverse_full_structural_reanalysis.py` | 25-item PLS path weighting, measurement indices, 5,000 bootstrap, 13 path p-values and Holm |
| Table 5 (R², effects) | `metaverse_full_structural_reanalysis.py` | R², adjusted R², indirect/total, specific PEU→PE→BI, 5,000 bootstrap |
| Figure 4 IPMA | `metaverse_full_structural_reanalysis.py`, `metaverse_figures_from_results.py` | Total effects and observed mean performance; exploratory for PIIT/PCR |
| Table S2 | `metaverse_common_method_fit.py` | Harman's first component, observed-mean VIF, PLS-score VIF, approximate SRMR |
| Table S3 | `metaverse_faculty_micom_mga.py` | Faculty-specific paths, MICOM permutation, 13-path permutation MGA, group bootstrap |
| Table 5 Panel C / Table S4 | `Comment4_fold_safe_PLS_predict.py` | Leakage-controlled 10×10 CV; fold-mean/LM baselines, Q²CV, paired sign-flip tests |
| Table S5 | `metaverse_full_structural_reanalysis.py`, `metaverse_PU4_predictive_comparison.py` | 25-item versus administered 26-item model and prediction |
| Table S6 | `PLS_PIIT_PCR_sensitivity_reproducible.py`, `metaverse_full_structural_reanalysis.py` | Exploratory reduced five-construct 17-item PLS; 2,000 bootstrap in dedicated script |
| Table S7 | `audit_PCR_items.py` | 740 PCR item-value cross-checks from raw exports |
| Table S8 | `Sampling_faculty_calibration_PLS_reproducible.py` | Faculty calibration 25-/17-item model, stratified bootstrap 2,000 |
| Table S9 | `Comment5_power_sensitivity_reproducible.py` | Approximate noncentral-F and Fisher-z retrospective sensitivity |
| Figures 1–6 | `metaverse_descriptives_and_figures.py`, `metaverse_figures_from_results.py` | Independently regenerated chart images; check legends and design before use |

### Requirements
Python 3.13.5 with pinned requirements in `requirements.txt`. Versions below are those observed in the verification environment, not a guarantee that all other systems produce bit-identical floating-point results.

### Privacy and data
The archive intentionally **does not include any student-level, raw survey, demographic, or scored CSV files**.
To run it, use a private local folder that contains:
- `Metaverse_25Item_Primary_Scored_Data.csv` (required)
- `Metaverse_Cleaned_Scored_Data.csv` (required for full raw provenance audit)
- `MedicalMetaverse.csv` and `Paramedical Metaverse.csv` (required for item-by-item raw response audit)

The CSV columns include `Faculty_EN`, item identifiers PU1–PU4, PEU1–PEU4, SE1–SE3, PE1–PE4, PCR1–PCR4, PIIT1–PIIT4, BI1–BI3, and original study identifiers.

**Reference file hashes (SHA256) for local provenance verification:**
- `Metaverse_25Item_Primary_Scored_Data.csv`: `8541d94d512b02f07641a9b04dc36859bbae4a27125c3ada4ab6bdb53b9e33ac`
- `Metaverse_Cleaned_Scored_Data.csv`: `b1f3ce44e490b489cdcba308b2fdfe10e6ad6457ea568f6c1bc969078c9565ad`
- `MedicalMetaverse.csv`: `73018fdf7b7dd4ea63f005058ec0de1d3e8d0930599d4cf37da583683f34e879`
- `Paramedical Metaverse.csv`: `c58778f600c3191e42afe1625a3d366752554dfea3e4d0a25fd1c99139337b37`


### Reproduce
```bash
python -m pip install -r requirements.txt
python test_numerical_regression.py --input PRIVATE_DATA/Metaverse_25Item_Primary_Scored_Data.csv
python run_all.py --data-dir PRIVATE_DATA --output results --mode quick
python run_all.py --data-dir PRIVATE_DATA --output results --mode full
```
`--mode quick` is **only a computational smoke test** (35 bootstrap draws; 25 permutations; 1×10 CV), not manuscript inference. `--mode full` uses 5,000 bootstrap draws / 5,000 permutations / 10×10 CV; weighted bootstrap uses 2,000 resamples, matching the revised sensitivity analysis. The fixed random seed is 20261029 for newly reconstructed scripts; reduced standalone script originally used 20261028.

The full routine saves tables/plots into the user-selected `results` folder; no network calls and no repository publication occurs automatically.

### Important methodological distinctions
- The reversed indicators were `PIIT3`, `PCR1`, and `PCR2`. PCR3 and PCR4 were direct-scored. Absence of mismatches does not establish the construct's validity.
- PU4 was administered to students but excluded **post hoc** from primary analysis due to tourism-domain wording. The full 26-item model is a sensitivity specification.
- The 17-item model excluding PIIT/PCR was specified **post hoc** and is not confirmatory evidence.
- The 13 primary paths are explicitly output with both empirical unadjusted and 13-family Holm-adjusted p-values. Decisions should be revised to match the chosen multiplicity policy.
- Bootstrapped sign-tail p-values are empirically approximated; their granularity is limited by resample count and they need not equal a t-based p-value.
- MICOM uses custom group-specific composition weights and re-estimation in each permutation; it is not independently cross-checked with standard PLS software.
- The Q²CV values are repeated cross-validation scores against **training-fold means**; do NOT re-label them as numerically verified commercial `Q²predict`. Predictor standardization, weight fitting and indicator loadings occur within each training fold.
- **S2 VIF values are based on observed construct means.** PLS latent-composite VIF values are different and are separately output. Harman's PCA ≈ 46.036%; approximate SRMR ≈ 0.105.
- Heterogeneous faculties and small programs must not be given causal interpretation; tests may not establish invariance.
- Faculty weights are population-to-sample *calibration weights*, not individual inclusion-probability weights.
- The power calculations use linear-regression/Fisher-z approximations and are not prospective or exact PLS-MGA power estimates.
- Figures are **new renderings**, not guaranteed exact source-figure replications.

### Before claiming full reproducibility in an editor response
1. Run the full pipeline on the actual locked dataset, review all generated tables, and reconcile every final manuscript statistic with output or document deviations.
2. Validate the core PLS/MICOM/PLSpredict results with an independently established package if such equivalence is claimed. **This cross-check remains outstanding.**
3. Archive the complete code in a recognised DOI-assigning repository (for example Zenodo) with the chosen final license, a release version, README, requirements, and code hashes.
4. Replace the DOI placeholder in the manuscript's `Code availability` and the point-by-point response **only after the host provides the actual DOI**.
5. Leave participant-level data out of a public code-only release unless ethics and consent explicitly permit disclosure.

### Scope of checks already carried out
- 185 total, faculty counts 98 and 87.
- Rebuilt primary model BI R² = 0.700636818; complete 26-item R² = 0.697508100; reduced 17-item BI R² = 0.629549835.
- PIIT→BI = 0.419378, PE→BI = 0.387747; reduced-model PE→BI = 0.558545.
- MICOM original group-specific composition correlations: PIIT ≈ 0.942; PCR ≈ 0.761; SE ≈ 0.997; PEU ≈ 0.992; PU ≈ 1.000; PE ≈ 0.998; BI ≈ 1.000.
- Harman first PCA share ≈ 46.036%, observed-mean VIFs and approximate SRMR reproduce the values in S2.
- Quick smoke execution of new full-structural, MICOM/MGA, descriptive, 26-item CV and diagnostics scripts succeeded.
- **5,000-iteration full pipeline, independent third-party package equivalence and code DOI deposit are NOT claimed completed by this package.**

### License and authorship
A research group should select its authorized copyright/license terms before public repository release. The code is provided here as a reproduction aid; no assertion of transfer of ownership to another party is made.
