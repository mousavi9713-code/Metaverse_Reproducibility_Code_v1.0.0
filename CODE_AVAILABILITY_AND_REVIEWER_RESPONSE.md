# Manuscript and reviewer response — code availability

## Original text to replace
`Code availability: No custom software, algorithm, or computational tool was developed in this study. The analyses were conducted using standard statistical procedures. The analytical outputs and detailed results required to understand and evaluate the findings are provided in the Supplementary Information.`

## Replacement, ONLY AFTER a working DOI has been assigned
`Code availability: Custom Python analysis scripts were used to implement the scoring audit, reflective Mode-A PLS path modelling, bootstrap and permutation procedures, MICOM, multi-group comparisons, predictive cross-validation, and sensitivity analyses. A versioned reproducibility package containing the source scripts, parameter settings, dependency specifications, and instructions is available at [INSERT ACTUAL REPOSITORY DOI] (version 1.0.0). The repository is code-only and does not contain participant-level survey records. Access to participant-level research data is governed by the Data Availability statement.`

**Do not represent independent reconstruction as the original historical source code.** If the deposited version comprises reconstructed scripts rather than the authors' historical implementations, explicitly describe that status:
`The deposited package contains independently reconstructed, documented Python implementations of the analyses used to verify and revise the manuscript. The original historical execution environment and all third-party PLS software equivalence checks could not be independently reconstructed.`

## Methods: Statistical analysis
Keep existing reporting that path weighting was implemented with custom Python routines and list exact dependent packages. Add the following only after verifying runtime:
`The documented source code used in the revised analyses, including fixed random seeds and fold-level preprocessing, is archived in the versioned reproducibility package linked under Code availability.`

## CRediT
Keep `Software` only if it accurately describes authors' contribution to developing or adapting scripts used for the study; do not remove merely to hide bespoke coding.

## Reviewer 3 comment 2.3 — after real DOI deposit and final full-data check
**Response:**
We thank the reviewer for identifying the inconsistency between the description of the Python analytical procedures and our previous Code Availability statement. We agree that the implementation scripts, even when they operationalize established statistical methods rather than novel algorithms, constitute a bespoke computational workflow and must be accessible for independent evaluation.

We have corrected the Code Availability statement and clarified that the PLS Mode-A estimation, bootstrap and permutation procedures, MICOM, multi-group analysis, cross-validation, and sensitivity analyses were implemented using Python scripts. A documented, versioned code package has been prepared and its reconstruction tests have been checked against the source scored dataset. The package includes the source scripts, a reproducibility guide, pinned library dependencies, analysis parameters, and numerical regression tests.

**After successful DOI deposit, append:**
The finalized code package has been deposited in [REPOSITORY NAME] at [VERIFIED DOI], with version 1.0.0 identified in the manuscript's Code Availability statement. No participant-level survey records were included in the public code release; research data availability is described separately.

We have retained the first author's Software CRediT role because Python implementation contributed to the study analyses. We have also clarified that the package reconstructs established methods; it does not represent a newly proposed statistical algorithm. Independent validation using an established third-party PLS-SEM implementation is a separate methodological check and is not implied by publication of these scripts alone.

## Outstanding
- Set the final DOI after actual deposit; NEVER invent.
- Run `--mode full`; check and reconcile Table 4, Table 5, S2, S3, S4, S5 and all revised results before publication.
- Independently validate key results with an established PLS-SEM implementation if asserting numerical equivalence.
- Ensure the complete original/reconstructed analytical scripts actually used for all claims are represented and the Methods accurately describe that distinction.
