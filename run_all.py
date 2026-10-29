#!/usr/bin/env python3
"""Execute all independent reproductions of manuscript analyses.

python run_all.py --data-dir PRIVATE_DATA --output results --mode quick
python run_all.py --data-dir PRIVATE_DATA --output results --mode full

No participant-level data are included in the code release.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=Path("outputs"))
    ap.add_argument("--mode",choices=["quick","full"],default="full")
    ap.add_argument("--skip-raw-audit",action="store_true",
                    help="Only if the two original raw Persian exports are unavailable")
    args=ap.parse_args()
    source=args.data_dir/"Metaverse_25Item_Primary_Scored_Data.csv"
    if not source.exists():
        ap.error(f"Required scored dataset not found: {source}")
    args.output.mkdir(parents=True,exist_ok=True)
    B=35 if args.mode=="quick" else 5000
    P=25 if args.mode=="quick" else 5000
    W=25 if args.mode=="quick" else 2000
    reps=1 if args.mode=="quick" else 10
    def call(fname,*argv,env=None):
        cmd=[sys.executable,str(SCRIPTS/fname),*map(str,argv)]
        print("\n$"," ".join(cmd),flush=True)
        subprocess.run(cmd,check=True,env=env)

    if not args.skip_raw_audit:
        for name in ("Paramedical Metaverse.csv","MedicalMetaverse.csv",
                     "Metaverse_Cleaned_Scored_Data.csv"):
            if not (args.data_dir/name).exists():
                raise FileNotFoundError(f"Raw-origin audit requires {name}")
        call("metaverse_initial_audit.py","--data-dir",args.data_dir)
        call("audit_PCR_items.py","--data-dir",args.data_dir)
    call("metaverse_descriptives_and_figures.py","--input",source,"--out",args.output)
    call("metaverse_common_method_fit.py","--input",source,"--out",args.output)
    call("metaverse_full_structural_reanalysis.py","--input",source,"--out",args.output,
         "--bootstrap",B,"--models","25item","26item","reduced17")
    call("metaverse_faculty_micom_mga.py","--input",source,"--out",args.output,
         "--permutations",P,"--bootstrap",B)
    call("Sampling_faculty_calibration_PLS_reproducible.py","--csv",source,
         "--bootstrap",W,"--out",args.output)
    call("Comment4_fold_safe_PLS_predict.py","--input",source,
         "--reps",reps,"--prefix",args.output/"primary_25item_CV")
    call("metaverse_PU4_predictive_comparison.py","--input",source,
         "--repetitions",reps,"--out",args.output)
    env=os.environ.copy()
    env["METAVERSE_POWER_OUTPUT"]=str(args.output/"power_sensitivity.csv")
    call("Comment5_power_sensitivity_reproducible.py",env=env)
    call("metaverse_figures_from_results.py","--output",args.output)
    print("\nReconstruction run complete. See README for non-equivalences and outstanding verification.")
if __name__=="__main__":
    main()
