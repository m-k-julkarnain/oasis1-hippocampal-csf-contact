"""Run the analysis cells in one shared namespace, exactly as in the Kaggle notebook.

    python run_all.py --mode full      # from raw OASIS-1 (hours; GPU needed for Cells 9-10)
    python run_all.py --mode restore   # from intermediate tables already on Google Drive (about an hour; no GPU)
    python run_all.py --list           # print the order and exit

Cells communicate through shared variables (OUT, lab, D, y, FR, rid, ...), so they must run in this order in one process.
Inside Kaggle, paste each file into a notebook cell in the same order instead (see README).
"""
import argparse, glob, os, sys

FULL = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "19", "20", "21", "22", "18", "23"]
RESTORE = ["01", "01c", "03", "12", "15", "19", "18", "23"]   # 18 is only needed if paper/table1_main.csv is not yet on Drive; it is cheap

def path(tag):
    return glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "*", f"cell{tag}_*.py"))[0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["full", "restore"], default="restore"); ap.add_argument("--list", action="store_true")
    a = ap.parse_args(); order = FULL if a.mode == "full" else RESTORE
    if a.list:
        print("\n".join(os.path.basename(path(t)) for t in order)); sys.exit(0)
    g = {"__name__": "__main__"}
    for t in order:
        p = path(t); print(f"\n######## {os.path.basename(p)} ########", flush=True)
        exec(compile(open(p).read(), p, "exec"), g)
