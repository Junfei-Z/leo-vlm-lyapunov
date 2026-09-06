#!/usr/bin/env python3
"""Master: full re-run suite at rho=0.3, U=0.30 (the new mainline)."""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

def run(name, *args):
    print(f"\n########## {name} ##########", flush=True)
    subprocess.run([PY, os.path.join(HERE, name), *args], check=True)

if __name__ == "__main__":
    run("tleP30_rerun.py")           # core + stability + 4 sweeps
    run("tleP30_isl.py")             # ISL sweep
    run("qsrc45_u30.py")             # 45-source
    run("contrast_u30.py")           # staggered frontier
    run("exp1_target_boundary.py")   # boundary at rho=0.3 (uses exp_common primary; see below)
    print("ALL DONE", flush=True)
