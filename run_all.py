"""Run the whole pipeline in order: Part I -> Part II -> Part III -> error sample.

Usage:  python run_all.py            (full run)
        QUICK=1 python run_all.py    (small smoke test, writes to outputs_quick/)
Extra args select parts, e.g.  python run_all.py 2 3
"""
import subprocess
import sys
import time

STEPS = {
    "1": "part1_language_models.py",
    "2": "part2_embeddings.py",
    "3": "part3_downstream.py",
    "4": "part3_error_sample.py",   # 20 sampled errors for the report's Appendix C
}

parts = sys.argv[1:] or list(STEPS)
for p in parts:
    script = STEPS[p]
    print(f"\n{'=' * 70}\n>>> {script}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    subprocess.run([sys.executable, script], check=True)
    print(f">>> {script} finished in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
