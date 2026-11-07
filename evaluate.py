"""Local scorer for ARC-AGI tasks, using the Kaggle metric (2 attempts per test output).

Usage: python3 evaluate.py <task_dir> [--limit N] [--show]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import arc_solver

TIME_LIMIT = 20  # seconds per task, enforced via the solver's own budget


def run_task(path):
    task = json.load(open(path))
    t0 = time.time()
    try:
        preds, names = arc_solver.solve_task(task, time_budget=TIME_LIMIT, return_names=True)
    except Exception as e:  # never let one task kill the run
        preds, names = [[[[0]], [[0]]] for _ in task["test"]], [f"error:{e}"]
    score = 0.0
    for p, t in zip(preds, task["test"]):
        if "output" in t and (p[0] == t["output"] or p[1] == t["output"]):
            score += 1
    return os.path.basename(path)[:-5], score / len(task["test"]), names, time.time() - t0


def main():
    d = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))[:limit]
    with Pool(4) as pool:
        results = pool.map(run_task, files, chunksize=4)
    total = sum(r[1] for r in results)
    solved = [r for r in results if r[1] > 0]
    print(f"{d}: {total:.2f}/{len(results)} = {100 * total / len(results):.2f}%  "
          f"(max task time {max(r[3] for r in results):.1f}s)")
    if "--show" in sys.argv:
        from collections import Counter
        c = Counter(r[2][0] if r[2] else "none" for r in solved)
        for k, v in c.most_common():
            print(f"  {v:4d}  {k}")
    json.dump({r[0]: r[1] for r in results}, open(os.path.basename(d.rstrip("/")) + "_scores.json", "w"))


if __name__ == "__main__":
    main()
