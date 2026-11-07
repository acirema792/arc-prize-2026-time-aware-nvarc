"""CPU-only checks of the pipeline logic that does not need a GPU or the model.

  PYTHONPATH=tests/stubs python3 tests/test_pipeline.py

Covers: the merge order of model and rule-solver answers, the per-task time budget, the
second-pass helpers in llm_worker.worker (vote counting, weak-task claims), and the merge
diagnostic that scores a run with and without its second-pass candidates.
"""
import bz2
import json
import os
import pickle
import queue
import shutil
import sys
import tempfile
import textwrap
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))


def grid(v, n=3):
    return [[v] * n for _ in range(n)]


def fake_tasks():
    """Four small tasks; the answer to each test input is the input recolored to task-specific value."""
    tasks, solutions = {}, {}
    for t, (name, size) in enumerate([("aaaa0001", 2), ("bbbb0002", 3), ("cccc0003", 4), ("dddd0004", 5)]):
        n_tests = 2 if t == 3 else 1
        tasks[name] = {"train": [{"input": grid(1, size), "output": grid(t + 2, size)}] * 2,
                       "test": [{"input": grid(1, size)} for _ in range(n_tests)]}
        solutions[name] = [grid(t + 2, size) for _ in range(n_tests)]
    return tasks, solutions


def test_merge_order():
    import importlib
    import merge
    ch = {"t": {"train": [], "test": [{"input": [[5]]}]}}
    for rule_second, expected in (("1", [[9]]), ("0", [[2]])):
        os.environ["ARC_RULE_SECOND"] = rule_second
        importlib.reload(merge)
        sub = merge.build_submission(ch, {"t_0": [[[1]], [[2]]]}, {"t": [[[[9]]]]})
        assert sub["t"][0]["attempt_1"] == [[1]], sub
        assert sub["t"][0]["attempt_2"] == expected, (rule_second, sub)
    os.environ.pop("ARC_RULE_SECOND")
    print("merge order: ok")


def test_task_budget():
    src = open(os.path.join(ROOT, "pipeline", "llm_worker.py")).read()
    fn = src[src.index("def task_budget"):src.index("def plan_training")]

    class Q:
        def qsize(self):
            return 40

    for fixed, expected in ((True, 1200), (False, 400)):
        ns = {"time": time, "MIN_TASK_SECONDS": 120, "MAX_TASK_SECONDS": 1200, "FIXED_BUDGET": fixed}
        exec(fn, ns)
        got = round(ns["task_budget"](Q(), 4, time.time() + 4000))
        assert got == expected, (fixed, got)
    print("task budget: ok")


def test_second_pass_and_diagnostic():
    from arc_loader import ArcDataset

    tmp = tempfile.mkdtemp()
    work = os.path.join(tmp, "kaggle", "working")
    os.makedirs(work)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        tasks, sol = fake_tasks()
        json.dump(tasks, open("c.json", "w"))
        json.dump(sol, open("s.json", "w"))
        arc_test_set = ArcDataset.from_file("c.json")
        dir_outputs = "../inference_outputs"
        os.makedirs(dir_outputs)

        # the exact helper code from llm_worker.worker, run outside the GPU loop
        src = open(os.path.join(ROOT, "pipeline", "llm_worker.py")).read()
        block = src[src.index("    # v15 second pass"):src.index("    for key, train_seed, view_seed, second in jobs():")]
        ns = dict(os=os, bz2=bz2, pickle=pickle, defaultdict=defaultdict, time=time, arc_test_set=arc_test_set,
                  dir_outputs=dir_outputs, rank=0, SECOND_PASS=True, SECOND_PASS_MAX_VOTES=8,
                  MIN_DECODE_SECONDS=240, end_time=time.time() + 3600)
        q = queue.Queue()
        keys = sorted(tasks)
        for k in keys:
            q.put(k)
        q.put(None)
        ns["queue"] = q
        exec(textwrap.dedent(block), ns)

        def write(name, grids, second=False):
            with bz2.BZ2File(os.path.join(dir_outputs, name), "w") as f:
                pickle.dump([{"beam_score": 0.1, "score_aug": [0.2] * 8, "solution": np.array(g),
                              "loose": False, "second": second} for g in grids], f)

        k0, k1, k2, k3 = keys
        for i in range(12):                      # k0: strong right answer
            write(f"{k0}_0.v{i}", [sol[k0][0]])
        for i in range(3):                       # k1: weak right answer, outvoted by a wrong one
            write(f"{k1}_0.v{i}", [sol[k1][0]])
        for i in range(5):
            write(f"{k1}_0.w{i}", [[[0]]])
        for j in range(2):                       # k3: strong on both test outputs
            for i in range(9):
                write(f"{k3}_{j}.v{i}", [sol[k3][j]])
        for k in keys:
            open(os.path.join(ns["dir_done"], k), "w").close()

        votes = {k: ns["top_votes"](k) for k in keys}
        assert votes == {k0: 12, k1: 5, k2: 0, k3: 9}, votes

        jobs = list(ns["jobs"]())
        first = [j for j in jobs if not j[3]]
        second = [j for j in jobs if j[3]]
        assert [j[0] for j in first] == keys
        assert [j[0] for j in second] == sorted([k1, k2], key=ns["task_cells"]), second
        assert all(j[1:] == (7, 3, True) for j in second)

        for i in range(6):                       # the second adapter adds votes for k1's right answer
            write(f"{k1}_0.s{i}", [sol[k1][0]], second=True)

        os.environ.update(ARC_TEST_PATH=os.path.abspath("c.json"), ARC_SOLUTIONS_PATH=os.path.abspath("s.json"),
                          ARC_LLM_DIR=dir_outputs, ARC_SUBMISSION_PATH="submission.json",
                          ARC_SYMBOLIC_PATH="none.json")
        sys.argv = ["merge.py", "final"]
        exec(open(os.path.join(ROOT, "pipeline", "merge.py")).read(), {"__name__": "__main__"})
        sub = json.load(open("submission.json"))
        assert set(sub) == set(tasks)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp)
    print("second pass and diagnostic: ok")


if __name__ == "__main__":
    test_merge_order()
    test_task_budget()
    test_second_pass_and_diagnostic()
    print("all tests passed")
