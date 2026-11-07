"""Build submission.json from the LLM's ranked guesses plus the rule solver.

  python merge.py symbolic   # run the rule solver, write a safety submission.json
  python merge.py final      # merge LLM outputs (../inference_outputs) on top

Ranking: the LLM's top two guesses come first (NVARC's kgmon selection).
Rule-solver programs that reproduce every training pair only fill slots the
LLM left empty: on the ARC-AGI-2 public eval they fit some tasks without
generalising, so they must never displace an LLM guess.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

TEST_PATH = os.environ.get("ARC_TEST_PATH", "")
SOLUTIONS_PATH = os.environ.get("ARC_SOLUTIONS_PATH", "")
SYMBOLIC_PATH = os.environ.get("ARC_SYMBOLIC_PATH", "symbolic_candidates.json")
LLM_DIR = os.environ.get("ARC_LLM_DIR", "../inference_outputs")
OUT_PATH = os.environ.get("ARC_SUBMISSION_PATH", "submission.json")
RULE_SECOND = os.environ.get("ARC_RULE_SECOND", "1") == "1"


def valid_grid(g):
    try:
        a = np.asarray(g)
    except Exception:
        return False
    return a.ndim == 2 and 0 < a.shape[0] <= 30 and 0 < a.shape[1] <= 30 and \
        np.issubdtype(a.dtype, np.integer) and a.min() >= 0 and a.max() <= 9


def build_submission(challenges, llm_ranked, symbolic):
    """challenges: {task_id: task}; llm_ranked: {"<task_id>_<i>": [grid, ...]} best first;
    symbolic: {task_id: [[grid, ...] per test input]}."""
    sub = {}
    for tid, task in challenges.items():
        entries = []
        for i, t in enumerate(task["test"]):
            cands = []

            def add(g):
                if valid_grid(g):
                    g = np.asarray(g).tolist()
                    if g not in cands:
                        cands.append(g)

            llm = llm_ranked.get(f"{tid}_{i}", [])[:2]
            sym = symbolic.get(tid, [])
            sym = sym[i] if i < len(sym) else []
            # a rule program that reproduces every train pair (97% precise on public training
            # tasks) outranks the model's second guess, but never its first; v17 NVARC-faithful
            # mode (ARC_RULE_SECOND=0) keeps both model guesses and only fills empty slots
            if RULE_SECOND:
                order = llm[:1] + sym[:1] + llm[1:] + sym[1:]
            else:
                order = llm + sym
            for g in order:
                add(g)
            add(t["input"])
            add([[0]])
            entries.append({"attempt_1": cands[0], "attempt_2": cands[1]})
        sub[tid] = entries
    return sub


def score(sub, solutions):
    total = 0.0
    for tid, outs in solutions.items():
        for i, truth in enumerate(outs):
            e = sub[tid][i]
            if e["attempt_1"] == truth or e["attempt_2"] == truth:
                total += 1 / len(outs)
    return total, len(solutions)


def _symbolic_one(item):
    import arc_solver
    tid, task = item
    try:
        return tid, arc_solver.verified_candidates(task, time_budget=20.0)[1]
    except Exception:
        return tid, [[] for _ in task["test"]]


def run_symbolic(challenges, processes=4):
    with Pool(processes) as pool:
        return dict(pool.map(_symbolic_one, sorted(challenges.items()), chunksize=2))


def load_llm_ranked(test_path, llm_dir):
    if not os.path.isdir(llm_dir) or not os.listdir(llm_dir):
        return {}
    from arc_decoder import ArcDecoder, score_kgmon
    decoder = ArcDecoder(dataset=None, n_guesses=2)
    decoder.load_decoded_results(llm_dir)
    return decoder.run_selection_algo(score_kgmon)


def diagnose(challenges, llm_dir, solutions):
    """Commit runs only: is the answer lost in generation (never among the candidates) or in
    selection (a candidate, but not in the top two)? Also scores each selection rule."""
    from arc_decoder import ArcDecoder, score_kgmon, selection_algorithms
    decoder = ArcDecoder(dataset=None, n_guesses=2)
    decoder.load_decoded_results(llm_dir)
    found, loose_only, hits = 0.0, 0, []
    for tid, outs in solutions.items():
        for i, truth in enumerate(outs):
            cands = decoder.decoded_results.get(f"{tid}_{i}", {}).values()
            right = [c for c in cands if np.array_equal(c["solution"], truth)]
            if right:
                found += 1 / len(outs)
                hits.append(f"{tid}_{i}:{len(right)}/{len(cands)}")
                loose_only += all(c.get("loose") for c in right)
    print(f"*** answer among the candidates: {found:.2f}/{len(solutions)} "
          f"({loose_only} only from the looser pass) {' '.join(hits)}")
    for algo in selection_algorithms:
        s, n = score(build_submission(challenges, decoder.run_selection_algo(algo), {}), solutions)
        print(f"*** selection {algo.__name__}: {s:.2f}/{n}")
    # v15: the same run without the second adapter's candidates, so its effect is measured on
    # identical first-pass results (no run-to-run noise)
    second = sum(c.get("second", False) for v in decoder.decoded_results.values() for c in v.values())
    if second:
        tasks = sorted({bk.rsplit("_", 1)[0] for bk, v in decoder.decoded_results.items()
                        if any(c.get("second") for c in v.values())})
        decoder.decoded_results = {bk: {k: c for k, c in v.items() if not c.get("second")}
                                   for bk, v in decoder.decoded_results.items()}
        s, n = score(build_submission(challenges, decoder.run_selection_algo(score_kgmon), {}), solutions)
        print(f"*** first pass only (kgmon): {s:.2f}/{n}; second pass added {second} candidates "
              f"on {len(tasks)} tasks: {' '.join(tasks)}")


def write(sub, challenges):
    assert set(sub) == set(challenges), "every task id must be present"
    with open(OUT_PATH, "w") as f:
        json.dump(sub, f)
    if SOLUTIONS_PATH and os.path.exists(SOLUTIONS_PATH):
        sol = json.load(open(SOLUTIONS_PATH))
        sol = {k: v for k, v in sol.items() if k in challenges}
        s, n = score(sub, sol)
        print(f"*** local score: {s:.2f}/{n} = {100 * s / max(n, 1):.2f}%")


if __name__ == "__main__":
    challenges = json.load(open(TEST_PATH))
    mode = sys.argv[1]
    if mode == "symbolic":
        symbolic = run_symbolic(challenges)
        json.dump(symbolic, open(SYMBOLIC_PATH, "w"))
        n = sum(any(c for c in v) for v in symbolic.values())
        print(f"rule solver: programs found for {n}/{len(challenges)} tasks")
        write(build_submission(challenges, {}, symbolic), challenges)
    elif mode == "final":
        symbolic = json.load(open(SYMBOLIC_PATH)) if os.path.exists(SYMBOLIC_PATH) else {}
        llm = load_llm_ranked(TEST_PATH, LLM_DIR)
        print(f"LLM produced guesses for {len(llm)} test outputs")
        write(build_submission(challenges, llm, symbolic), challenges)
        if llm and SOLUTIONS_PATH and os.path.exists(SOLUTIONS_PATH):
            sol = json.load(open(SOLUTIONS_PATH))
            try:  # a diagnostic must never fail the run
                diagnose(challenges, LLM_DIR, {k: v for k, v in sol.items() if k in challenges})
            except Exception as e:
                print("diagnostic failed:", repr(e))
    else:
        raise SystemExit(f"unknown mode {mode}")
