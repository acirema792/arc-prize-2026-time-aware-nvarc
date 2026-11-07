"""Spawn one test-time-training worker per GPU over a shared task queue.

Adapted from NVARC's ARC Prize 2025 starter: paths come from the environment,
tasks are queued smallest-first so the dynamic per-task budget in
llm_worker.task_budget can hand any time saved on easy tasks to later ones,
and ARC_TASK_KEYS can restrict the run to a subset (smoke tests).
"""
import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time, n_workers):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"../worker{rank-1}"):
            time.sleep(5)

    from llm_worker import worker

    with open(f"../worker{rank}", "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")

    # an exception in one worker makes mp.spawn stop all of them, so the rest of the queue would go
    # unsolved: log it, free the GPU and restart (only the task in progress is lost)
    import gc
    import traceback
    for attempt in range(4):
        try:
            worker(rank, queue, end_time, n_workers)
            break
        except Exception:
            traceback.print_exc()
        # outside the except block, so the traceback no longer holds the crashed model
        gc.collect()
        torch.cuda.empty_cache()
        if time.time() > end_time - 300:
            break
        print(f"[Rank {rank}] restarting after a crash")

    print(f"[Rank {rank}] done!")


def task_size(task):
    return sum(len(g) * len(g[0]) for pair in task["train"] for g in pair.values()) + \
        sum(len(t["input"]) * len(t["input"][0]) for t in task["test"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    with open(os.environ["ARC_TEST_PATH"], "r") as f:
        data = json.load(f)

    keys = sorted(data.keys(), key=lambda k: (task_size(data[k]), k))
    if os.getenv("ARC_TASK_ORDER") == "file":  # v17 NVARC-faithful mode: the input file's order
        keys = list(data.keys())
    subset = os.getenv("ARC_TASK_KEYS")
    if subset:
        keys = [k for k in keys if k in set(subset.split(","))]

    n_workers = max(1, torch.cuda.device_count())
    print(f"Queueing {len(keys)} tasks on {n_workers} GPUs")

    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)
    for _ in range(n_workers):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, args.end_time, n_workers), nprocs=n_workers)
