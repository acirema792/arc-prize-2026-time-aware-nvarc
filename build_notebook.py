"""Assemble the Kaggle submission notebook from the files in pipeline/.

  python3 build_notebook.py   ->  arc2026_nvarc_plus.ipynb
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--top", default="8", help="most training copies per task (8, 16, 24, or auto: 16 only where cheap, else 8)")
parser.add_argument("--epochs", type=int, default=1, help="test-time training epochs")
parser.add_argument("--lr", default="5e-5", help="test-time training learning rate")
parser.add_argument("--second-pass", action="store_true", help="v15: idle GPUs retrain weak tasks with a second adapter")
parser.add_argument("--second-model", action="store_true", help="v16: the second pass uses the 2B NVARC model")
parser.add_argument("--nvarc", action="store_true", help="v17: NVARC-faithful mode (fixed 1200 s per task, file order, rule solver never takes attempt 2)")
parser.add_argument("--smoke",default="small", choices=["small", "realistic"], help="commit-run test mode")
parser.add_argument("--seed", type=int, default=0, help="realistic smoke: which random task sample")
parser.add_argument("--out", default="arc2026_nvarc_plus.ipynb")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "pipeline")


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def writefile(name, path):
    return code(f"%%writefile {name}\n" + open(path).read())


patch = open(os.path.join(P, "unsloth_qwen3_flash.patch"), "rb").read()

cells = [
    md("# ARC Prize 2026 · ARC-AGI-2 — NVARC test-time training + rule solver\n\n"
       "Built on the open-source ARC Prize 2025 winner [NVARC](https://github.com/1ytic/NVARC) "
       "(Qwen3-4B fine-tuned on synthetic puzzles, per-task LoRA test-time training, DFS sampling, "
       "augmentation scoring). Changes for 2026 (v17: optional NVARC-faithful mode for A/B tests; v16: optional second pass on NVARC's separately trained 2B model; v15: optional second pass, where GPUs left idle after the queue retrain finished tasks with a weak answer on a new example order and colors and decode 16 new views, votes pooled over both adapters; v14: training epochs and learning rate are settings, and 4 copies is a training size; v13: optional extra views in a task's leftover time, off by default; v12: optional 'auto' training size, 16 copies only where they take at most a quarter of the task's budget; commit runs report whether answers are lost in generation or selection; v11: a GPU worker that crashes restarts instead of stopping all four; v10: at most 8 training copies, looser second decode for test outputs with no guess; v7: time-aware test-time training that shrinks or skips training when it would not fit, decoding guaranteed after training, GPU-side scoring patch, verified rule programs take attempt 2, rule solver runs alongside the GPUs):\n\n"
       "* **Dynamic per-task time budget.** The hidden set has 240 tasks (2025: 120); NVARC's fixed 1200 s "
       "cap would leave the tail of the queue unsolved. Remaining time is split over remaining tasks.\n"
       "* **Smallest tasks first**, so time saved on easy tasks flows to hard ones.\n"
       "* **Safety submission first.** A rule-solver `submission.json` is written before the GPU stage, "
       "so a crash or timeout still scores.\n"
       "* **Rule solver takes attempt 2** when it finds a program that reproduces every training pair (97% precise on public training tasks).\n"
       "* Commit runs (not competition reruns) run a short smoke test on public eval tasks and print a local score.\n\n"
       "Inputs: competition `arc-prize-2026-arc-agi-2`, model `sorokin/qwen3_4b_grids15_sft139` "
       "(Transformers, bfloat16, v1). Accelerator: GPU L4 x4. Internet: off."),
    code("import time\nglobal_start = time.time()\n"
         "# Kaggle stops at 12 h. Training no longer runs past the deadline (v7), so 25 min covers\n"
         "# the final merge and runtime variance (NVARC itself kept 10 min)\n"
         "global_end_time = global_start + 12 * 3600 - 25 * 60"),
    code(f"TOP_COPIES = {args.top!r}  # most test-time training copies per task (time-aware: may use fewer)\n"
         f"TTT_EPOCHS = {args.epochs}  # test-time training epochs\n"
         f"TTT_LR = {args.lr!r}  # test-time training learning rate\n"
         f"SECOND_PASS = {args.second_pass}  # idle GPUs retrain weak tasks with a second adapter\n"
         f"SECOND_MODEL = {args.second_model}  # second pass on sorokin/qwen3_2b_grids15_sft141 (if attached)\n"
         f"NVARC_MODE = {args.nvarc}  # fixed 1200 s per task, input file order, rule solver only fills empty slots\n"
         f"SMOKE_MODE = {args.smoke!r}  # commit runs only: 'small' (8 smallest, 15 min) or 'realistic'\n"
         f"SMOKE_SEED = {args.seed}  # realistic mode: which random sample of public eval tasks\n"
         + r'''import glob, json, os, subprocess

RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None

# prefer the 2026 competition mount even if an older ARC Prize input is still attached
comp_dir = os.path.dirname(find_one("/kaggle/input/*2026*arc-agi-2*/**/arc-agi_test_challenges.json")
                           or find_one("/kaggle/input/**/arc-agi_test_challenges.json"))
# the 4B model by name: with the 2B one also attached, "*grids15*" would sort the 2B first
model_cfg = find_one("/kaggle/input/**/*4b_grids15*/**/config.json") or find_one("/kaggle/input/**/*grids15*/**/config.json")
assert comp_dir and model_cfg, (comp_dir, model_cfg)
model_dir = os.path.dirname(model_cfg)

os.environ["ARC_MODEL_PATH"] = model_dir
second_cfg = find_one("/kaggle/input/**/*2b_grids15*/**/config.json")
if SECOND_PASS and SECOND_MODEL and second_cfg:
    os.environ["ARC_SECOND_MODEL_PATH"] = os.path.dirname(second_cfg)
print("second pass:", SECOND_PASS, "| second model:", os.getenv("ARC_SECOND_MODEL_PATH"))
if RERUN:
    os.environ["ARC_TEST_PATH"] = f"{comp_dir}/arc-agi_test_challenges.json"
else:
    # commit run: smoke test on a few public eval tasks that have solutions
    os.environ["ARC_TEST_PATH"] = f"{comp_dir}/arc-agi_evaluation_challenges.json"
    os.environ["ARC_SOLUTIONS_PATH"] = f"{comp_dir}/arc-agi_evaluation_solutions.json"
    os.environ["ARC_SMOKE_MODE"] = SMOKE_MODE
    if SMOKE_MODE == "realistic":
        # random mixed-size eval tasks on the hidden run's clock per task: 240 tasks share
        # 12 h - 25 min, of which ~8 min is setup, so ~166 s of wall time per task on 4 GPUs
        os.environ["ARC_SMOKE_TASKS"] = os.getenv("ARC_SMOKE_TASKS", "24")
        per_task = (12 * 3600 - 25 * 60 - 8 * 60) / 240
        global_end_time = global_start + 8 * 60 + int(os.environ["ARC_SMOKE_TASKS"]) * per_task
    else:
        os.environ["ARC_SMOKE_TASKS"] = os.getenv("ARC_SMOKE_TASKS", "8")
        # a short clock forces the time-aware training paths (8 copies, base-model decode) to run
        global_end_time = min(global_end_time, global_start + 15 * 60)
os.environ["ARC_TOP_COPIES"] = TOP_COPIES
os.environ["ARC_TTT_EPOCHS"] = str(TTT_EPOCHS)
os.environ["ARC_TTT_LR"] = TTT_LR
os.environ["ARC_SECOND_PASS"] = "1" if SECOND_PASS else "0"
os.environ["ARC_FIXED_BUDGET"] = "1" if NVARC_MODE else "0"
os.environ["ARC_TASK_ORDER"] = "file" if NVARC_MODE else "size"
os.environ["ARC_RULE_SECOND"] = "0" if NVARC_MODE else "1"
os.environ["ARC_LLM_DIR"] = "../inference_outputs"
os.environ["ARC_SUBMISSION_PATH"] = "/kaggle/working/submission.json"
print("rerun:", RERUN, "| model:", model_dir, "| test:", os.environ["ARC_TEST_PATH"])'''),
    code("!pip uninstall -y tensorflow > /dev/null 2>&1"),
    code("# NVARC's flash-attention patch for batched decoding in unsloth's Qwen3 fast path (idempotent)\n"
         f"patch = {patch!r}\n"
         r'''import importlib.util
# locate unsloth without importing it (importing would grab a CUDA context in this kernel)
target = os.path.join(os.path.dirname(importlib.util.find_spec("unsloth").origin), "models", "qwen3.py")
if "flash_attn_func(Qnn, Knn, Vnn)" in open(target).read():
    print("unsloth qwen3 patch already applied")
else:
    open("qwen3.patch", "wb").write(patch)
    print(subprocess.run(["patch", "--binary", target, "qwen3.patch"], capture_output=True, text=True).stdout)'''),
    writefile("arc_solver.py", os.path.join(HERE, "arc_solver.py")),
    writefile("arc_loader.py", os.path.join(P, "arc_loader.py")),
    writefile("arc_decoder.py", os.path.join(P, "arc_decoder.py")),
    writefile("llm_worker.py", os.path.join(P, "llm_worker.py")),
    writefile("starter.py", os.path.join(P, "starter.py")),
    writefile("merge.py", os.path.join(P, "merge.py")),
    code(r'''# smoke-test subset for commit runs: the N smallest public eval tasks, or N random ones
if not RERUN:
    import random
    from starter import task_size
    data = json.load(open(os.environ["ARC_TEST_PATH"]))
    n = int(os.environ["ARC_SMOKE_TASKS"])
    if os.environ["ARC_SMOKE_MODE"] == "realistic":
        keys = random.Random(SMOKE_SEED).sample(sorted(data), n)
    else:
        keys = sorted(data, key=lambda k: task_size(data[k]))[:n]
    sub_path = "smoke_challenges.json"
    json.dump({k: data[k] for k in keys}, open(sub_path, "w"))
    os.environ["ARC_TEST_PATH"] = os.path.abspath(sub_path)
    print("smoke tasks:", keys)'''),
    code("# 1) rule solver on CPU, in the background at low priority so the GPUs start at once\n"
         "# (it took up to ~20 min of idle GPU time when it ran first)\n"
         "import sys\n"
         "sym_proc = subprocess.Popen([\"nice\", \"-n\", \"19\", sys.executable, \"merge.py\", \"symbolic\"],\n"
         "                            stdout=open(\"symbolic.log\", \"w\"), stderr=subprocess.STDOUT)"),
    code("# 2) NVARC test-time training + DFS decoding on all GPUs\n"
         "!UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 "
         "python starter.py --end-time {global_end_time}"),
    code("# 3) merge LLM guesses with the rule solver's programs; the rule solver must be done first,\n"
         "# or its own submission.json would overwrite this one\n"
         "try:\n"
         "    sym_proc.wait(timeout=300)\n"
         "except subprocess.TimeoutExpired:\n"
         "    sym_proc.kill(); sym_proc.wait()\n"
         "    print(\"rule solver stopped unfinished; merging without it\")\n"
         "print(open(\"symbolic.log\").read()[-500:])\n"
         "!python merge.py final"),
    code("!ls -la /kaggle/working/ && rm -f ../worker*"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        # no "kaggle" block: accelerator, internet and environment come from the
        # forked notebook's settings (an L4 request here made draft saves fail)
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = os.path.join(HERE, args.out)
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, len(cells), "cells")
