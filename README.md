# Time-aware NVARC for ARC-AGI-2

Code and experiment record for our ARC Prize 2026 (ARC-AGI-2) entry: NVARC's Qwen3-4B test-time-training pipeline, made time-aware and crash-safe, plus the measurement harness behind 31 Kaggle runs and 6 hidden submissions. Best hidden public score: **30.56**.

Official Kaggle Solution Writeup: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/writeups/where-test-time-training-stops-time-aware-nvarc-o (same text as [`WRITEUP.md`](WRITEUP.md)). Public notebook: https://www.kaggle.com/code/oscarariza3/arc2-qwen3-unsloth-flash-lora-batch4-queue

Read [`WRITEUP.md`](WRITEUP.md) first. It explains what limits the score (the base model's generation, not training, sampling or selection) and documents a validation-leakage trap.

## Contents
| Path | What it is |
|---|---|
| `WRITEUP.md` | The solution writeup: method, theory, evaluation method, negative results |
| `EXPERIMENTS.md` | Every local test and hidden submission, with scores, timeouts and diagnostics |
| `pipeline/llm_worker.py` | Per-GPU worker: time-aware LoRA test-time training, DFS decoding, augmentation scoring, optional second pass |
| `pipeline/starter.py` | Spawns one worker per GPU over a shared task queue; restarts a crashed worker |
| `pipeline/merge.py` | Writes `submission.json`; on commit runs, prints the generation/selection diagnostics |
| `pipeline/arc_loader.py`, `arc_decoder.py` | NVARC's data formatting, augmentation and selection rules |
| `pipeline/unsloth_qwen3_flash.patch` | NVARC's flash-attention patch for batched decoding |
| `arc_solver.py` | Rule-based program search (13 generators plus composition), verified on every training pair |
| `build_notebook.py` | Assembles everything into one Kaggle notebook |
| `notebooks/` | The two final configurations, prebuilt |
| `evaluate.py`, `results/` | Local scorer for the rule solver and its per-task results on the public sets |
| `docs/LOOP.md`, `docs/loop-run-log.md` | The iteration method (gates, budget, human approvals) and its action log |
| `tests/test_pipeline.py` | CPU-only tests of the merge, time budget, second pass and diagnostics |

## Run on Kaggle
1. Fork NVARC's notebook `sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue` (**Copy & Edit**). This keeps its Docker image, which has unsloth and flash-attn installed; submissions run without internet.
2. Inputs:
   - the competition **ARC Prize 2026 - ARC-AGI-2**;
   - the model **`sorokin/qwen3_4b_grids15_sft139`** (Transformers, bfloat16, v1);
   - the utility script **`pip-install-unsloth-flash-patch`**.
3. Settings: accelerator **GPU L4 x4**, internet **off**.
4. **File → Import Notebook** with one of the notebooks in `notebooks/`:

| Notebook | Configuration | Hidden score |
|---|---|---|
| `arc2026_v17_nvarc_faithful.ipynb` | 16 copies, NVARC's fixed 1200 s per task and file order, rule solver only fills empty slots | 30.56 |
| `arc2026_top16.ipynb` | 16 copies, dynamic per-task budget, smallest tasks first | v7, the closest earlier version, also scored 30.56 |

5. **Save Version → Save & Run All**. A commit run tests on the 8 smallest public evaluation tasks and prints `*** local score`. A competition rerun processes the hidden set instead.

## Build other configurations
```bash
python3 build_notebook.py --top 16 --nvarc --out notebooks/arc2026_v17_nvarc_faithful.ipynb
```

| Flag | Effect |
|---|---|
| `--top 4/8/16/24/auto` | Most test-time training copies per task (time-aware: may use fewer) |
| `--epochs N`, `--lr X` | Test-time training epochs and learning rate (default 1, 5e-5) |
| `--nvarc` | NVARC's fixed per-task budget, file order, rule solver fills empty slots only |
| `--second-pass` | Idle GPUs retrain weak tasks with a second adapter (measured: no gain) |
| `--second-model` | That second pass uses `sorokin/qwen3_2b_grids15_sft141`. **Do not use on public eval:** that model was trained on it (see `WRITEUP.md`). |
| `--smoke realistic --seed N` | Commit runs test 24 random public eval tasks on the hidden run's clock |

## Test
```bash
PYTHONPATH=tests/stubs python3 tests/test_pipeline.py
```
This needs only numpy. The stub replaces `transformers`, which the tested code imports but does not use.

## License and credits
Apache License 2.0 (see `LICENSE` and `NOTICE`). The pipeline is derived from **NVARC** by Ivan Sorokin et al. (Apache 2.0), the ARC Prize 2025 winning solution. The GPU-side scoring is adapted from **mikelou1**'s public perf patch (Apache 2.0). ARC-AGI-2 is by the ARC Prize Foundation.
