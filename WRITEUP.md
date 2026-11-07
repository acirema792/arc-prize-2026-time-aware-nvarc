# Measuring where test-time training stops: a controlled study of NVARC on ARC-AGI-2 under a fixed compute budget

*ARC Prize 2026 · ARC-AGI-2 · Solution Writeup · Kaggle notebook `oscarariza3/arc2-qwen3-unsloth-flash-lora-batch4-queue`*

## Summary
We took NVARC, the ARC Prize 2025 winner (Qwen3-4B fine-tuned on synthetic puzzles, per-task LoRA test-time training, depth-first sampling, augmentation-scored answer selection), and ran **eleven loops of controlled experiments (31 Kaggle runs and 6 hidden submissions)** to find out which part of the system limits its score under the 2026 budget: 240 hidden tasks in 12 hours on 4×L4.

Best hidden public score: **30.56** (v7, and again with v17).

Contributions:
1. **A robust, time-aware pipeline.** It decides per task how much to train from measured GPU speed, so no task trains past its deadline. Every trained task is decoded. A crashed GPU worker restarts instead of stopping all four.
2. **A measurement method** that separates the three places an ARC answer can be lost:
   - **Generation:** the model never proposes the right grid.
   - **Selection:** the right grid is proposed but not ranked in the top two.
   - **Time:** the task never finishes.
   Every run reports all three, and any extension also reports the same run's score without it, so effects are measured on identical first-pass results.
3. **A decomposition of where the score comes from.** On a representative 24-task sample:
   - The right answer is generated for **13 of 24** tasks (54%), and selection keeps **11.5** of them (88% of what is found).
   - No test-time change we tried generated a single new answer: more training, less training, more views, looser decoding, a second adapter, or other selection rules.
   - **The remaining gap is almost entirely generation, and generation is fixed by the base model.**
4. **A documented validation-leakage trap.** Adding NVARC's 2B model raised local scores by 7 tasks of 24, but that model was trained on the public evaluation set. On the hidden set the same change *lost* 1.67 points. We show how to detect this before submitting.

The practical conclusion for anyone aiming at 85%: on this hardware, test-time compute around a fixed 4B generator is saturated near 30%. Progress has to come from the generator itself (pre-training data, model size or search over programs), not from more sampling or more test-time training.

## Final results (hidden set, public leaderboard)
| Version | Configuration | Hidden score |
|---|---|---|
| v1 | NVARC, 16 copies, dynamic per-task budget, smallest tasks first | 30.14 |
| v3 | v1 with 24 copies | 25.00 |
| **v7** | time-aware training (max 16), decode guarantee, GPU-side scoring, background rule solver | **30.56** |
| v12 | max 8 copies, crash isolation, looser fallback decode | 28.89 |
| v16 | v12 + second pass with NVARC's 2B model in idle GPU time | 27.22 |
| **v17** | NVARC-faithful settings (16 copies, fixed 1200 s per task, file order, rule solver only fills empty slots) + crash isolation + GPU-side scoring | **30.56** (scored after the deadline) |

Public reproductions of plain NVARC score 32.22 on the same leaderboard. Our NVARC-faithful v17 scored 30.56, the same as v7, so the remaining difference is not explained by the time budget, task order or rule solver; we report it as unexplained rather than guess.

## Method
The pipeline is NVARC's, with the following changes (all in `pipeline/`, assembled into one Kaggle notebook by `build_notebook.py`):

1. **Time-aware test-time training** (`llm_worker.plan_training`).
   - Each GPU worker measures its own training speed in seconds per training token, keeping the slower of its last two measurements.
   - Before each task it estimates the cost of training on 24, 16, 8 or 4 augmented copies and picks the largest that leaves at least 4 minutes of decoding before both the task's deadline and the run's.
   - Training is the one step that cannot be interrupted. This removes the failure where a large task near the end trains past the deadline and returns nothing.
2. **Dynamic per-task budget, smallest tasks first** (`task_budget`, `starter.py`). Remaining time is split over remaining tasks, so time saved on easy tasks flows to hard ones. An `--nvarc` switch restores NVARC's fixed 1200 s per task and file order.
3. **Guaranteed decoding.** Every trained task is decoded, even if its training overran its budget.
4. **GPU-side scoring.** DFS and augmentation scoring compute the 12 ARC-token log-probabilities on the GPU and transfer only those. Results are identical and run time about 20% lower. Adapted from mikelou1's public perf patch (Apache 2.0).
5. **Crash isolation** (`starter.local_worker`).
   - Before: an exception in one worker made `torch.multiprocessing.spawn` stop all four GPUs, leaving every queued task unsolved. We hit this in a local test: 2.00/24 instead of 6.50/24.
   - Now: a crashed worker logs the error, frees its GPU, reloads the model and keeps draining the queue. Only the task in progress is lost.
6. **CPU work off the critical path.**
   - A rule-based program search runs at low priority alongside the GPUs: 13 grid-operation generators plus composition, verified against every training pair.
   - It writes a safety `submission.json` first, so a crash still scores.
7. **In-run diagnostics** (`merge.diagnose`, commit runs only). Each run reports:
   - for how many tasks the right answer appears among the candidates;
   - what each of six selection rules would score;
   - for any second pass, the same run's score without that pass's candidates.
8. **Optional extensions, kept as switches for reproducibility** (all measured, none adopted):
   - extra views in leftover time;
   - a second adapter trained on a new example order and colors in idle GPU time;
   - a second base model in that pass;
   - other training intensities.

## Why it works, and where it stops
Each claim below is a measurement on the same 24 tasks unless stated otherwise.

**Answers are lost to generation, not selection.** The right grid appears among the candidates for 13.0 of 24 tasks, and kgmon (count of votes minus mean augmentation negative log-likelihood) ranks it in the top two for 11.5. Across every run not affected by the leakage below, no alternative selection rule beat kgmon on more than one run:

| Rule | Result against kgmon |
|---|---|
| Votes only (4 runs) | +1.33, 0, 0, −1.0 |
| Augmentation score only (4 runs) | 0, −1.0, 0, 0 |
| Half vote weight (6 runs) | tie every time |
| Double vote weight (6 runs) | one −1.0, otherwise tie |
| probmul_3 (12 runs) | one +1.0, one −1.0, one −0.5, otherwise tie (net −0.5) |

**Test-time training sharpens; it does not discover.** On the 24 tasks:

| Training setting | Tasks with the answer among candidates |
|---|---|
| 4 copies | 12.0 |
| 8 copies | 13.0 |
| 8 copies × 2 epochs | 13.0 |
| 8 copies at twice NVARC's learning rate | 11.5 |

Training moves probability mass toward answers the base model can already produce. Below a threshold it misses some; above it, it only costs time (2 epochs caused 3 timeouts and lost 1.5 tasks it had found).

**Sampling is saturated at 16 augmented views.** Three extensions together produced several hundred extra candidates and **zero** new correct answers:
- a 10× looser DFS cutoff for outputs with no guess (0 of about 20);
- 16 more views per task in leftover time (11.0 vs 11.5);
- a second adapter trained on a new example order and new colors, on 17 weak tasks (paired: 11.50 → 11.50 and 8.50 → 8.50).

The answers this 4B model cannot reach in 16 views it does not reach in 32.

**So the binding constraint is the generator.** About half the tasks never produce a correct candidate. Neither more test-time compute nor better ranking can recover them; only a stronger prior can. This is consistent with the leaderboard: the leading 2026 entries reportedly use much larger reasoning models in program-writing loops, not more NVARC sampling.

**Why time-awareness matters, and why its gains are modest.** The hidden run sits close to its clock: 24 copies scored 25.00 against 30.14 for 16, because the queue tail went unsolved. Time-aware training removes the worst case (a task that trains past the deadline) and let v7 run 16 copies safely; it does not add answers.

## Evaluation method
- **Hidden-clock local test.** 24 random public evaluation tasks, run under the hidden set's time per task (about 166 s of wall time on 4 GPUs). Re-running a configuration on the same sample reproduces its score exactly (11.50 twice); different samples vary by about ±1 task. So we compare configurations on the same sample, and require a gain on at least two samples.
- **Paired in-run comparisons.** Extensions that add candidates are scored with and without their candidates inside the same run, which removes run-to-run noise.
- **Leakage check.** No model or data used may have been trained on the validation tasks (below).
- **Where local and hidden disagree.**
  - On six local samples, 8 copies scored at least as well as 16. On the hidden set, 16 copies led: 30.14 and 30.56 against 28.89.
  - The local set uses public evaluation tasks, whose difficulty and size distribution may differ from the hidden set's.
  - We therefore used the hidden set for the final configuration and the local test for mechanisms (generation vs selection), where it is consistent.

## Validation leakage: a trap worth documenting
NVARC's author also published a 2B model. Used as the generator in our idle-GPU second pass, it raised the local score from 11.50 to **18.50** and from 9.50 to **14.00** within the same runs.

Three signs pointed to memorization:
1. NVARC's write-up describes a training subset that uses "both training and evaluation puzzles from ARC-AGI-2", which is exactly our validation set.
2. A model that scored lower than the 4B on the 2025 hidden set (22.50 vs 27.22) suddenly solved 7 of the 4B model's 13 weak tasks.
3. Its added answers won selection with **1 vote in 20**, because its augmentation scores were near-certain.

We flagged it before submitting, then measured it: on the hidden set the configuration scored **27.22 against 28.89** without the 2B pass. The general rule: check every model and dataset for overlap with the validation tasks before believing a local gain, and treat near-certain likelihoods on hard tasks as a warning sign.

## Universality
None of the contributions is specific to ARC:
- **Time-aware training:** measure the training throughput, then choose the training size that still leaves decoding time before the deadline. It applies to any per-instance fine-tuning under a wall-clock limit, such as test-time training for other reasoning benchmarks or per-user adaptation.
- **Crash isolation for multi-GPU worker pools:** contain one worker's failure instead of cancelling the whole job. It applies to any batch inference job.
- **The generation / selection / time decomposition** with in-run paired diagnostics: a general way to find the binding constraint of any sample-and-rank system.
- **The leakage signature** (high-confidence, low-vote answers that win selection) applies to any pipeline that adds a second model whose training data is not fully known.

## Progress toward 85%
Our measurements show what *not* to spend effort on with a 4B NVARC-style generator under this budget: more test-time training, more sampling, a second adapter, re-ranking. Each is saturated. They point at the generator:
- **Better pre-training data** for the base model: the synthetic puzzles that taught it the priors it has.
- **Larger or reasoning models** that can produce programs rather than grids.
- **Search over programs** whose proposals can be verified on the training pairs.

Our rule solver is 97% precise on public training tasks (a program that reproduces every training pair is almost always right), but it finds such a program for 0 of 120 evaluation tasks. Hand-written program families do not reach ARC-AGI-2 difficulty; learned program proposal is needed.

## Novelty
- **Time-aware test-time training.** Choosing the per-task LoRA training size from measured throughput and the remaining clock. NVARC uses a fixed size and a fixed cap. To our knowledge no public ARC solution plans training against the deadline.
- **In-run paired diagnostics** for additive inference passes, and the resulting quantitative decomposition of NVARC's losses into generation, selection and time.
- **A measured negative result for a second adapter in idle compute,** and a measured, documented case of validation leakage through a released checkpoint, with its detection signature.

## Negative results
- **Agentic program synthesis with Qwen3.8-27B** (vLLM 0.19, tensor parallel over 4 GPUs, Python sandbox).
  - On 4×L4, each reasoning turn took 3–5 minutes. With a 25-minute cap per task the agent answered 2 of 15 public evaluation tasks, and 0 of 11 with 32 concurrent streams.
  - We found no way to serve the stronger Qwen3.8-Flash-Next on L4 with the offline vLLM build.
- **Heavier training:** 24 copies lost 5 points on the hidden set.
- **Other training intensities:** 4 copies, 2 epochs and twice the learning rate all scored lower locally (see above).
- **More sampling:** looser cutoff, extra views and a second adapter added no answers (see above).
- **Selection rules:** no alternative beat kgmon on more than one sample.
- **Second base model:** a validation-leakage artifact; lower on the hidden set (see above).
- **NVARC-faithful settings:** a fixed per-task budget and file order instead of our dynamic budget and smallest-first order tied our best hidden score (30.56), and did not reach the 32.22 of public NVARC copies.

## Reproducibility
- **Code:** `pipeline/` (worker, decoder, loader, merge, starter), `arc_solver.py`, and `build_notebook.py`, which writes the Kaggle notebook.
- **Configurations:** every one in this write-up is a `build_notebook.py` flag combination:

| Configuration | Flags |
|---|---|
| v7-style | `--top 16` |
| v12 | `--top 8` |
| v17 | `--top 16 --nvarc` |
| second adapter | `--second-pass` |
| 2B pass | `--second-pass --second-model` |
| local test | `--smoke realistic --seed N` |

- **Kaggle inputs:** the competition data, model `sorokin/qwen3_4b_grids15_sft139` (Apache 2.0), and utility script `pip-install-unsloth-flash-patch`. The 2B pass also needs `sorokin/qwen3_2b_grids15_sft141`.
- **Experiment log:** `EXPERIMENTS.md` (every run, with its score, timeouts and diagnostics) and `docs/loop-run-log.md`.
- **Tests:** `tests/test_pipeline.py` checks the merge, time budget, second pass and diagnostics on CPU.
- **Method of iteration:** `docs/LOOP.md`. Each loop had a reject-by-default checker requiring a gain on two samples and clean validation.
- **License:** Apache 2.0.
- **Credits:** NVARC (Ivan Sorokin et al.) for the base solution and models; mikelou1 for the GPU scoring patch; the ARC Prize team for the benchmark.
