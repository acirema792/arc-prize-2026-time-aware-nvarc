# Experiment log

Every local test run and hidden submission, in order, with its configuration, score, run time, timeouts and diagnostics, and the reading drawn from it at the time. Kaggle "version" numbers refer to versions of the notebook `oscarariza3/arc2-qwen3-unsloth-flash-lora-batch4-queue`. "Seed N" is the random sample of 24 public evaluation tasks used by the realistic local test (`build_notebook.py --smoke realistic --seed N`).

Hidden public scores: v1 30.14, v3 25.00, v7 30.56, v12 28.89, v16 27.22, v17 30.56 (scored after the deadline).

## Realistic test loops (24 random public eval tasks, seed 0, clock scaled to the hidden run: ~166 s wall per task on 4 GPUs, 77 min total)
Build: `python3 build_notebook.py --top <8|16> --smoke realistic --out <file>` (hidden runs are unaffected by `--smoke`).
| Loop | Version | Config | Score | Run time | Outputs with a model guess |
|---|---|---|---|---|---|
| 1 | 11 (loopA top16) | v7 settings, max 16 copies | 6.00/24 = 25.0% | 4496 s of 4600 | 30 |
| 1 | 10 (loopA top8) | v9 settings, max 8 copies | **6.50/24 = 27.1%** | 3916 s | 25 |
| 2 | 12 (loopB v10) | v9 + looser second decode (2% cutoff, ≤120 s) for outputs with no guess | **crashed**: 2.00/24 | 627 s | 6 |
| 2b | 13 (loopC v11) | v10 with the fallback bug fixed + worker restart guard | **6.50/24 = 27.1%** | 3957 s | 30 (9 looser decodes, none correct) |
| 3 | 14 (loopD) | v11, max 8 copies, **seed 1** sample | **8.50/24 = 35.4%** | 4523 s | 33 (7 looser decodes) |
| 3 | 15 (loopD) | v11, max 16 copies, seed 1 sample | 6.00/24 = 25.0% | 4579 s of 4600 | 34 (8 task timeouts) |

Loop 1 reading: 16 copies only just fits the hidden-run clock (100 s to spare), consistent with 24 copies (v3) losing 5 points; 8 copies is at least as accurate and 13% faster, but leaves 5 outputs without a guess, which loop 2 targets.

Loop 2 reading: the looser decode batched views of different grid shapes (`subkeys[:4]` mixes rot90 views; prompt lengths 1637 vs 1640), raised in one worker, and `mp.spawn` then stopped all four GPUs at 627 s. Two fixes in v11: the fallback reuses the shape-matched batch (views 0, 1, 4, 5; checked offline on all 167 public eval test outputs: 0 mismatched, versus 104 for the old batch), and `starter.local_worker` restarts a crashed worker instead of letting one exception end the run. The second fix also protects v1–v9-style runs, which had no guard: any exception on one hidden task (for example out-of-memory) would have lost every task still queued. Versions 9/10 do not contain the fallback, so they were never exposed to this bug.

Loop 3 reading: over both samples (48 tasks), max 8 copies scores 15.0 and max 16 copies 12.0; 8 copies wins on both samples. With 16 copies, 8 of 24 tasks hit their per-task timeout and the run used 4579 of 4600 s, so on the hidden set (same clock per task) heavier training costs answers, as v3 (24 copies, 25.00) already showed. **Decision: submit v11 = Kaggle version 13 (max 8 copies, crash guard, fixed fallback).** Version 14 is the same code with a different commit-run sample and behaves identically on the hidden run.

Loop 3 time profile (version 14 log): training took 40–190 s per task (410 s for the largest) out of 150–1430 s, so decoding and augmentation scoring, not training, use most of each task's time. Most tasks finished well under their budget, and one GPU was idle for the last ~1000 s while the largest task ran. So 16 copies is affordable on small tasks, and only the large ones need 8. Leaderboard check (Oct 30): top 88.06, 30th 66.67; the best public notebooks are still NVARC copies at 31.4–33.9, and the newest one (Boyko NVARC+ V22: 64 augmentations, two pooled passes, validated on 4 tasks) scores 32.64, which is no better.

Loop 4 (v12, **seed 2** sample, both started 13:55 local): version 16 = max 8 copies (v11 behaviour) and version 17 = `--top auto` (16 copies only when their estimated training takes ≤ 25% of the task's budget, else 8). Both print a new diagnostic: how many test outputs have the answer anywhere among the candidates (lost in generation vs in selection), how many come only from the looser pass, and the score of each selection rule (kgmon vs probmul_3).

| Loop | Version | Config | Score | Run time | Answer among candidates | kgmon / probmul_3 |
|---|---|---|---|---|---|---|
| 4 | 16 (loopE) | v12, max 8 copies, seed 2 | **11.50/24 = 47.9%** | 3509 s of 4600 | 13.0 | 11.5 / 11.5 |
| 4 | 17 (loopE) | v12 auto (16 copies on 13 tasks, 8 on 11), seed 2 | 11.00/24 = 45.8% | 4026 s | 12.5 | 11.0 / 10.0 |

Loop 4 reading: 8 copies wins a third time (26.5/72 vs 23.0/72 for 16 or auto over three samples) and is 13% faster, so extra training does not pay even where it is cheap. Generation, not selection, is the bottleneck: the answer never appears among the candidates for 11 of 24 tasks, and kgmon loses only 1.5 of the 13 it could get (probmul_3 is no better, so kgmon stays). The looser pass is 0 for about 20 across loops 2b–4. With 8 copies the GPUs finish ~1100 s (24%) early.

Loop 5 (v13, version 18, seed 2, started 15:50 local): v12 max 8 + **extra views**. After the usual 16 views (and the fallback), a task with ≥ 60 s of budget left decodes 16 more views (`augment(n=2, seed=3)`: new color permutations and example orders, same batch shapes; checked offline on all 167 public eval test outputs: 0 mismatched batches, 0 view names shared with the first pass). Wrapped in try/except so it can never cost the task. GPU quota at start: 27:40 of 30 h used (weekly reset Saturday).

| 5 | 18 (loopF) | v13 = v12 max 8 + extra views, seed 2 | 11.00/24 = 45.8% | 4592 s of 4600 (3 timeouts) | 12.5 | 11.0 / 11.0 |

Loop 5 reading: extra views ran on 20 tasks (94 batches, no errors), but scored 0.5 lower than version 16 on the same tasks: one new answer (b99e7126), two lost (62593bfd, dfadab01, each found by a single view before), and the run used the whole clock. More views of the same model do not find new answers, which matches the looser pass (0/20). Extra views are now off by default (`ARC_EXTRA_VIEWS=1` turns them on).

**Decision (Oct 30, 17:15 local): submit version 16 (v12, max 8 copies) in the Oct 31 UTC slot.** On a hidden rerun it behaves exactly like version 13 (v11); the only differences are a candidate tag and the commit-run diagnostic. Five realistic samples now agree: max 8 copies is the best configuration of this pipeline, and further gains need a stronger generator (model or training data), not more compute per task.

Submitted (Oct 31 01:20 UTC): version 16 ("loop4 v12 top8 seed2"). v7 finished at **30.56** (best so far; v1 30.14, v3 25.00), rank 1246, level with the best public NVARC notebooks. GPU quota reset to 0 of 30 h on Oct 31 UTC. Kaggle runs at most two GPU commits at a time.

Grand Prize note (from the overview): the $275k Grand Prize goes to the best **Solution Writeup**, scored 0–5 on accuracy, universality, progress, theory (why it works), completeness and novelty, with all artifacts open-sourced and attached within 7 days of Nov 2 (by Nov 9). The write-up deserves as much care as the score.

Loop 6 (v14, seed 2, paired with version 16's 11.50): **training intensity**. 8 copies beat 16 and 24, so map the other directions. v14 makes epochs (`ARC_TTT_EPOCHS`) and learning rate (`ARC_TTT_LR`) settings and adds 4 copies as a training size (a task that cannot fit 8 now trains on 4 instead of none).

| Loop | Version | Config | Score | Run time | Answer among candidates | kgmon / probmul_3 |
|---|---|---|---|---|---|---|
| 6 | 19 (loopG, id 364075814) | max 4 copies | 9.50/24 = 39.6% | 3081 s of 4600 | 12.0 | 9.5 / 10.5 |
| 6 | 20 (loopG, id 364076086) | max 8 copies, 2 epochs | 10.00/24 = 41.7% | 4276 s (3 timeouts) | 13.0 | 10.0 / 10.0 |
| 6 | 21 (loopG) | max 8 copies, lr 1e-4 (NVARC 5e-5) | 11.00/24 = 45.8% | 3712 s (2 timeouts) | 11.5 | 11.0 / 11.0 |

Loop 6 reading (19, 20): version 16's setting (8 copies, 1 epoch, 5e-5) is a peak on the copies axis and on the epochs axis. 4 copies finds fewer answers (12.0 among candidates vs 13.0) and frees time that more decoding cannot use (loop 5). 2 epochs finds the same 13.0 but costs 22% more time and 3 timeouts, so 1.5 fewer are kept. Test-time training sharpens the model just enough at 8×1; past that it costs time without new answers, and below it the model misses answers.

Version 21 (lr 1e-4) closes the loop: 11.0 kept, 11.5 among candidates (vs 13.0), 2 timeouts (62593bfd, e3721c99). A higher learning rate does not add answers; it loses some, which fits overfitting a handful of training pairs. All three directions away from 8 copies × 1 epoch × 5e-5 score lower, so the v12 setting (submitted as version 16) stays the default. The gap of 0.5 is within the ±1 run-to-run noise, so this is "no gain", not proof of harm.

Loop 7 (v15, **second pass**): where the remaining answers are lost. Version 21's log shows the last 8 tasks all at the 1200 s cap (2 timeouts), then each GPU idle for 800–1350 s before the clock ends; every realistic run so far left ~24% of the GPU time unused. Loop 5 showed more views from the *same* adapter add nothing, so v15 varies the adapter: once the queue is empty, an idle GPU claims a finished task whose weakest test output has a top answer with fewer than 8 votes (smallest first, a claim file per task so GPUs never duplicate), trains a fresh LoRA on 8 copies with a new example order and colors (training seed 7 instead of 1), decodes 16 new views (seed 3), and writes its candidates next to the first pass's, so kgmon votes over both adapters. The first pass is unchanged. The commit-run diagnostic now also prints **the same run's score without the second-pass candidates**, which isolates the effect with no run-to-run noise, plus kgmon with half and double vote weight. Tested locally on fake outputs over 4 real eval tasks (vote counting, weak-task order, claims, paired score). Build: `python3 build_notebook.py --second-pass --smoke realistic --seed N`.

| Loop | Version | Config | Score | Run time | Answer among candidates | First pass only | kgmon / probmul_3 / ½ / 2× |
|---|---|---|---|---|---|---|---|
| 7 | 22 (loopH, id 364314548) | v15 second pass, seed 2 (version 16: 11.50) | 11.50/24 = 47.9% | 4619 s (1 first-pass timeout; second passes on 11 tasks, 4 skipped as too big) | 13.0 | 11.50 (+0) | 11.5 / 11.0 / 11.5 / 10.5 |
| 7 | 23 (loopH) | v15 second pass, seed 1 (version 14: 8.50) | 8.50/24 = 35.4% | 4638 s (second passes on 6 tasks from 3689 s) | 9.5 | 8.50 (+0) | 8.5 / 8.5 / 8.5 / 8.5 |

Loop 7 reading: the second adapter added 178 + 66 candidates on 17 weak tasks and changed nothing: no new answer among the candidates (13.0, as in version 16), and no first-pass answer lost. A second LoRA trained the same way on the same base model only re-finds what the first found, which confirms loop 5 from a different angle: coverage is set by the base model, not by the adapter's training order, colors or views. The selection variants agree with loop 4: kgmon is best or tied (double vote weight loses 1.0 on seed 2). An external harness (Actualjudgepresident/arc2026-nvarc-harness, 120 eval tasks) saw the same pattern at scale: pooling two full NVARC runs raised coverage 36.3 → 40.4% but top-2 only 28.5 → 29.4%.

Next (loop 8): the second pass is the right slot for a *different generator*. NVARC's author also published `sorokin/qwen3_2b_grids15_sft141` (Apache 2.0, same special tokens 13/14/15, 2.8 GB), trained separately, so its errors should differ from the 4B model's.

Loop 8 (v16, **second model**): the v15 second pass, but before its first job each idle GPU frees the 4B model and loads `qwen3_2b_grids15_sft141` (same LoRA settings, 8 copies, training speed re-measured); if loading fails the GPU stops instead of crash-looping. The notebook now picks the 4B model by name (`*4b_grids15*`): with both models attached, the old `*grids15*` lookup would have sorted the 2B model first and made it the main model. Both models are attached to the Kaggle notebook from version 24 on; hidden behaviour without `--second-model` is unchanged. Build: `python3 build_notebook.py --second-pass --second-model --smoke realistic --seed N`.

| Loop | Version | Config | Score | Run time | Answer among candidates | First pass only | kgmon / probmul_3 / ½ / 2× |
|---|---|---|---|---|---|---|---|
| 8 | 24 (loopI, id 364367002) | v16 2B second pass, seed 2 | 18.50/24 = 77.1% (**contaminated**, see below) | 4651 s (2B loaded on all 4 GPUs, 13 second passes, 2 end timeouts) | 19.5 | 11.50 (+7.0) | 18.5 / 18.5 / 18.5 / 18.5 |
| 8 | 25 (loopI) | v16 2B second pass, seed 1 | 14.00/24 = 58.3% (**contaminated**) | 4633 s (16 second passes, 11 skipped near the end) | 14.0 | 9.50 (+4.5) | 14.0 / 14.0 / 14.0 / 14.0 |

Loop 8 reading: **these local gains cannot be trusted.** NVARC's write-up describes a "full" training subset that "uses both training and evaluation puzzles from ARC-AGI-2", and on the 2025 hidden set the 2B model scored 22.50 against 27.22 for the 4B. The 4B model scores about the same on public eval (28.5% in the external harness) as on the hidden set, so it was not trained on the eval tasks; the 2B model behaves as if it was: a weaker model that solves 7 of 13 weak tasks the 4B missed, and every answer it adds wins selection even when it holds 1 of 12–20 votes, because its augmentation scores are near-certain (memorization's signature). Public eval is therefore no longer a valid test for anything involving the 2B model; only the hidden leaderboard can measure it. What the runs do show cleanly: the swap works (no load failures, no crashes, no first-pass slowdown), and the second pass finishes inside the clock.

Loop 9 (v17, **ours vs plain NVARC**, fresh seed 3). Public evidence (Kaggle code tab, Nov 1): plain NVARC reproductions score **32.22** on the public leaderboard (Reproduce NVARC 2025 Results; ARC AGI2 LB33.89 Minimal Perfpatch re-scored at 32.22, so identical code varies by about ±1.7 points), Boyko NVARC+ 32.64; our v1 = 30.14 and v7 = 30.56. Every earlier local test compared variants of our own pipeline, never our pipeline against NVARC itself. v17 adds an NVARC-faithful mode (`--nvarc`): NVARC's fixed 1200 s per task instead of the dynamic share, the input file's task order instead of smallest-first, and the rule solver only filling empty slots instead of taking attempt 2. The selection diagnostic also prints votes-only and augmentation-score-only.

| Loop | Version | Config | Score | Run time | Answer among candidates | Selection: kgmon / probmul_3 / ½ / 2× / votes / aug |
|---|---|---|---|---|---|---|
| 9 | 26 (loopJ, A) | v12 as submitted (max 8 copies), seed 3 | 8.33/24 = 34.7% | 3781 s (2 timeouts) | 9.67 | 8.33 / 8.33 / 8.33 / 8.33 / **9.67** / 8.33 |
| 9 | 27 (loopJ, B) | NVARC-faithful: 16 copies, fixed 1200 s, file order, no rule attempt 2, seed 3 | 7.67/24 = 32.0% | 4052 s (3 timeouts) | 9.00 | 7.67 / 7.67 / 7.67 / 7.67 / 7.67 / 6.67 |

Version 26 reading: **votes-only selection kept every answer the model found (9.67) while kgmon kept 8.33.** kgmon ranks by votes minus the mean augmentation NLL, so a wrong grid the adapter scores confidently can outrank the right grid with more votes. Evidence is mixed so far (double vote weight lost 1.0 on seed 2 in loop 7), so this needs another sample before it may change the submission (G3). Version 28 reruns v12 on seed 2 with the new diagnostics: it gives votes-only vs kgmon on 24 more tasks and a run-to-run noise check against version 16 (11.50).

| 9 | 28 (loopK) | v12 rerun, seed 2 (version 16: 11.50) | 11.50/24 = 47.9% | 3527 s (0 timeouts) | 12.5 | 11.50 on all six rules |

Version 27 reading: plain NVARC's settings do not beat ours on a sample neither was tuned on (7.67 vs 8.33; found 9.00 vs 9.67; 3 timeouts vs 2, and 7% slower). The public 32.22 vs our 30.56 is therefore most likely hidden-run variance (identical public code scored both 33.89 and 32.22), not a cost of our changes. Direction closed. Votes-only selection so far: +1.33 (version 26) and +0.00 (version 27), never below kgmon.

| 9 | 29 (loopK) | v12, seed 1 (version 14: 8.50), third selection sample | 7.50/24 = 31.3% | 4450 s (3 timeouts) | 9.50 | 7.50 / 7.50 / 7.50 / 7.50 / **6.50** / 7.50 |

**Hidden score, version 16 (v12, max 8 copies): 28.89** (Nov 1). Hidden so far: v1 (16 copies) 30.14, v7 (time-aware, max 16) 30.56, v12 (max 8) 28.89, v3 (24) 25.00. On the hidden set 16 copies leads 8 by 1.3–1.7 points, the opposite of five local samples (8 copies ≥ 16 every time); both gaps are within the hidden run-to-run spread (±1.7), so neither is settled, but the hidden evidence is what scores. v7 stays the best confirmed submission.

Version 28 reading: the same configuration on the same 24 tasks reproduced 11.50 exactly (version 16 locally: 11.50), so local noise on a fixed sample is small; the ±1 spread seen earlier comes from changing the sample. Votes-only selection: +1.33, +0, +0 so far.

Version 29 reading: votes-only selection lost 1.0 here. Over four samples it is +1.33, +0, +0, −1.0 (net +0.33 of ~96 tasks), so the checker rejects it (G3) and kgmon stays; selection direction closed after 3 loops. Seed 1 scored 7.50 against version 14's 8.50 on the same tasks, with 3 timeouts against 0, so on this sample timing noise costs about one task.

**Loop 9 conclusion:** on the local harness nothing beats v12 (NVARC-faithful −0.67, selection variants net +0.33 or worse). On the hidden set, 16 copies leads 8 by 1.3–1.7 points. The Nov 2 candidate is therefore built on 16 copies (v7's setting), with the 2B second pass only if version 24's hidden score beats version 16's 28.89 (same first pass, so the difference is the 2B pass).

Version 30 (**Nov 2 candidate**: v18 = 16 copies + 2B second pass; small smoke, 8 smallest tasks, 15 min): passed the crash check (G1, G2). 2B loaded on 3 GPUs, 5 second passes, no load failure, traceback or restart, 836 s. It trained 8 copies per task here because the small smoke's per-task budget is under 600 s; on the hidden run's budgets (≥ 664 s) the same rule allows 16, as in v7. Its 6.33/8 against 2.83 for the first pass alone is the same contamination as loop 8 (G4), not evidence. Submit it on Nov 2 only if version 24's hidden score beats version 16's 28.89.

Loop 10 (user asked for more smoke tests, Nov 01 07:20 UTC; quota 25:46 of 30 h before it): **version 31** = version 30's exact config (v18: 16 copies + 2B second pass) at the realistic clock on seed 2. Purpose: G2 at full scale (does 16 copies plus a second pass fit the hidden run's clock without extra timeouts?) and, through the first-pass-only line, a clean 16-copy result on current code to compare with v12's 11.50 on the same tasks. Its total score is contaminated by the 2B model (G4). After it, about 1.5–2 h of quota remain; Nov 2 needs no further commits (version 30 and v7's version exist).

| 10 | 31 (loopL, id 364464897) | v18: 16 copies + 2B second pass, realistic, seed 2 | 13.50/24 (contaminated, G4) | 4630 s, at the clock's edge (4 timeouts) | 15.0 | **first pass only 10.00** (v12, 8 copies: 11.50); 186 candidates on 7 tasks from 15 second passes |

Loop 10 reading: the Nov 2 candidate runs cleanly at full scale (G2) but uses the whole clock: 16 copies leave about half the idle time that 8 copies did (7 tasks got a second pass vs 13 in version 24). Its first pass alone scored **10.00 against 11.50 for 8 copies on the same 24 tasks**, the sixth local sample where 8 copies beats 16. On the hidden set 16 copies leads (30.14 and 30.56 vs one 8-copy draw of 28.89). The two sources still disagree, and both gaps are within their noise. Decision for Nov 2 waits on version 24 (8 copies + 2B): if it beats 28.89, the 2B pass helps, and 8 copies gives it twice the idle time, so resubmitting version 24's configuration is as defensible as version 30.

**Hidden score, version 24 (v16: v12 first pass + 2B second pass): 27.22** (2026-11-02 06:18 UTC). Same first pass as version 16 (28.89), so the 2B second pass cost 1.67 points on the hidden set (within the ±1.7 noise, but the wrong sign). Its +7.0 and +4.5 local gains were memorization, as the G4 check predicted. Direction closed; version 30 (16 copies + 2B) is not submitted.

Hidden scores so far: v1 (16 copies) 30.14, v3 (24) 25.00, **v7 (16, time-aware) 30.56**, v12 (8) 28.89, v16 (8 + 2B) 27.22. Every hidden result is consistent with "16 copies, nothing else added": the two 16-copy runs are the two best.

**Nov 2 recommendation:** submit Kaggle version 15 (loopD, v11 code with max 16 copies): v7's training setting plus the crash guard (a crashed GPU worker restarts instead of stopping all four) and the looser fallback decode. It is a second independent 16-copy draw with lower crash risk on the private set. Finals: v7 (30.56) and version 15 if it scores at least v1's 30.14; otherwise v7 and v1.

**Revised Nov 2 recommendation (after the user asked what else the other versions teach):** on the hidden set, plain NVARC with 16 copies scores 32.22, 32.22, 32.22 and 33.89 across public notebooks (mean ≈ 32.6), while our 16-copy runs v1 and v7 scored 30.14 and 30.56 (mean 30.35). The 2.3-point gap exceeds the 1.7-point spread between identical notebooks, so it is likely a real cost of what v1 and v7 share and NVARC lacks: the dynamic per-task budget and smallest-first order (v7 also lets the rule solver take attempt 2). None of these could be measured locally (the rule solver finds 0 programs on public eval, and local tests disagree with the hidden set on training settings). **Kaggle version 27** (v17 NVARC-faithful mode: 16 copies, fixed 1200 s per task, file order, rule solver only fills empty slots) reverts all three while keeping the crash guard, GPU-side scoring and deadline-safe training, so it is the closest available match to the 32.22 configuration. It replaces version 15 as the recommendation.
