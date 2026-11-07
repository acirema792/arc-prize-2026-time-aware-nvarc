# Loop run log (append-only)

| UTC | Action | Result |
|---|---|---|
| 2026-10-31 ~05:20 | Read version 21 (lr 1e-4, seed 2) | 11.00/24 vs 11.50 baseline; direction closed |
| 2026-10-31 ~18:05 | Committed versions 22, 23 (v15 second adapter, seeds 2 and 1) | running |
| 2026-10-31 ~22:00 | Read versions 22, 23 | 11.50 and 8.50, paired +0; direction closed |
| 2026-10-31 ~22:25 | Attached `sorokin/qwen3_2b_grids15_sft141`; committed versions 24, 25 (v16, 2B second pass) | running |
| 2026-10-31 23:55 | Read versions 24, 25 | 18.50 (paired 11.50) and 14.00 (paired 9.50); checker G4 fail: 2B trained on public eval |
| 2026-11-01 ~00:05 | **Submitted version 24** (user approved) | hidden run started |
| 2026-11-01 00:10 | Adopted loop-engineering structure: `LOOP.md`, `STATE.md`, this log | |
| 2026-11-01 00:40 | Read Kaggle code tab: plain NVARC reproductions 32.22, LB33.89 notebook re-scored 32.22 (±1.7 noise), ours 30.14/30.56 | new open direction: ours vs plain NVARC |
| 2026-11-01 00:46 | Built v17 NVARC-faithful mode (offline tests pass); committed versions 26 (A: v12) and 27 (B: NVARC-faithful), seed 3 | running |
| 2026-11-01 01:44 | Read version 26 (A: v12, seed 3) | 8.33/24, found 9.67; votes-only selection 9.67 vs kgmon 8.33; selection reopened |
| 2026-11-01 01:47 | Committed version 28 (v12 rerun, seed 2) for a second selection sample and a noise check | running |
| 2026-11-01 02:15 | Read version 27 (B: NVARC-faithful, seed 3) | 7.67/24 vs A 8.33; direction closed; votes-only = kgmon here |
| 2026-11-01 02:20 | Committed version 29 (v12, seed 1) as third selection sample | running; quota will reach ~25.5 h, no further smoke runs planned |
| 2026-11-01 03:08 | Hidden score version 16 (v12, 8 copies) = 28.89 | below v7 30.56; hidden favours 16 copies, within noise |
| 2026-11-01 03:10 | Read version 28 (v12 rerun, seed 2) | 11.50 (reproduces 11.50); all selection rules tie |
| 2026-11-01 05:24 | Read version 29 (v12, seed 1) | 7.50/24; votes-only 6.50; selection direction closed (net +0.33 over 4 samples) |
| 2026-11-01 05:26 | Built and committed version 30 (Nov 2 candidate: 16 copies + 2B second pass, small smoke for a crash check) | running |
| 2026-11-01 05:58 | Read version 30 (Nov 2 candidate, small smoke) | crash check passed: 2B on 3 GPUs, 5 second passes, no errors; version 24 hidden still running |
| 2026-11-01 06:57 | Version 24 hidden still running (7 h); rewrote WRITEUP summary for the hidden 28.89 result, added theory and evaluation-method sections | |
| 2026-11-01 07:20 | User asked why no smoke tests (quota guard had paused them). Quota 25:46/30 h; committed version 31 (v18 realistic, seed 2) | running |
| 2026-11-01 08:20 | Version 31 live (id 364464897) at 3538 s: 21/24 tasks done, 16 copies on all, 2 timeouts, 1 second pass started; version 24 hidden still running (8 h) | |
| 2026-11-01 08:42 | Read version 31 (v18 realistic, seed 2) | 13.50 total (G4 void); first pass 16 copies 10.00 vs 8 copies 11.50; runs at clock edge, 4 timeouts; version 24 hidden still running (9 h) |
| 2026-11-01 09:42 | Version 24 hidden still running (10 h) | no action |
| 2026-11-01 10:42 | Version 24 hidden still running (11 h). Version 16 took ~26 h from submit to score, so version 24 may land around Nov 2 02:00 UTC | no action |
| 2026-11-01 11:42 | Version 24 hidden still running (12 h) | no action |
| 2026-11-01 12:42 | Version 24 hidden still running (13 h) | no action |
| 2026-11-01 13:42 | Version 24 hidden still running (14 h) | no action |
| 2026-11-01 14:42 | Version 24 hidden still running (15 h) | no action |
| 2026-11-01 15:42 | Version 24 hidden still running (16 h) | no action |
| 2026-11-01 16:19 | Quota 28:20/30 h. User chose to keep the remaining 1:40 as a reserve for a Nov 2 recommit; no more GPU tests | |
| 2026-11-01 17:18 | Version 24 hidden still running (17 h) | no action |
| 2026-11-01 18:18 | Version 24 hidden still running (18 h) | no action |
| 2026-11-01 19:18 | Version 24 hidden still running (19 h) | no action |
| 2026-11-01 20:18 | Version 24 hidden still running (20 h) | no action |
| 2026-11-01 21:18 | Version 24 hidden still running (21 h) | no action |
| 2026-11-01 22:18 | Version 24 hidden still running (~22 h) | no action |
| 2026-11-01 23:18 | Version 24 hidden still running (~23 h) | no action |
| 2026-11-02 00:18 | Version 24 hidden still running (~24 h); Nov 2 slot now open | no action |
| 2026-11-02 01:18 | Version 24 hidden still running (~25 h) | no action |
| 2026-11-02 02:18 | Version 24 hidden still running (~26 h) | no action |
| 2026-11-02 03:18 | Version 24 hidden still running (~27 h, past version 16's 26 h) | no action; fallback decision due ~12:00 UTC if still pending |
| 2026-11-02 04:18 | Version 24 hidden still running (~28 h) | no action; fallback decision ~12:00 UTC if still pending |
| 2026-11-02 05:18 | Version 24 hidden still running (~29 h) | no action; fallback decision ~12:00 UTC if still pending |
| 2026-11-02 06:18 | Hidden score version 24 (8 copies + 2B) = 27.22 vs 28.89 without 2B; 2B direction closed; recommending version 15 (v11, 16 copies) for Nov 2 | asking user |
| 2026-11-02 16:51 | Re-examined public vs our hidden scores (NVARC ~32.6 vs ours ~30.35 at 16 copies); recommendation changed to version 27 (NVARC-faithful) | asking user |
| 2026-11-02 16:54 | **Submitted version 27** (v17 NVARC-faithful, 16 copies) as the Nov 2 final submission (user approved) | hidden run started |
| 2026-11-02 16:54 | Final selection left at 0/2 so Kaggle auto-selects the two best scores (running version 27 has no Select box) | no more submission slots |
