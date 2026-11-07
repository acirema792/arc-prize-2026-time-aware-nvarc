from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling

import logging
from contextlib import redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import bz2
import pickle

logging.disable(logging.WARNING)

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


class UnslothFixedTrainer(UnslothTrainer):

    # Issue https://github.com/unslothai/unsloth/issues/2435

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # 🔧 KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()  # Converts view tensor to independent tensor
        # Now safe for DDP gradient scaling
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:

    n = logits.size(0)

    # perf patch (mikelou1, Apache 2.0): same NLL, but only the 12 ARC-token columns leave the GPU
    logits_f = logits.float()
    token_ids = torch.tensor(ARC_TOKENS, dtype=torch.long, device=logits.device)
    nll = (
        torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
        + torch.logsumexp(logits_f, dim=-1, keepdim=True)
        - logits_f.index_select(-1, token_ids)
    ).cpu()

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for token_idx, t in enumerate(ARC_TOKENS):
            score = nll[i, token_idx].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0]) #[:5]
    
    while time.time() - start_time < 540 and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)
    # perf patch: teacher-forced scoring needs no KV cache, and only target-token log-probs leave the GPU
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    batch_logits = outputs.logits.float()
    batch_log_norm = torch.logsumexp(batch_logits, dim=-1)
    result = []
    for row_id, (query_tokens, answer_tokens) in enumerate(zip(batch_query_tokens, batch_answer_tokens)):
        query_length = len(query_tokens)
        positions = torch.arange(query_length - 1, query_length - 1 + len(answer_tokens), device=model.device)
        targets = torch.tensor(answer_tokens, dtype=torch.long, device=model.device)
        answer_score = (batch_logits[row_id, positions, targets] - batch_log_norm[row_id, positions]).sum()
        result.append(-answer_score.item())
    return result


def task_budget(queue, n_workers, end_time):
    """Seconds this worker may spend on its next task.

    The 2026 hidden test set has 240 tasks (2025 had 120), so NVARC's fixed
    1200s cap would run out of time with tasks still queued. Instead, split
    the remaining time evenly over the tasks this worker is still expected to
    take (the queue also holds one None sentinel per worker).
    """
    if FIXED_BUDGET:  # v17 NVARC-faithful mode: NVARC's fixed per-task cap
        return MAX_TASK_SECONDS
    remaining_tasks = max(0, queue.qsize() - n_workers)
    my_share = remaining_tasks / n_workers + 1
    budget = (end_time - time.time()) / my_share
    return max(MIN_TASK_SECONDS, min(MAX_TASK_SECONDS, budget))


def plan_training(budget, time_left, tokens_16, sec_per_token, top=16, share_16=None):
    """How many augmented copies to train on for this task: top (4, 8, 16 or 24), 16, 8, 4, or 0
    (decode with the base model).

    Training cannot be interrupted, so near the end of the run a large task could train past the
    deadline and come back with nothing. With a measured training speed (seconds per training
    token, from this worker's earlier tasks), pick the largest set that still leaves time to
    decode. tokens_16 is the token count of the 16-copy set; other sizes scale linearly.
    share_16 (v12 "auto"): 16 copies only when their training takes at most this share of the
    task's budget, since decoding, not training, uses most of a task's time.
    """
    n = top if budget >= 600 else min(top, 8)
    if sec_per_token is None:
        return min(n, 16)  # first task: no estimate yet, keep the old rule
    for m in [c for c in (24, 16, 8, 4) if c <= n]:
        est = sec_per_token * tokens_16 * m / 16
        if m > 16:
            # copies beyond NVARC's 16 only when they fit comfortably in this task's budget
            fits_task = est <= 0.6 * budget
        elif m == 16 and share_16:
            fits_task = est <= share_16 * budget
        else:
            fits_task = budget > SCARCE_BUDGET or est <= 0.5 * budget
        if est + MIN_DECODE_SECONDS <= time_left and fits_task:
            return m
    return 0


MIN_TASK_SECONDS = 120
MAX_TASK_SECONDS = 1200
MIN_DECODE_SECONDS = 240
# below this per-task budget time is short, and training must fit in half of it
SCARCE_BUDGET = 300
# looser second pass for a test output that got no guess: 2% total probability instead of 20%
FALLBACK_MAX_SCORE = -np.log(0.02)
FALLBACK_DECODE_SECONDS = 120
# ARC_TOP_COPIES=auto: 16-copy training may take at most this share of a task's budget (else 8)
AUTO_SHARE_16 = 0.25
# v13: decode a second set of 16 views while the task's budget has at least this much time left.
# Off by default: loop 5 scored 11.0/24 with them vs 11.5 without, and used the whole clock
EXTRA_VIEWS = os.environ.get("ARC_EXTRA_VIEWS", "0") == "1"
EXTRA_VIEWS_MIN_SECONDS = 60
# v14: test-time training intensity (NVARC: 1 epoch at 5e-5)
TTT_EPOCHS = int(os.environ.get("ARC_TTT_EPOCHS", "1"))
TTT_LR = float(os.environ.get("ARC_TTT_LR", "5e-5"))
# v15: once the queue is empty, an idle GPU trains a second adapter (new example order and colors)
# on a finished task whose best answer is weak, and decodes 16 new views with it; the merge votes
# over both passes. Every realistic test so far ended with the GPUs idle for ~24% of the clock, and
# more views from the same adapter found nothing new (loop 5), so the second pass varies the adapter
SECOND_PASS = os.environ.get("ARC_SECOND_PASS", "0") == "1"
# a test output whose top answer has fewer votes (decoded beams) than this counts as weak
SECOND_PASS_MAX_VOTES = 8
# v17: NVARC-faithful mode uses NVARC's fixed 1200 s per task instead of the dynamic share. Plain NVARC
# reproductions score 32.22 on the public leaderboard, above our v1 (30.14) and v7 (30.56)
FIXED_BUDGET = os.environ.get("ARC_FIXED_BUDGET", "0") == "1"


def worker(rank, queue, end_time, n_workers=4):

    model_path = os.environ["ARC_MODEL_PATH"]
    test_path = os.environ["ARC_TEST_PATH"]

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=TTT_EPOCHS,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=TTT_LR,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        # Disable FSDP (use standard DDP)
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192

    def load_model(path):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,
            full_finetuning=False,
            load_in_4bit=False,
            local_files_only=True,
            use_gradient_checkpointing=False,
            max_seq_length=max_seq_length,
        )

        model = FastLanguageModel.get_peft_model(model, **peft_params)

        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

        default_weights = get_peft_model_state_dict(model, adapter_name="default")
        default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

        collator = QwenDataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            mlm=False,
        )

        formatter = QwenFormatter(tokenizer=tokenizer)

        return model, tokenizer, default_weights, collator, formatter, formatter.max_new_tokens()

    model, tokenizer, default_weights, collator, formatter, max_new_tokens = load_model(model_path)
    second_model_path = os.environ.get("ARC_SECOND_MODEL_PATH")  # v16: the second pass's generator

    max_score = -np.log(0.2)

    arc_test_set = ArcDataset.from_file(test_path)

    dir_outputs = "../inference_outputs"
    os.makedirs(dir_outputs, exist_ok=True)

    sec_per_token = None  # measured training speed, set after this worker's first training
    top_env = os.environ.get("ARC_TOP_COPIES", "16")  # 24 trains more where time allows
    share_16 = AUTO_SHARE_16 if top_env == "auto" else None
    top_copies = 16 if top_env == "auto" else int(top_env)

    # v15 second pass: a task is eligible once its first pass is done; a claim file keeps two
    # GPUs from taking the same task
    dir_done, dir_claims = "../pass1_done", "../pass2_claims"
    os.makedirs(dir_done, exist_ok=True)
    os.makedirs(dir_claims, exist_ok=True)

    def top_votes(key):
        """Fewest votes behind any of this task's test outputs' top answer (0: no guess)."""
        n_tests = len(arc_test_set.queries[key]["test"])
        votes = [defaultdict(int) for _ in range(n_tests)]
        for name in os.listdir(dir_outputs):
            bk = name.split(".")[0]
            if bk.rsplit("_", 1)[0] != key:
                continue
            with bz2.BZ2File(os.path.join(dir_outputs, name)) as f:
                for g in pickle.load(f):
                    votes[int(bk.rsplit("_", 1)[1])][tuple(map(tuple, g["solution"]))] += 1
        return min(max(v.values(), default=0) for v in votes)

    def task_cells(key):
        q = arc_test_set.queries[key]
        return sum(len(g) * len(g[0]) for p in q["train"] for g in p.values()) + \
            sum(len(t["input"]) * len(t["input"][0]) for t in q["test"])

    def claim_weak_task():
        """Smallest finished task with a weak answer that no GPU has claimed yet, or None."""
        done = [k for k in os.listdir(dir_done) if not os.path.exists(os.path.join(dir_claims, k))]
        for key in sorted(done, key=task_cells):
            votes = top_votes(key)
            if votes >= SECOND_PASS_MAX_VOTES:
                continue
            try:
                os.close(os.open(os.path.join(dir_claims, key), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            print(f"[Rank {rank}] second pass for {key} (top answer has {votes} votes)")
            return key
        return None

    def jobs():
        """(task, training seed, view seed, second pass): the queue, then v15 second passes."""
        while not queue.empty():
            if time.time() > end_time:
                print(f"[Rank {rank}] stop!")
                return
            key = queue.get()
            if key is None:
                break
            yield key, 1, 2, False
        while SECOND_PASS and time.time() < end_time - MIN_DECODE_SECONDS:
            key = claim_weak_task()
            if key is None:
                if len(os.listdir(dir_done)) >= len(arc_test_set.keys):
                    return
                time.sleep(30)  # other GPUs are still on their last first-pass tasks
                continue
            yield key, 7, 3, True

    for key, train_seed, view_seed, second in jobs():

        if second and second_model_path:
            # v16: swap in a separately trained base model once, so the second pass's errors differ
            # from the first's (a second adapter on the same base model found nothing new, loop 7)
            print(f"[Rank {rank}] loading {second_model_path} for the second pass")
            try:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model, tokenizer, default_weights, collator, formatter, max_new_tokens = load_model(second_model_path)
            except Exception as e:  # never retry: a crash restart would reload and fail again
                print(f"[Rank {rank}] second model failed to load, stopping this GPU: {e!r}")
                return
            second_model_path = None
            sec_per_token = None  # training speed differs between the two models

        start_time = time.time()
        # a second pass has the rest of the clock: the queue is empty
        budget = end_time - start_time if second else task_budget(queue, n_workers, end_time)
        task_deadline = min(end_time, start_time + budget)
        print(f"[Rank {rank}] puzzle {key}: budget {budget:.0f}s")

        torch.cuda.reset_peak_memory_stats()

        load_result = set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )

        model = FastLanguageModel.for_training(model)

        puzzle_ds = arc_test_set.change_keys([key])

        def train_list(n):
            ds = puzzle_ds.augment(n=n, shfl_keys=True, seed=train_seed)
            return ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length).as_list(formatter)

        # v7: size test-time training to the time this task can afford
        samples = train_list(16)
        tokens_16 = sum(len(tokenizer.encode(s["text"])) for s in samples)
        n_copies = plan_training(budget, end_time - time.time(), tokens_16, sec_per_token, top_copies, share_16)
        print(f"[Rank {rank}] puzzle {key}: training on {n_copies} copies ({tokens_16} tokens at 16)")
        if second and not n_copies:
            print(f"[Rank {rank}] second pass for {key} skipped: training does not fit")
            continue
        if n_copies not in (0, 16):
            samples = train_list(n_copies)

        stats = "skipped (decoding with the base model)"
        if n_copies:
            train_start = time.time()
            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):

                trainer = UnslothFixedTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    train_dataset=Dataset.from_list(samples),
                    dataset_text_field="text",
                    max_seq_length=max_seq_length,
                    args=UnslothTrainingArguments(**train_args),
                )

                stats = trainer.train()

                model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

                del trainer
            # keep the slower of the last two measurements so estimates err long
            rate = (time.time() - train_start) / max(1, tokens_16 * n_copies / 16)
            sec_per_token = rate if sec_per_token is None else max(rate, 0.5 * (rate + sec_per_token))

        model = FastLanguageModel.for_inference(model)
        
        gc.collect()
        torch.cuda.empty_cache()
            
        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for training")

        torch.cuda.reset_peak_memory_stats()
        
        print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

        # v4: a task whose training overran its budget still gets decoded (it used to get nothing)
        task_deadline = min(end_time, max(task_deadline, time.time() + MIN_DECODE_SECONDS))

        puzzle_ds_multi = puzzle_ds.split_multi_replies()

        def view_batches(seed):
            eval_ds = puzzle_ds_multi.augment(n=2, seed=seed)
            eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

            test_id_to_subkeys = defaultdict(list)
            for subkey in sorted(eval_ds.keys):
                test_id = subkey.split(".")[0].split("_")[1]
                test_id_to_subkeys[test_id].append(subkey)

            batches = []
            for test_id, subkeys in test_id_to_subkeys.items():
                # 0: permute x 2
                # 4: rot90.rot90.permute x 2
                batch = []
                for offset in [0, 4]:
                    batch.extend(subkeys[offset:offset+2])
                batches.append(batch)
                # 2: permute.rot90 x 2
                # 6: rot90.rot90.rot90.permute x 2
                batch = []
                for offset in [2, 6]:
                    batch.extend(subkeys[offset:offset+2])
                batches.append(batch)
            for test_id, subkeys in test_id_to_subkeys.items():
                # 8: transpose.permute x 2
                # 12: transpose.rot90.rot90.permute x 2
                batch = []
                for offset in [8, 12]:
                    batch.extend(subkeys[offset:offset+2])
                batches.append(batch)
                # 10: transpose.rot90.permute x 2
                # 14: transpose.rot90.rot90.rot90.permute x 2
                batch = []
                for offset in [10, 14]:
                    batch.extend(subkeys[offset:offset+2])
                batches.append(batch)
            return eval_ds, test_id_to_subkeys, batches

        eval_ds, test_id_to_subkeys, batches = view_batches(seed=view_seed)

        with torch.inference_mode():

            known_scores = {}
            covered = set()  # test outputs ("<task>_<i>") with at least one saved guess

            def decode_batch(subkeys, score_limit, deadline):
                tokens = []
                for subkey in subkeys:
                    data = eval_ds.get(subkey, formatter)
                    tokens.append(tokenizer.encode(data["input"]))

                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, score_limit, deadline)

                for subkey_id, scored_beams in dfs_result:

                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]
                    decoded_result = []

                    for beam_score, tokens in scored_beams:

                        array = formatter.convert_tokens_to_array(tokens)
                        if array is None:
                            continue

                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                        grid_id = (bk, tuple(map(tuple, solution)))

                        if grid_id in known_scores:
                            augmented_scores = known_scores[grid_id]
                        else:
                            print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                            aug_dataset = ArcDataset(
                                keys=[bk],
                                queries={bk: puzzle_ds_multi.queries.get(bk)},
                                replies={bk: [solution.tolist()]},
                            )
                            aug_dataset = aug_dataset.augment(seed=hash(bk) % 1024**2)
                            aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                            aug_queries = []
                            aug_answers = []
                            for augmented_sample in aug_dataset.as_list(formatter):
                                aug_queries.append(augmented_sample["input"])
                                aug_answers.append(augmented_sample["reply"])
                            augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                            augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                            augmented_scores = augmented_scores1 + augmented_scores2
                            known_scores[grid_id] = augmented_scores
                        
                        decoded_result.append({
                            "beam_score": beam_score,
                            "score_aug": augmented_scores,
                            "solution": solution,
                            "loose": score_limit != max_score,  # from the looser fallback pass
                            "second": second,  # from the v15 second adapter
                        })

                    if len(decoded_result):
                        covered.add(bk)
                        with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                            pickle.dump(decoded_result, f)

            for subkeys in batches:

                spend_time = time.time() - start_time
                if time.time() > task_deadline:
                    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                    break

                print(f"[Rank {rank}] decoding {subkeys}")
                decode_batch(subkeys, max_score, task_deadline)

            # v10: a test output with no guess above the usual 20% cutoff gets one looser pass over
            # 4 of its views, instead of falling back to a placeholder grid. The views must share the
            # grid shape (identity and rot180, as in the first batch above): a batch mixing rot90 views
            # has unequal prompt lengths and crashed the worker in loop 2
            for test_id, subkeys in test_id_to_subkeys.items():
                bk = f"{key}_{test_id}"
                fallback_deadline = min(end_time, time.time() + FALLBACK_DECODE_SECONDS)
                if second or bk in covered or time.time() >= fallback_deadline - 10:
                    continue
                print(f"[Rank {rank}] no guess for {bk}; looser decode")
                try:
                    decode_batch(subkeys[0:2] + subkeys[4:6], FALLBACK_MAX_SCORE, fallback_deadline)
                except Exception as e:  # a lost fallback must never cost the rest of the queue
                    print(f"[Rank {rank}] looser decode failed for {bk}: {e!r}")
                    torch.cuda.empty_cache()

            # v13: time left in this task's budget goes to 16 more views (new color permutations and
            # example orders), which adds candidates and votes; loop 4 found the answer among the
            # candidates for 13/24 tasks and runs ended ~20% early. Last, because eval_ds changes
            # and decode_batch reads it
            if EXTRA_VIEWS and not second and time.time() < task_deadline - EXTRA_VIEWS_MIN_SECONDS:
                try:
                    eval_ds, _, extra_batches = view_batches(seed=3)
                    for subkeys in extra_batches:
                        if time.time() > task_deadline:
                            break
                        print(f"[Rank {rank}] extra views {subkeys}")
                        decode_batch(subkeys, max_score, task_deadline)
                except Exception as e:  # extra views are optional; never lose the task over them
                    print(f"[Rank {rank}] extra views failed for {key}: {e!r}")
                    torch.cuda.empty_cache()

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")
        
        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")
        if not second:
            open(os.path.join(dir_done, key), "w").close()