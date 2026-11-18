"""LoRA fine-tuning of OpenJev on the survey answers, then scoring by candidate log-likelihood.

Run with torchrun on one Hangang node (4 GPUs, one full bf16 copy per GPU, DDP on the LoRA weights):
  torchrun --nproc_per_node 4 finetune_openjev.py --model <openjev dir> --data <dir with train/eval.jsonl> --out <dir>

Scoring: for each question the model's log-probability of every allowed answer string
('{"level": "보통"}', ...) is summed over its tokens; the most likely answer is the prediction.
The same scoring runs before training (base model) and after (LoRA), so the gain is measured on equal terms.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist


def log(*a):
    if int(os.environ.get("RANK", "0")) == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_model(path, device):
    import transformers
    cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None) or transformers.AutoModelForImageTextToText
    model = cls.from_pretrained(path, dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa")
    tok = transformers.AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def prompt_ids(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def batches(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


@torch.no_grad()
def score(model, tok, items, device, per_forward=6):
    """Return {id: {"pred", "logp": {candidate: value}}} for this rank's items."""
    model.eval()
    seqs = []  # (item index, candidate index, ids, answer start)
    for n, it in enumerate(items):
        p = prompt_ids(tok, it["messages"])
        for k, c in enumerate(it["candidates"]):
            a = tok(c, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            seqs.append((n, k, p + a, len(p)))
    logp = {}
    for chunk in batches(seqs, per_forward):
        L = max(len(s[2]) for s in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (_, _, s, _) in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        out = model(input_ids=ids.to(device), attention_mask=att.to(device))
        for j, (n, k, s, start) in enumerate(chunk):
            tgt = torch.tensor(s[start:], device=device)
            lp = torch.log_softmax(out.logits[j, start - 1:len(s) - 1].float(), dim=-1)  # answer positions only
            logp[(n, k)] = lp.gather(1, tgt[:, None]).sum().item()
        del out
    res = {}
    for n, it in enumerate(items):
        vals = [logp[(n, k)] for k in range(len(it["candidates"]))]
        best = max(range(len(vals)), key=vals.__getitem__)
        res[it["id"]] = {"pred": json.loads(it["candidates"][best])["level"], "logp": vals}
    return res


def gather_scores(local, out_path, rank, world):
    tmp = f"{out_path}.rank{rank}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(local, f, ensure_ascii=False)
    dist.barrier()
    if rank == 0:
        merged = {}
        for r in range(world):
            with open(f"{out_path}.rank{r}", encoding="utf-8") as f:
                merged.update(json.load(f))
            os.remove(f"{out_path}.rank{r}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False)
        log(f"wrote {out_path} ({len(merged)} items)")
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8, help="sequences per GPU per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank-lora", type=int, default=16)
    ap.add_argument("--max-train-minutes", type=float, default=100)
    ap.add_argument("--skip-base-eval", action="store_true")
    ap.add_argument("--limit-train", type=int, default=0)
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    os.makedirs(a.out, exist_ok=True)

    train = [json.loads(x) for x in open(os.path.join(a.data, "train.jsonl"), encoding="utf-8")]
    evals = [json.loads(x) for x in open(os.path.join(a.data, "eval.jsonl"), encoding="utf-8")]
    if a.limit_train:
        train = train[:a.limit_train]
    my_evals = evals[rank::world]

    model, tok = load_model(a.model, device)
    log("model loaded", type(model).__name__, f"world={world}")

    if not a.skip_base_eval:
        t0 = time.time()
        gather_scores(score(model, tok, my_evals, device), os.path.join(a.out, "scores_base.json"), rank, world)
        log(f"base scoring {time.time() - t0:.0f}s")

    from peft import LoraConfig, get_peft_model
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    pattern = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    import re
    targets = [n for n in names if re.match(pattern, n)]
    if not targets:
        raise SystemExit("no LoRA target modules matched; sample names: " + ", ".join(names[:20]))
    log(f"LoRA targets: {len(targets)} linear layers (e.g. {targets[0]})")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank_lora, lora_alpha=2 * a.rank_lora, lora_dropout=0.05,
                                             target_modules=pattern, task_type="CAUSAL_LM"))
    if rank == 0:
        model.print_trainable_parameters()
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=False)

    # tokenise: loss only on the answer tokens
    rng = random.Random(0)
    examples = []
    for r in train:
        p = prompt_ids(tok, r["messages"])
        ans = tok(r["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        examples.append((p + ans, len(p)))
    steps_per_epoch = len(examples) // (a.batch * world)
    total = max(1, int(steps_per_epoch * a.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    log(f"train examples {len(examples)}, steps {total} (batch {a.batch} x {world} GPUs)")

    model.train()
    t0 = time.time()
    order = list(range(len(examples)))
    step = 0
    stop = torch.zeros(1, device=device)
    while step < total:
        rng.shuffle(order)
        for i in range(0, len(order) - a.batch * world + 1, a.batch * world):
            if step >= total:
                break
            mine = [examples[j] for j in order[i + rank * a.batch: i + (rank + 1) * a.batch]]
            L = max(len(s) for s, _ in mine)
            ids = torch.full((len(mine), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(mine), L), dtype=torch.long)
            lab = torch.full((len(mine), L), -100, dtype=torch.long)
            for j, (s, start) in enumerate(mine):
                ids[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
                lab[j, start:len(s)] = torch.tensor(s[start:])
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            out = ddp(input_ids=ids, attention_mask=att)
            mask = lab[:, 1:] != -100  # loss on the answer tokens only; avoids a float copy of all logits
            loss = torch.nn.functional.cross_entropy(out.logits[:, :-1][mask].float(), lab[:, 1:][mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                log(f"step {step}/{total} loss {loss.item():.4f} {el / step:.1f}s/step eta {(total - step) * el / step / 60:.0f}min")
            stop[0] = 1.0 if (time.time() - t0) / 60 > a.max_train_minutes else 0.0
            dist.all_reduce(stop)
            if stop.item() > 0:
                log(f"time budget reached at step {step}")
                total = step
                break
    log(f"training done: {step} steps in {(time.time() - t0) / 60:.1f} min")
    if rank == 0:
        model.save_pretrained(os.path.join(a.out, "lora"))
        with open(os.path.join(a.out, "train_meta.json"), "w") as f:
            json.dump({"steps": step, "examples_seen": step * a.batch * world, "train_rows": len(examples),
                       "batch_per_gpu": a.batch, "gpus": world, "lr": a.lr, "lora_rank": a.rank_lora,
                       "minutes": round((time.time() - t0) / 60, 1)}, f)
    dist.barrier()
    t1 = time.time()
    gather_scores(score(model, tok, my_evals, device), os.path.join(a.out, "scores_lora.json"), rank, world)
    log(f"lora scoring {time.time() - t1:.0f}s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
