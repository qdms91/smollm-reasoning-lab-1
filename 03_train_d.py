"""Method D: full-parameter REINFORCE for a single direct answer on CPU."""

import argparse
import hashlib
import json
import random
import re
import resource
import shutil
import tempfile
from collections import Counter
from importlib import import_module
from itertools import combinations_with_replacement
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

control = import_module("01_baseline_inference")
INSTRUCTION = "Answer the question directly in one short sentence."
SIZES = {"train": 200, "validation": 50, "test": 100}
DATA_SEED, TRAIN_SEED = 41, 73
LR = 2e-6
MAX_TOKENS = control.TOKEN_BUDGET
CHECKS = (25, 50, 75, 100)


# 1. Freeze the same questions for all future methods.
def frozen_data():
    folder = control.ROOT / "data"
    probes = control.read_jsonl(folder / "questions.jsonl")
    pattern = r"Is (\d+) \+ (\d+) greater than (\d+)\?"
    reserved = set()
    for item in probes:
        match = re.fullmatch(pattern, item["question"], re.I)
        if match:
            reserved.add(tuple(sorted(map(int, match.groups()[:2]))))

    rng = random.Random(DATA_SEED)
    pool = [p for p in combinations_with_replacement(
        range(1, 33), 2) if p not in reserved]
    rng.shuffle(pool)
    rows = []
    for split, size in SIZES.items():
        block = []
        while len(block) < size:
            first = pool.pop()
            # Pair two different sums near one threshold. Each threshold gets both labels.
            candidates = [p for p in pool if 1 <=
                          abs(sum(first) - sum(p)) <= 3]
            if not candidates:
                continue
            second = rng.choice(candidates)
            pool.remove(second)
            low, high = sorted((sum(first), sum(second)))
            threshold = rng.randrange(low, high)
            for a, b in (first, second):
                if rng.randrange(2):
                    a, b = b, a
                block.append({"question": f"Is {a} + {b} greater than {threshold}?",
                              "answer": "Yes" if a + b > threshold else "No", "split": split})
        rng.shuffle(block)
        rows.extend(block)

    path = folder / "arithmetic_v1.jsonl"
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(
                f"{path} differs from this split recipe. Check the file and probes.")
    else:
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as tmp:
            tmp.write(data)
        Path(tmp.name).replace(path)
    digest = hashlib.sha256(data).hexdigest()
    print(
        f"Frozen splits: 200 train / 50 validation / 100 test | SHA-256 {digest}", flush=True)
    # The training and benchmark loops receive only the examples they are allowed to use.
    return ([r for r in rows if r["split"] == "train"],
            [r for r in rows if r["split"] == "validation"], digest)


# 2. Extract the answer without consulting the reference. A missing answer scores zero.
POLAR = r"(yes|no)\b"
END = (r"(?=[ \t]*(?:$|[.,!;:()—–-]|\d|(?:it|this|that|the|there|we|I|because|since|as|"
       r"indeed|definitely|absolutely|unfortunately|but|and)\b))")
MARKED = re.compile(
    r"\b(?:final\b(?:[ \t]+answer\b)?[ \t]*(?:is\b[ \t]*[:=]?|[:=\-])?|"
    r"(?:answer|response|choice|conclusion|result)[ \t]*(?:is\b[ \t]*[:=]?|would be\b|must be\b|[:=]))"
    r"[ \t]*(?:(?:clearly|definitely|simply)[ \t]+)?\(?[ \t]*" + POLAR + END, re.I | re.M)
STATED = re.compile(
    r"(?:^|[.,!;:\n]|\b(?:so|therefore|thus|hence|actually|however|but|in conclusion|I would say)\b[ ,]*)"
    r"[ \t]*\(?[ \t]*" + POLAR + END, re.I | re.M)
ALTERNATIVES = re.compile(
    r"\b(?:yes\s*[,;]?\s*(?:/|or|and)\s*no|no\s*[,;]?\s*(?:/|or|and)\s*yes)\b", re.I)
UNCERTAIN = re.compile(
    r"\b(?:if|whether|maybe|perhaps|possibly|not sure|unsure|not saying)\b", re.I)


def boolean_answer(text):
    clean = re.sub(r"""[*_`"'“”]""", "", text).strip()
    clean = re.sub(r"[^.!?;\n]*\?", "\n",
                   clean)  # a repeated question makes no assertion
    if not clean:
        return None
    # A labelled final decision can resolve earlier deliberation. Conflicting final
    # declarations, a choice list, or a mere question are still ambiguous.

    def declarations(pattern, passage):
        for match in pattern.finditer(passage):
            lead = re.split(r"[.!?;,\n]", passage[:match.start()])[-1]
            if not UNCERTAIN.search(lead):
                yield match

    marked = list(declarations(MARKED, clean))
    explicit = [m for m in marked if m.group().lower().startswith("final")]
    if explicit:
        marked = [m for m in marked if m.start() >= explicit[0].start()]
    scope = clean[marked[0].start():] if marked else clean
    if ALTERNATIVES.search(scope):
        return None
    votes = {m.group(1).capitalize() for m in marked}
    votes.update(m.group(1).capitalize() for m in declarations(STATED, scope))
    return votes.pop() if len(votes) == 1 else None


# 3. Sample from the model. Replay just its generated actions for the gradient.
def encode(tokenizer, question):
    messages = [{"role": "user", "content": f"{question}\n{INSTRUCTION}"}]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True)


@torch.no_grad()
def rollout(model, tokenizer, question, sample):
    inputs = encode(tokenizer, question)
    # At temperature 1 with truncation disabled, the sampling policy is the same
    # softmax that cross_entropy uses in the gradient pass below.
    sampling = {"temperature": 1.0, "top_k": 0, "top_p": 1.0} if sample else {}
    ids = model.generate(**inputs, do_sample=sample, num_beams=1, max_new_tokens=MAX_TOKENS,
                         repetition_penalty=1.0, use_cache=True, **sampling)
    start = inputs["input_ids"].shape[1]
    answer_ids = ids[0, start:]
    text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return ids, start, {"text": text, "prediction": boolean_answer(text), "tokens": len(answer_ids)}


def update(model, tokenizer, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    observations = []
    for item in batch:
        ids, start, reply = rollout(
            model, tokenizer, item["question"], sample=True)
        ids = ids.clone()  # normal token tensor, also with inference-mode generation backends
        reward = int(reply["prediction"] == item["answer"])
        # Logit at position start-1 predicts the first generated token. Include EOS
        # when sampled. Each completion is unpadded. Prompt tokens receive no loss.
        logits = model(input_ids=ids[:, :-1], use_cache=False).logits
        nll = F.cross_entropy(
            logits[0, start-1:, :], ids[0, start:], reduction="sum")
        loss = (reward - 0.5) * nll / len(batch)
        loss.backward()
        del logits, nll, loss
        observations.append({**item, **reply, "reward": reward})
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return observations, float(grad_norm)


def validate(model, tokenizer, items, step, log):
    started = perf_counter()
    replies = []
    for item in items:
        _, _, reply = rollout(model, tokenizer, item["question"], sample=False)
        replies.append(
            {**item, **reply, "correct": reply["prediction"] == item["answer"]})
    counts = Counter(r["prediction"] for r in replies)
    correct = sum(r["correct"] for r in replies)
    print(f"Validation {step}: {correct}/{len(items)} correct | Yes {counts['Yes']}, "
          f"No {counts['No']}, invalid {counts[None]} | {perf_counter()-started:.1f} s", flush=True)
    log({"event": "validation", "step": step,
        "correct": correct, "responses": replies})
    return correct


def run(mode):
    torch.set_num_threads(4)
    train, validation, digest = frozen_data()
    # One pass over all 200 questions, with one example of each label per update.
    rng = random.Random(TRAIN_SEED)
    by_label = [[r for r in train if r["answer"] == label]
                for label in ("Yes", "No")]
    for group in by_label:
        rng.shuffle(group)
    batches = list(zip(*by_label))
    limit = 5 if mode == "benchmark" else len(batches)
    tokenizer = AutoTokenizer.from_pretrained(control.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        control.MODEL_ID, dtype=torch.float32).eval()
    # eval() disables dropout while allowing full gradients.
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.0, foreach=False)
    witness = model.model.norm.weight  # small, named trainable vector in SmolLM2
    original = witness.detach().clone()
    runs = control.ROOT / "runs"
    runs.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"d_{mode}_", dir=runs))
    print(f"Run directory: {out}", flush=True)

    with (out / "events.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
        def log(event):
            stream.write(json.dumps(event) + "\n")

        log({"event": "settings", "mode": mode, "model": control.MODEL_ID,
             "revision": getattr(model.config, "_commit_hash", None), "data_sha256": digest,
             "seed": TRAIN_SEED, "prompt": INSTRUCTION, "lr": LR, "max_new_tokens": MAX_TOKENS,
             "torch": str(torch.__version__), "transformers": transformers.__version__,
             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        if mode == "train":
            validate(model, tokenizer, validation, 0, log)
        # benchmark and train start from the same RNG state
        torch.manual_seed(TRAIN_SEED)
        best_count, best_step = -1, None
        durations = []
        for step, batch in enumerate(batches[:limit], 1):
            started = perf_counter()
            replies, grad_norm = update(model, tokenizer, optimizer, batch)
            seconds = perf_counter() - started
            durations.append(seconds)
            reward = sum(r["reward"] for r in replies)
            log({"event": "update", "step": step, "seconds": seconds, "grad_norm": grad_norm,
                 "responses": replies})
            print(f"Update {step}/{limit}: {seconds:.1f} s, rewarded {reward}/{len(batch)}",
                  flush=True)
            if mode == "train" and step in CHECKS:
                correct = validate(model, tokenizer, validation, step, log)
                if correct >= best_count:  # latest trained checkpoint wins ties
                    checkpoint = out / f"checkpoint_{step}"
                    model.save_pretrained(checkpoint)
                    tokenizer.save_pretrained(checkpoint)
                    selected = {
                        "step": step, "validation_correct": correct, "path": checkpoint.name}
                    pending = out / "best.tmp"
                    pending.write_text(json.dumps(
                        selected) + "\n", encoding="utf-8")
                    pending.replace(out / "best.json")
                    if best_step is not None:
                        shutil.rmtree(out / f"checkpoint_{best_step}")
                    best_count, best_step = correct, step

        delta = float((witness.detach() - original).abs().max())
        changed = int((witness.detach() != original).sum())
        mean = sum(durations) / len(durations)
        peak = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # Linux KiB -> GiB
        log({"event": "summary", "mean_update_s": mean, "peak_rss_gib": peak,
             "witness": "model.norm.weight", "changed_elements": changed, "max_abs_change": delta,
             "best_trained_step": best_step})
        print(f"Mean {mean:.2f} s/update | 100 updates: {100*mean/60:.1f} min "
              f"(updates only) | peak process RAM {peak:.2f} GiB")
        print(f"model.norm.weight: {changed}/{original.numel()} elements changed, "
              f"max absolute change {delta:.3g}")
        if best_step is not None:
            print(
                f"Selected trained checkpoint: step {best_step}, validation {best_count}/50.")
        if not changed:
            raise RuntimeError(
                "The tracked parameter did not change. Inspect the update log.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("benchmark", "train"), required=True)
    run(parser.parse_args().mode)
