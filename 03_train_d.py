"""Method D: full-parameter REINFORCE for a single direct answer on CPU."""

import argparse
import hashlib
import json
import math
import random
import re
import resource
import shutil
import tempfile
from collections import Counter
from importlib import import_module
from itertools import combinations_with_replacement, islice
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
KEEP = (25, 50, 100)
ORDERS = ("balanced", "blocked-training", "threshold-paired")
LABELS = ("Yes", "No")
EXPRESSION = re.compile(r"Is (\d+) \+ (\d+) greater than (\d+)\?", re.I)


# 1. Freeze the same questions for all future methods.
def frozen_data():
    folder = control.ROOT / "data"
    path = folder / "arithmetic_v1.jsonl"
    existing = path.is_file()
    probes = control.read_jsonl(folder / "questions.jsonl")
    reserved = set()
    for item in probes:
        match = EXPRESSION.fullmatch(item["question"])
        if match:
            reserved.add(tuple(sorted(map(int, match.groups()[:2]))))

    rng = random.Random(DATA_SEED)
    pool = [p for p in combinations_with_replacement(
        range(1, 33), 2) if p not in reserved]
    rng.shuffle(pool)
    rows = []
    for split, size in SIZES.items():
        if existing and split == "test":
            break
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

    lines = [(json.dumps(row) + "\n").encode() for row in rows]
    dev_count = SIZES["train"] + SIZES["validation"]
    development = b"".join(lines[:dev_count])
    if existing:
        # The frozen file is ordered train, validation, test. Stop at validation.
        with path.open("rb") as stream:
            if b"".join(islice(stream, dev_count)) != development:
                raise ValueError(f"Training/validation in {path} differs from the frozen recipe.")
    else:
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as tmp:
            tmp.writelines(lines)
        Path(tmp.name).replace(path)
    digest = hashlib.sha256(development).hexdigest()
    return ([r for r in rows if r["split"] == "train"],
            [r for r in rows if r["split"] == "validation"], digest)


def read_validation():
    """Use the original file directly, even if the exploratory probes have since changed."""
    path = control.ROOT / "data" / "arithmetic_v1.jsonl"
    start, size = SIZES["train"], SIZES["validation"]
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in islice(stream, start, start + size)]
    counts = Counter(r["answer"] for r in rows)
    if (counts != {"Yes": size // 2, "No": size // 2}
            or any(r["split"] != "validation" for r in rows)
            or len({r["question"] for r in rows}) != size):
        raise ValueError(f"Expected the {size} original balanced validation questions in {path}.")
    return rows


def training_batches(train, order):
    """Each schedule uses every example once. Keep the original within-label shuffles."""
    rng = random.Random(TRAIN_SEED)
    yes, no = ([r for r in train if r["answer"] == label] for label in LABELS)
    rng.shuffle(yes)
    rng.shuffle(no)
    if len(yes) != len(no) or len(yes) + len(no) != len(train):
        raise ValueError("The schedules require equal numbers of Yes and No examples.")
    if order == "balanced":
        return list(zip(yes, no))
    if order == "blocked-training":
        if len(yes) % 10:
            raise ValueError("Each five-update block needs ten examples of its label.")
        batches = []
        for start in range(0, len(yes), 10):
            for group in (yes, no):
                batches.extend((group[i], group[i + 1]) for i in range(start, start + 10, 2))
        return batches
    if order != "threshold-paired":
        raise ValueError(f"Unknown batch order: {order}")

    def threshold_and_sum(row):
        a, b, threshold = map(int, EXPRESSION.fullmatch(row["question"]).groups())
        return threshold, a + b

    # Match by threshold, then by sum within a threshold. Repeated thresholds can
    # admit multiple valid partners. Preserve the balanced schedule's Yes order.
    partners = {}
    for y, n in zip(sorted(yes, key=threshold_and_sum), sorted(no, key=threshold_and_sum)):
        ty, sy = threshold_and_sum(y)
        tn, sn = threshold_and_sum(n)
        if ty != tn or not (sn <= ty < sy and 1 <= sy - sn <= 3):
            raise ValueError(f"No valid threshold pairing for {y['question']} and {n['question']}")
        partners[y["question"]] = n
    return [(y, partners[y["question"]]) for y in yes]


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
def answer_token_ids(tokenizer, messages, prompt_ids):
    """Find the exact one-token continuations in the actual assistant context."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = prompt_ids[0].tolist()
    ids = []
    for label in LABELS:
        full = tokenizer(prompt + label, add_special_tokens=False)["input_ids"]
        if len(full) != len(prefix) + 1 or full[:-1] != prefix:
            raise ValueError(f"{label!r} is not one next token after this chat prompt.")
        ids.append(full[-1])
    if ids[0] == ids[1] or any(i in tokenizer.all_special_ids for i in ids):
        raise ValueError("The two answer tokens must be distinct vocabulary tokens.")
    return ids


@torch.no_grad()
def rollout(model, tokenizer, question, sample, probabilities=False):
    messages = [{"role": "user", "content": f"{question}\n{INSTRUCTION}"}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
    # At temperature 1 with truncation disabled, the sampling policy is the same
    # softmax that cross_entropy uses in the gradient pass below.
    sampling = {"temperature": 1.0, "top_k": 0, "top_p": 1.0} if sample else {}
    options = {"output_logits": True, "return_dict_in_generate": True} if probabilities else {}
    generated = model.generate(**inputs, do_sample=sample, num_beams=1, max_new_tokens=MAX_TOKENS,
                               repetition_penalty=1.0, use_cache=True, **sampling, **options)
    ids = generated.sequences if probabilities else generated
    start = inputs["input_ids"].shape[1]
    answer_ids = ids[0, start:]
    text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    reply = {"text": text, "prediction": boolean_answer(text), "tokens": len(answer_ids)}
    if probabilities:
        token_ids = answer_token_ids(tokenizer, messages, inputs["input_ids"])
        # Unprocessed scores for the first assistant token, from this same greedy call.
        logits = generated.logits[0][0].double()
        reply["raw_yes"], reply["raw_no"] = logits.softmax(dim=0)[token_ids].tolist()
        reply["q_yes"], reply["q_no"] = logits[token_ids].softmax(dim=0).tolist()
    return ids, start, reply


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


def validate(model, tokenizer, items, step=None, log=None):
    started = perf_counter()
    replies = []
    for item in items:
        _, _, reply = rollout(model, tokenizer, item["question"], sample=False, probabilities=True)
        replies.append(
            {**item, **reply, "correct": reply["prediction"] == item["answer"]})
    yes = [r["q_yes"] for r in replies if r["answer"] == "Yes"]
    no = [r["q_no"] for r in replies if r["answer"] == "No"]
    size = len(items)
    scores = {
        "correct": sum(r["correct"] for r in replies), "size": size,
        "q_correct": (sum(yes) + sum(no)) / size,
        "q_wrong": sum(r["q_no"] if r["answer"] == "Yes" else r["q_yes"] for r in replies) / size,
        "yes_class": sum(yes) / len(yes), "no_class": sum(no) / len(no),
        "raw_yes": sum(r["raw_yes"] for r in replies) / size,
        "raw_no": sum(r["raw_no"] for r in replies) / size,
    }
    if log is not None:
        counts = Counter(r["prediction"] for r in replies)
        print(f"Validation {step}: {scores['correct']}/{size} correct | Yes {counts['Yes']}, "
              f"No {counts['No']}, invalid {counts[None]} | {perf_counter()-started:.1f} s", flush=True)
        log({"event": "validation", "step": step, **scores, "responses": replies})
    return scores


def probability_table(rows):
    width = max(len("Model"), *(len(name) for name, _ in rows))
    print(f"\n{'Model':<{width}}  qCorrect   qWrong    qC|Yes     qC|No   ΔqCorrect"
          "     raw Yes/No      Greedy")
    base = rows[0][1]["q_correct"]
    for name, s in rows:
        print(f"{name:<{width}}  {s['q_correct']:8.5f} {s['q_wrong']:8.5f} "
              f"{s['yes_class']:9.5f} {s['no_class']:9.5f} {s['q_correct']-base:+10.5f}  "
              f"{s['raw_yes']:.5f}/{s['raw_no']:.5f}  {s['correct']:2}/{s['size']}")


def discover_checkpoints(order, lr, validation):
    """Find the legacy 25/100 pair or the milestones of one matching new run."""
    matches = {step: [] for step in KEEP}
    references = [(r["question"], r["answer"]) for r in validation]
    for events in sorted((control.ROOT / "runs").rglob("events.jsonl")):
        paths = {step: events.parent / f"checkpoint_{step}" for step in KEEP}
        if not any(path.is_dir() for path in paths.values()):
            continue
        with events.open(encoding="utf-8") as stream:
            settings = json.loads(next(stream, "{}"))
            if (settings.get("mode") != "train" or settings.get("model") != control.MODEL_ID
                    or settings.get("batch_order", "balanced") != order or settings.get("lr") != lr
                    or settings.get("prompt") != INSTRUCTION or settings.get("seed") != TRAIN_SEED
                    or settings.get("max_new_tokens") != MAX_TOKENS):
                continue
            # Both old and new trainers recorded step-zero validation immediately after settings.
            initial = json.loads(next(stream, "{}"))
            recorded = [(r["question"], r["answer"]) for r in initial.get("responses", [])]
            if initial.get("event") != "validation" or initial.get("step") != 0 or recorded != references:
                continue
        for step, path in paths.items():
            if path.is_dir():
                matches[step].append(path)
    if not matches[25] or not matches[100] or any(len(paths) > 1 for paths in matches.values()):
        found = "\n".join(str(p) for paths in matches.values() for p in paths) or "(none)"
        raise ValueError("Expected one checkpoint each at 25 and 100, and at most one at 50. "
                         "Choose exact directories with --checkpoints PATH [PATH ...]. "
                         f"Matching directories:\n{found}")
    return [p for paths in matches.values() for p in paths]


def compare_checkpoints(validation, order, lr, paths):
    paths = paths or discover_checkpoints(order, lr, validation)
    for path in paths:
        if not path.is_dir():
            raise ValueError(f"Checkpoint directory does not exist: {path}")
    tokenizer = AutoTokenizer.from_pretrained(control.MODEL_ID)
    sources = [("original", control.MODEL_ID)]
    names = Counter(path.name for path in paths)
    for path in paths:
        label = path.name if names[path.name] == 1 else f"{path.name} ({path.parent.name})"
        sources.append((label, path))
    rows = []
    for name, source in sources:
        model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32).eval()
        rows.append((name, validate(model, tokenizer, validation)))
        del model  # Load one set of weights at a time on the laptop.
    probability_table(rows)


def train_model(mode, order, lr, train, validation, digest):
    batches = training_batches(train, order)
    limit = 5 if mode == "benchmark" else len(batches)
    tokenizer = AutoTokenizer.from_pretrained(control.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        control.MODEL_ID, dtype=torch.float32).eval()
    # eval() disables dropout while allowing full gradients.
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.0, foreach=False)
    witness = model.model.norm.weight  # small, named trainable vector in SmolLM2
    original = witness.detach().clone()
    runs = control.ROOT / "runs"
    runs.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"d_{mode}_{order}_lr{lr:g}_", dir=runs))
    print(f"Run directory: {out}", flush=True)

    with (out / "events.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
        def log(event):
            stream.write(json.dumps(event) + "\n")

        log({"event": "settings", "mode": mode, "model": control.MODEL_ID,
             "revision": getattr(model.config, "_commit_hash", None), "development_sha256": digest,
             "seed": TRAIN_SEED, "prompt": INSTRUCTION, "lr": lr, "batch_order": order,
             "max_new_tokens": MAX_TOKENS,
             "torch": str(torch.__version__), "transformers": transformers.__version__,
             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        summaries = []
        if mode == "train":
            summaries.append(("original", validate(model, tokenizer, validation, 0, log)))
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
                scores = validate(model, tokenizer, validation, step, log)
                summaries.append((f"step {step}", scores))
                correct = scores["correct"]
                new_best = correct >= best_count  # latest trained checkpoint wins ties
                if step in KEEP or new_best:
                    checkpoint = out / f"checkpoint_{step}"
                    # Publish the directory after both weights and tokenizer have been written.
                    with tempfile.TemporaryDirectory(dir=out, prefix=".saving-") as temp:
                        staged = Path(temp) / checkpoint.name
                        model.save_pretrained(staged)
                        tokenizer.save_pretrained(staged)
                        staged.rename(checkpoint)
                if new_best:
                    selected = {
                        "step": step, "validation_correct": correct, "path": checkpoint.name}
                    pending = out / "best.tmp"
                    pending.write_text(json.dumps(
                        selected) + "\n", encoding="utf-8")
                    pending.replace(out / "best.json")
                    if best_step is not None and best_step not in KEEP:
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
            probability_table(summaries)
        if not changed:
            raise RuntimeError(
                "The tracked parameter did not change. Inspect the update log.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "train", "probabilities"), required=True)
    parser.add_argument("--batch-order", "--batch_order", choices=ORDERS, default="balanced")
    parser.add_argument("--lr", type=float, default=LR, help="AdamW learning rate (default 2e-6)")
    parser.add_argument("--checkpoints", nargs="+", type=Path, metavar="DIR",
                        help="exact checkpoint directories for probabilities; the original is included")
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be positive and finite")
    if args.checkpoints and args.mode != "probabilities":
        parser.error("--checkpoints applies to --mode probabilities")
    torch.set_num_threads(4)
    if args.mode == "probabilities":
        compare_checkpoints(read_validation(), args.batch_order, args.lr, args.checkpoints)
    else:
        train, validation, digest = frozen_data()
        train_model(args.mode, args.batch_order, args.lr, train, validation, digest)
