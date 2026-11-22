"""Method C. Train one optional two-turn policy and compare it with its frozen baseline."""
import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
from importlib import import_module
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

b = import_module("02_baseline_inference")
d = import_module("03_train_d")
a = import_module("04_train_a")

FIRST = ("Either ask yourself one short question about the sum using both numbers, "
         "or answer the original problem directly in words. If asking, write only the question.")
FOLLOW_UP = b.FOLLOW_UP + " Write the addition equation on its own line before your final answer."
GENERIC = "What should I check before answering?"
SPECIFIC = "What is {x} + {y}?"
FIRST_LIMIT = 32
# Fix the original A reward independently of a locally edited A shaping run.
QUESTION_BONUS, SUM_BONUS, BASELINE = 0.10, 0.20, 0.50


def first_history(problem):
    return [{"role": "user", "content": f"{problem}\n{FIRST}"}]


def second_history(history, question):
    return history + [{"role": "assistant", "content": question},
                      {"role": "user", "content": FOLLOW_UP}]


def single_question(text):
    # The bare form and the ASK-prefixed form take the same branch.
    body = re.sub(r"^ASK\b[ \t:]*", "", text.strip(), flags=re.I)
    return bool(re.fullmatch(r"[^.!?\r\n]+\?", body)
                and re.search(r"[A-Za-z]", body)
                and not re.match(r"(?:yes|no|final|ask)\b", body, re.I)
                and not re.search(r"\bFINAL[ \t:]+(?:yes|no)\b", body, re.I)
                and d.boolean_answer(body[:-1]) is None)


def grade_final(text, item, *, first_call=False):
    parsed = a.score(text, item)
    # Call one must choose an action. Call two may restate the question before answering.
    # An explicit new ASK or a trailing question cannot request a third model call.
    invalid = ((first_call and "?" in text) or text.rstrip().endswith("?")
               or re.search(r"^\s*ASK\b", text, re.I | re.M))
    prediction = None if invalid else parsed["prediction"]
    x, y, _ = map(int, d.EXPRESSION.fullmatch(item["question"]).groups())
    equations = re.findall(r"(?<![\w.+*/-])(\d+)\s*\+\s*(\d+)\s*=\s*(-?\d+)(?![\w+*/-]|\.\d)", text)
    # Descriptive only: an inline equation after an answer can be arithmetically right.
    literal_sum = any((int(u), int(v)) in ((x, y), (y, x)) and int(n) == x + y
                      for u, v, n in equations)
    return {"prediction": prediction, "correct": prediction == item["answer"],
            "verified_sum": bool(not invalid and parsed["verified_sum"]),
            "literal_sum_in_final_text": literal_sum}


@torch.no_grad()
def generate(model, tokenizer, messages, limit, sample):
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
    options = {"temperature": 1.0, "top_k": 0, "top_p": 1.0} if sample else {}
    begin = perf_counter()
    ids = model.generate(**inputs, do_sample=sample, num_beams=1, max_new_tokens=limit,
                         repetition_penalty=1.0, use_cache=True, **options)
    seconds = perf_counter() - begin
    start = inputs["input_ids"].shape[1]
    completion = ids[0, start:]
    turn = {"messages": messages, "text": tokenizer.decode(completion, skip_special_tokens=True),
            "tokens": len(completion), "limit": limit, "seconds": seconds}
    return turn, (ids, start)


def episode(model, tokenizer, item, sample):
    history = first_history(item["question"])
    first, action = generate(model, tokenizer, history, FIRST_LIMIT, sample)
    turns, actions = [first], [action]
    asked = single_question(first["text"])
    if asked:
        # Route on the generated text alone. Even an irrelevant question gets a second turn.
        history = second_history(history, first["text"])
        final, action = generate(model, tokenizer, history, d.MAX_TOKENS - first["tokens"], sample)
        turns.append(final)
        actions.append(action)

    final_scores = grade_final(turns[-1]["text"], item, first_call=not asked)
    targeted = asked and a.score(first["text"], item)["targeted_question"]
    final_reward = float(final_scores["correct"]) + SUM_BONUS * final_scores["verified_sum"]
    reward = QUESTION_BONUS * targeted + final_reward
    # The first action can influence the entire outcome. The second cannot earn the first bonus.
    turns[0]["return_to_go"] = reward
    if asked:
        turns[1]["return_to_go"] = final_reward
    row = {**item, **final_scores, "asked_question": asked, "targeted_question": targeted,
           "reward": reward, "tokens": sum(t["tokens"] for t in turns), "turns": turns}
    return row, actions


def update(model, tokenizer, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    rows = []
    for item in batch:
        row, actions = episode(model, tokenizer, item, sample=True)
        for (ids, start), turn in zip(actions, row["turns"], strict=True):
            ids = ids.clone()
            logits = model(input_ids=ids[:, :-1], use_cache=False).logits
            # The earlier assistant reply is context in call two. Score its tokens only in call one.
            nll = F.cross_entropy(logits[0, start-1:, :], ids[0, start:], reduction="sum")
            ((turn["return_to_go"] - BASELINE) * nll / len(batch)).backward()
            del logits, nll
        rows.append(row)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return rows, float(norm)


def paired_changes(before, after):
    """Count corrections and regressions on the same questions, including invalids as wrong."""
    if before.keys() != after.keys():
        raise ValueError("The validation questions changed between conditions.")
    fixed = sum(after[q] and not before[q] for q in before)
    spoiled = sum(before[q] and not after[q] for q in before)
    return {"fixed": fixed, "spoiled": spoiled}


def validate(model, tokenizer, items, step, log, frozen=None):
    rows = [episode(model, tokenizer, item, sample=False)[0] for item in items]
    outcomes = {r["question"]: r["correct"] for r in rows}
    counts = Counter(r["prediction"] for r in rows)
    correct = sum(outcomes.values())
    asked = sum(r["asked_question"] for r in rows)
    targeted = sum(r["targeted_question"] for r in rows)
    after_ask = sum(r["correct"] and r["asked_question"] for r in rows)
    verified = sum(r["verified_sum"] for r in rows)
    anywhere = sum(r["literal_sum_in_final_text"] for r in rows)
    joint = sum(r["verified_sum"] and r["correct"] for r in rows)
    hits = {label: sum(r["correct"] and r["answer"] == label for r in rows) for label in d.LABELS}
    change = paired_changes(frozen, outcomes) if frozen is not None else {}
    label = "B frozen" if step == 0 else f"C step {step}"
    conditional = f"{after_ask}/{asked}" if asked else "n/a"
    print(f"{label} | correct {correct}/{len(rows)} (true Yes/No {hits['Yes']}/{hits['No']}) | "
          f"outputs Yes/No/invalid {counts['Yes']}/{counts['No']}/{counts[None]} | "
          f"asked {asked}, targeted {targeted}, correct after asking {conditional}\n"
          f"  Credited sums {verified}, literal sums in final replies {anywhere}, sum and final correct {joint}"
          + (f" | fixed/spoiled vs B {change['fixed']}/{change['spoiled']}" if change else ""),
          flush=True)
    log({"event": "validation", "step": step, "correct": correct, "asked": asked,
         "targeted": targeted, "correct_after_ask": after_ask, "verified_sum": verified,
         "literal_sum_in_final_text": anywhere, "correct_by_label": hits,
         "sum_and_final_correct": joint, **change, "responses": rows})
    return correct, outcomes


def cue_check(model, tokenizer, items, step, log):
    """Read-only paired intervention. Supply a first reply, then score only the model's final turn."""
    records = []
    for item in items:
        x, y, _ = d.EXPRESSION.fullmatch(item["question"]).groups()
        history = first_history(item["question"])
        record = dict(item)
        for name, cue in (("generic", GENERIC), ("specific", SPECIFIC.format(x=x, y=y))):
            messages = second_history(history, cue)
            turn, _ = generate(model, tokenizer, messages, d.MAX_TOKENS - FIRST_LIMIT, sample=False)
            record[name] = {"supplied_question": cue, **grade_final(turn["text"], item), "turn": turn}
        records.append(record)
    by_cue = {name: {r["question"]: r[name]["correct"] for r in records}
              for name in ("generic", "specific")}
    changes = paired_changes(by_cue["generic"], by_cue["specific"])
    sums = {name: sum(r[name]["verified_sum"] for r in records) for name in by_cue}
    ns = len(items)
    print(f"Cue check step {step} | generic {sum(by_cue['generic'].values())}/{ns}, "
          f"operand-specific {sum(by_cue['specific'].values())}/{ns} | "
          f"fixed {changes['fixed']}, spoiled {changes['spoiled']} | "
          f"credited sums generic/specific {sums['generic']}/{sums['specific']}", flush=True)
    log({"event": "cue_check", "step": step, **changes, "verified_sum_by_cue": sums, "responses": records})


def save_best(model, tokenizer, out, step, correct, previous):
    path = out / f"checkpoint_{step}"
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        staged = Path(tmp) / "model"
        model.save_pretrained(staged)
        tokenizer.save_pretrained(staged)
        staged.rename(path)
    pending = out / "best.tmp"
    pending.write_text(json.dumps(dict(step=step, validation_correct=correct, path=path.name))
                       + "\n", encoding="utf-8")
    pending.replace(out / "best.json")
    if previous is not None:
        shutil.rmtree(previous)
    return path


def run(mode):
    torch.set_num_threads(4)
    validation = d.read_validation()  # Requires the existing split. Stops before the held-out rows.
    train, _, digest = d.frozen_data() if mode == "train" else (None, None, None)
    runs = d.control.ROOT / "runs"
    runs.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"c_{mode}_", dir=runs))
    print(f"Run directory: {out}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(d.control.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(d.control.MODEL_ID, dtype=torch.float32).eval()
    with (out / "events.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
        def log(event):
            stream.write(json.dumps(event) + "\n")

        log(dict(event="settings", mode=mode, model=d.control.MODEL_ID, seed=d.TRAIN_SEED,
                 lr=d.LR, batch_order="balanced", dev_sha256=digest, first_prompt=FIRST,
                 follow_up=FOLLOW_UP, first_limit=FIRST_LIMIT, max_new_tokens=d.MAX_TOKENS,
                 question_bonus=QUESTION_BONUS, sum_bonus=SUM_BONUS, correct_reward=1.0,
                 baseline=BASELINE, generic_cue=GENERIC, specific_cue=SPECIFIC))
        _, frozen = validate(model, tokenizer, validation, 0, log)
        cue_check(model, tokenizer, validation, 0, log)
        if mode == "controls":
            return

        model.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=d.LR, weight_decay=0., foreach=False)
        torch.manual_seed(d.TRAIN_SEED)
        batches = d.training_batches(train, "balanced")
        best_count, best_step, best_path = -1, None, None
        sampled = Counter()
        for step, batch in enumerate(batches, 1):
            started = perf_counter()
            rows, norm = update(model, tokenizer, optimizer, batch)
            log(dict(event="update", step=step, grad_norm=norm, seconds=perf_counter()-started,
                     responses=rows))
            for r in rows:
                sampled.update(trials=1, asked=int(r["asked_question"]), targeted=int(r["targeted_question"]),
                               correct_after_ask=int(r["asked_question"] and r["correct"]))
            if step % 5 == 0:
                print(f"Update {step}/{len(batches)}", flush=True)
            if step in d.CHECKS:
                correct, _ = validate(model, tokenizer, validation, step, log, frozen)
                if correct >= best_count:
                    best_path = save_best(model, tokenizer, out, step, correct, best_path)
                    best_step, best_count = step, correct

        log(dict(event="sampled_training", **sampled))
        print(f"Sampled training | asked {sampled['asked']}/{sampled['trials']}, "
              f"targeted {sampled['targeted']}, correct after asking {sampled['correct_after_ask']}",
              flush=True)
        del optimizer
        if best_step != len(batches):
            del model
            model = AutoModelForCausalLM.from_pretrained(best_path, dtype=torch.float32).eval()
        print(f"Selected {best_path.name} | validation {best_count}/{len(validation)}", flush=True)
        cue_check(model, tokenizer, validation, best_step, log)
        direct = d.validate(model, tokenizer, validation)
        log(dict(event="direct_prompt_transfer", checkpoint=best_path.name, **direct))
        print(f"D's direct prompt | correct {direct['correct']}/{direct['size']} | "
              f"qCorrect {direct['q_correct']:.5f} | qC on Yes/No "
              f"{direct['yes_class']:.5f}/{direct['no_class']:.5f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("controls", "train"),
                        help="train includes the frozen controls before the first update")
    run(parser.parse_args().mode)
