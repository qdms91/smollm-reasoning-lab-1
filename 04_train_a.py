"""Method A: one generated response, two prompt conditions, one REINFORCE reward."""
import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
from importlib import import_module
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

d = import_module("03_train_d")
INSTRUCTION = ("Ask yourself one short question about the sum using both numbers. Answer it with an addition "
               "equation on its own line. Then give your final Boolean answer in words.")
EXAMPLE = "Example of a self-question: What is the sum of the two numbers?"
ARMS = ("plain", "scaffold")
QUESTION_BONUS, SUM_BONUS = 0.40, 0.20


def prompt(question, arm):
    lines = [INSTRUCTION]
    if arm == "scaffold":
        lines.append(EXAMPLE)
    return "\n".join([*lines, question])


def score(text, item):
    """Conservative observable proxies. Only work preceding the decision earns a bonus."""
    a, b, _ = map(int, d.EXPRESSION.fullmatch(item["question"]).groups())
    prediction = d.boolean_answer(text)
    correct = prediction == item["answer"]
    clean = re.sub(r"[*_`]", "", text)
    decision = re.search(r"\b(?:yes|no)\b", clean, re.I)
    work = clean[:decision.start()] if decision else clean
    questions = re.findall(r"[^.!?\n]+\?", work)
    # This domain admits an objectively identifiable subproblem: add these operands.
    # A verbatim repetition of the original comparison does not ask for that value.
    pairs = ((a, b), (b, a))
    relevant = False
    for x, y in pairs:
        operands = rf"(?:{x}\s*\+\s*{y}|(?:the\s+)?sum of {x} and {y})"
        query = rf"\b(?:what(?:\s+is|'s)(?: the (?:value|result) of)?|how much is)\s+{operands}\s*\?"
        relevant |= any(re.search(query, q, re.I)
                        is not None for q in questions)

    # Accept explicit integer equations, ignoring repeated copies of the same result.
    # Do not credit a threshold, an echoed question, or the right number in other prose.
    results = set()
    for match in re.finditer(
        r"^\s*(?:\d+[.)]\s+)?(?:[Aa]nswer:\s*)?(\d+)\s*\+\s*(\d+)\s*=\s*(-?\d+)[ \t]*\.?[ \t]*$",
        work, re.M,
    ):
        x, y, result = map(int, match.groups())
        if (x, y) in pairs:
            results.add(result)
    verified = results == {a + b}
    return dict(prediction=prediction, correct=correct, asked_question=bool(questions),
                targeted_question=relevant, verified_sum=verified,
                reward=float(correct) + QUESTION_BONUS * relevant + SUM_BONUS * verified)


def response(model, tokenizer, item, arm, sample):
    messages = [{"role": "user", "content": prompt(item["question"], arm)}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
    sampling = dict(temperature=1.0, top_k=0, top_p=1.0) if sample else {}
    with torch.no_grad():
        ids = model.generate(**inputs, do_sample=sample, num_beams=1, use_cache=True,
                             max_new_tokens=d.MAX_TOKENS, repetition_penalty=1.0, **sampling)
    start = inputs["input_ids"].shape[1]
    text = tokenizer.decode(ids[0, start:], skip_special_tokens=True)
    row = {**item, "text": text,
           "tokens": ids.shape[1] - start, **score(text, item)}
    return ids, start, row


def update(model, tokenizer, optimizer, batch, arm):
    optimizer.zero_grad(set_to_none=True)
    rows = []
    for item in batch:
        ids, start, row = response(model, tokenizer, item, arm, sample=True)
        ids = ids.clone()
        logits = model(input_ids=ids[:, :-1], use_cache=False).logits
        nll = F.cross_entropy(logits[0, start-1:],
                              ids[0, start:], reduction="sum")
        ((row["reward"] - 0.5) * nll / len(batch)).backward()
        del logits, nll
        rows.append(row)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return rows, float(norm)


def validate(model, tokenizer, items, arm, step, log):
    rows = [response(model, tokenizer, item, arm, False)[2] for item in items]
    counts = Counter(r["prediction"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    totals = {k: sum(r[k] for r in rows) for k in (
        "asked_question", "targeted_question", "verified_sum")}
    joint = sum(r["verified_sum"] and r["correct"] for r in rows)
    per_label = [sum(r["correct"] and r["answer"] == label for r in rows)
                 for label in d.LABELS]
    print(f"{arm} step {step} | correct {correct}/{len(rows)} (true Yes/No {per_label}) | "
          f"outputs Yes/No/invalid {counts['Yes']}/{counts['No']}/{counts[None]} | "
          f"question/targeted/sum {totals['asked_question']}/{totals['targeted_question']}/{totals['verified_sum']} "
          f"| sum and final correct {joint}", flush=True)
    log(dict(event="validation", arm=arm, step=step, correct=correct, **totals,
             sum_and_final_correct=joint, responses=rows))
    return correct


def run(mode, arm):
    torch.set_num_threads(4)
    # This project already has D's frozen data. Never create or score a test set here.
    validation = d.read_validation()
    root = d.control.ROOT
    train, _, digest = d.frozen_data()
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(dir=runs, prefix=f"a_{mode}_{arm or 'both'}_"))
    print(f"Run: {out}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(d.control.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        d.control.MODEL_ID, dtype=torch.float32).eval()
    with (out / "events.jsonl").open("w", buffering=1, encoding="utf-8") as events:
        def log(record):
            events.write(json.dumps(record) + "\n")

        log(dict(event="settings", mode=mode, arm=arm, model=d.control.MODEL_ID,
                 seed=d.TRAIN_SEED, lr=d.LR, batch_order="balanced", dev_sha256=digest,
                 instruction=INSTRUCTION, example=EXAMPLE, max_new_tokens=d.MAX_TOKENS,
                 question_bonus=QUESTION_BONUS, sum_bonus=SUM_BONUS, final_reward=1.0))
        if mode == "controls":
            for condition in ARMS:
                validate(model, tokenizer, validation, condition, 0, log)
            return
        validate(model, tokenizer, validation, arm, 0, log)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=d.LR, weight_decay=0., foreach=False)
        torch.manual_seed(d.TRAIN_SEED)
        best_count, best_path = -1, None
        for step, batch in enumerate(d.training_batches(train, "balanced"), 1):
            rows, norm = update(model, tokenizer, optimizer, batch, arm)
            log(dict(event="update", step=step, grad_norm=norm, responses=rows))
            if step % 5 == 0:
                print(f"Update {step}/100", flush=True)
            if step in d.CHECKS:
                correct = validate(
                    model, tokenizer, validation, arm, step, log)
                if correct >= best_count:
                    path = out / f"checkpoint_{step}"
                    with tempfile.TemporaryDirectory(dir=out) as temp:
                        staged = Path(temp) / "model"
                        model.save_pretrained(staged)
                        tokenizer.save_pretrained(staged)
                        staged.rename(path)
                    info = dict(
                        step=step, validation_correct=correct, path=path.name)
                    tmp = out / "best.tmp"
                    tmp.write_text(json.dumps(info) + "\n", encoding="utf-8")
                    tmp.replace(out / "best.json")
                    if best_path is not None:
                        shutil.rmtree(best_path)
                    best_count, best_path = correct, path
        # Reuse D's exact prompt, parser and next-token diagnostic for a common readout.
        del optimizer, model
        best = AutoModelForCausalLM.from_pretrained(
            best_path, dtype=torch.float32).eval()
        direct = d.validate(best, tokenizer, validation)
        log(dict(event="direct_prompt_transfer",
            checkpoint=best_path.name, **direct))
        print(f"{best_path.name} under D's direct prompt | correct {direct['correct']}/50 | "
              f"qCorrect {direct['q_correct']:.5f} | qC on Yes/No "
              f"{direct['yes_class']:.5f}/{direct['no_class']:.5f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("controls", "train"), required=True)
    parser.add_argument("--arm", choices=ARMS,
                        help="required for training; controls evaluates both")
    args = parser.parse_args()
    if (args.mode == "train") != bool(args.arm):
        parser.error("Use --arm plain or --arm scaffold for training only.")
    run(args.mode, args.arm)
