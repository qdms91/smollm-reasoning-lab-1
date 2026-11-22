"""Final, greedy, inference-only comparison on the 100 original held-out questions."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from collections import Counter
from importlib import import_module
from itertools import groupby
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

c = import_module("05_train_c")
a, d = c.a, c.d
ROOT = d.control.ROOT
# Recorded when the dataset was created, before the first trained checkpoint.
DATA_SHA = "38de04efc4d661f5344c95f639279aea57e9ee09eaed4ad2196db6e3492c4be3"
DEV_SHA = "2a6fbc681fd5ad5991bf8303679467ddb4f94fe395392900e4da2b956e2ad27c"
CONTROLS = {"direct": "Frozen direct", "loop": "B frozen", "plain": "Frozen A plain",
            "scaffold": "Frozen A scaffold"}
# One recorded validation winner per completed training run. The first balanced D
# run used earliest-on-tie; the repeat used latest-on-tie. Report both as related runs.
EXPECTED = {"A plain q0.10", "A scaffold q0.10", "A scaffold q0.40", "C optional"}
EXPECTED.update(f"D {order} {lr:g}" for order in d.ORDERS for lr in (2e-6, 1e-5))
EARLY_D = "D threshold-paired 1e-05"  # Its step-25 validation qCorrect was 0.50425.


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path, text):
    # All writers hold the directory lock. A partial write never replaces a complete file.
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(text, encoding="utf-8")
    pending.replace(path)


def source_hashes():
    files = [d.control.__file__, c.b.__file__, d.__file__, a.__file__, c.__file__, __file__]
    return {Path(p).name: sha(Path(p).read_bytes()) for p in files}


def identify(family, settings):
    if settings["model"] != d.control.MODEL_ID or settings["max_new_tokens"] != d.MAX_TOKENS:
        raise ValueError("Model or token budget differs from the registered evaluation.")
    if settings["seed"] != d.TRAIN_SEED:
        raise ValueError("Unexpected training seed.")
    digest = settings.get("development_sha256") or settings.get("dev_sha256") or settings.get("data_sha256")
    if digest not in (DATA_SHA, DEV_SHA):
        raise ValueError("Training used an unrecognized dataset.")
    if family == "d":
        if settings["prompt"] != d.INSTRUCTION:
            raise ValueError("The direct prompt differs from the training record.")
        return f"D {settings.get('batch_order', 'balanced')} {settings['lr']:g}", "direct"
    if settings["lr"] != 2e-6 or settings["batch_order"] != "balanced":
        raise ValueError("Unexpected A/C training settings.")
    if family == "a":
        arm = settings["arm"]
        if settings["instruction"] != a.INSTRUCTION or (arm == "scaffold" and settings["example"] != a.EXAMPLE):
            raise ValueError("Method A prompt differs from the training record.")
        return f"A {arm} q{settings['question_bonus']:.2f}", arm
    if (settings["first_prompt"], settings["follow_up"], settings["first_limit"]) != (c.FIRST, c.FOLLOW_UP, c.FIRST_LIMIT):
        raise ValueError("Method C prompts or first-turn limit differ from the training record.")
    return "C optional", "loop"


def make_plan():
    """Use only training metadata and validation scores. This routine never opens the split file."""
    cases = [dict(name=name, protocol=p, checkpoint=None) for p, name in CONTROLS.items()]
    seen = set()
    for events in sorted((ROOT / "runs").glob("*_train_*/events.jsonl")):
        folder = events.parent
        family = folder.name.split("_train_", 1)[0]
        if family not in ("a", "c", "d") or not (folder / "best.json").is_file():
            continue
        with events.open(encoding="utf-8") as stream:
            settings = json.loads(next(stream))
            scores = {r["step"]: r["correct"] for r in map(json.loads, stream)
                      if r["event"] == "validation" and r["step"] > 0}
        if settings.get("mode") != "train" or 100 not in scores:
            continue
        label, protocol = identify(family, settings)
        if label not in EXPECTED:
            raise ValueError(f"Unregistered training condition in {folder}: {label}")
        best = read_json(folder / "best.json")
        step = best["step"]
        if (best["path"] != f"checkpoint_{step}" or scores.get(step) != best["validation_correct"]
                or scores[step] != max(scores.values())):
            raise ValueError(f"Recorded winner disagrees with the validation log in {folder}.")

        steps = [(step, "recorded validation winner")]
        if label == EARLY_D and step != 25:
            steps.append((25, "preselected validation probability diagnostic"))
        for selected, reason in steps:
            checkpoint = folder / f"checkpoint_{selected}"
            cases.append(dict(name=f"{label} @{selected}", protocol=protocol,
                              checkpoint=str(checkpoint.relative_to(ROOT)), step=selected,
                              validation_correct=scores[selected], selection=reason))
        seen.add((label, step))

    missing = EXPECTED - {label for label, step in seen}
    if missing:
        raise FileNotFoundError("Missing completed training records in runs/: " + ", ".join(sorted(missing)))
    # Include both actual saved winners of the repeated first D experiment.
    for step in (25, 100):
        if ("D balanced 2e-06", step) not in seen:
            raise FileNotFoundError(f"Missing the original balanced D run with the step-{step} winner.")
    # Replicate runs can have the same label/step; never overwrite or silently merge a row.
    counts = Counter(x["name"] for x in cases)
    for case in cases:
        if counts[case["name"]] > 1:
            case["name"] += f" ({Path(case['checkpoint']).parent.name})"
    return dict(dataset="data/arithmetic_v1.jsonl", dataset_sha256=DATA_SHA,
                endpoint="native greedy Boolean accuracy; invalid counts as wrong",
                model=d.control.MODEL_ID, script_sha256=source_hashes(), cases=cases)


def require_weights(cases):
    missing = []
    for case in cases:
        if case["checkpoint"] is None:
            continue
        folder = ROOT / case["checkpoint"]
        weights = [folder / "model.safetensors"]
        index = folder / "model.safetensors.index.json"
        if index.is_file():
            weights = [folder / n for n in set(read_json(index)["weight_map"].values())]
        if not weights or not all(p.is_file() for p in [folder / "config.json", *weights]):
            missing.append(str(folder))
    if missing:
        raise FileNotFoundError("Missing saved checkpoint files. Expected:\n" + "\n".join(missing))


def read_test(plan):
    data = (ROOT / plan["dataset"]).read_bytes()
    if sha(data) != plan["dataset_sha256"]:
        raise ValueError("The split file does not match the SHA-256 from the original experiment.")
    items = [r for r in map(json.loads, data.splitlines()) if r["split"] == "test"]
    if len({r["question"] for r in items}) != 100 or Counter(r["answer"] for r in items) != {"Yes": 50, "No": 50}:
        raise ValueError("Expected the original 100 distinct, balanced test questions.")
    return items


def trial(model, tokenizer, item, case):
    # Call the same inference function, with the same prompt and parser, as validation.
    # In A and C the reference is used by their scorer only after generation.
    protocol = case["protocol"]
    started = perf_counter()
    if protocol == "direct":
        _, _, reply = d.rollout(model, tokenizer, item["question"], sample=False, probabilities=True)
    elif protocol == "loop":
        reply, _ = c.episode(model, tokenizer, item, sample=False)
    else:
        _, _, reply = a.response(model, tokenizer, item, protocol, sample=False)
    seconds = perf_counter() - started
    turns = [t["text"] for t in reply["turns"]] if protocol == "loop" else [reply["text"]]
    # This audit never changes a route, the native prediction, or the denominator.
    routed = len(turns) == 2
    echo = routed and turns[0].strip() == item["question"]
    final_audit = c.grade_final(turns[-1], item)
    prediction = reply["prediction"]
    record = dict(case=case["name"], question=item["question"], reference=item["answer"],
                  turns=turns, prediction=prediction, correct=prediction == item["answer"],
                  second_turn=routed, exact_echo=echo, asked_question=reply.get("asked_question"),
                  targeted_question=reply.get("targeted_question"),
                  verified_sum=reply.get("verified_sum", final_audit["verified_sum"]),
                  literal_sum=final_audit["literal_sum_in_final_text"], tokens=reply["tokens"], seconds=seconds)
    if protocol == "direct":
        record.update({k: reply[k] for k in ("q_yes", "q_no", "raw_yes", "raw_no")})
    return record


def summarize(case, rows, controls):
    n = len(rows)
    pred = Counter(r["prediction"] for r in rows)
    hits = [sum(r["correct"] and r["reference"] == k for r in rows) for k in ("Yes", "No")]
    counts = {k: sum(r[k] for r in rows) for k in ("second_turn", "exact_echo", "verified_sum", "literal_sum")}
    for key in ("asked_question", "targeted_question"):
        counts[key] = None if case["protocol"] == "direct" else sum(r[key] for r in rows)
    counts["correct_after_second_turn"] = sum(r["second_turn"] and r["correct"] for r in rows)
    counts["sum_and_final_correct"] = sum(r["verified_sum"] and r["correct"] for r in rows)
    matched = CONTROLS[case["protocol"]]
    base = {r["question"]: r["correct"] for r in controls[matched]}
    paired = c.paired_changes(base, {r["question"]: r["correct"] for r in rows})
    result = dict(condition=case["name"], matched_control=matched, n=n, correct=sum(hits),
                  accuracy=sum(hits)/n, correct_true_yes=hits[0], correct_true_no=hits[1],
                  predicted_yes=pred["Yes"], predicted_no=pred["No"], invalid=pred[None],
                  **counts, **paired, mean_tokens=sum(r["tokens"] for r in rows)/n,
                  mean_seconds=sum(r["seconds"] for r in rows)/n,
                  q_correct=None, q_correct_true_yes=None, q_correct_true_no=None)
    if case["protocol"] == "direct":
        yes = [r["q_yes"] for r in rows if r["reference"] == "Yes"]
        no = [r["q_no"] for r in rows if r["reference"] == "No"]
        result.update(q_correct=(sum(yes) + sum(no))/n, q_correct_true_yes=sum(yes)/len(yes),
                      q_correct_true_no=sum(no)/len(no))
    return result


def evaluate(out):
    manifest = out / "manifest.json"
    plan = read_json(manifest) if manifest.exists() else make_plan()
    if plan["script_sha256"] != source_hashes():
        raise ValueError("The evaluation scripts differ from the saved manifest. Restore the versions used for this test.")
    cases = plan["cases"]
    raw = out / "responses.jsonl"
    completed = {}
    if raw.exists():
        if not manifest.exists():
            raise ValueError("Responses exist without their manifest. Restore the manifest to resume.")
        with raw.open(encoding="utf-8") as stream:
            for r in map(json.loads, stream):
                completed.setdefault(r["case"], []).append(r)
    require_weights([x for x in cases if x["name"] not in completed])
    if not manifest.exists():
        write_atomic(manifest, json.dumps(plan, indent=2) + "\n")
    print(f"Fixed {len(cases)} conditions in {manifest}.", flush=True)
    items = read_test(plan)
    expected = [(r["question"], r["answer"]) for r in items]
    if completed.keys() - {x["name"] for x in cases} or any(
        [(r["question"], r["reference"]) for r in group] != expected for group in completed.values()
    ):
        raise ValueError("Saved responses do not match the locked condition list and test rows.")

    torch.set_num_threads(4)
    tokenizer = None
    print(f"{'Condition':<35}  correct  true Yes/No  invalid  2nd/copy", flush=True)
    summaries = []
    for checkpoint, group in groupby(cases, key=lambda x: x["checkpoint"]):
        model = None
        for case in group:
            name = case["name"]
            if name not in completed:
                print(f"Evaluating {name} ...", flush=True)
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(d.control.MODEL_ID, local_files_only=True)
                if model is None:
                    source = ROOT / checkpoint if checkpoint else d.control.MODEL_ID
                    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32, local_files_only=True).eval()
                with torch.inference_mode():
                    completed[name] = [trial(model, tokenizer, item, case) for item in items]
                # Commit each complete condition. An interrupted condition can be run again.
                write_atomic(raw, "".join(json.dumps(r) + "\n" for batch in completed.values() for r in batch))
            s = summarize(case, completed[name], completed)
            summaries.append(s)
            print(f"{name:<35}  {s['correct']:3}/{s['n']:<3}  {s['correct_true_yes']:5}/{s['correct_true_no']:<3} "
                  f"{s['invalid']:7}  {s['second_turn']:3}/{s['exact_echo']}", flush=True)
        del model
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=summaries[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(summaries)
    write_atomic(out / "summary.csv", table.getvalue())
    print(f"Final comparison: {out / 'summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="show selections without reading the test split or loading weights")
    args = parser.parse_args()
    out = ROOT / "results" / "final_test"
    if args.plan:
        manifest = out / "manifest.json"
        plan = read_json(manifest) if manifest.exists() else make_plan()
        for case in plan["cases"]:
            print(f"{case['name']:<35} {case['checkpoint'] or d.control.MODEL_ID}")
        return
    out.mkdir(parents=True, exist_ok=True)
    # Linux laptop: allow one writer, including during a resume.
    handle = os.open(out, os.O_RDONLY)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate(out)
    finally:
        os.close(handle)


if __name__ == "__main__":
    main()
