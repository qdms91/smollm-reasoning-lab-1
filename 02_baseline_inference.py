"""Method B, an instruction-order comparison, and a four-probe diagnostic."""

import argparse
import json
import re
from functools import partial
from importlib import import_module

# A filename beginning with a digit can be imported by name with import_module.
# Importing 01 defines its functions. It runs the control only when launched directly.
control = import_module("01_baseline_inference")

ASK_OPTION = (
    "Alternatively, you may write ASK followed by one short question "
    "for yourself that would help you solve the problem."
)
FOLLOW_UP = "Use your question to solve the original problem. Give your final answer now."
REQUEST_QUESTION = (
    "Write one short question for yourself that would help you solve the problem. "
    "Start the line with ASK and end it with a question mark. "
    "Leave the final answer for your next turn."
)


def is_ask(text):
    # Require one ASK line with a question mark. Reject mixed ASK and FINAL answers.
    question = re.fullmatch(
        r"ASK[ \t]+[^\s?][^?\r\n]*\?", text.strip(), re.IGNORECASE)
    mixed = re.search(r"\bFINAL[ \t]+(?:Yes|No)\b", text, re.IGNORECASE)
    return bool(question and not mixed)


def final_turn(generate, messages, first, instruction):
    # Build a fresh history so two follow-ups can use exactly the same first reply.
    history = messages + [
        {"role": "assistant", "content": first["text"]},
        {"role": "user", "content": f"{FOLLOW_UP}\n{instruction}"},
    ]
    return generate(history, max_new_tokens=control.TOKEN_BUDGET - first["tokens"])


def loop_trial(generate, question, instruction, *, ask_first=False):
    # The two optional modes move one whole sentence. All words stay the same.
    lines = [question, ASK_OPTION, instruction] if ask_first else [
        question, instruction, ASK_OPTION]
    messages = [{"role": "user", "content": "\n".join(lines)}]
    first = generate(messages, max_new_tokens=32)
    if not is_ask(first["text"]):
        # The shared scorer recognizes FINAL or records an invalid reply.
        return [first]

    return [first, final_turn(generate, messages, first, instruction)]


def compare_with_saved(rows, filename):
    path = control.ROOT / "runs" / filename
    if not path.exists():
        print(f"Paired comparison skipped because {path.name} is missing.")
        return

    def key(row):
        return row["question"], row["order"], row["reference"]

    old_rows = control.read_jsonl(path)
    previous = {key(row): row["correct"] for row in old_rows}
    if len(previous) != len(old_rows) or previous.keys() != {key(row) for row in rows}:
        print(
            f"Paired comparison skipped because {path.name} contains different trials.")
        return

    print(f"\nCompared with {filename} (an invalid final counts as incorrect)")
    for order in control.INSTRUCTIONS:
        group = [row for row in rows if row["order"] == order]
        fixed = spoiled = 0
        for row in group:
            was_correct = previous[key(row)]
            if row["correct"] and not was_correct:
                fixed += 1
            elif was_correct and not row["correct"]:
                spoiled += 1
        print(f"{order}: fixed {fixed}, spoiled {spoiled}")


def requested_question_check():
    items = control.read_jsonl(control.ROOT / "data" / "questions.jsonl")
    probes = [item for item in items if item["split"] == "probe"][:4]
    if not probes:
        raise ValueError(
            "The diagnostic needs at least one probe in data/questions.jsonl.")
    generate = control.load_generator()
    records = []
    for item in probes:
        question = item["question"]
        messages = [
            {"role": "user", "content": f"{question}\n{REQUEST_QUESTION}"}]
        first = generate(messages, max_new_tokens=32)
        print(f'{question} | requested question | {first["text"]!r}')
        finals = {}
        text = first["text"].strip()
        valid_question = is_ask(text) or (
            text.endswith("?")
            and "\n" not in text
            and not re.search(r"\bFINAL\b", text, re.IGNORECASE)
        )

        if valid_question:
            # Each option order starts from the same model-generated question.
            for order, instruction in control.INSTRUCTIONS.items():
                reply = final_turn(generate, messages, first, instruction)
                prediction = control.parse_final(reply["text"])
                finals[order] = {"turn": reply, "prediction": prediction,
                                 "correct": prediction == item["answer"]}
                print(f'  {order} | {reply["text"]!r}')
        records.append({"question": question, "reference": item["answer"],
                        "first_turn": first, "finals": finals})

    path = control.ROOT / "runs" / "method_b_requested_no_prefix.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    completed = [row for row in records if row["finals"]]
    print(
        f"\nSaved {path.name}. Valid questions: {len(completed)}/{len(probes)}.")
    if completed:
        for order in control.INSTRUCTIONS:
            finals = [row["finals"][order] for row in completed]
            correct = sum(reply["correct"] for reply in finals)
            invalid = sum(reply["prediction"] is None for reply in finals)
            print(
                f"{order}: correct {correct}/{len(finals)} after a valid ASK, invalid {invalid}")
    else:
        print("There were no second turns to score.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("original", "ask-first", "request-question"),
                        default="original")
    mode = parser.parse_args().mode
    if mode == "request-question":
        requested_question_check()
    elif mode == "ask-first":
        results = control.run_probes(
            partial(loop_trial, ask_first=True), "method_b_ask_first.jsonl")
        compare_with_saved(results, "method_b.jsonl")
    else:
        results = control.run_probes(loop_trial, "method_b.jsonl")
        compare_with_saved(results, "direct.jsonl")
