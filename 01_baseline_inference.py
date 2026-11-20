"""Direct-answer control. Script 02 reuses the same generator and scorer."""

import json
import re
from pathlib import Path
from time import perf_counter

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
TOKEN_BUDGET = 64
INSTRUCTIONS = {
    "Yes-first": 'Write exactly one line. Use either "FINAL Yes" or "FINAL No".',
    "No-first": 'Write exactly one line. Use either "FINAL No" or "FINAL Yes".',
}


def load_generator():
    """Load the model once. Return a function that generates one assistant turn."""
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32).eval()
    print(f"{MODEL_ID} | PyTorch {torch.__version__} | Transformers {transformers.__version__}")

    def generate(messages, max_new_tokens):
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        start = perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        seconds = perf_counter() - start
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return {
            "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
            "tokens": len(new_tokens),
            "seconds": round(seconds, 4),
        }

    return generate


def parse_final(text):
    match = re.fullmatch(r"FINAL[ \t]+(Yes|No)\.?",
                         text.strip(), re.IGNORECASE)
    return match.group(1).capitalize() if match else None


def direct_trial(generate, question, instruction):
    messages = [{"role": "user", "content": f"{question}\n{instruction}"}]
    return [generate(messages, TOKEN_BUDGET)]


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def run_probes(trial, output_name):
    items = read_jsonl(ROOT / "data" / "questions.jsonl")
    probes = [item for item in items if item["split"] == "probe"]
    if not probes or len({item["question"] for item in probes}) != len(probes):
        raise ValueError(
            "Supply at least one probe and give each probe a unique question.")
    if any(item["answer"] not in ("Yes", "No") for item in probes):
        raise ValueError("Each probe needs a reference answer of Yes or No.")

    generate = load_generator()
    rows = []
    for order, instruction in INSTRUCTIONS.items():
        for item in probes:
            # Only these two strings reach the model. The reference stays in the scorer.
            turns = trial(generate, item["question"], instruction)
            prediction = parse_final(turns[-1]["text"])
            rows.append({
                "order": order, "question": item["question"], "reference": item["answer"],
                "turns": turns, "prediction": prediction, "correct": prediction == item["answer"],
                "total_tokens": sum(turn["tokens"] for turn in turns),
                "generation_seconds": round(sum(turn["seconds"] for turn in turns), 4),
            })
            replies = " -> ".join(repr(turn["text"]) for turn in turns)
            print(f'{order} | {item["question"]} | {replies}')

    path = ROOT / "runs" / output_name
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"\nSaved {len(rows)} trials to {path}")
    for order in INSTRUCTIONS:
        group = [row for row in rows if row["order"] == order]
        correct = sum(row["correct"] for row in group)
        invalid = sum(row["prediction"] is None for row in group)
        asked = sum(len(row["turns"]) == 2 for row in group)
        print(
            f"{order}: correct {correct}/{len(group)}, invalid {invalid}, ASK {asked}")

    # A flip needs a valid Yes/No response in both orders for the same question.
    predictions = {(row["question"], row["order"])
                    : row["prediction"] for row in rows}
    flips = valid_pairs = 0
    for item in probes:
        question = item["question"]
        first = predictions[question, "Yes-first"]
        second = predictions[question, "No-first"]
        if first is not None and second is not None:
            valid_pairs += 1
            flips += first != second
    print(f"Order flips: {flips}; valid pairs: {valid_pairs}/{len(probes)}")
    return rows


if __name__ == "__main__":
    run_probes(direct_trial, "direct.jsonl")
