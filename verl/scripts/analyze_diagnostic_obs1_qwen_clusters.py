#!/usr/bin/env python3
"""Task-aware Qwen clustering for open-ended diagnostic rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERL_ROOT = Path(__file__).resolve().parents[1]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.utils.reward_score.ttrl.direct_answer import (  # noqa: E402
    UNKNOWN_LABEL,
    normalize_direct_answer,
)


OPEN_BENCHMARKS = ("visualsimpleqa", "ocrbench_v1", "aokvqa_da_val")


def canonical(value: Any) -> str | None:
    answer = normalize_direct_answer(value)
    return None if answer == UNKNOWN_LABEL else answer


def canonical_labels(labels: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    output: list[int] = []
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
        output.append(mapping[label])
    return output


def clean_question(prompt: Any) -> str:
    text = str(prompt or "")
    if "</img>" in text:
        text = text.split("</img>", 1)[1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_jsonl_stable(path: Path, attempts: int = 60, delay_seconds: float = 2.0) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            text = path.read_text(encoding="utf-8")
            lines = [line for line in text.splitlines() if line.strip()]
            if not lines:
                raise ValueError(f"Shard is empty: {path}")
            return [json.loads(line) for line in lines]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            last_error = error
            time.sleep(delay_seconds)
    raise RuntimeError(f"Shard did not stabilize: {path}: {last_error}")


def load_questions(experiment_root: Path) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for benchmark in OPEN_BENCHMARKS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        path = experiment_root / "dumps" / benchmark / "real.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                groups[str(row["question_id"])].append(row)
        if len(groups) != 250:
            raise ValueError(f"{benchmark}: expected 250 questions, got {len(groups)}")
        for question_id, records in groups.items():
            records.sort(key=lambda row: int(row["sample_idx"]))
            if [int(row["sample_idx"]) for row in records] != list(range(32)):
                raise ValueError(f"{benchmark}/{question_id}: expected sample_idx=0..31")
            answers: list[str] = []
            counter: Counter[str] = Counter()
            answer_correct: dict[str, int] = {}
            for record in records:
                if not bool(record.get("is_valid")):
                    continue
                # Match the reward exactly: count the parser output without a
                # second analysis-only normalization pass.
                answer = str(record.get("parsed_answer") or "").strip()
                if not answer or answer.lower() == UNKNOWN_LABEL:
                    continue
                if answer not in counter:
                    answers.append(answer)
                counter[answer] += 1
                answer_correct[answer] = max(
                    answer_correct.get(answer, 0), int(record.get("is_correct") or 0)
                )
            key = f"{benchmark}\t{question_id}"
            questions[key] = {
                "case_id": key,
                "benchmark": benchmark,
                "question_id": question_id,
                "question": clean_question(records[0].get("prompt")),
                "answers": answers,
                "counter": counter,
                "answer_correct": answer_correct,
                "n_valid": sum(counter.values()),
            }
    return questions


def pilot_cases() -> list[dict[str, Any]]:
    raw = [
        ("number_alias", "How many objects are shown?", ["2", "two", "3"], [0, 0, 1]),
        ("numeric_format", "What number is shown?", ["1,000", "1000", "1001"], [0, 0, 1]),
        (
            "marker_aliases",
            "How many markers are on the tray behind the man? Answer with a short direct answer.",
            ["one", "2", "two", "0", "zero"],
            [0, 1, 1, 2, 2],
        ),
        (
            "phone_synonyms",
            "What device is sitting next to the laptop? Answer with a short direct answer.",
            ["smartphone", "cell phone", "cellphone", "aptop"],
            [0, 0, 0, 1],
        ),
        (
            "city_contrast",
            "In which city is the headquarters of this truck brand? Answer with a short direct answer.",
            ["lyon", "london", "rome", "munich germany"],
            [0, 1, 2, 3],
        ),
        (
            "number_contrast",
            "How many chromosomes does the animal have? Answer with a short direct answer.",
            ["60", "sixty four", "64", "78"],
            [0, 1, 1, 2],
        ),
        (
            "ocr_names",
            "Who is the youngest gold medalist in the Olympic Games? Answer with the exact text from the image.",
            ["marjorie gestring", "marjorie gesting", "marie geiringer", "larisa latynina"],
            [0, 1, 2, 3],
        ),
        (
            "ocr_noise",
            "What word is shown? Answer with the exact text from the image.",
            ["thankful", "thrmful", "thathamful"],
            [0, 1, 2],
        ),
        ("city_alias", "Which city is shown?", ["seattle", "seattle wa", "portland"], [0, 0, 1]),
        ("object_alias", "What vehicle is shown?", ["bicycle", "bike", "bus"], [0, 0, 1]),
        (
            "country_alias",
            "Which country is represented?",
            ["usa", "united states", "canada"],
            [0, 0, 1],
        ),
        ("color_alias", "What color is the object?", ["red", "the color red", "blue"], [0, 0, 1]),
        ("binary_alias", "Is the object present?", ["yes", "yeah", "no"], [0, 0, 1]),
        ("specificity", "What animal is shown?", ["poodle", "dog", "cat"], [0, 1, 2]),
        (
            "date_format",
            "What date is shown?",
            ["april 4 2018", "4 april 2018", "april 5 2018"],
            [0, 0, 1],
        ),
        (
            "ocr_number_format",
            "What number is printed? Answer with the exact text from the image.",
            ["12 721", "12721", "12731"],
            [0, 0, 1],
        ),
        (
            "direction_contrast",
            "Which direction is the person looking?",
            ["forward", "upward", "right"],
            [0, 1, 2],
        ),
        (
            "phone_sentence_aliases",
            "What type of device is next to the computer? Answer with a short direct answer.",
            [
                "smartphone",
                "cell phone",
                "cellphone",
                "phone",
                "the device next to the computer is a mobile phone",
                "laptop",
            ],
            [0, 0, 0, 0, 0, 1],
        ),
        (
            "negative_sentence_alias",
            "Is a body of water visible? Answer with a short direct answer.",
            ["no", "no there is no body of water visible", "yes"],
            [0, 0, 1],
        ),
        (
            "positive_sentence_alias",
            "Can this instrument play several pitches? Answer with a short direct answer.",
            ["yes", "yes this instrument can play several pitches", "no"],
            [0, 0, 1],
        ),
        (
            "profession_aliases",
            "What is the profession of the person near the net? Answer with a short direct answer.",
            ["tennis player", "male tennis player", "tennis professional", "umpire"],
            [0, 0, 0, 1],
        ),
        (
            "object_number_aliases",
            "Which writing object is on the desk? Answer with a short direct answer.",
            ["pen", "pens", "a pen is on the desk", "pencil"],
            [0, 0, 0, 1],
        ),
        (
            "exact_code_leading_zero",
            "What are the final two characters? Answer with the exact text from the image.",
            ["0", "00", "01"],
            [0, 1, 2],
        ),
        (
            "exact_text_typo",
            "What is the first word? Answer with the exact text from the image.",
            ["inspired", "insprired", "inspire"],
            [0, 1, 2],
        ),
        (
            "exact_text_spacing",
            "What is the printed code? Answer with the exact text from the image.",
            ["AB1 18", "AB118", "AB1 19"],
            [0, 0, 1],
        ),
        (
            "formula_wrapper",
            "Write the formula using LaTeX. Answer with the exact text from the image.",
            ["x+1", "\\boxed{x+1}", "x-1"],
            [0, 0, 1],
        ),
        (
            "contextual_vehicle_aliases",
            "What vehicle is closest to the guard? Answer with a short direct answer.",
            ["silver car", "silver sedan", "red car", "bus"],
            [0, 0, 1, 2],
        ),
    ]
    return [
        {
            "case_id": f"pilot\t{category}",
            "benchmark": "pilot",
            "question_id": category,
            "question": question,
            "answers": answers,
            "expected_labels": canonical_labels(labels),
        }
        for category, question, answers, labels in raw
    ]


def case_permutation(case_id: str, size: int, permutation_id: int) -> list[int]:
    order = list(range(size))
    if permutation_id == 0 or size < 2:
        return order
    seed = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16], 16)
    random.Random(seed).shuffle(order)
    if order == list(range(size)):
        order = order[1:] + order[:1]
    return order


def build_prompt(case: dict[str, Any], order: list[int]) -> str:
    answer_lines = "\n".join(
        f"{position}. {json.dumps(case['answers'][answer_index], ensure_ascii=False)}"
        for position, answer_index in enumerate(order)
    )
    return (
        "Cluster candidate answers for one visual question. Use only the question and answer text; "
        "do not guess the image or decide which answer is factually correct. Two answers share a "
        "cluster only when they are interchangeable as the same correct answer under this exact "
        "question and scoring instruction. Synonyms, abbreviations, and equivalent number words or "
        "digits may share a cluster. Answers that merely concern the same topic, differ in entity, "
        "quantity, direction, or specificity must be separate. For an exact-text/OCR instruction, "
        "different letters or digits are separate; harmless punctuation or number-group spacing may "
        "be equivalent.\n\n"
        f"Question and scoring instruction:\n{case['question']}\n\n"
        f"Candidate answers:\n{answer_lines}\n\n"
        "Return exactly one JSON array of non-negative integers, one integer per candidate in the "
        "shown order. Equal integers mean the same cluster. Use 0 for the first cluster and introduce "
        "1, 2, ... in first-occurrence order. Return no explanation."
    )


def parse_cluster_labels(text: str, expected_size: int) -> list[int] | None:
    for match in re.finditer(r"\[[^\[\]]*\]", text, flags=re.DOTALL):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, list)
            and len(value) == expected_size
            and all(isinstance(item, int) and item >= 0 for item in value)
        ):
            return canonical_labels(value)
    return None


def build_pair_prompt(case: dict[str, Any], left: str, right: str) -> str:
    return (
        "Judge whether two candidate answers are interchangeable as the same correct answer for one "
        "specific visual question. Do not decide which candidate is factually correct and do not use "
        "an image. Choose SAME only for genuine synonyms, abbreviations, or equivalent number formats. "
        "Answers that merely share a topic, or differ in entity, quantity, direction, or specificity, "
        "are DIFFERENT. For an exact-text/OCR instruction, changed letters or digits are DIFFERENT; "
        "harmless punctuation or number-group spacing may be SAME. The decision must not depend on "
        "candidate order.\n\n"
        "General examples:\n"
        "- 7 and seven: SAME.\n"
        "- mobile phone and smartphone: SAME.\n"
        "- Boston and Boston, MA when the question asks for a city: SAME.\n"
        "- 45 600 and 45600 when they denote a printed number: SAME.\n"
        "- Paris and London: DIFFERENT.\n"
        "- 40 and forty-two: DIFFERENT.\n"
        "- center and centre under an exact-text OCR instruction: DIFFERENT.\n\n"
        f"Question and scoring instruction:\n{case['question']}\n\n"
        f"Candidate 1: {json.dumps(left, ensure_ascii=False)}\n"
        f"Candidate 2: {json.dumps(right, ensure_ascii=False)}\n\n"
        "A. SAME\nB. DIFFERENT\nOutput only A or B."
    )


def build_semantic_pair_prompt(
    case: dict[str, Any], left: str, right: str, *, adjudication: bool
) -> str:
    exact_text = case.get("benchmark") == "ocrbench_v1" or "exact text" in case["question"].lower()
    if exact_text:
        task_rule = (
            "This is an exact-text/OCR or formula task. Compare the intended literal transcription or "
            "mathematical expression. Ignore answer wrappers, harmless whitespace/tokenization, and "
            "purely presentational LaTeX wrappers. Changed letters, digits, operators, names, or values "
            "are DIFFERENT even when the strings look similar."
        )
        examples = (
            "- Code ZX2 41 and ZX241: SAME when only grouping spaces differ.\n"
            "- Formula y+2 and boxed y+2: SAME when boxing is only presentation.\n"
            "- Code ZX241 and ZX242: DIFFERENT.\n"
            "- Words sale and sake: DIFFERENT.\n"
        )
    else:
        task_rule = (
            "This is a short direct-answer task. Compare the answer denotation, not surface form. Ignore "
            "capitalization, spacing, singular/plural wording, response scaffolding, and short-answer "
            "versus full-sentence phrasing. Synonyms, abbreviations, number words/digits, and a concise "
            "answer versus a sentence asserting that same answer are SAME. Different entities, values, "
            "polarities, directions, or materially different attributes are DIFFERENT. Merely belonging "
            "to the same broad category, or one answer being a more specific subtype, is not enough."
        )
        examples = (
            "- mobile phone and cellphone: SAME.\n"
            "- eight and 8: SAME.\n"
            "- no and no, the object is not present: SAME.\n"
            "- bicycle and bike: SAME.\n"
            "- yes and no: DIFFERENT.\n"
            "- 40 and forty-two: DIFFERENT.\n"
            "- pen and pencil: DIFFERENT.\n"
            "- sparrow and bird: DIFFERENT because they differ in specificity.\n"
        )
    if adjudication:
        arbitration = (
            "Classify the semantic relation, not just topical similarity. Use A only when both outputs "
            "have the same answer denotation and can be substituted without changing the answer. Use B "
            "when they are related but one is broader, narrower, or a different member of the same "
            "category. Use C when they conflict or are otherwise distinct. Examples: cellphone versus "
            "cell phone is A; bicycle versus bike is A; 8 versus eight is A; a short answer versus a "
            "sentence asserting exactly that answer is A; poodle versus dog is B; sedan versus vehicle "
            "is B; pen versus pencil is B; Paris versus London is C. "
        )
        output_rule = (
            "A. SAME DENOTATION / SURFACE VARIANT\n"
            "B. RELATED OR HIERARCHICAL, BUT NOT EQUIVALENT\n"
            "C. CONFLICTING OR DISTINCT\n"
            "Output only A, B, or C."
        )
    else:
        arbitration = "Judge the two candidates independently and make the decision invariant to their order. "
        output_rule = "A. SAME SEMANTIC ANSWER\nB. DIFFERENT SEMANTIC ANSWERS\nOutput only A or B."
    return (
        "Determine whether two model outputs belong to the same semantic answer cluster for one visual "
        "question. Do not decide whether either answer is factually correct. "
        f"{arbitration}{task_rule}\n\n"
        f"General examples:\n{examples}\n"
        f"Question and scoring instruction:\n{case['question']}\n\n"
        f"Candidate 1: {json.dumps(left, ensure_ascii=False)}\n"
        f"Candidate 2: {json.dumps(right, ensure_ascii=False)}\n\n"
        f"{output_rule}"
    )


_NUMBER_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _pure_number_value(answer: str) -> int | None:
    text = answer.strip().lower()
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    tokens = text.replace("-", " ").split()
    if len(tokens) == 1:
        return _NUMBER_WORD_VALUES.get(tokens[0])
    if (
        len(tokens) == 2
        and tokens[0] in _NUMBER_WORD_VALUES
        and 20 <= _NUMBER_WORD_VALUES[tokens[0]] <= 90
        and _NUMBER_WORD_VALUES[tokens[0]] % 10 == 0
        and tokens[1] in _NUMBER_WORD_VALUES
        and 0 < _NUMBER_WORD_VALUES[tokens[1]] < 10
    ):
        return _NUMBER_WORD_VALUES[tokens[0]] + _NUMBER_WORD_VALUES[tokens[1]]
    return None


def _formula_signature(answer: str) -> str:
    text = answer.strip().lower()
    previous = None
    while text != previous:
        previous = text
        text = re.sub(
            r"\\(?:boxed|fbox|mathrm|mathbf|mathit|text)\s*\{([^{}]*)\}",
            r"\1",
            text,
        )
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", "", text)


_SENTENCE_SCAFFOLD_TOKENS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "there",
    "it",
    "its",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "has",
    "have",
    "had",
    "does",
    "do",
    "did",
    "can",
    "could",
    "will",
    "would",
    "answer",
}


def _singular_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _lexical_tokens(text: str) -> list[str]:
    return [_singular_token(token) for token in re.findall(r"[a-z0-9]+", text.lower())]


def _short_answer_sentence_equivalent(case: dict[str, Any], left: str, right: str) -> bool:
    left_tokens = _lexical_tokens(left)
    right_tokens = _lexical_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if len(left_tokens) > len(right_tokens):
        left_tokens, right_tokens = right_tokens, left_tokens

    remaining = Counter(right_tokens)
    for token in left_tokens:
        if remaining[token] <= 0:
            return False
        remaining[token] -= 1
    question_tokens = set(_lexical_tokens(case["question"]))
    allowed_extras = question_tokens | _SENTENCE_SCAFFOLD_TOKENS
    return all(count == 0 or token in allowed_extras for token, count in remaining.items())


def semantic_pair_override(case: dict[str, Any], left: str, right: str) -> bool | None:
    question = case["question"].lower()
    exact_text = case.get("benchmark") == "ocrbench_v1" or "exact text" in question
    formula_task = "formula" in question or "latex" in question
    if formula_task:
        # Formula operators carry semantics. Generic punctuation stripping would
        # incorrectly collapse x+1 and x-1.
        return _formula_signature(left) == _formula_signature(right)

    left_compact = re.sub(r"[^0-9a-z]+", "", left.lower())
    right_compact = re.sub(r"[^0-9a-z]+", "", right.lower())
    if left_compact and left_compact == right_compact:
        return True

    if exact_text and not formula_task:
        # Literal OCR content is the task target. Distinct alphanumeric content
        # cannot be rescued by a semantic judge.
        return False

    left_number = _pure_number_value(left)
    right_number = _pure_number_value(right)
    if left_number is not None and right_number is not None:
        return left_number == right_number

    polarities = {"yes", "no"}
    left_first = left.lower().split(maxsplit=1)[0] if left.strip() else ""
    right_first = right.lower().split(maxsplit=1)[0] if right.strip() else ""
    if {left_first, right_first} == polarities:
        return False
    if left_first == right_first and left_first in polarities:
        return True
    if _short_answer_sentence_equivalent(case, left, right):
        return True
    return None


def parse_pair_label(text: str) -> str | None:
    match = re.search(r"(?:^|[^A-Z])([ABC])(?:[^A-Z]|$)", text.strip().upper())
    return match.group(1) if match else None


def create_llm(args: argparse.Namespace):
    from vllm import LLM

    return LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=True,
    )


def infer(args: argparse.Namespace, cases: list[dict[str, Any]]) -> int:
    from transformers import AutoProcessor
    from vllm import SamplingParams

    tasks: list[tuple[dict[str, Any], int, list[int]]] = []
    for case in cases:
        if len(case["answers"]) < 2:
            continue
        for permutation_id in (0, 1):
            tasks.append(
                (
                    case,
                    permutation_id,
                    case_permutation(case["case_id"], len(case["answers"]), permutation_id),
                )
            )
    tasks = tasks[args.rank :: args.world_size]
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    llm = create_llm(args)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    rows: list[dict[str, Any]] = []
    for offset in range(0, len(tasks), args.chunk_size):
        batch = tasks[offset : offset + args.chunk_size]
        prompts: list[str] = []
        for case, _, order in batch:
            messages = [{"role": "user", "content": [{"type": "text", "text": build_prompt(case, order)}]}]
            prompts.append(
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for (case, permutation_id, order), output in zip(batch, outputs):
            raw_output = output.outputs[0].text.strip()
            shown_labels = parse_cluster_labels(raw_output, len(order))
            original_labels: list[int] | None = None
            if shown_labels is not None:
                original_labels = [0] * len(order)
                for shown_index, original_index in enumerate(order):
                    original_labels[original_index] = shown_labels[shown_index]
                original_labels = canonical_labels(original_labels)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "benchmark": case["benchmark"],
                    "question_id": case["question_id"],
                    "permutation_id": permutation_id,
                    "order": order,
                    "labels_original_order": original_labels,
                    "parse_valid": original_labels is not None,
                    "raw_output": raw_output,
                }
            )
        print(f"[qwen-cluster rank={args.rank}] {min(offset + len(batch), len(tasks))}/{len(tasks)}", flush=True)
    output_path = args.output_dir / f"judge.rank{args.rank}.jsonl"
    write_jsonl(output_path, rows)
    return 0


def pair_directions(args: argparse.Namespace) -> tuple[int, ...]:
    return (0, 1, 2) if args.pair_protocol == "semantic_v2" else (0, 1)


def infer_pairwise(args: argparse.Namespace, cases: list[dict[str, Any]]) -> int:
    from transformers import AutoProcessor
    from vllm import SamplingParams

    tasks: list[tuple[dict[str, Any], int, int, int]] = []
    for case in cases:
        for left_index in range(len(case["answers"])):
            for right_index in range(left_index + 1, len(case["answers"])):
                for direction in pair_directions(args):
                    tasks.append((case, left_index, right_index, direction))
    tasks = tasks[args.rank :: args.world_size]
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    llm = create_llm(args)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    rows: list[dict[str, Any]] = []
    for offset in range(0, len(tasks), args.chunk_size):
        batch = tasks[offset : offset + args.chunk_size]
        prompts: list[str] = []
        for case, left_index, right_index, direction in batch:
            left = case["answers"][left_index]
            right = case["answers"][right_index]
            if direction == 1:
                left, right = right, left
            elif direction == 2 and (right.casefold(), right) < (left.casefold(), left):
                left, right = right, left
            if args.pair_protocol == "semantic_v2":
                prompt = build_semantic_pair_prompt(
                    case, left, right, adjudication=direction == 2
                )
            else:
                prompt = build_pair_prompt(case, left, right)
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
            prompts.append(
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for (case, left_index, right_index, direction), output in zip(batch, outputs):
            raw_output = output.outputs[0].text.strip()
            label = parse_pair_label(raw_output)
            deterministic_override = (
                semantic_pair_override(
                    case, case["answers"][left_index], case["answers"][right_index]
                )
                if args.pair_protocol == "semantic_v2"
                else None
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "benchmark": case["benchmark"],
                    "question_id": case["question_id"],
                    "left_index": left_index,
                    "right_index": right_index,
                    "direction": direction,
                    "pair_protocol": args.pair_protocol,
                    "deterministic_override": deterministic_override,
                    "label": label,
                    "parse_valid": label is not None,
                    "raw_output": raw_output,
                }
            )
        print(
            f"[qwen-pair rank={args.rank}] {min(offset + len(batch), len(tasks))}/{len(tasks)}",
            flush=True,
        )
    write_jsonl(args.output_dir / f"pair_judge.rank{args.rank}.jsonl", rows)
    return 0


def infer_nli_pairs(args: argparse.Namespace, cases: list[dict[str, Any]]) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.nli_model is None:
        raise ValueError("--nli-model is required for stage infer-nli")
    if not torch.cuda.is_available():
        raise RuntimeError("NLI veto inference requires a CUDA worker")

    pairs = {
        pair
        for case in cases
        for left_index, left in enumerate(case["answers"])
        for right in case["answers"][left_index + 1 :]
        for pair in ((left, right), (right, left))
    }
    ordered = sorted(pairs)[args.rank :: args.world_size]
    tokenizer = AutoTokenizer.from_pretrained(args.nli_model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.nli_model,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval().to("cuda:0")
    label2id = {str(key).upper(): int(value) for key, value in model.config.label2id.items()}
    entailment_id = label2id.get("ENTAILMENT")
    if entailment_id is None:
        raise ValueError(f"Missing ENTAILMENT label: {model.config.label2id}")

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for offset in range(0, len(ordered), args.nli_batch_size):
            batch = ordered[offset : offset + args.nli_batch_size]
            encoded = tokenizer(
                [left for left, _ in batch],
                [right for _, right in batch],
                padding=True,
                truncation=True,
                max_length=args.nli_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to("cuda:0", non_blocking=True) for key, value in encoded.items()}
            logits = model(**encoded).logits.float()
            probabilities = torch.softmax(logits, dim=-1)
            labels = logits.argmax(dim=-1)
            for (left, right), label, probability in zip(
                batch, labels.tolist(), probabilities[:, entailment_id].tolist()
            ):
                rows.append(
                    {
                        "premise": left,
                        "hypothesis": right,
                        "predicted_label_id": int(label),
                        "entailment_probability": float(probability),
                        "is_entailment": bool(label == entailment_id),
                    }
                )
            print(
                f"[nli-veto rank={args.rank}] {min(offset + len(batch), len(ordered))}/{len(ordered)}",
                flush=True,
            )
    write_jsonl(args.output_dir / f"nli_pair.rank{args.rank}.jsonl", rows)
    return 0


def same_relation(labels: list[int], left: int, right: int) -> bool:
    return labels[left] == labels[right]


def intersect_labels(first: list[int], second: list[int]) -> list[int]:
    signatures = [(first[index], second[index]) for index in range(len(first))]
    mapping: dict[tuple[int, int], int] = {}
    labels: list[int] = []
    for signature in signatures:
        if signature not in mapping:
            mapping[signature] = len(mapping)
        labels.append(mapping[signature])
    return labels


def load_judgments(args: argparse.Namespace) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    merged: list[dict[str, Any]] = []
    for rank in range(args.world_size):
        path = args.output_dir / f"judge.rank{rank}.jsonl"
        for row in read_jsonl_stable(path):
            permutation_id = int(row["permutation_id"])
            case_id = str(row["case_id"])
            if permutation_id in grouped[case_id]:
                raise ValueError(f"Duplicate judgment: {case_id}/{permutation_id}")
            grouped[case_id][permutation_id] = row
            merged.append(row)
    merged.sort(key=lambda row: (str(row["case_id"]), int(row["permutation_id"])))
    write_jsonl(args.output_dir / "judge_outputs.jsonl", merged)
    return grouped


def load_pair_judgments(args: argparse.Namespace) -> dict[tuple[str, int, int], dict[int, dict[str, Any]]]:
    grouped: dict[tuple[str, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    merged: list[dict[str, Any]] = []
    for rank in range(args.world_size):
        path = args.output_dir / f"pair_judge.rank{rank}.jsonl"
        for row in read_jsonl_stable(path):
            key = (str(row["case_id"]), int(row["left_index"]), int(row["right_index"]))
            direction = int(row["direction"])
            if direction in grouped[key]:
                raise ValueError(f"Duplicate pair judgment: {key}/{direction}")
            grouped[key][direction] = row
            merged.append(row)
    merged.sort(
        key=lambda row: (
            str(row["case_id"]),
            int(row["left_index"]),
            int(row["right_index"]),
            int(row["direction"]),
        )
    )
    write_jsonl(args.output_dir / "pair_judge_outputs.jsonl", merged)
    return grouped


def load_nli_judgments(args: argparse.Namespace) -> dict[tuple[str, str], dict[str, Any]]:
    paths = (
        [args.nli_judgments]
        if args.nli_judgments is not None
        else [args.output_dir / f"nli_pair.rank{rank}.jsonl" for rank in range(args.world_size)]
    )
    judgments: dict[tuple[str, str], dict[str, Any]] = {}
    merged: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl_stable(path):
            key = (str(row["premise"]), str(row["hypothesis"]))
            value = bool(row["is_entailment"])
            if key in judgments and bool(judgments[key]["is_entailment"]) != value:
                raise ValueError(f"Conflicting NLI judgments for {key}")
            judgments[key] = row
            merged.append(row)
    if args.nli_judgments is None:
        merged.sort(key=lambda row: (str(row["premise"]), str(row["hypothesis"])))
        write_jsonl(args.output_dir / "nli_pair_outputs.jsonl", merged)
    return judgments


def annotate_nli_veto(
    cases: list[dict[str, Any]],
    pair_judgments: dict[tuple[str, int, int], dict[int, dict[str, Any]]],
    nli_judgments: dict[tuple[str, str], dict[str, Any]],
) -> None:
    case_by_id = {str(case["case_id"]): case for case in cases}
    missing: list[tuple[str, str]] = []
    for (case_id, left_index, right_index), directions in pair_judgments.items():
        case = case_by_id[case_id]
        left = str(case["answers"][left_index])
        right = str(case["answers"][right_index])
        forward = nli_judgments.get((left, right))
        reverse = nli_judgments.get((right, left))
        if forward is None:
            missing.append((left, right))
        if reverse is None:
            missing.append((right, left))
        if forward is None or reverse is None:
            continue
        mutual = bool(forward["is_entailment"] and reverse["is_entailment"])
        for row in directions.values():
            row["nli_forward_entailment"] = bool(forward["is_entailment"])
            row["nli_reverse_entailment"] = bool(reverse["is_entailment"])
            row["nli_forward_entailment_probability"] = float(forward["entailment_probability"])
            row["nli_reverse_entailment_probability"] = float(reverse["entailment_probability"])
            row["nli_mutual_entailment"] = mutual
    if missing:
        raise ValueError(f"NLI veto coverage missing {len(missing)} directions; examples={missing[:10]}")


def validate_pair_judgment_coverage(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    judgments: dict[tuple[str, int, int], dict[int, dict[str, Any]]],
) -> tuple[int, int]:
    expected: set[tuple[str, int, int]] = set()
    for case in cases:
        for left in range(len(case["answers"])):
            for right in range(left + 1, len(case["answers"])):
                expected.add((str(case["case_id"]), left, right))

    actual = set(judgments)
    missing = expected - actual
    extra = actual - expected
    expected_directions = set(pair_directions(args))
    incomplete = {
        key: sorted(expected_directions - set(judgments[key]))
        for key in expected & actual
        if set(judgments[key]) != expected_directions
    }
    if missing or extra or incomplete:
        raise ValueError(
            "Pair judgment coverage mismatch: "
            f"expected_pairs={len(expected)}, actual_pairs={len(actual)}, "
            f"missing={len(missing)}, extra={len(extra)}, incomplete={len(incomplete)}; "
            f"missing_examples={sorted(missing)[:3]}, extra_examples={sorted(extra)[:3]}, "
            f"incomplete_examples={list(sorted(incomplete.items()))[:3]}"
        )
    return len(expected), len(expected_directions) * len(expected)


def pair_decision(
    args: argparse.Namespace, pair: dict[int, dict[str, Any]]
) -> tuple[bool, bool, dict[int, str | None], bool]:
    directions = pair_directions(args)
    labels = {direction: pair.get(direction, {}).get("label") for direction in directions}
    valid = all(labels[direction] is not None for direction in directions)
    override_values = {
        row.get("deterministic_override")
        for row in pair.values()
        if row.get("deterministic_override") is not None
    }
    if len(override_values) > 1:
        raise ValueError(f"Inconsistent deterministic pair overrides: {override_values}")
    deterministic_override = next(iter(override_values), None)
    if deterministic_override is not None:
        predicted_same = bool(valid and deterministic_override)
    elif args.pair_protocol == "semantic_v2":
        # The third judgment is deliberately conservative: it can veto an
        # otherwise permissive directional match. At least one directional
        # judgment must still support the edge.
        predicted_same = bool(
            valid
            and labels.get(2) == "A"
            and (labels.get(0) == "A" or labels.get(1) == "A")
        )
    else:
        predicted_same = bool(valid and labels.get(0) == "A" and labels.get(1) == "A")
    if args.nli_veto and deterministic_override is None and predicted_same:
        probability_pairs = {
            (
                row.get("nli_forward_entailment_probability"),
                row.get("nli_reverse_entailment_probability"),
            )
            for row in pair.values()
            if row.get("nli_forward_entailment_probability") is not None
            and row.get("nli_reverse_entailment_probability") is not None
        }
        if len(probability_pairs) != 1:
            raise ValueError(f"Missing or inconsistent NLI veto probabilities: {probability_pairs}")
        forward_probability, reverse_probability = next(iter(probability_pairs))
        strongly_non_equivalent = (
            max(float(forward_probability), float(reverse_probability))
            < args.nli_veto_max_entailment_prob
        )
        predicted_same = not strongly_non_equivalent
    order_agree = bool(
        deterministic_override is not None
        or (labels.get(0) is not None and labels.get(0) == labels.get(1))
    )
    return predicted_same, valid, labels, order_agree


def analyze_pairwise_pilot(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    judgments: dict[tuple[str, int, int], dict[int, dict[str, Any]]],
) -> int:
    expected_pairs, expected_directional_judgments = validate_pair_judgment_coverage(
        args, cases, judgments
    )
    tp = fp = fn = tn = parsed = order_agree = total = 0
    rows: list[dict[str, Any]] = []
    for case in cases:
        expected = case["expected_labels"]
        for left in range(len(expected)):
            for right in range(left + 1, len(expected)):
                total += 1
                pair = judgments.get((case["case_id"], left, right), {})
                prediction, valid, labels, agrees = pair_decision(args, pair)
                label_forward = labels.get(0)
                label_reverse = labels.get(1)
                truth = same_relation(expected, left, right)
                parsed += int(valid)
                order_agree += int(valid and agrees)
                tp += int(truth and prediction)
                fp += int(not truth and prediction)
                fn += int(truth and not prediction)
                tn += int(not truth and not prediction)
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "question": case["question"],
                        "left": case["answers"][left],
                        "right": case["answers"][right],
                        "expected_same": truth,
                        "forward_label": label_forward,
                        "reverse_label": label_reverse,
                        "adjudication_label": labels.get(2),
                        "deterministic_override": next(
                            (
                                row.get("deterministic_override")
                                for row in pair.values()
                                if row.get("deterministic_override") is not None
                            ),
                            None,
                        ),
                        "nli_mutual_entailment": next(
                            (
                                row.get("nli_mutual_entailment")
                                for row in pair.values()
                                if row.get("nli_mutual_entailment") is not None
                            ),
                            None,
                        ),
                        "nli_forward_entailment_probability": next(
                            (
                                row.get("nli_forward_entailment_probability")
                                for row in pair.values()
                                if row.get("nli_forward_entailment_probability") is not None
                            ),
                            None,
                        ),
                        "nli_reverse_entailment_probability": next(
                            (
                                row.get("nli_reverse_entailment_probability")
                                for row in pair.values()
                                if row.get("nli_reverse_entailment_probability") is not None
                            ),
                            None,
                        ),
                        "parse_valid": valid,
                        "predicted_same": prediction,
                        "correct": prediction == truth,
                    }
                )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    summary = {
        "pair_protocol": args.pair_protocol,
        "pair_cluster_method": args.pair_cluster_method,
        "nli_veto": bool(args.nli_veto),
        "nli_veto_max_entailment_probability": (
            args.nli_veto_max_entailment_prob if args.nli_veto else None
        ),
        "pairs": total,
        "expected_pairs": expected_pairs,
        "expected_directional_judgments": expected_directional_judgments,
        "parse_rate": parsed / total,
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_specificity": specificity,
        "order_label_agreement": order_agree / total,
        "pairwise_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    summary["gate_passed"] = bool(
        summary["parse_rate"] == 1.0
        and precision >= 0.95
        and recall >= (0.90 if args.pair_protocol == "semantic_v2" else 0.85)
        and specificity >= 0.95
        and summary["order_label_agreement"] >= 0.95
    )
    write_jsonl(args.output_dir / "pair_pilot_cases.jsonl", rows)
    (args.output_dir / "pair_pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "pair_pilot_report.md").write_text(
        "\n".join(
            [
                "# Obs1 Qwen pairwise-equivalence pilot",
                "",
                f"- Pair protocol: {args.pair_protocol}",
                f"- Cluster method: {args.pair_cluster_method}",
                f"- NLI strong-non-equivalence veto: {args.nli_veto}",
                f"- NLI max-entailment threshold: {args.nli_veto_max_entailment_prob if args.nli_veto else 'NA'}",
                f"- Parse rate: {summary['parse_rate']:.2%}",
                f"- Pairwise precision: {precision:.2%}",
                f"- Pairwise recall: {recall:.2%}",
                f"- Pairwise specificity: {specificity:.2%}",
                f"- Order-label agreement: {summary['order_label_agreement']:.2%}",
                f"- Gate passed: {summary['gate_passed']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    return 0 if summary["gate_passed"] else 3


def complete_link_pair_labels(
    args: argparse.Namespace,
    case_id: str,
    size: int,
    judgments: dict[tuple[str, int, int], dict[int, dict[str, Any]]],
) -> tuple[list[int], int, float, float]:
    equivalent: dict[tuple[int, int], bool] = {}
    parsed = 0
    order_agree = 0
    total = size * (size - 1) // 2
    for left in range(size):
        for right in range(left + 1, size):
            pair = judgments.get((case_id, left, right), {})
            predicted_same, valid, _, agrees = pair_decision(args, pair)
            parsed += int(valid)
            order_agree += int(valid and agrees)
            equivalent[(left, right)] = predicted_same

    clusters: list[list[int]] = []
    for answer_index in range(size):
        for cluster in clusters:
            if all(
                equivalent[(min(answer_index, member), max(answer_index, member))]
                for member in cluster
            ):
                cluster.append(answer_index)
                break
        else:
            clusters.append([answer_index])
    labels = [0] * size
    for label, cluster in enumerate(clusters):
        for answer_index in cluster:
            labels[answer_index] = label
    if total == 0:
        return labels, 0, 1.0, 1.0
    return labels, total - parsed, parsed / total, order_agree / total


def connected_component_pair_labels(
    args: argparse.Namespace,
    case_id: str,
    size: int,
    judgments: dict[tuple[str, int, int], dict[int, dict[str, Any]]],
) -> tuple[list[int], int, float, float]:
    parent = list(range(size))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    parsed = 0
    order_agree = 0
    total = size * (size - 1) // 2
    for left in range(size):
        for right in range(left + 1, size):
            predicted_same, valid, _, agrees = pair_decision(
                args, judgments.get((case_id, left, right), {})
            )
            parsed += int(valid)
            order_agree += int(valid and agrees)
            if predicted_same:
                union(left, right)

    root_to_label: dict[int, int] = {}
    labels: list[int] = []
    for answer_index in range(size):
        root = find(answer_index)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels.append(root_to_label[root])
    if total == 0:
        return labels, 0, 1.0, 1.0
    return labels, total - parsed, parsed / total, order_agree / total


def analyze_pairwise_full(
    args: argparse.Namespace,
    questions: dict[str, dict[str, Any]],
    judgments: dict[tuple[str, int, int], dict[int, dict[str, Any]]],
) -> int:
    expected_pairs, expected_directional_judgments = validate_pair_judgment_coverage(
        args, list(questions.values()), judgments
    )
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for case_id in sorted(questions):
        question = questions[case_id]
        answers = question["answers"]
        counter = question["counter"]
        if not answers:
            rows.append(
                {
                    "benchmark": question["benchmark"],
                    "question_id": question["question_id"],
                    "n_total": 32,
                    "n_valid": 0,
                    "all_invalid": 1,
                }
            )
            continue

        cluster_fn = (
            connected_component_pair_labels
            if args.pair_cluster_method == "mutual_cc"
            else complete_link_pair_labels
        )
        final_labels, invalid_pairs, pair_parse_rate, pair_order_agreement = cluster_fn(
            args, case_id, len(answers), judgments
        )
        judge_invalid = int(invalid_pairs > 0)
        clusters: list[list[str]] = []
        for label in sorted(set(final_labels)):
            clusters.append(
                [answer for answer, answer_label in zip(answers, final_labels) if answer_label == label]
            )
        masses = [sum(counter[answer] for answer in cluster) for cluster in clusters]
        semantic_max = max(masses)
        semantic_tie = int(sum(mass == semantic_max for mass in masses) > 1)
        semantic_cluster = clusters[masses.index(semantic_max)]
        string_answer, string_count = counter.most_common(1)[0]
        string_tie = int(sum(count == string_count for count in counter.values()) > 1)
        disagreement = int(string_answer not in semantic_cluster)
        correct = question["answer_correct"]
        string_eval_correct = int(correct.get(string_answer, 0) > 0)
        semantic_eval_correct = int(any(correct.get(answer, 0) > 0 for answer in semantic_cluster))
        fatal_strict = int(
            disagreement
            and not string_eval_correct
            and semantic_eval_correct
            and not string_tie
            and not semantic_tie
            and not judge_invalid
        )
        row = {
            "benchmark": question["benchmark"],
            "question_id": question["question_id"],
            "pair_protocol": args.pair_protocol,
            "pair_cluster_method": args.pair_cluster_method,
            "nli_veto": int(args.nli_veto),
            "nli_veto_max_entailment_probability": (
                args.nli_veto_max_entailment_prob if args.nli_veto else None
            ),
            "n_total": 32,
            "n_valid": question["n_valid"],
            "all_invalid": 0,
            "judge_invalid": judge_invalid,
            "judge_invalid_pairs": invalid_pairs,
            "judge_pair_parse_rate": pair_parse_rate,
            "string_cluster_count": len(counter),
            "semantic_cluster_count": len(clusters),
            "fragmentation_ratio": len(counter) / len(clusters),
            "order_pair_agreement": pair_order_agreement,
            "string_majority_answer": string_answer,
            "string_majority_count": string_count,
            "string_majority_tie": string_tie,
            "semantic_majority_answers": json.dumps(semantic_cluster, ensure_ascii=False),
            "semantic_majority_mass": semantic_max,
            "semantic_majority_tie": semantic_tie,
            "disagreement": disagreement,
            "string_eval_correct": string_eval_correct,
            "semantic_eval_correct": semantic_eval_correct,
            "fatal_strict": fatal_strict,
        }
        rows.append(row)
        if disagreement or fatal_strict or judge_invalid:
            cases.append(
                {
                    **row,
                    "question": question["question"],
                    "string_counter": dict(counter),
                    "final_semantic_labels": final_labels,
                    "semantic_clusters": [
                        {"answers": cluster, "mass": mass}
                        for cluster, mass in zip(clusters, masses)
                    ],
                }
            )

    summary = summarize_full(rows)
    write_csv(args.output_dir / "qwen_pair_question_level.csv", rows)
    write_csv(args.output_dir / "qwen_pair_fragmentation.csv", summary)
    write_jsonl(args.output_dir / "qwen_pair_cases.jsonl", cases)
    report = [
        "# Obs1: Qwen bidirectional pairwise answer clustering",
        "",
        f"Pair protocol: {args.pair_protocol}; cluster method: {args.pair_cluster_method}.",
        "Semantic-v2 creates an edge only when the three-way adjudicator selects same denotation and at least one directional judge selects SAME.",
        (
            f"A DeBERTa strong-non-equivalence veto rejects a proposed edge only when both directional entailment probabilities are below {args.nli_veto_max_entailment_prob}."
            if args.nli_veto
            else "No NLI veto is enabled."
        ),
        "Deterministic formatting, numeric, formula-wrapper, and short-answer/sentence equivalences take precedence over the NLI veto.",
        "Invalid rollout outputs are excluded. Strict fatal excludes string ties, semantic ties, and judge parse failures.",
        f"Coverage: {expected_pairs} unordered pairs / {expected_directional_judgments} directional judgments.",
        "",
        "| Benchmark | valid n | all invalid | judge invalid | F mean | order agree | string acc | semantic acc | strict fatal / frequency-wrong |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        fatal = "NA" if row["fatal_strict_rate"] is None else f"{row['fatal_strict_rate']:.2%}"
        report.append(
            f"| {row['benchmark']} | {row['questions_valid']} | {row['questions_all_invalid']} | "
            f"{row['judge_invalid_questions']} | {row['fragmentation_mean']:.3f} | "
            f"{row['order_pair_agreement_mean']:.2%} | {row['string_eval_accuracy']:.2%} | "
            f"{row['semantic_eval_accuracy']:.2%} | {fatal} "
            f"({row['fatal_strict_successes']}/{row['fatal_strict_n']}) |"
        )
    (args.output_dir / "qwen_pair_fragmentation.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "expected_pairs": expected_pairs,
                "expected_directional_judgments": expected_directional_judgments,
                "cases": len(cases),
                "summary": summary,
            }
        ),
        flush=True,
    )
    return 0


def analyze_pilot(args: argparse.Namespace, cases: list[dict[str, Any]], judgments: dict[str, Any]) -> int:
    tp = fp = fn = tn = exact = parsed = order_agree = order_pairs = 0
    rows: list[dict[str, Any]] = []
    for case in cases:
        pair = judgments.get(case["case_id"], {})
        labels_a = pair.get(0, {}).get("labels_original_order")
        labels_b = pair.get(1, {}).get("labels_original_order")
        valid = labels_a is not None and labels_b is not None
        final_labels = intersect_labels(labels_a, labels_b) if valid else list(range(len(case["answers"])))
        expected = case["expected_labels"]
        exact += int(valid and final_labels == expected)
        parsed += int(valid)
        for left in range(len(expected)):
            for right in range(left + 1, len(expected)):
                truth = same_relation(expected, left, right)
                prediction = same_relation(final_labels, left, right)
                tp += int(truth and prediction)
                fp += int(not truth and prediction)
                fn += int(truth and not prediction)
                tn += int(not truth and not prediction)
                if valid:
                    order_agree += int(
                        same_relation(labels_a, left, right) == same_relation(labels_b, left, right)
                    )
                    order_pairs += 1
        rows.append(
            {
                **case,
                "labels_permutation_0": labels_a,
                "labels_permutation_1": labels_b,
                "final_intersection_labels": final_labels,
                "parse_valid": valid,
                "exact_partition": int(valid and final_labels == expected),
            }
        )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    summary = {
        "cases": len(cases),
        "parse_rate": parsed / len(cases),
        "exact_partition_rate": exact / len(cases),
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_specificity": specificity,
        "order_pair_agreement": order_agree / order_pairs if order_pairs else 0.0,
        "pairwise_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    summary["gate_passed"] = bool(
        summary["parse_rate"] == 1.0
        and precision >= 0.95
        and recall >= 0.85
        and specificity >= 0.95
        and summary["order_pair_agreement"] >= 0.95
    )
    write_jsonl(args.output_dir / "pilot_cases.jsonl", rows)
    (args.output_dir / "pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "pilot_report.md").write_text(
        "\n".join(
            [
                "# Obs1 Qwen clustering pilot",
                "",
                f"- Parse rate: {summary['parse_rate']:.2%}",
                f"- Exact partition rate: {summary['exact_partition_rate']:.2%}",
                f"- Pairwise precision: {precision:.2%}",
                f"- Pairwise recall: {recall:.2%}",
                f"- Pairwise specificity: {specificity:.2%}",
                f"- Order-pair agreement: {summary['order_pair_agreement']:.2%}",
                f"- Gate passed: {summary['gate_passed']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    return 0 if summary["gate_passed"] else 3


def summarize_full(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes = [(name, [row for row in rows if row["benchmark"] == name]) for name in OPEN_BENCHMARKS]
    scopes.append(("OPEN_MICRO", rows))
    for name, selected_all in scopes:
        selected = [row for row in selected_all if not row["all_invalid"]]
        wrong = [row for row in selected if not row["string_eval_correct"]]
        output.append(
            {
                "benchmark": name,
                "questions_total": len(selected_all),
                "questions_valid": len(selected),
                "questions_all_invalid": len(selected_all) - len(selected),
                "judge_invalid_questions": sum(row["judge_invalid"] for row in selected),
                "fragmentation_mean": statistics.fmean(row["fragmentation_ratio"] for row in selected),
                "fragmentation_median": statistics.median(row["fragmentation_ratio"] for row in selected),
                "order_pair_agreement_mean": statistics.fmean(row["order_pair_agreement"] for row in selected),
                "disagreement_rate": sum(row["disagreement"] for row in selected) / len(selected),
                "string_eval_accuracy": sum(row["string_eval_correct"] for row in selected) / len(selected),
                "semantic_eval_accuracy": sum(row["semantic_eval_correct"] for row in selected) / len(selected),
                "fatal_strict_successes": sum(row["fatal_strict"] for row in wrong),
                "fatal_strict_n": len(wrong),
                "fatal_strict_rate": sum(row["fatal_strict"] for row in wrong) / len(wrong) if wrong else None,
            }
        )
    return output


def analyze_full(
    args: argparse.Namespace,
    questions: dict[str, dict[str, Any]],
    judgments: dict[str, Any],
) -> int:
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for case_id in sorted(questions):
        question = questions[case_id]
        answers = question["answers"]
        counter = question["counter"]
        if not answers:
            rows.append(
                {
                    "benchmark": question["benchmark"],
                    "question_id": question["question_id"],
                    "n_total": 32,
                    "n_valid": 0,
                    "all_invalid": 1,
                }
            )
            continue
        if len(answers) == 1:
            labels_a = labels_b = final_labels = [0]
            judge_invalid = 0
            pair_agreement = 1.0
        else:
            pair = judgments.get(case_id, {})
            labels_a = pair.get(0, {}).get("labels_original_order")
            labels_b = pair.get(1, {}).get("labels_original_order")
            judge_invalid = int(labels_a is None or labels_b is None)
            if judge_invalid:
                labels_a = labels_b = final_labels = list(range(len(answers)))
                pair_agreement = 0.0
            else:
                final_labels = intersect_labels(labels_a, labels_b)
                comparisons = [
                    int(same_relation(labels_a, left, right) == same_relation(labels_b, left, right))
                    for left in range(len(answers))
                    for right in range(left + 1, len(answers))
                ]
                pair_agreement = sum(comparisons) / len(comparisons)
        clusters: list[list[str]] = []
        for label in sorted(set(final_labels)):
            clusters.append([answer for answer, answer_label in zip(answers, final_labels) if answer_label == label])
        masses = [sum(counter[answer] for answer in cluster) for cluster in clusters]
        semantic_max = max(masses)
        semantic_tie = int(sum(mass == semantic_max for mass in masses) > 1)
        semantic_cluster = clusters[masses.index(semantic_max)]
        string_answer, string_count = counter.most_common(1)[0]
        string_tie = int(sum(count == string_count for count in counter.values()) > 1)
        disagreement = int(string_answer not in semantic_cluster)
        correct = question["answer_correct"]
        string_eval_correct = int(correct.get(string_answer, 0) > 0)
        semantic_eval_correct = int(any(correct.get(answer, 0) > 0 for answer in semantic_cluster))
        fatal_strict = int(
            disagreement
            and not string_eval_correct
            and semantic_eval_correct
            and not string_tie
            and not semantic_tie
            and not judge_invalid
        )
        row = {
            "benchmark": question["benchmark"],
            "question_id": question["question_id"],
            "n_total": 32,
            "n_valid": question["n_valid"],
            "all_invalid": 0,
            "judge_invalid": judge_invalid,
            "string_cluster_count": len(counter),
            "semantic_cluster_count": len(clusters),
            "fragmentation_ratio": len(counter) / len(clusters),
            "order_pair_agreement": pair_agreement,
            "string_majority_answer": string_answer,
            "string_majority_count": string_count,
            "string_majority_tie": string_tie,
            "semantic_majority_answers": json.dumps(semantic_cluster, ensure_ascii=False),
            "semantic_majority_mass": semantic_max,
            "semantic_majority_tie": semantic_tie,
            "disagreement": disagreement,
            "string_eval_correct": string_eval_correct,
            "semantic_eval_correct": semantic_eval_correct,
            "fatal_strict": fatal_strict,
        }
        rows.append(row)
        if disagreement or fatal_strict or judge_invalid:
            cases.append(
                {
                    **row,
                    "question": question["question"],
                    "string_counter": dict(counter),
                    "labels_permutation_0": labels_a,
                    "labels_permutation_1": labels_b,
                    "final_intersection_labels": final_labels,
                    "semantic_clusters": [
                        {"answers": cluster, "mass": mass}
                        for cluster, mass in zip(clusters, masses)
                    ],
                }
            )
    summary = summarize_full(rows)
    write_csv(args.output_dir / "qwen_question_level.csv", rows)
    write_csv(args.output_dir / "qwen_fragmentation.csv", summary)
    write_jsonl(args.output_dir / "qwen_cases.jsonl", cases)
    report = [
        "# Obs1: Qwen task-aware answer clustering",
        "",
        "Two deterministic answer-order permutations are judged independently. The final partition is their intersection.",
        "Invalid rollout outputs are excluded; judge parse failures are reported and conservatively left as singleton clusters.",
        "Strict fatal excludes string-majority ties, semantic-majority ties, and judge parse failures.",
        "",
        "| Benchmark | valid n | all invalid | judge invalid | F mean | order agree | string acc | semantic acc | strict fatal / frequency-wrong |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        fatal = "NA" if row["fatal_strict_rate"] is None else f"{row['fatal_strict_rate']:.2%}"
        report.append(
            f"| {row['benchmark']} | {row['questions_valid']} | {row['questions_all_invalid']} | "
            f"{row['judge_invalid_questions']} | {row['fragmentation_mean']:.3f} | "
            f"{row['order_pair_agreement_mean']:.2%} | {row['string_eval_accuracy']:.2%} | "
            f"{row['semantic_eval_accuracy']:.2%} | {fatal} "
            f"({row['fatal_strict_successes']}/{row['fatal_strict_n']}) |"
        )
    (args.output_dir / "qwen_fragmentation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "cases": len(cases), "summary": summary}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--judge-mode", choices=("cluster", "pairwise"), default="cluster")
    parser.add_argument(
        "--pair-protocol", choices=("strict_v1", "semantic_v2"), default="strict_v1"
    )
    parser.add_argument(
        "--pair-cluster-method", choices=("complete_link", "mutual_cc"), default="complete_link"
    )
    parser.add_argument("--stage", choices=("infer", "infer-nli", "analyze"), required=True)
    parser.add_argument("--nli-veto", action="store_true")
    parser.add_argument("--nli-model", type=Path)
    parser.add_argument("--nli-judgments", type=Path)
    parser.add_argument("--nli-batch-size", type=int, default=256)
    parser.add_argument("--nli-max-length", type=int, default=64)
    parser.add_argument("--nli-veto-max-entailment-prob", type=float, default=0.25)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.experiment_root) if args.mode == "full" else {}
    cases = list(questions.values()) if args.mode == "full" else pilot_cases()
    if args.stage == "infer":
        return infer_pairwise(args, cases) if args.judge_mode == "pairwise" else infer(args, cases)
    if args.stage == "infer-nli":
        return infer_nli_pairs(args, cases)
    if args.judge_mode == "pairwise":
        pair_judgments = load_pair_judgments(args)
        if args.nli_veto:
            annotate_nli_veto(cases, pair_judgments, load_nli_judgments(args))
        if args.mode == "pilot":
            return analyze_pairwise_pilot(args, cases, pair_judgments)
        return analyze_pairwise_full(args, questions, pair_judgments)
    judgments = load_judgments(args)
    if args.mode == "pilot":
        return analyze_pilot(args, cases, judgments)
    return analyze_full(args, questions, judgments)


if __name__ == "__main__":
    raise SystemExit(main())
