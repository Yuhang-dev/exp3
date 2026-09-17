"""Re-score saved generations and classify synthetic retrieval failures."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import re

from scoring import score_prediction


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def contains_token(text, token, ignore_case=False):
    flags = re.IGNORECASE if ignore_case else 0
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
        text,
        flags=flags,
    ) is not None


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stored_score(prediction):
    value = prediction["score"]
    return float(value["score"] if isinstance(value, dict) else value)


def classify_target(text, parsed_value, expected_value, strict_correct, key_present):
    exact_value_present = contains_token(text, expected_value)
    folded_value_present = contains_token(text, expected_value, ignore_case=True)
    if strict_correct:
        failure = "correct"
    elif parsed_value is not None and parsed_value.casefold() == expected_value.casefold():
        failure = "case_only"
    elif folded_value_present and parsed_value is not None:
        failure = "expected_value_misbound"
    elif folded_value_present:
        failure = "correct_value_unparsed"
    elif parsed_value is not None:
        failure = "wrong_value_after_key"
    elif key_present:
        failure = "key_present_no_value"
    else:
        failure = "omitted"
    return failure, exact_value_present, folded_value_present


def synthetic_value_pattern(target_key, expected_value, parsed_value, answer_values):
    if parsed_value is None:
        return "no_value"
    if parsed_value == expected_value:
        return "correct"
    if parsed_value.casefold() == expected_value.casefold():
        return "case_only"
    if parsed_value in answer_values:
        return "duplicate_other_target"
    namespace = target_key[3:-1]
    shared_prefix = f"VAL{namespace}"
    suffix_start = len(shared_prefix) + 1
    if (
        parsed_value.startswith(shared_prefix)
        and len(parsed_value) >= suffix_start
        and parsed_value[suffix_start:] == expected_value[suffix_start:]
    ):
        return "correct_suffix_wrong_index"
    if expected_value.startswith(parsed_value):
        return "truncated_correct_value"
    if parsed_value.startswith(shared_prefix):
        return "same_namespace_hallucination"
    return "other"


def analyze(run_dir):
    inputs = {
        item["sample_id"]: item
        for item in read_jsonl(run_dir / "inputs.jsonl")
    }
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    sample_rows = []
    target_rows = []

    for prediction in predictions:
        sample = inputs[prediction["sample_id"]]
        rescored = score_prediction(sample, prediction["text"])
        stored = stored_score(prediction)
        common = {
            "task": prediction["task"],
            "length_label": prediction["length_label"],
            "sample_id": prediction["sample_id"],
            "method": prediction["method"],
            "config_id": prediction["config_id"],
        }
        sample_row = {
            **common,
            "stored_score": stored,
            "rescored_score": float(rescored["score"]),
            "score_matches": int(abs(stored - float(rescored["score"])) < 1e-9),
            "value_substring_score": "",
            "strict_correct_targets": "",
            "value_present_targets": "",
            "target_count": "",
            "end_reason": prediction["end_reason"],
            "generated_tokens": len(prediction["generated_token_ids"]),
            "parsed_answer": json.dumps(rescored["parsed_answer"], ensure_ascii=False),
            "text": prediction["text"],
        }

        if sample["task"] == "synthetic_kv_retrieval":
            parsed = rescored["parsed_answer"]
            strict_count = 0
            value_count = 0
            answer_values = list(sample["answers"].values())
            for target_index, key in enumerate(sample["target_keys"]):
                expected = sample["answers"][key]
                parsed_value = parsed.get(key)
                strict_correct = parsed_value == expected
                key_present = contains_token(prediction["text"], key, ignore_case=True)
                failure, exact_present, folded_present = classify_target(
                    prediction["text"],
                    parsed_value,
                    expected,
                    strict_correct,
                    key_present,
                )
                strict_count += int(strict_correct)
                value_count += int(folded_present)
                evidence = sample["evidence_positions"][target_index]
                target_rows.append({
                    **common,
                    "target_index": target_index,
                    "evidence_token_start": evidence["token_span"][0],
                    "evidence_token_end": evidence["token_span"][1],
                    "evidence_relative_position": evidence["token_span"][0] / sample["actual_tokens"],
                    "target_key": key,
                    "expected_value": expected,
                    "parsed_value": "" if parsed_value is None else parsed_value,
                    "strict_correct": int(strict_correct),
                    "key_present": int(key_present),
                    "value_present_exact_case": int(exact_present),
                    "value_present_casefold": int(folded_present),
                    "classification": failure,
                    "value_pattern": synthetic_value_pattern(
                        key,
                        expected,
                        parsed_value,
                        answer_values,
                    ),
                })
            total = len(sample["target_keys"])
            sample_row.update({
                "value_substring_score": 100.0 * value_count / total,
                "strict_correct_targets": strict_count,
                "value_present_targets": value_count,
                "target_count": total,
            })
        sample_rows.append(sample_row)

    return sample_rows, target_rows


def make_markdown(sample_rows, target_rows):
    groups = defaultdict(list)
    target_groups = defaultdict(list)
    for row in sample_rows:
        groups[(row["task"], row["length_label"], row["config_id"])].append(row)
    for row in target_rows:
        target_groups[(row["task"], row["length_label"], row["config_id"])].append(row)

    lines = [
        "# Saved-quality diagnosis",
        "",
        "本报告只重算已保存的生成，不重新运行模型。`Value recall` 仅检查标准答案值是否出现在输出中，",
        "用于区分格式/解析损失与真正的检索失败；它不替代主指标。",
        "",
        "| Task | Length | Config | n | Stored | Rescored | Value recall | Value-present gap | Wrong value | No value | Max-token ends |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(groups):
        rows = groups[key]
        targets = target_groups.get(key, [])
        stored = sum(row["stored_score"] for row in rows) / len(rows)
        rescored = sum(row["rescored_score"] for row in rows) / len(rows)
        if targets:
            value_recall = 100.0 * sum(row["value_present_casefold"] for row in targets) / len(targets)
            value_gap = sum(
                row["classification"] in {
                    "case_only", "correct_value_unparsed", "expected_value_misbound"
                }
                for row in targets
            )
            wrong = sum(row["classification"] == "wrong_value_after_key" for row in targets)
            omitted = sum(
                row["classification"] in {"key_present_no_value", "omitted"}
                for row in targets
            )
            value_cell = f"{value_recall:.2f}"
        else:
            value_gap = wrong = omitted = 0
            value_cell = "—"
        max_ends = sum(row["end_reason"] == "max_new_tokens" for row in rows)
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(rows)} | "
            f"{stored:.2f} | {rescored:.2f} | {value_cell} | {value_gap} | "
            f"{wrong} | {omitted} | {max_ends} |"
        )

    mismatches = [row for row in sample_rows if not row["score_matches"]]
    lines.extend([
        "",
        "## Consistency checks",
        "",
        f"- Saved predictions: {len(sample_rows)}",
        f"- Stored/rescored mismatches: {len(mismatches)}",
        f"- Synthetic target failures: {sum(not row['strict_correct'] for row in target_rows)} / {len(target_rows)}",
        "",
        "## Synthetic failure morphology",
        "",
        "| Pattern | Count |",
        "| --- | ---: |",
    ])
    patterns = Counter(
        row["value_pattern"] for row in target_rows if not row["strict_correct"]
    )
    for pattern, count in patterns.most_common():
        lines.append(f"| {pattern} | {count} |")
    lines.extend([
        "",
        "## Failed synthetic targets",
        "",
        "| Sample | Config | Position | Key | Expected | Parsed | Class | Pattern |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ])
    for row in target_rows:
        if row["strict_correct"]:
            continue
        lines.append(
            f"| {row['sample_id']} | {row['config_id']} | "
            f"{100 * row['evidence_relative_position']:.0f}% | {row['target_key']} | "
            f"{row['expected_value']} | {row['parsed_value'] or '—'} | "
            f"{row['classification']} | {row['value_pattern']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = arguments()
    sample_rows, target_rows = analyze(args.run_dir)
    write_csv(
        args.run_dir / "quality_rescored.csv",
        sample_rows,
        [
            "task", "length_label", "sample_id", "method", "config_id",
            "stored_score", "rescored_score", "score_matches",
            "value_substring_score", "strict_correct_targets",
            "value_present_targets", "target_count", "end_reason",
            "generated_tokens", "parsed_answer", "text",
        ],
    )
    write_csv(
        args.run_dir / "target_diagnosis.csv",
        target_rows,
        [
            "task", "length_label", "sample_id", "method", "config_id",
            "target_index", "evidence_token_start", "evidence_token_end",
            "evidence_relative_position", "target_key", "expected_value",
            "parsed_value", "strict_correct",
            "key_present", "value_present_exact_case", "value_present_casefold",
            "classification", "value_pattern",
        ],
    )
    report = make_markdown(sample_rows, target_rows)
    (args.run_dir / "QUALITY_DIAGNOSIS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
