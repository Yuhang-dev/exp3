"""Task scorers used by the end-to-end generation loop."""

from collections import Counter
import re
import string


SCORER_VERSION = "exp3-scorers-v5-2026-09-25"


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value):
        excluded = set(string.punctuation)
        return "".join(character for character in value if character not in excluded)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def token_f1(prediction, ground_truth):
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(ground_truth).split()
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_hotpotqa(prediction, answers):
    f1 = max(token_f1(prediction, answer) for answer in answers)
    exact = max(
        float(normalize_answer(prediction) == normalize_answer(answer))
        for answer in answers
    )
    return {
        "metric": "longbench_official_token_f1",
        "scorer_version": SCORER_VERSION,
        "score": 100.0 * f1,
        "exact_match": exact,
        "parsed_answer": prediction.strip(),
        "target_accuracy": None,
        "all_target_em": None,
    }


def parse_synthetic(prediction, target_keys):
    parsed = {}
    for key in target_keys:
        match = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(key)}\s*[:=]\s*([A-Za-z0-9_-]+)",
            prediction,
            flags=re.IGNORECASE,
        )
        if match:
            parsed[key] = match.group(1)
    return parsed


def score_synthetic(prediction, answers, target_keys):
    parsed = parse_synthetic(prediction, target_keys)
    correct = {
        key: int(parsed.get(key) == answers[key])
        for key in target_keys
    }
    accuracy = sum(correct.values()) / len(target_keys)
    all_correct = float(all(correct.values()))
    return {
        "metric": "per_target_exact_match",
        "scorer_version": SCORER_VERSION,
        "score": 100.0 * accuracy,
        "exact_match": all_correct,
        "parsed_answer": parsed,
        "target_accuracy": accuracy,
        "all_target_em": all_correct,
        "target_correct": correct,
    }


def score_ruler(prediction, answers, scorer_prefix):
    scorer_text = scorer_prefix + prediction
    official_hits = [int(answer.lower() in scorer_text.lower()) for answer in answers]
    raw_hits = [int(answer.lower() in prediction.lower()) for answer in answers]
    normalized_prediction = normalize_answer(scorer_text)
    normalized_hits = [
        int(normalize_answer(answer) in normalized_prediction)
        for answer in answers
    ]
    recall = sum(official_hits) / len(official_hits)
    all_correct = float(all(official_hits))
    return {
        "metric": "ruler_official_substring_recall",
        "scorer_version": SCORER_VERSION,
        "score": 100.0 * recall,
        "exact_match": all_correct,
        "parsed_answer": scorer_text,
        "target_accuracy": recall,
        "all_target_em": all_correct,
        "target_correct": official_hits,
        "raw_target_correct": raw_hits,
        "normalized_target_correct": normalized_hits,
        "raw_substring_score": 100.0 * sum(raw_hits) / len(raw_hits),
        "normalized_substring_score": 100.0 * sum(normalized_hits) / len(normalized_hits),
        "scorer_text": scorer_text,
    }


def parse_ruler_output(output, prefix="Answer:"):
    """FlashPrefill ruler/utils.py parse_output (baa6120)."""
    patterns = [
        re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(output)
        if match is not None:
            return re.sub(f"^{re.escape(prefix)}", "", match[1].strip(), flags=re.IGNORECASE).strip()
    return None


def score_ruler_qa(prediction, answers, scorer_prefix):
    """RULER qa_1/qa_2: upstream default_post_process substring_exact_match, max over raw and parsed."""
    scorer_text = scorer_prefix + prediction
    parsed = parse_ruler_output(scorer_text)
    texts = [scorer_text] + ([parsed] if parsed is not None else [])
    hit = max(
        float(any(normalize_answer(answer) in normalize_answer(text) for answer in answers))
        for text in texts
    )
    return {
        "metric": "ruler_qa_substring_exact_match",
        "scorer_version": SCORER_VERSION,
        "score": 100.0 * hit,
        "exact_match": hit,
        "parsed_answer": parsed,
        "target_accuracy": hit,
        "all_target_em": hit,
        "scorer_text": scorer_text,
    }


def score_longbench_v2(prediction, answers):
    """Mirror LongBench v2 pred.py's exact answer extraction."""
    response = prediction.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match is None:
        match = re.search(r"The correct answer is ([A-D])", response)
    parsed = match.group(1) if match is not None else None
    expected = str(answers[0])
    correct = float(parsed == expected)
    return {
        "metric": "longbench_v2_official_accuracy",
        "scorer_version": SCORER_VERSION,
        "score": 100.0 * correct,
        "exact_match": correct,
        "parsed_answer": parsed,
        "target_accuracy": correct,
        "all_target_em": correct,
    }


def score_prediction(sample, prediction):
    if sample["task"] == "synthetic_kv_retrieval":
        return score_synthetic(prediction, sample["answers"], sample["target_keys"])
    if sample["task"] == "hotpotqa":
        return score_hotpotqa(prediction, sample["answers"])
    if sample["task"] == "ruler" and sample["variant"].startswith("ruler_qa"):
        return score_ruler_qa(prediction, sample["answers"], sample["scorer_prefix"])
    if sample["task"] == "ruler":
        return score_ruler(prediction, sample["answers"], sample["scorer_prefix"])
    if sample["task"] == "longbench_v2":
        return score_longbench_v2(prediction, sample["answers"])
    raise ValueError(f"unsupported task: {sample['task']}")
