"""Task scorers used by the end-to-end generation loop."""

from collections import Counter
import re
import string


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
        "score": 100.0 * accuracy,
        "exact_match": all_correct,
        "parsed_answer": parsed,
        "target_accuracy": accuracy,
        "all_target_em": all_correct,
        "target_correct": correct,
    }


def score_prediction(sample, prediction):
    if sample["task"] == "synthetic_kv_retrieval":
        return score_synthetic(prediction, sample["answers"], sample["target_keys"])
    if sample["task"] == "hotpotqa":
        return score_hotpotqa(prediction, sample["answers"])
    raise ValueError(f"unsupported task: {sample['task']}")
