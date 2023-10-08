"""Summary evaluation script."""
import jsonlines
import pandas as pd
from rouge import Rouge
from typing import Tuple, Dict
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import sent_tokenize, word_tokenize
from summarizer.util import GOLD_SUM_FIELD, MODEL_SUM_FIELD

ROUGE = Rouge()
SM_FUNC = SmoothingFunction().method4


def get_eval_score(candidate: str, reference: str) -> Dict:
    """Return rouge and bleu scores."""
    score = ROUGE.get_scores(candidate, reference)[0]

    reference_bleu = [word_tokenize(reference)]
    candidate_bleu = word_tokenize(candidate)
    score["bleu-1"] = sentence_bleu(
        reference_bleu, candidate_bleu, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=SM_FUNC
    )

    reference_bleu = [word_tokenize(sentence) for sentence in sent_tokenize(reference)]
    score["bleu-2"] = sentence_bleu(
        reference_bleu, candidate_bleu, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=SM_FUNC
    )

    return score


def get_eval_metric(eval_file_path: str) -> Tuple[Dict, pd.DataFrame]:
    """Return average rouge and bleu scores."""
    avg_score = {
        "bleu-1": 0.0,
        "bleu-2": 0.0,
        "rouge-1": {
            "f": 0.0,
            "p": 0.0,
            "r": 0.0
        },
        "rouge-2": {
            "f": 0.0,
            "p": 0.0,
            "r": 0.0
        },
        "rouge-l": {
            "f": 0.0,
            "p": 0.0,
            "r": 0.0
        }
    }

    flat_scores = []
    for line in jsonlines.open(eval_file_path):
        score = line["eval_score"]
        flat_scores.append([
            score["bleu-1"], score["bleu-2"], score["rouge-1"]["p"], score["rouge-1"]["r"], score["rouge-1"]["f"],
            score["rouge-l"]["p"], score["rouge-l"]["r"], score["rouge-l"]["f"],
            line["metadata"]["chapter_path"], line["metadata"]["summary_path"]
        ])
        avg_score["bleu-1"] += score["bleu-1"]
        avg_score["bleu-2"] += score["bleu-2"]
        avg_score["rouge-1"]["f"] += score["rouge-1"]["f"]
        avg_score["rouge-1"]["p"] += score["rouge-1"]["p"]
        avg_score["rouge-1"]["r"] += score["rouge-1"]["r"]
        avg_score["rouge-2"]["f"] += score["rouge-2"]["f"]
        avg_score["rouge-2"]["p"] += score["rouge-2"]["p"]
        avg_score["rouge-2"]["r"] += score["rouge-2"]["r"]
        avg_score["rouge-l"]["f"] += score["rouge-l"]["f"]
        avg_score["rouge-l"]["p"] += score["rouge-l"]["p"]
        avg_score["rouge-l"]["r"] += score["rouge-l"]["r"]

    scores_df = pd.DataFrame(flat_scores, columns=["bleu-1", "bleu-2", "rouge-1-p", "rouge-1-r", "rouge-1-f",
                                                   "rouge-l-p", "rouge-l-r", "rouge-l-f", "chapter-path", "summary-path"])
    scores_df.sort_values(by=["rouge-l-f", "bleu-2"], ascending=False, inplace=True)

    num_of_samples = len(flat_scores)
    avg_score["bleu-1"] = float(avg_score["bleu-1"]) / num_of_samples
    avg_score["bleu-2"] = float(avg_score["bleu-2"]) / num_of_samples
    avg_score["rouge-1"]["f"] = float(avg_score["rouge-1"]["f"]) / num_of_samples
    avg_score["rouge-1"]["p"] = float(avg_score["rouge-1"]["p"]) / num_of_samples
    avg_score["rouge-1"]["r"] = float(avg_score["rouge-1"]["r"]) / num_of_samples
    avg_score["rouge-2"]["f"] = float(avg_score["rouge-2"]["f"]) / num_of_samples
    avg_score["rouge-2"]["p"] = float(avg_score["rouge-2"]["p"]) / num_of_samples
    avg_score["rouge-2"]["r"] = float(avg_score["rouge-2"]["r"]) / num_of_samples
    avg_score["rouge-l"]["f"] = float(avg_score["rouge-l"]["f"]) / num_of_samples
    avg_score["rouge-l"]["p"] = float(avg_score["rouge-l"]["p"]) / num_of_samples
    avg_score["rouge-l"]["r"] = float(avg_score["rouge-l"]["r"]) / num_of_samples

    return avg_score, scores_df


def evaluate(file_path: str) -> Tuple[Dict, pd.DataFrame]:
    eval_file_path = file_path.split(".")[0] + "_eval.jsonl"
    with jsonlines.open(eval_file_path, mode="w") as writer:
        for line in jsonlines.open(file_path):
            line["eval_score"] = get_eval_score(candidate=line[MODEL_SUM_FIELD], reference=line[GOLD_SUM_FIELD])
            writer.write(line)

    return get_eval_metric(eval_file_path)

