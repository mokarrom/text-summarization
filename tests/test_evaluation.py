import os
import json
from pytest import approx
import jsonlines

from summarizer.eval.evaluation import get_eval_score, get_eval_metric, evaluate


def test_get_score():
    reference = 'John really loves data science very much and studies it a lot.'
    candidate = 'John very much loves data science and enjoys it a lot.'
    score = get_eval_score(candidate, reference)

    assert score["bleu-1"] > 0.80
    assert score["bleu-2"] > 0.80

    assert score["rouge-1"]["p"] > 0.90
    assert score["rouge-1"]["r"] > 0.80
    assert score["rouge-1"]["f"] > 0.85

    print(json.dumps(score, indent=4, sort_keys=True))

    ref = "this is a dog. it is dog. dog it is. a dog, it is"
    cand = 'it is dog'
    score = get_eval_score(cand, ref)

    assert score["bleu-1"] < 0.10
    assert score["bleu-2"] > 0.70

    print(json.dumps(score, indent=4, sort_keys=True))


def test_get_eval_metric(resource_path, rel_tol):
    eval_file = os.path.join(resource_path, "summary", "booksum-10chapt-sum_davinci.jsonl")
    score, _ = evaluate(eval_file)
    print(f"davinci-003 4k stats:\n{json.dumps(score, indent=4, sort_keys=True)}")

    assert score["bleu-2"] == approx(0.398, rel_tol)
    assert score["rouge-l"]["p"] == approx(0.329, rel_tol)
    assert score["rouge-l"]["r"] == approx(0.152, rel_tol)
    assert score["rouge-l"]["f"] == approx(0.194, rel_tol)

    eval_file = os.path.join(resource_path, "summary", "booksum-10chapt-sum_curie.jsonl")
    score, _ = evaluate(eval_file)
    print(f"curie-001 2k stats:\n{json.dumps(score, indent=4, sort_keys=True)}")

    assert score["bleu-2"] == approx(0.449, rel_tol)
    assert score["rouge-l"]["p"] == approx(0.372, rel_tol)
    assert score["rouge-l"]["r"] == approx(0.090, rel_tol)
    assert score["rouge-l"]["f"] == approx(0.130, rel_tol)

