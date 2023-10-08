import os
import json
import logging
import jsonlines
import pandas as pd
from summarizer.util import SRC_RESOURCES_DIR
from summarizer.eval.evaluation import evaluate


def evaluation(file_path):
    data = []
    for line in jsonlines.open(file_path, "r"):
        data.append([line["summary"], line["system_sum"], len(line["summary"]), len(line["system_sum"])])
    df = pd.DataFrame(data, columns=["gold_sum", "system_sum", "gold_sum_len", "system_sum_len"])
    df.to_json("eval.json", lines=True, orient="records")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_name = "booksum-10chapt_summary.jsonl"
    file_path = os.path.join(SRC_RESOURCES_DIR, "booksum", file_name)
    eval_metric, scores_df = evaluate(file_path)
    scores_df.to_csv(file_path.split(".")[-2] + ".csv")
    print(scores_df.describe(datetime_is_numeric=True))
    print(json.dumps(eval_metric, indent=4, sort_keys=True))


