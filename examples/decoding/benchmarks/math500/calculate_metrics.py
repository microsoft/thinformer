# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import re
from benchmarks.math500.grader import grade_answer

def extract_boxed(pred_answer):
    try:
        return str(pred_answer.split("boxed{")[1].split("}")[0])
    except IndexError:
        return None


def score_aime(pred_answer, true_answer):
    return extract_boxed(pred_answer) == str(true_answer)


def extract_boxed_regex(pred_answer):
    """Extract the last occurrence of a boxed answer.
    Args:
        pred_answer: The predicted answer string.
    Returns:
        The last occurrence of a boxed answer.
    """
    pattern = r".*\\boxed\{(.*)\}(.*?)$"
    match = re.search(pattern, pred_answer, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def score_math500(pred_answer, true_answer):
    """Score using the grader recommended by the MATH500 dataset authors.

    Ref: https://github.com/openai/prm800k?tab=readme-ov-file#answer-grading
    Args:
        pred_answer: The predicted answer string.
        true_answer: The true answer string.
    Returns:
        The score.
    """
    return grade_answer(extract_boxed_regex(pred_answer), true_answer)


def calculate_metrics(df: pd.DataFrame) -> dict:
    correct = 0
    answered = 0
    for index, row in df.iterrows():
        if False:
            correct += score_aime(row["predicted_answer"], row["answer"])
        else:
            correct += score_math500(row["predicted_answer"], row["answer"])
        answered += "boxed{" in row["predicted_answer"]
    return {"correct": correct, "answered": answered, "accuracy": correct / len(df), "total": len(df), "mean_cache_size": df["mean_cache_size"].mean(), "std_cache_size": df["mean_cache_size"].std(), "mean_completion_tokens": df["completion_tokens"].mean(), "std_completion_tokens": df["completion_tokens"].std()}
