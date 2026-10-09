from argparse import ArgumentParser
import os
from glob import glob
from evaluate import _load_yaml_config
from joblib import Memory, Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt

from benchmarks.longbench.calculate_metrics import dataset2metric
from benchmarks.math500.calculate_metrics import score_math500

memory = Memory(location="cachedir", compress=True)

RULER_METRICS = ["cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery", "niah_multivalue", "niah_single_1", "niah_single_2", "niah_single_3", "qa_1", "qa_2", "vt"]

def get_base_parser():
    parser = ArgumentParser(description="Plot accuracy vs cache size")
    parser.add_argument("--output_dir", "-od", type=str, default="out", help="Output directory containing the results")
    parser.add_argument("--model", "-m", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--dataset", "-d", type=str, default="math500", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="hotpotqa_e", help="Data dir")
    return parser


def score_longbench_e_single(dataset, prediction, ground_truths, all_classes):
    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        prediction = prediction.lstrip("\n").split("\n")[0]
    score = 0.0
    for ground_truth in ground_truths:
        score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
    return score


def get_data(model, output_dir, dataset="math500", data_dir="hotpotqa_e", score=True):
    """
    Load data from output directory and compute scores if needed.
    
    Args:
        model (str): Model name.
        output_dir (str): Output directory containing results.
        dataset (str): Dataset name.
        data_dir (str): Data directory name (not needed for math500).
        score (bool): Whether to compute scores.
        
    Returns:
        pd.DataFrame: DataFrame containing results and scores.
    
    """
    components = [
        dataset,
        str(data_dir) if data_dir else "",
        model.replace("/", "--"),
        "*",
    ]
    dir_name = "__".join(filter(None, components))  # Filter None/empty strings
    subdirs = glob(os.path.join(output_dir, dir_name))
    # more_subdirs = [glob(os.path.join(s, "*/")) for s in subdirs]
    # collapse into a single list
    # subdirs += [item for sublist in more_subdirs for item in sublist]
    print(f"Found {len(subdirs)} subdirectories")

    df_list = []
    for subdir in subdirs:
        config_file = os.path.join(subdir, "config.yaml")
        if not os.path.exists(config_file):
            print(f"Config file {config_file} does not exist, skipping...")
            continue
        config = _load_yaml_config(config_file)
        preds_file = os.path.join(subdir, "predictions.parquet")
        print(f"Reading {preds_file}...")
        preds = pd.read_parquet(preds_file)
        # add config to metrics
        for config_name, config_value in config.items():
            preds[config_name] = config_value
        df_list.append(preds)
    # return df_list
    df = pd.concat(df_list, ignore_index=True)
    if score:
        match dataset:
            case "math500":
                predictions = df["predicted_answer"].tolist()
                answers = df["answer"].tolist()
                results = Parallel(n_jobs=8, verbose=1)(
                    delayed(score_math500)(pred, ans)
                    for pred, ans in zip(predictions, answers)
                )
            case "longbench-e":
                tasks = df["task"].apply(lambda x: x.removesuffix("-e")).tolist()
                predictions = df["predicted_answer"].tolist()
                answers = df["answers"].tolist()
                all_classes_list = df["all_classes"].tolist()
                results = Parallel(n_jobs=8, verbose=1)(
                    delayed(score_longbench_e_single)(task, pred, ans, all_classes)
                    for task, pred, ans, all_classes in zip(tasks, predictions, answers, all_classes_list)
                )
            case _:
                raise ValueError(f"Unsupported dataset: {dataset}")
        df['all'] = results

    return df


# Backward compatibility wrapper
@memory.cache
def get_data_math500(model, output_dir, score=True):
    return get_data(model, output_dir, dataset="math500", data_dir=None, score=score)


@memory.cache
def get_times(dataset, data_dir, model, output_dir):
    components = [
        dataset,
        str(data_dir) if data_dir else "",
        model.replace("/", "--"),
        "*", #args.press_name,
        "*", #f"{args.compression_ratio:.2f}",
    ]
    subdirs = glob(os.path.join(output_dir, "__".join(components)))
    # more_subdirs = [glob(os.path.join(s, "*/")) for s in subdirs]
    # collapse into a single list
    # subdirs += [item for sublist in more_subdirs for item in sublist]
    print(f"Found {len(subdirs)} subdirectories")

    data = []
    for subdir in subdirs:
        config_file = os.path.join(subdir, "config.yaml")
        if not os.path.exists(config_file):
            print(f"Config file {config_file} does not exist, skipping...")
            continue
        config = _load_yaml_config(config_file)
        predictions_file = os.path.join(subdir, "times.csv")
        # HACK: replace fraction0.100 with fraction0.010
        if "fraction0.100" in predictions_file:
            predictions_file = predictions_file.replace("fraction0.100", "fraction0.010")
        if not os.path.exists(predictions_file):
            print(f"Predictions file {predictions_file} does not exist, skipping...")
            continue
        predictions = pd.read_csv(predictions_file)
        # join config with predictions
        for name, value in config.items():
            predictions[name] = value
        # predictions_and_config = pd.concat([pd.DataFrame([config]), predictions], axis=1)
        data.append(predictions)
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

# 
# Plotting parameters
# 
plt.rcParams.update(
    {
        # "font.family": "serif",
        # "font.serif": ["Times New Roman"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "no_press": "grey",
    "streaming_llm": "cyan",
    "decoding_streaming_llm": "cyan",
    "pyramidkv": "magenta",
    "snapkv": "teal",
    "decoding_snapkv": "teal",
    "random": "grey",
    "express_32_32": "maroon",
    "express_new_32_32": "maroon",
    "decoding_adakv_expected_attention_e2": "red",
    "decoding_keydiff": "orange",
    "decoding_knorm": "green",
    "decoding_balancekv": "blue",
    "balancekv_express": "blue",
}
    
MARKERS = {
    "no_press": "s",
    "streaming_llm": "D",
    "decoding_streaming_llm": "D",
    "pyramidkv": "v",
    "decoding_snapkv": "v",
    "snapkv": "x",
    "random": "P",
    "express_new_32_32": "o",
    "decoding_balancekv": "^",
    "balancekv_express": "^",
}
LINE_STYLES = {
    "express_new_32_32": "-",
    # Other methods (decoding_X and X share a style since they share an alias)
    "snapkv": "--",
    "decoding_snapkv": "--",
    "streaming_llm": "-.",
    "decoding_streaming_llm": "-.",
    "pyramidkv": ":",
    "decoding_pyramidkv": ":",
    "decoding_expected_attention": (0, (3, 1, 1, 1)),  # dash-dot-dot
    "decoding_keydiff": (0, (5, 1)),                    # long dash
    "decoding_knorm": (0, (1, 1)),                      # densely dotted
    "decoding_balancekv": (0, (3, 1)),                  # short dash
    "balancekv_express": "-",
}

TICK_FONT_SIZE = 7
MINOR_TICK_FONT_SIZE = 3
LABEL_FONT_SIZE = 7.75
LEGEND_FONT_SIZE = 7
MARKER_ALPHA = 0.75
MARKER_SIZE = 3
LINE_ALPHA = 0.75
OUTWARD = 4
TITLE_FONT_SIZE = 12
NEURIPS_WIDTH = 5.5
NEURIPS_HEIGHT = 2

METHOD_ALIASES = {
    "snapkv": "SnapKV",
    "streaming_llm": "StreamingLLM",
    "pyramidkv": "PyramidKV",
    "express_32_32": "Thinformer Express",
    "express_new_32_32": "Thinformer Express",
    "decoding_snapkv": "SnapKV",
    "decoding_streaming_llm": "StreamingLLM",
    "decoding_pyramidkv": "PyramidKV",
    "decoding_expected_attention": "Expected Attention",
    "decoding_keydiff": "KeyDiff",
    "decoding_knorm": "Knorm",
    "decoding_balancekv": "BalanceKV",
    "balancekv_express": "BalanceKV Express",
}
LONGBENCH_E_TASK_ALIASES = {
    "2wikimqa_e": "2WikiMultihopQA-E",
    "gov_report_e": "GovReport-E",
    "hotpotqa_e": "HotpotQA-E",
    "lcc_e": "LCC-E",
    "multi_news_e": "MultiNews-E",
    "multifieldqa_en_e": "MultiFieldQA-en-E",
    "passage_count_e": "PassageCount-E",
    "passage_retrieval_en_e": "PassageRetrieval-en-E",
    "qasper_e": "Qasper-E",
    "repobench-p_e": "RepoBench-P-E",
    "samsum_e": "SAMSum-E",
    "trec_e": "TREC-E",
    "triviaqa_e": "TriviaQA-E",
}
LONGBENCH_E_TASK_METRIC_NAMES = {
    "multifieldqa_en_e": "F1 Score",
    "qasper_e": "F1 Score",
    "lcc_e" : "Edit Sim",
    "repobench-p_e": "Edit Sim"
}