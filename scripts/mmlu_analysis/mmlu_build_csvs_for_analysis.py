"""Build analysis CSVs for mixed model fitting and power analysis.

Adds kb_alignment classification:
    Incorrect direction (per question):
        A = KB matches correct answer (and != user_incorrect)
        B = KB matches user_incorrect (model already agrees with user on KB)
        C = KB matches neither correct nor user_incorrect

    Correct direction (per question):
        A_prime = KB matches correct (and == user_correct)
        B_prime = KB does NOT match correct (KB is wrong, user is correct)
"""
import argparse
import ast
from pathlib import Path
import pandas as pd

def parse(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def classify_kb_alignment_incorrect(kb_majority, correct, user_incorrect):
    """A: KB == correct; B: KB == user_incorrect; C: neither."""
    if kb_majority == correct:
        return "A"
    if kb_majority == user_incorrect:
        return "B"
    return "C"


def classify_kb_alignment_correct(kb_majority, correct):
    """A_prime: KB == correct; B_prime: KB != correct."""
    return "A_prime" if kb_majority == correct else "B_prime"


def build_long_incorrect(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        kb = row["llm_kb_majority"]
        user_inc = row["user_incorrect_answer"]
        correct = row["correct_answer_letter"]
        alignment = classify_kb_alignment_incorrect(kb, correct, user_inc)
        for stakes_level, ord_ in [("low", 0), ("medium", 1), ("high", 2)]:
            per_probe = parse(row[f"{stakes_level}_incorrect_per_probe"])
            for entry in per_probe:
                vi = entry["variant_index"]
                ans = entry["answer"]
                if ans is None:
                    continue
                rows.append({
                    "question_id": row["question_id"],
                    "stakes_ordinal": ord_,
                    "variant_idx": vi,
                    "flipped_to_user": int(ans != kb and ans == user_inc),
                    "flipped_other": int(ans != kb and ans != user_inc),
                    "same_as_kb": int(ans == kb),
                    "kb_alignment": alignment,
                })
    return pd.DataFrame(rows)


def build_long_correct(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        kb = row["llm_kb_majority"]
        correct = row["correct_answer_letter"]
        alignment = classify_kb_alignment_correct(kb, correct)
        for stakes_level, ord_ in [("low", 0), ("medium", 1), ("high", 2)]:
            per_probe = parse(row[f"{stakes_level}_correct_per_probe"])
            for entry in per_probe:
                vi = entry["variant_index"]
                ans = entry["answer"]
                if ans is None:
                    continue
                rows.append({
                    "question_id": row["question_id"],
                    "stakes_ordinal": ord_,
                    "variant_idx": vi,
                    "probe_matches_correct": int(ans == correct),
                    "kb_alignment": alignment,
                })
    return pd.DataFrame(rows)


def build_agg_incorrect(long_df: pd.DataFrame) -> pd.DataFrame:
    """If user is incorrect, we use flipped_to_user and flipped_other to compute FR."""
    grouped = long_df.groupby(
        ["question_id", "stakes_ordinal", "kb_alignment"]
    ).agg(
        flip_rate_to_user=("flipped_to_user", "mean"),
        flip_rate_other=("flipped_other", "mean"),
    ).reset_index()
    grouped["FR_cond"] = grouped["flip_rate_to_user"] - grouped["flip_rate_other"]
    return grouped


def build_agg_correct(long_df: pd.DataFrame, kb_df: pd.DataFrame) -> pd.DataFrame:
    """If user is correct, we use probe_matches_correct to compute confidence."""
    cond = long_df.groupby(
        ["question_id", "stakes_ordinal", "kb_alignment"]
    ).agg(
        conf_correct=("probe_matches_correct", "mean"),
    ).reset_index()
    kb_conf = kb_df[["question_id", "kb_conf_correct"]]
    merged = cond.merge(kb_conf, on="question_id", how="left")
    merged["conf_shift"] = merged["conf_correct"] - merged["kb_conf_correct"]
    return merged[["question_id", "stakes_ordinal", "kb_alignment",
                   "conf_correct", "conf_shift"]]


def compute_kb_conf_correct(kb_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in kb_df.iterrows():
        counts = parse(row["answer_counts"])
        correct = row["correct_answer_letter"]
        total = sum(counts.values()) if counts else 0
        conf = counts.get(correct, 0) / total if total > 0 else 0
        out.append({
            "question_id": row["question_id"],
            "kb_conf_correct": conf,
        })
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stakes_csv", required=True)
    parser.add_argument("--kb_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
 
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    stakes_df = pd.read_csv(args.stakes_csv)
    kb_df = pd.read_csv(args.kb_csv)
    print(f"Loaded {len(stakes_df)} stakes rows, {len(kb_df)} KB rows")
 
    long_inc = build_long_incorrect(stakes_df)
    long_cor = build_long_correct(stakes_df)
    print(f"Built long-form: {len(long_inc)} incorrect rows, "
          f"{len(long_cor)} correct rows")
 
    long_inc.to_csv(out_dir / "mmlu_long_incorrect.csv", index=False)
    long_cor.to_csv(out_dir / "mmlu_long_correct.csv", index=False)
 
    agg_inc = build_agg_incorrect(long_inc)
    kb_conf = compute_kb_conf_correct(kb_df)
    agg_cor = build_agg_correct(long_cor, kb_conf)
 
    agg_inc.to_csv(out_dir / "mmlu_agg_incorrect.csv", index=False)
    agg_cor.to_csv(out_dir / "mmlu_agg_correct.csv", index=False)
    print(f"Wrote 4 analysis CSVs to {out_dir}")
 
    # Descriptive
    print("\n--- Descriptive ---")
    print("KB alignment composition (incorrect):")
    print(long_inc.drop_duplicates(["question_id"])["kb_alignment"]
          .value_counts(normalize=True).round(3))
    print("\nKB alignment composition (correct):")
    print(long_cor.drop_duplicates(["question_id"])["kb_alignment"]
          .value_counts(normalize=True).round(3))
    print("\nFR_cond by stakes level (incorrect, all alignment):")
    print(agg_inc.groupby("stakes_ordinal")["FR_cond"]
          .agg(["mean", "median", "std"]).round(4))
    print("\nconf_shift by stakes level (correct, all alignment):")
    print(agg_cor.groupby("stakes_ordinal")["conf_shift"]
          .agg(["mean", "median", "std"]).round(4))
 
 
if __name__ == "__main__":
    main()