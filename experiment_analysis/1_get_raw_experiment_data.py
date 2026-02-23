"""
1_get_raw_experiment_data.py

Stage 1 of the analysis pipeline. The script merges Prolific exports, pulls
data from the experiment database, and writes out CSVs for downstream stages.
It now also produces a sanitised copy of the Stage 1 outputs so later steps can
run without exposing personal identifiers.

Expected Inputs:
- Prolific CSV exports in the folder specified by CONFIG["prolific_csv_folder"]
- Access to the experiment database (see CONFIG["db"])

Outputs (private copy):
- results/0_private_raw/*.csv

Outputs (sanitised public copy used by subsequent stages):
- results/1_raw/prolific_merged.csv (hashed participant IDs)
- results/1_raw/users_raw.csv (surrogate user IDs)
- results/1_raw/events_raw.csv
- results/1_raw/user_completed_raw.csv (includes prolific_start/end columns)
- results/4_overview/prolific_participant_overview.md
"""

import argparse
import os
import shutil
from typing import List, Optional

import pandas as pd
from analysis_config import (
    PROLIFIC_CSV_FOLDER,
    RESULTS_DIR,
    DB_CONFIG,
    EXPLICITELY_REPLACE_GROUP_NAME,
    GROUP,
    COMPARING_GROUP,
    DATASET,
    EXPERIMENT_HANDLE,
    EXPERIMENT_DATE,
)
from analysis_utils import merge_prolific_csvs, connect_db, make_stage_dirs, fetch_data_as_dataframe, \
    get_data_for_prolific_users, get_study_name_description_if_possible
from sanitize_for_publication import sanitize_stage_one


REPORT_FILENAME = "prolific_participant_overview.md"


def _resolve_prolific_folder(group_name: str) -> str:
    """Construct the prolific CSV directory for a given study group."""
    return f"data_{DATASET}_{EXPERIMENT_HANDLE}_{group_name}_{EXPERIMENT_DATE}/"


def _format_value_counts(series: pd.Series, top_n: int = 5) -> List[str]:
    """Return formatted lines for the top N values in a categorical series."""
    if series.empty:
        return ["No data available"]
    counts = series.fillna("Missing").value_counts().head(top_n)
    total = float(series.size) if series.size else 1.0
    lines = []
    for value, count in counts.items():
        percent = (count / total) * 100 if total else 0
        lines.append(f"- {value}: {count} ({percent:.1f}%)")
    return lines


def build_prolific_report(
    prolific_df: pd.DataFrame,
    output_dir: str,
    condition_column: Optional[str] = None,
) -> str:
    """Generate a textual overview of the Prolific export and save it as Markdown."""
    if prolific_df is None or prolific_df.empty:
        raise ValueError("Cannot build report: merged Prolific dataset is empty.")

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, REPORT_FILENAME)
    total_records = len(prolific_df)

    lines: List[str] = ["# Prolific Participant Overview", ""]
    lines.append(f"- Total participants in merged export: {total_records}")
    if "source_file" in prolific_df.columns:
        source_counts = prolific_df["source_file"].value_counts()
        lines.append("- Source files merged: " + ", ".join(sorted(source_counts.index)))
    lines.append("")

    if condition_column and condition_column in prolific_df.columns:
        condition_counts = prolific_df[condition_column].fillna("Unknown").value_counts()
        lines.append("## Study Conditions")
        for condition, count in condition_counts.items():
            percent = (count / total_records) * 100 if total_records else 0
            lines.append(f"- {condition}: {count} ({percent:.1f}%)")
        lines.append("")

    # Status breakdown
    if "Status" in prolific_df.columns:
        status_counts = prolific_df["Status"].fillna("Missing").value_counts()
        lines.append("## Submission Status")
        for status, count in status_counts.items():
            percent = (count / total_records) * 100 if total_records else 0
            lines.append(f"- {status}: {count} ({percent:.1f}%)")
        lines.append("")

    # Time metadata
    if "Started at" in prolific_df.columns:
        start_times = pd.to_datetime(prolific_df["Started at"], errors="coerce").dropna()
        if not start_times.empty:
            lines.append("## Timeline")
            lines.append(f"- First start: {start_times.min().isoformat()}")
            lines.append(f"- Last start: {start_times.max().isoformat()}")
            if "Completed at" in prolific_df.columns:
                completion_times = pd.to_datetime(prolific_df["Completed at"], errors="coerce").dropna()
                if not completion_times.empty:
                    lines.append(f"- Last completion: {completion_times.max().isoformat()}")
            if "Time taken" in prolific_df.columns:
                durations = pd.to_numeric(prolific_df["Time taken"], errors="coerce").dropna()
                if not durations.empty:
                    median_minutes = durations.median() / 60.0
                    percentile_minutes = durations.quantile(0.95) / 60.0
                    lines.append(f"- Median time taken: {median_minutes:.1f} minutes")
                    lines.append(f"- 95th percentile time taken: {percentile_minutes:.1f} minutes")
            lines.append("")

    # Age summary (values above 120 are assumed to be birth years)
    if "Age" in prolific_df.columns:
        raw_age = pd.to_numeric(prolific_df["Age"], errors="coerce")
        valid_age = raw_age.dropna()
        if not valid_age.empty:
            likely_birth_year = valid_age[valid_age > 120]
            if len(likely_birth_year) > (len(valid_age) * 0.25):
                # Treat values as birth year and derive age assuming current year
                current_year = pd.Timestamp.utcnow().year
                derived_age = (current_year - valid_age).clip(lower=0)
                metric_series = derived_age
                age_note = "interpreted as birth years"
            else:
                metric_series = valid_age
                age_note = "interpreted as ages"
            lines.append("## Age Distribution")
            lines.append(f"- Valid entries: {len(metric_series)}")
            lines.append(f"- Median age: {metric_series.median():.1f}")
            lines.append(f"- Range: {metric_series.min():.0f} – {metric_series.max():.0f}")
            lines.append(f"- Note: values {age_note}. {len(raw_age) - len(valid_age)} entries missing or non-numeric.")
            lines.append("")

    categorical_columns = [
        "Sex",
        "Ethnicity simplified",
        "Country of residence",
        "Country of birth",
        "Nationality",
        "Language",
        "Employment status",
        "Student status",
    ]
    available_categoricals = [c for c in categorical_columns if c in prolific_df.columns]
    if available_categoricals:
        lines.append("## Demographic Fields")
        for column in available_categoricals:
            lines.append(f"### {column}")
            lines.extend(_format_value_counts(prolific_df[column]))
            lines.append("")

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))

    print(f"Participant overview written to {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect raw experiment data and produce sanitised outputs.")
    parser.add_argument(
        "--skip-sanitization",
        action="store_true",
        help="Keep raw CSVs in results/1_raw without creating a sanitised copy."
    )
    parser.add_argument(
        "--drop-free-text",
        action="store_true",
        help="Remove logs/questionnaires/feedback and event details from the sanitised copy."
    )
    parser.add_argument(
        "--salt",
        default=None,
        help="Optional salt used to hash participant identifiers in the sanitised Prolific export."
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None):
    if args is None:
        args = parse_args()

    # Create stage dirs inside the experiment-specific results folder
    stage_dirs = make_stage_dirs(RESULTS_DIR, stages=["raw", "unpacked", "filtered", "overview"])

    # 1. Merge Prolific CSVs
    merged_csv = os.path.join(stage_dirs["raw"], "prolific_merged.csv")
    prolific_df = merge_prolific_csvs(PROLIFIC_CSV_FOLDER, merged_csv)

    # Participant overview report (optionally pooling with a comparison group)
    report_dir = stage_dirs.get("overview", stage_dirs["raw"])
    report_frames = [prolific_df.assign(Condition=GROUP)]

    if COMPARING_GROUP:
        comparing_folder = _resolve_prolific_folder(COMPARING_GROUP)
        if os.path.isdir(comparing_folder):
            try:
                comparing_df = merge_prolific_csvs(comparing_folder)
            except ValueError as exc:
                print(f"Warning: Unable to load comparison group Prolific data: {exc}")
            else:
                if not comparing_df.empty:
                    report_frames.append(comparing_df.assign(Condition=COMPARING_GROUP))
                    print(
                        f"Including comparison group '{COMPARING_GROUP}' in Prolific overview "
                        f"(participants: {len(comparing_df)})"
                    )
        else:
            print(
                f"Warning: Comparison group folder '{comparing_folder}' not found. "
                "Generating overview for the primary group only."
            )

    combined_report_df = pd.concat(report_frames, ignore_index=True, sort=False)
    if "Participant id" in combined_report_df.columns:
        combined_report_df = combined_report_df.drop_duplicates(subset=["Participant id"])

    build_prolific_report(combined_report_df, report_dir, condition_column="Condition")

    # 2. Pull from DB (using fetch_data_as_dataframe for flexibility)
    conn = connect_db(DB_CONFIG)
    users = fetch_data_as_dataframe("SELECT * FROM users", conn)
    events = fetch_data_as_dataframe("SELECT * FROM events", conn)
    user_completed = fetch_data_as_dataframe("SELECT * FROM user_completed", conn)

    # 3. Filter by Prolific users
    users = get_data_for_prolific_users(users, merged_csv)
    # For events and user_completed, filter by user_id column matching users['id']
    user_ids = set(users['id'].astype(str))
    events = events[events['user_id'].astype(str).isin(user_ids)]
    user_completed = user_completed[user_completed['user_id'].astype(str).isin(user_ids)]

    # Apply group name replacement if configured
    if EXPLICITELY_REPLACE_GROUP_NAME:
        import json
        # Load prolific data to get source_file information
        prolific_df = pd.read_csv(merged_csv)
        
        # Merge source_file info into users dataframe based on prolific_id
        if 'source_file' in prolific_df.columns:
            prolific_source_map = prolific_df.set_index('Participant id')['source_file'].to_dict()
            users['source_file'] = users['prolific_id'].map(prolific_source_map)
        
        for idx, row in users.iterrows():
            try:
                profile_data = json.loads(row['profile'])
                if 'study_group_name' in profile_data:
                    current_name = profile_data['study_group_name']
                    source_file = row.get('source_file', '')
                    
                    # Check if we have replacement mappings for this source file
                    if source_file and source_file in EXPLICITELY_REPLACE_GROUP_NAME:
                        source_replacements = EXPLICITELY_REPLACE_GROUP_NAME[source_file]
                        if current_name in source_replacements:
                            profile_data['study_group_name'] = source_replacements[current_name]
                            users.at[idx, 'profile'] = json.dumps(profile_data)
                            print(f"Replaced '{current_name}' with '{source_replacements[current_name]}' for user {row.get('id', idx)} from {source_file}")
            except (json.JSONDecodeError, TypeError):
                # Skip rows with invalid JSON
                continue

    # 4. Filter by study group
    users.to_csv(os.path.join(stage_dirs["raw"], "users_raw.csv"), index=False)
    events.to_csv(os.path.join(stage_dirs["raw"], "events_raw.csv"), index=False)
    user_completed.to_csv(os.path.join(stage_dirs["raw"], "user_completed_raw.csv"), index=False)

    user_experiment_name = get_study_name_description_if_possible(users, GROUP)
    print(f"Raw data collection complete. Results saved in: {RESULTS_DIR}.")
    print(f"Found data for {len(users)} users with experiment names: {user_experiment_name}")

    if not args.skip_sanitization:
        print("\nPreparing private and sanitised Stage 1 copies...")

        # Move raw CSVs to a private sub-folder first; sanitisation reads from there.
        private_raw_dir = os.path.join(RESULTS_DIR, "0_private_raw")
        os.makedirs(private_raw_dir, exist_ok=True)

        raw_files = [f for f in os.listdir(stage_dirs["raw"]) if f.endswith(".csv")]
        for filename in raw_files:
            src = os.path.join(stage_dirs["raw"], filename)
            dst = os.path.join(private_raw_dir, filename)
            shutil.move(src, dst)

        sanitize_stage_one(
            results_dir=RESULTS_DIR,
            output_root=RESULTS_DIR,
            salt=args.salt,
            drop_free_text=args.drop_free_text,
            emit_mapping_path=os.path.join(private_raw_dir, "user_id_mapping.csv"),
            raw_subdir="0_private_raw",
        )

        print(
            "Sanitised CSVs written to results/1_raw. Original raw files preserved in "
            f"{private_raw_dir}."
        )
        print(
            "Downstream scripts now operate on anonymised data in 1_raw. "
            "Do not publish 0_private_raw or the user_id_mapping.csv file."
        )

if __name__ == "__main__":
    main()
