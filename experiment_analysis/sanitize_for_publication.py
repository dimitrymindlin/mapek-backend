"""
sanitize_for_publication.py

Utility for preparing public-release copies of raw experiment data. The script:
- Generates surrogate user identifiers that stay consistent across tables.
- Computes Prolific timing columns once (if the raw Prolific export is available).
- Drops direct Prolific identifiers and optional free-text columns.

By default the script reads from the active RESULTS_DIR configured in
analysis_config and writes a sanitized copy to <results>/public_release/1_raw.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Dict, Iterable

import pandas as pd

from analysis_config import RESULTS_DIR
from analysis_utils import add_prolific_times


DEFAULT_OUTPUT_ROOT = os.path.join(RESULTS_DIR, "public_release")


def _build_surrogate_ids(ids: Iterable[str], prefix: str = "user") -> Dict[str, str]:
    """Return a deterministic mapping from raw IDs to surrogate IDs."""
    unique_ids = sorted({str(_id) for _id in ids if pd.notna(_id)})
    return {raw_id: f"{prefix}_{index + 1:04d}" for index, raw_id in enumerate(unique_ids)}


def _hash_identifier(identifier: str, salt: str) -> str:
    """Return a short salted hash for an identifier (used for participant IDs)."""
    if pd.isna(identifier):
        return ""
    digest = hashlib.sha256(f"{salt}{identifier}".encode("utf-8")).hexdigest()
    return digest[:16]


def sanitize_stage_one(
    results_dir: str,
    output_root: str,
    salt: str | None = None,
    drop_free_text: bool = False,
    emit_mapping_path: str | None = None,
    raw_subdir: str = "1_raw",
) -> None:
    """Create sanitized copies of the 1_raw assets for publication."""

    raw_dir = os.path.join(results_dir, raw_subdir)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    output_dir = os.path.join(output_root, "1_raw")
    os.makedirs(output_dir, exist_ok=True)

    users_path = os.path.join(raw_dir, "users_raw.csv")
    events_path = os.path.join(raw_dir, "events_raw.csv")
    user_completed_path = os.path.join(raw_dir, "user_completed_raw.csv")
    prolific_path = os.path.join(raw_dir, "prolific_merged.csv")

    users = pd.read_csv(users_path)
    events = pd.read_csv(events_path)
    user_completed = pd.read_csv(user_completed_path)
    prolific_df = pd.read_csv(prolific_path) if os.path.exists(prolific_path) else None

    # Build surrogate user IDs before mutating tables
    surrogate_map = _build_surrogate_ids(users["id"], prefix="participant")
    users["id"] = users["id"].map(surrogate_map)
    events["user_id"] = events["user_id"].map(surrogate_map)
    user_completed["user_id"] = user_completed["user_id"].map(surrogate_map)

    for frame_name, frame, column in (
        ("events_raw.csv", events, "user_id"),
        ("user_completed_raw.csv", user_completed, "user_id"),
    ):
        missing_ids = frame[column].isna().sum()
        if missing_ids:
            print(
                f"WARNING: {missing_ids} rows in {frame_name} referenced unknown user IDs and will be dropped."
            )
            frame.dropna(subset=[column], inplace=True)

    # Compute timing columns if they are not already present
    timing_columns = {"prolific_start_time", "prolific_end_time"}
    has_timings = timing_columns.issubset(user_completed.columns)
    if not has_timings and prolific_df is None:
        raise FileNotFoundError(
            "Prolific export missing and timing columns absent. Provide prolific_merged.csv "
            "or run stage 1 before sanitisation."
        )

    if not has_timings and prolific_df is not None:
        print("Computing timing columns using prolific_merged.csv before dropping identifiers...")
        user_completed, _ = add_prolific_times(user_completed, prolific_df, events)

    # Drop columns that directly identify participants
    columns_to_drop = ["prolific_id", "source_file"]
    for column in columns_to_drop:
        if column in users.columns:
            users = users.drop(columns=[column])
        if column in user_completed.columns:
            user_completed = user_completed.drop(columns=[column])

    if prolific_df is not None:
        # Create a hashed participant column if the caller provided a salt
        if salt:
            prolific_df["participant_hash"] = prolific_df["Participant id"].apply(
                lambda value: _hash_identifier(value, salt)
            )
        # Drop raw identifiers and sensitive timestamps before saving (if requested)
        prolific_columns_to_remove = [
            "Participant id",
            "Submission id",
            "Custom study tncs accepted at",
            "Completion code",
        ]
        prolific_df = prolific_df.drop(
            columns=[col for col in prolific_columns_to_remove if col in prolific_df.columns],
            errors="ignore",
        )

    if drop_free_text:
        free_text_columns = ["logs", "questionnaires", "feedback"]
        removed = [col for col in free_text_columns if col in users.columns]
        if removed:
            print(f"Dropping free-text columns from users_raw.csv: {', '.join(removed)}")
            users = users.drop(columns=removed)
        if "details" in events.columns:
            print("Removing events.details column at caller request.")
            events = events.drop(columns=["details"])
    else:
        print(
            "Free-text columns retained. Review questionnaires/feedback manually before public release."
        )

    # Save sanitized assets
    users.to_csv(os.path.join(output_dir, "users_raw.csv"), index=False)
    events.to_csv(os.path.join(output_dir, "events_raw.csv"), index=False)
    user_completed.to_csv(os.path.join(output_dir, "user_completed_raw.csv"), index=False)

    # Only persist the Prolific export if we stripped identifiers; otherwise callers can discard it
    if prolific_df is not None:
        prolific_output = os.path.join(output_dir, "prolific_summary.csv")
        prolific_df.to_csv(prolific_output, index=False)
        print(f"Wrote redacted Prolific summary to {prolific_output}")

    if emit_mapping_path:
        mapping_df = (
            pd.DataFrame(sorted(surrogate_map.items()), columns=["raw_user_id", "participant_id"])
            if surrogate_map
            else pd.DataFrame(columns=["raw_user_id", "participant_id"])
        )
        mapping_df.to_csv(emit_mapping_path, index=False)
        print(f"Internal ID mapping stored at {emit_mapping_path}. Do not publish this file.")

    print(f"Sanitized raw assets written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create sanitized Stage 1 data for publication.")
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Path to the experiment results directory containing 1_raw." \
             " Defaults to analysis_config.RESULTS_DIR.",
    )
    parser.add_argument(
        "--raw-subdir",
        default="1_raw",
        help="Sub-directory under results-dir from which raw CSVs should be read (e.g., 0_private_raw).",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the sanitized copy will be written (subdir 1_raw).",
    )
    parser.add_argument(
        "--salt",
        default=None,
        help="Optional salt for hashing participant identifiers inside the redacted Prolific summary.",
    )
    parser.add_argument(
        "--drop-free-text",
        action="store_true",
        help="Remove logs/questionnaires/feedback and event details from the public copy.",
    )
    parser.add_argument(
        "--emit-mapping",
        default=None,
        help="Write the raw→surrogate ID mapping to this CSV (for internal reference only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sanitize_stage_one(
        results_dir=args.results_dir,
        output_root=args.output_root,
        salt=args.salt,
        drop_free_text=args.drop_free_text,
        emit_mapping_path=args.emit_mapping,
        raw_subdir=args.raw_subdir,
    )


if __name__ == "__main__":
    main()
