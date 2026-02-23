# Public Release Workflow

This guide explains how to prepare experiment data for publication while keeping
personally identifiable information (PII) outside of the repository.

## 1. Run Stage 1 (Automatic Sanitisation)

Execute `python experiment_analysis/1_get_raw_experiment_data.py`. The script:

- Collects the raw Prolific exports and database tables into
  `results/<handle>/0_private_raw/` for internal use only.
- Builds the participant overview report at `4_overview/prolific_participant_overview.md`.
- Generates a sanitised copy in `results/<handle>/1_raw/` with surrogate IDs,
  precomputed Prolific timings, and (optionally) stripped free-text columns.

Use `--salt <private-salt>` to hash Prolific participant IDs consistently and
`--drop-free-text` if you want to remove questionnaires/logs before publication.
The script also emits `results/<handle>/0_private_raw/user_id_mapping.csv` for
internal reconciliation—do not ship this file.

## 2. (Optional) Re-sanitise with Custom Settings

If you need to regenerate the sanitised copy with different options, use:

```bash
python experiment_analysis/sanitize_for_publication.py \
  --results-dir experiment_analysis/results/<handle> \
  --output-root experiment_analysis/results/<handle> \
  --raw-subdir 0_private_raw \
  --salt <private-salt> \
  --drop-free-text
```

This utility reads the private copy in `0_private_raw` (created during Stage 1)
and overwrites `1_raw/` directly with the new sanitised CSVs for downstream stages.

## 3. Re-run Stages 2–4 on Sanitised Data

With the redacted raw assets in place you can recompute the derived tables:

```bash
python experiment_analysis/2_extract_information_to_columns.py
python experiment_analysis/3_filter_and_clean_data.py
python experiment_analysis/4_get_raw_overview.py
python experiment_analysis/5_create_irt_scores.py
python experiment_analysis/6_run_unified_analysis.py --condition <path_to_cond_a> chat --condition <path_to_cond_b> interactive
```

`2_extract_information_to_columns.py` now detects when `prolific_merged.csv` is
absent and reuses the pre-computed timing columns. Scripts 3 and 4 depend only on
the outputs of previous stages and do not need database access or the Prolific
export.

Stage 5 augments `users_filtered.csv` with Rasch-style IRT ability estimates and
stores the result in `results/<handle>/5_irt/users_with_irt_scores.csv`.

Stage 6 consolidates all sanitised artefacts into `results/<handle>/6_unified/`,
producing `users_unified.csv`, a feedback export, and a compact
`summary_metrics.json` for reporting.

## 4. Publish Only Sanitised Artefacts

Share the following:

- `results/<handle>/1_raw/` from the sanitised copy (optionally excluding
  `prolific_summary.csv` if it is not required).
- `results/<handle>/2_unpacked/`, `3_filtered/`, and `4_overview/` generated from
  the sanitised inputs.
- A short note in your README or appendix describing the sanitisation choices,
  wording consent/IRB constraints, and clarifying that Prolific exports and raw
  database dumps stay private.

Never publish the ID mapping emitted with `--emit-mapping`; keep it in a secure
location for internal reconciliation only.
