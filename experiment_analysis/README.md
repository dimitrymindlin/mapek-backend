# Experiment Analysis Workflow

This folder bundles the data-preparation and reporting scripts that power the experiment dashboards. Use the recipes below to regenerate the per-condition datasets (steps `1_` to `4_`), configure single-condition analysis, and run the condition-comparison pipeline.

## 1. Configure `analysis_config.py`

Update `experiment_analysis/analysis_config.py` before running any scripts. The key fields are:

- `DATASET`: dataset identifier (for example `adult`).
- `EXPERIMENT_HANDLE`: experiment family (`mapek`, `interactive`, etc.).
- `GROUP`: condition name (`chat`, `interactive`, `static`, ...); this becomes part of the output folder name.
- `EXPERIMENT_DATE`: date tag in `MM_YYYY` format; aligns folders across runs.
- `INTRO_TEST_CYCLES`, `TEACHING_CYCLES`, `FINAL_TEST_CYCLES`: cycle counts used when filtering.
- `PROLIFIC_CSV_FOLDER`/`RESULTS_DIR`: auto-derived paths; leave them untouched unless the directory layout differs.
- `DB_CONFIG`: credentials only needed if `1_get_raw_experiment_data.py` fetches from Postgres instead of local CSV exports.

Every script in the `1_`–`4_` sequence reads these variables, so double-check the values before executing the pipeline.

## 2. Generate the per-condition dataset (`1_` to `4_`)

Run the numbered scripts in order inside an activated virtual environment with the required packages installed (`pip install -r requirements.txt`). Each step populates the next output folder under `experiment_analysis/results/<data_folder>`:

1. `python experiment_analysis/1_get_raw_experiment_data.py`
   - Pulls the raw Prolific export / DB dump into `results/<data_folder>/1_raw/`.
2. `python experiment_analysis/2_extract_information_to_columns.py`
   - Normalises raw logs into flat CSVs in `.../2_unpacked/`.
3. `python experiment_analysis/3_filter_and_clean_data.py`
   - Applies quality filters, creating the analysis-ready CSVs within `.../3_filtered/`.
4. `python experiment_analysis/4_get_raw_overview.py`
   - Builds quick descriptive summaries (`.../4_overview/`) and caches counts for the dashboards.

Re-run the scripts whenever `analysis_config.py` changes; downstream folders are overwritten.

## 3. Compare conditions in `compare_conditions`

The comparison pipeline expects both conditions to have completed the four preparation steps above (each with its own `analysis_config.py` settings). Copy or point the filtered outputs (`results/data_<dataset>_<handle>_<condition>_<date>/3_filtered`) into the comparison config.

1. Edit `experiment_analysis/compare_conditions/1_analysis_config.yaml`:
   - `data_sources.condition_a_dir` / `condition_b_dir`: absolute or repo-relative paths to the per-condition `results/...` folders (without the `3_filtered` suffix; the script appends it).
   - `experiment_settings.condition_a` / `condition_b`: identifiers used in plots and summary tables.
   - Adjust `filtered_subdir`, `output.results_dir`, and any enabled analyses as needed.
2. (Optional) Re-run `experiment_analysis/compare_conditions/2_calculate_and_save_avg_irt_scores.py` if you need fresh IRT score merges for each condition.
3. Execute `python experiment_analysis/compare_conditions/3_unified_analysis_pipeline.py`.
   - Results are written under `experiment_analysis/compare_conditions/results/<comparison_name>/` with subfolders for `plots`, `data`, and `reports`.

If the comparison script cannot find a module (for example `statistical_tests`), ensure you run it from the repository root so Python picks up the helper modules added there.

### Not currently supported: single-condition dashboard

The historical single-condition pipeline (`experiment_analysis/unified_analysis_pipeline.py`) is no longer maintained in this repo snapshot. Focus on the two-condition comparison flow above; any single-condition reporting will need the older tooling restored first.

## Quick Checklist

- [ ] `analysis_config.py` updated for each condition.
- [ ] Steps `1_`–`4_` executed per condition; verify `results/data_*/*` folders exist.
- [ ] `compare_conditions/1_analysis_config.yaml` points to both condition folders.
- [ ] `2_calculate_and_save_avg_irt_scores.py` run when new IRT outputs are needed.
- [ ] `3_unified_analysis_pipeline.py` executed to regenerate plots and reports.

This workflow keeps the raw exports, intermediate cleaned data, and final analytics reproducible for the cross-condition comparisons used in this project.
