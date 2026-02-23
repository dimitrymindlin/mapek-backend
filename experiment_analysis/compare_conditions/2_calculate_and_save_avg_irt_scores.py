import numpy as np
import pandas as pd
from irt_analysis import get_irt_trainer_for_users_from_multiple_dfs, prepare_data
from dialogue_analysis_base import UnifiedDataLoader
import statsmodels.api as sm
import os
import yaml
from pathlib import Path


def filter_by_group(df, group_name):
    return df[df['study_group'] == group_name]


def dif_test(data_condition_a, data_condition_b):
    # Get final test predictions and add a group indicator
    df_interactive = data_condition_a["predictions"][
        data_condition_a["predictions"]["source"] == "final-test"
    ].copy()
    df_chat = data_condition_b["predictions"][
        data_condition_b["predictions"]["source"] == "final-test"
    ].copy()

    df_interactive = prepare_data(df_interactive)
    df_interactive['group'] = 'interactive'
    df_chat = prepare_data(df_chat)
    df_chat['group'] = 'chat'

    # Combine data from both groups
    df_combined = pd.concat([df_interactive, df_chat], ignore_index=True)

    # Assume item columns are all columns except 'user_id' and 'group'
    item_cols = [col for col in df_combined.columns if col not in ['user_id', 'group']]

    dif_results = {}
    for item in item_cols:
        # Prepare a temporary DataFrame for the current item
        df_item = df_combined[['user_id', 'group', item]].dropna().copy()

        # If no data remains, skip this item with a warning or assign NaN p-value
        if df_item.empty:
            dif_results[item] = float('nan')
            continue

        # Create a dummy variable for group membership (1 for chat, 0 for interactive)
        df_item['group_dummy'] = (df_item['group'] == 'chat').astype(int)

        # Check if both groups are present
        if df_item['group_dummy'].nunique() < 2:
            dif_results[item] = float('nan')
            continue

        # Set up the logistic regression model: response ~ group_dummy
        X = sm.add_constant(df_item['group_dummy'])
        y = df_item[item]

        # If X or y is empty, skip
        if X.empty or y.empty:
            dif_results[item] = float('nan')
            continue

        try:
            model = sm.Logit(y, X).fit(disp=0)
            # p-value for the group coefficient indicates DIF significance
            dif_results[item] = model.pvalues['group_dummy']
        except Exception as e:
            print(f"Could not fit model for item {item}: {e}")
            dif_results[item] = float('nan')

    return dif_results


def load_config(config_path: str = '1_analysis_config.yaml'):
    """Load configuration from YAML file, resolving relative paths to this directory."""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file {cfg_path} not found.")
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def load_data_from_config(config_path: str = '1_analysis_config.yaml'):
    """Load all CSV files from configured directories using unified data loader."""
    config = load_config(config_path)
    config_dir = Path(config_path)
    if not config_dir.is_absolute():
        config_dir = Path(__file__).resolve().parent / config_dir
    config_dir = config_dir.parent

    data_sources = config['data_sources']
    exp_settings = config['experiment_settings']
    filtered_subdir = exp_settings.get('filtered_subdir', '3_filtered')

    repo_root = config_dir.parent if config_dir.name != '' else Path.cwd()
    repo_root = repo_root.parent  # config_dir is compare_conditions; parent = experiment_analysis; parent again = repo root
    repo_root = repo_root.resolve()

    # Define condition to directory mapping with filtered_subdir appended
    if exp_settings['comparison_mode']:
        condition_dirs = {
            exp_settings['condition_a']: str((repo_root / data_sources['condition_a_dir']).resolve() / filtered_subdir),
            exp_settings['condition_b']: str((repo_root / data_sources['condition_b_dir']).resolve() / filtered_subdir)
        }
    else:
        # For single condition, use condition_a_dir
        condition_dirs = {
            exp_settings['single_condition']: str((repo_root / data_sources['condition_a_dir']).resolve() / filtered_subdir)
        }

    # Get output directory
    output_dir = (repo_root / config['output']['results_dir']).resolve() / "data"

    # Use the unified data loader
    loader = UnifiedDataLoader()
    data_frames = loader.load_from_directories(condition_dirs)

    return data_frames, str(output_dir), list(condition_dirs.keys())


def main(config_path: str = '1_analysis_config.yaml'):
    """Main function that uses the unified data loading system."""
    print("Loading data from configuration using unified data loader...")

    # Load all data using unified approach
    data_frames, output_dir, conditions = load_data_from_config(config_path)

    if len(conditions) < 2:
        print("Error: Need at least 2 conditions for IRT analysis")
        return

    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for each condition
    condition_a, condition_b = conditions[0], conditions[1]
    print(f"Analyzing conditions: {condition_a} vs {condition_b}")
    print(f"Output directory: {output_dir}")
    
    # Extract condition dictionaries from the nested structure
    data_condition_a = data_frames[condition_a]
    data_condition_b = data_frames[condition_b]
    
    print(f"Condition A ({condition_a}): {len(data_condition_a['predictions'])} prediction rows")
    print(f"Condition B ({condition_b}): {len(data_condition_b['predictions'])} prediction rows")

    # Global storage for IRT scores
    user_id_irt_scores_mapping = {}

    # Run the IRT analysis loop
    print(f"🔄 Running 50 IRT iterations...")
    for i in range(50):
        if i % 10 == 0:
            print(f"  Progress: {i}/50 iterations completed")
        run_irt_iteration(data_condition_a, data_condition_b, user_id_irt_scores_mapping, i)

    # Calculate means and standard deviations
    calculate_irt_statistics(user_id_irt_scores_mapping)

    # Add IRT scores to user dataframes
    add_irt_scores_to_dataframes(data_condition_a, data_condition_b, user_id_irt_scores_mapping)

    # Save results using condition names from config
    save_results(data_condition_a, data_condition_b, condition_a, condition_b, output_dir)

    print("✅ IRT analysis completed successfully!")


def process_irt_scores_combined(data_condition_a, data_condition_b, score_col):
    """Extract and combine predictions from both conditions for IRT analysis."""
    source_filter = {"final_irt_score": "final-test", "intro_irt_score": "intro-test", "learning_irt_score": "test"}[score_col]
    
    predictions_list = [
        data_condition_a["predictions"][data_condition_a["predictions"]["source"] == source_filter],
        data_condition_b["predictions"][data_condition_b["predictions"]["source"] == source_filter]
    ]
    
    return get_irt_trainer_for_users_from_multiple_dfs(predictions_list, score_col)


def add_scores_to_mapping(scores_mapping, score_name, user_id_irt_scores_mapping):
    """Add IRT scores to the user mapping dictionary."""
    for user_id, score in scores_mapping.set_index('id')[score_name].items():
        if user_id not in user_id_irt_scores_mapping:
            user_id_irt_scores_mapping[user_id] = {"final_irt_score": [], "intro_irt_score": [], "learning_irt_score": []}
        user_id_irt_scores_mapping[user_id][score_name].append(score)


def run_irt_iteration(data_condition_a, data_condition_b, user_id_irt_scores_mapping, loop_idx):
    """Run a single iteration of IRT analysis."""
    # Process final and initial test predictions using combined approach
    # Each iteration will calibrate new models due to randomness in IRT optimization
    for score_col in ["final_irt_score", "intro_irt_score", "learning_irt_score"]:
        trainer, scores_mapping = process_irt_scores_combined(data_condition_a, data_condition_b, score_col)
        add_scores_to_mapping(scores_mapping, score_col, user_id_irt_scores_mapping)

        # Print progress on first iteration only
        if loop_idx == 0:
            num_items = len(trainer.best_params.get("diff", []))
            print(f"📊 {score_col}: {num_items} items processed")

    # Run the DIF tests only on first iteration (deterministic, so no need to repeat)
    if loop_idx == 0:
        dif_pvalues = dif_test(data_condition_a, data_condition_b)
        significant_items = sum(1 for p in dif_pvalues.values() if p < 0.05)
        print(f"🔍 DIF analysis: {significant_items}/{len(dif_pvalues)} items show significant differences")


def calculate_irt_statistics(user_id_irt_scores_mapping):
    """Calculate mean and standard deviation of IRT scores for each user."""
    for user_id, scores in user_id_irt_scores_mapping.items():
        for score_type in ["final_irt_score", "intro_irt_score", "learning_irt_score"]:
            scores_list = scores[score_type]
            user_id_irt_scores_mapping[user_id][f"{score_type}_mean"] = np.mean(scores_list) if scores_list else None
            user_id_irt_scores_mapping[user_id][f"{score_type}_std"] = np.std(scores_list) if scores_list else None


def add_irt_scores_to_dataframes(data_condition_a, data_condition_b, user_id_irt_scores_mapping):
    """Add the average IRT scores to the user dataframes."""
    for data in [data_condition_a, data_condition_b]:
        data["users"] = data["users"].merge(
            pd.DataFrame(user_id_irt_scores_mapping).T,
            left_on='id',
            right_index=True,
            how='left'
        )


def save_results(data_condition_a, data_condition_b, condition_a, condition_b, output_dir):
    """Save the user dataframes with IRT scores to the output directory."""
    # Create meaningful file names
    condition_a_file = os.path.join(output_dir, f"{condition_a}_users_with_irt.csv")
    condition_b_file = os.path.join(output_dir, f"{condition_b}_users_with_irt.csv")
    
    # Save the users with the IRT scores to CSV files
    data_condition_a["users"].to_csv(condition_a_file, index=False)
    data_condition_b["users"].to_csv(condition_b_file, index=False)
    
    print(f"💾 Saved {condition_a} results to: {condition_a_file}")
    print(f"💾 Saved {condition_b} results to: {condition_b_file}")


if __name__ == "__main__":
    # Use configuration-based file loading instead of hardcoded paths
    main()
