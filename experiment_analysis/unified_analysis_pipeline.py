"""Unified analysis pipeline for DialogueXAI-Continuum-Analysis."""

import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import pandas as pd

from dialogue_analysis_base import UnifiedDataLoader, QuestionnaireAnalyzer
from plot_overviews import extract_feedback_data
import likert_scale_analysis_refactored
# Import autosuggestion functions
from analyse_dialogues_refactored import is_autosuggested_question, get_suggestions


def normalize_question_text(text):
    """
    Normalize question text for consistent analysis.
    - Convert to lowercase
    - Strip whitespace
    - Preserve original structure for readability when needed
    """
    if pd.isna(text) or text == "":
        return text
    return str(text).lower().strip()


class AnalysisConfig:
    """Configuration management for analysis pipeline."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Add default column names if not present
        if 'column_names' not in config:
            config['column_names'] = {}
        
        column_defaults = {
            'final_score_column': 'final_irt_score_mean',
            'learning_score_column': 'learning_irt_score_mean',
            'subjective_understanding_column': 'subjective_understanding',
            'user_id_column': 'user_id'
        }
        
        for key, default_value in column_defaults.items():
            if key not in config['column_names']:
                config['column_names'][key] = default_value
        
        return config


class UnifiedAnalysisPipeline:
    """Main pipeline for orchestrating multiple analyses."""

    def __init__(self, config_path: str = 'analysis_config.yaml'):
        self.config = AnalysisConfig(config_path)
        self.results = {}
        self.setup_logging()
        self.data_loader = UnifiedDataLoader()

        # Create output directory based on config settings
        output_config = self.config.config['output']
        output_dir_str = output_config['results_dir']
        output_dir = Path(output_dir_str)  # Ensure this is a Path object
        output_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir
        self.plots_dir = self.output_dir / 'plots'
        self.data_dir = self.output_dir / 'data'
        self.reports_dir = self.output_dir / 'reports'

        self.plots_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

    def get_conditions(self) -> list:
        """Get the conditions to analyze from experiment_settings."""
        exp_settings = self.config.config['experiment_settings']
        if exp_settings['comparison_mode']:
            return [exp_settings['condition_a'], exp_settings['condition_b']]
        else:
            return [exp_settings['single_condition']]

    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.config['logging']['level'])
        log_file = self.config.config['logging']['file']

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def _load_data_frames(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all CSV files from configured directories into pandas DataFrames."""
        self.logger.info("Loading data from directories...")

        data_sources = self.config.config['data_sources']
        exp_settings = self.config.config['experiment_settings']
        filtered_subdir = exp_settings.get('filtered_subdir', '')

        # Define condition to directory mapping with filtered_subdir appended
        if exp_settings['comparison_mode']:
            condition_dirs = {
                exp_settings['condition_a']: str(Path(data_sources['condition_a_dir']) / filtered_subdir),
                exp_settings['condition_b']: str(Path(data_sources['condition_b_dir']) / filtered_subdir)
            }
        else:
            # For single condition, use condition_a_dir
            condition_dirs = {
                exp_settings['single_condition']: str(Path(data_sources['condition_a_dir']) / filtered_subdir)
            }

        # Use the centralized data loader
        data_frames = self.data_loader.load_from_directories(condition_dirs)

        # Add normalized question text columns
        data_frames = self._add_normalized_question_columns(data_frames)

        # Enhance with IRT scores if available
        data_frames = self._enhance_with_irt_scores(data_frames)

        # Process exit questionnaire data if available
        data_frames = self._process_exit_questionnaires(data_frames)

        return data_frames

    def _add_normalized_question_columns(self, data_frames: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Add normalized question text columns to relevant DataFrames."""
        question_columns = ['typed_question', 'question', 'message', 'text']
        
        for condition, dfs in data_frames.items():
            for df_name, df in dfs.items():
                # Check for question text columns and add normalized versions
                for col in question_columns:
                    if col in df.columns:
                        normalized_col = f"{col}_normalized" 
                        df[normalized_col] = df[col].apply(normalize_question_text)
                        self.logger.debug(f"Added {normalized_col} to {condition}.{df_name}")
        
        return data_frames

    def _enhance_with_irt_scores(self, data_frames: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[
        str, Dict[str, pd.DataFrame]]:
        """Replace user DataFrames with IRT-enhanced versions if available."""
        user_id_column = self.config.config['column_names']['user_id_column']
        
        for condition_name, condition_data in data_frames.items():
            irt_user_file = self.data_dir / f"{condition_name}_users_with_irt.csv"

            if irt_user_file.exists():
                irt_users_df = self.data_loader.load_csv(str(irt_user_file))

                # Standardize column names: rename 'id' to user_id_column if present
                if 'id' in irt_users_df.columns and user_id_column not in irt_users_df.columns:
                    irt_users_df = irt_users_df.rename(columns={'id': user_id_column})
                    self.logger.debug(f"Renamed 'id' column to '{user_id_column}' in {condition_name} IRT users data")

                condition_data['users'] = irt_users_df

        return data_frames

    def _process_exit_questionnaires(self, data_frames: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[
        str, Dict[str, pd.DataFrame]]:
        """Process exit questionnaire data and add item scores to users DataFrames."""
        self.logger.info("Processing exit questionnaire data...")
        
        user_id_column = self.config.config['column_names']['user_id_column']

        questionnaire_analyzer = QuestionnaireAnalyzer()
        prefix_groups = questionnaire_analyzer.configure_exit_questionnaire()

        for condition_name, condition_data in data_frames.items():
            if 'users' not in condition_data:
                continue

            users_df = condition_data['users']

            # Check if exit questionnaire data exists
            if not ('exit_questions' in users_df.columns and 'exit_answers' in users_df.columns):
                self.logger.info(
                    f"No exit questionnaire data found for {condition_name}. Available columns: {list(users_df.columns)}")
                continue

            # Extract individual question scores
            exit_scores_df = questionnaire_analyzer.extract_exit_questionnaire_data(users_df,
                                                                                    question_col='exit_questions',
                                                                                    answer_col='exit_answers')

            if exit_scores_df.empty:
                self.logger.warning(f"Failed to extract exit questionnaire data for {condition_name}")
                continue

            self.logger.debug(
                f"Extracted exit questionnaire data for {condition_name}: {len(exit_scores_df)} rows, columns: {list(exit_scores_df.columns)}")

            # Calculate item means for prefix groups (subscales)
            for prefix, question_codes in prefix_groups.items():
                existing_codes = [code for code in question_codes if code in exit_scores_df.columns]

                if not existing_codes:
                    continue

                if len(existing_codes) == 1:
                    # Single question - use it directly
                    item_col = f"{prefix}_score"
                    exit_scores_df[item_col] = exit_scores_df[existing_codes[0]]
                else:
                    # Multiple questions - calculate mean
                    item_col = f"{prefix}_mean"
                    exit_scores_df[item_col] = exit_scores_df[existing_codes].mean(axis=1, skipna=True)

                self.logger.debug(f"Created {item_col} for {condition_name} from {len(existing_codes)} questions")

            # Merge exit questionnaire scores back into users DataFrame
            enhanced_users_df = users_df.merge(exit_scores_df, on=user_id_column, how='left')
            condition_data['users'] = enhanced_users_df

            # Debug: Log what columns were actually added
            new_cols = [col for col in enhanced_users_df.columns if col not in users_df.columns]
            self.logger.debug(f"Added columns to {condition_name}: {new_cols}")

            # Save enhanced users DataFrame with exit questionnaire data
            exit_enhanced_file = self.data_dir / f"{condition_name}_users_with_exit_questionnaire.csv"
            enhanced_users_df.to_csv(exit_enhanced_file, index=False)
            self.logger.info(f"Saved exit questionnaire enhanced users data to {exit_enhanced_file}")

            # Log summary
            exit_cols = [col for col in enhanced_users_df.columns if
                         any(prefix in col for prefix in prefix_groups.keys())]
            self.logger.debug(
                f"Added {len(exit_cols)} exit questionnaire columns to {condition_name} users data: {exit_cols}")

        return data_frames

    def _get_worst_and_best_users(self, df_with_score, score_column=None, group_name=None):
        """Get the best and worst performing users based on a score."""
        if score_column is None:
            score_column = self.config.config['column_names']['final_score_column']
            
        if group_name:
            df_with_score = df_with_score[df_with_score["study_group"] == group_name]

        # Calculate thresholds for the top 15% and bottom 15% based on the score
        top_threshold = df_with_score[score_column].quantile(0.85)
        bottom_threshold = df_with_score[score_column].quantile(0.15)

        # Filter best and worst users
        best_users = df_with_score[df_with_score[score_column] >= top_threshold]
        worst_users = df_with_score[df_with_score[score_column] <= bottom_threshold]

        # Balance the groups to have equal size
        if len(worst_users) > len(best_users):
            worst_users = worst_users.sort_values(score_column, ascending=True).head(len(best_users))
        else:
            best_users = best_users.sort_values(score_column, ascending=False).head(len(worst_users))

        return best_users, worst_users

    def _make_question_count_df(self, questions_df, user_df, is_chat=True, final_score_column=None):
        """Create a dataframe with question counts per user."""
        if final_score_column is None:
            final_score_column = self.config.config['column_names']['final_score_column']
            
        if questions_df.empty or 'question_id' not in questions_df.columns:
            return pd.DataFrame()

        # For each user calculate the frequency of each question type
        user_question_freq = questions_df.groupby("user_id")["question_id"].value_counts().unstack()
        user_question_freq = user_question_freq.fillna(0)

        # Combine question types into categories
        if is_chat:
            # Combine ceteris paribus and feature statistics
            if 'ceterisParibus' in user_question_freq.columns and 'featureStatistics' in user_question_freq.columns:
                user_question_freq["feature_specific"] = (user_question_freq['ceterisParibus'] +
                                                          user_question_freq["featureStatistics"])

            # Combine general questions
            general_cols = ['counterfactualAnyChange', 'top3Features', 'anchor', 'least3Features', 'shapAllFeatures']
            existing_cols = [col for col in general_cols if col in user_question_freq.columns]
            if existing_cols:
                user_question_freq["general"] = user_question_freq[existing_cols].sum(axis=1)
        else:
            # Handle numeric question IDs from older format
            if 25 in user_question_freq.columns and 13 in user_question_freq.columns:
                user_question_freq["feature_specific"] = user_question_freq[25] + user_question_freq[13]

            general_cols = [7, 23, 11, 27, 24]
            existing_cols = [col for col in general_cols if col in user_question_freq.columns]
            if existing_cols:
                user_question_freq["general"] = user_question_freq[existing_cols].sum(axis=1)

        # Add final_score and study_group to the user_question_freq
        merge_cols = ["id"] if "id" in user_df.columns else ["user_id"]
        if final_score_column in user_df.columns:
            merge_cols.append(final_score_column)
        if "study_group" in user_df.columns:
            merge_cols.append("study_group")

        id_col = "id" if "id" in user_df.columns else "user_id"
        user_question_freq = user_question_freq.merge(user_df[merge_cols],
                                                      left_on="user_id", right_on=id_col)
        return user_question_freq

    def run_question_analysis(self, data_frames: Dict[str, Dict[str, pd.DataFrame]]):
        """Run question analysis across conditions."""
        try:
            self.logger.info("Running question analysis...")
            conditions = self.get_conditions()
            question_stats = self._run_question_analysis(data_frames, conditions)

            # Save summary to data directory
            if question_stats:
                # Create summary excluding performance_analysis from each condition's data
                summary_data = {}
                for condition, stats in question_stats.items():
                    if isinstance(stats, dict):
                        # Include all stats except performance_analysis
                        summary_data[condition] = {k: v for k, v in stats.items() if k != 'performance_analysis'}

                summary_df = pd.DataFrame(summary_data).T
                summary_path = self.data_dir / "question_analysis_summary.csv"
                summary_df.to_csv(summary_path)
                self.logger.info(f"Saved question analysis summary to {summary_path}")
            
            return {'question_analysis': question_stats}
        except Exception as e:
            self.logger.error(f"Question analysis failed: {e}")
            return {}

    def _run_question_analysis(self, data_frames: Dict[str, Dict[str, pd.DataFrame]], conditions: list):
        """Private method to run question analysis logic."""
        question_stats = {}

        for condition in conditions:
            if condition not in data_frames:
                continue

            condition_data = data_frames[condition]

            # Try to find questions in different DataFrames (predictions, dialogues, etc.)
            questions_df = pd.DataFrame()
            for df_name, df in condition_data.items():
                if df.empty:
                    continue

                if 'question_id' in df.columns:
                    if condition == 'interactive':
                        questions_df = df[df['question_id'].notna()]
                    else:
                        # Prefer normalized column if available, fallback to original
                        if 'typed_question_normalized' in df.columns:
                            questions_df = df[df['typed_question_normalized'].notna()]
                        else:
                            questions_df = df[df['typed_question'].notna()]
                    break
                elif 'event_type' in df.columns:
                    questions_df = df[df['event_type'] == 'question_asked']
                    break

            if not questions_df.empty:
                total_questions = len(questions_df)
                unique_users_asking = questions_df['user_id'].nunique() if 'user_id' in questions_df.columns else 0

                # Handle different question types based on condition
                if 'question_id' in questions_df.columns and questions_df['question_id'].notna().any():
                    # Interactive condition: structured questions with IDs
                    unique_questions = questions_df['question_id'].nunique()
                    # For structured questions, we don't have autosuggestion tracking (they use question IDs)
                    question_stats[condition] = {
                        'total_questions': total_questions,
                        'unique_question_types': unique_questions,
                        'unique_users_asking': unique_users_asking,
                        'avg_questions_per_user': total_questions / unique_users_asking if unique_users_asking > 0 else 0,
                        'question_type': 'structured',
                        'total_autosuggested': 0,  # N/A for structured questions
                        'total_user_generated': 0,  # N/A for structured questions
                        'autosuggestion_rate': 0.0  # N/A for structured questions
                    }
                else:
                    # Chat condition: free-form text questions
                    question_text_col = None
                    for col in ['typed_question', 'question', 'message', 'text']:
                        if col in questions_df.columns and questions_df[col].notna().any():
                            question_text_col = col
                            break

                    if question_text_col:
                        # Analyze free-form questions
                        question_texts = questions_df[question_text_col].dropna()

                        # Count unique exact question texts (for repetitive questions)
                        unique_exact_questions = question_texts.nunique()

                        # Calculate autosuggestion metrics for free-form questions
                        dataset = self.config.config.get('experiment_settings', {}).get('dataset', 'adult')
                        autosuggested_count = 0
                        for text in question_texts:
                            if is_autosuggested_question(text, dataset):
                                autosuggested_count += 1
                        
                        user_generated_count = total_questions - autosuggested_count
                        autosuggestion_rate = autosuggested_count / total_questions if total_questions > 0 else 0

                        question_stats[condition] = {
                            'total_questions': total_questions,
                            'unique_exact_questions': unique_exact_questions,
                            'unique_users_asking': unique_users_asking,
                            'avg_questions_per_user': total_questions / unique_users_asking if unique_users_asking > 0 else 0,
                            'question_diversity_ratio': round(unique_exact_questions / total_questions,
                                                              3) if total_questions > 0 else 0,
                            'question_type': 'free_form',
                            'total_autosuggested': autosuggested_count,
                            'total_user_generated': user_generated_count,
                            'autosuggestion_rate': round(autosuggestion_rate, 3)
                        }
                    else:
                        # Fallback if no question text column found
                        question_stats[condition] = {
                            'total_questions': total_questions,
                            'unique_users_asking': unique_users_asking,
                            'avg_questions_per_user': total_questions / unique_users_asking if unique_users_asking > 0 else 0,
                            'question_type': 'unknown'
                        }

                # Log appropriate statistics based on question type
                if question_stats[condition].get('question_type') == 'structured':
                    self.logger.debug(
                        f"{condition}: {total_questions} total questions, {question_stats[condition]['unique_question_types']} unique types")
                elif question_stats[condition].get('question_type') == 'free_form':
                    diversity = question_stats[condition]['question_diversity_ratio']
                    unique_q = question_stats[condition]['unique_exact_questions']
                    self.logger.debug(
                        f"{condition}: {total_questions} free-form questions, {unique_q} unique, diversity={diversity}")
                else:
                    self.logger.debug(f"{condition}: {total_questions} total questions")

                # Enhanced question analysis with categories using user data
                if 'users' in condition_data:
                    user_df = condition_data['users']
                    is_chat = "chat" in condition.lower()
                    final_score_column = self.config.config['column_names']['final_score_column']

                    question_count_df = self._make_question_count_df(questions_df, user_df, is_chat, final_score_column)
                    if not question_count_df.empty:
                        # Save detailed question counts to data directory
                        question_counts_path = self.data_dir / f"detailed_question_counts_{condition}.csv"
                        question_count_df.to_csv(question_counts_path, index=False)
                        self.logger.info(f"Saved detailed question counts to {question_counts_path}")

                        # Analyze performance by question usage (now with IRT scores!)
                        if final_score_column in question_count_df.columns:
                            best_users, worst_users = self._get_worst_and_best_users(question_count_df, final_score_column)

                            if not best_users.empty and not worst_users.empty:
                                performance_analysis = {
                                    'best_performers': {
                                        'count': len(best_users),
                                        'avg_score': best_users[final_score_column].mean(),
                                        'avg_questions': best_users.select_dtypes(include=[int, float]).sum(
                                            axis=1).mean()
                                    },
                                    'worst_performers': {
                                        'count': len(worst_users),
                                        'avg_score': worst_users[final_score_column].mean(),
                                        'avg_questions': worst_users.select_dtypes(include=[int, float]).sum(
                                            axis=1).mean()
                                    }
                                }
                                question_stats[condition]['performance_analysis'] = performance_analysis
                                self.logger.debug(f"Performance analysis completed for {condition}")

                # Save basic question counts to data directory
                # Determine which column to use for counting based on question type
                count_column = None
                if question_stats[condition].get('question_type') == 'structured':
                    # Use question_id for structured questions (Q001, Q002, etc.)
                    count_column = 'question_id'
                elif question_stats[condition].get('question_type') == 'free_form':
                    # Find the same text column that was used for analysis (prefer normalized columns)
                    for col in ['typed_question_normalized', 'question_normalized', 'message_normalized', 'text_normalized', 'typed_question', 'question', 'message', 'text']:
                        if col in questions_df.columns and questions_df[col].notna().any():
                            count_column = col
                            break
                else:
                    # Fallback - try question_id first, then any text column
                    if 'question_id' in questions_df.columns and questions_df['question_id'].notna().any():
                        count_column = 'question_id'
                    else:
                        # Prefer normalized columns when available
                        for col in ['typed_question_normalized', 'question_normalized', 'message_normalized', 'text_normalized', 'typed_question', 'question', 'message', 'text']:
                            if col in questions_df.columns and questions_df[col].notna().any():
                                count_column = col
                                break

                # Generate counts using the appropriate column with autosuggestion tracking
                if count_column and count_column in questions_df.columns:
                    # Get dataset from config for autosuggestion detection
                    dataset = self.config.config.get('experiment_settings', {}).get('dataset', 'adult')
                    
                    # Create enhanced question counts with autosuggestion indicator
                    question_data = []
                    for question_text in questions_df[count_column].dropna():
                        is_suggested = is_autosuggested_question(question_text, dataset)
                        question_data.append({
                            'question': question_text,
                            'from_autosuggestion': is_suggested
                        })
                    
                    if question_data:
                        enhanced_df = pd.DataFrame(question_data)
                        question_counts = enhanced_df.groupby(['question', 'from_autosuggestion']).size().reset_index(name='count')
                        
                        # Save enhanced question counts
                        enhanced_counts_path = self.data_dir / f"question_counts_{condition}.csv"
                        question_counts.to_csv(enhanced_counts_path, index=False)
                        self.logger.info(f"Saved enhanced question counts to {enhanced_counts_path} using column '{count_column}'")
                        
                        # Calculate autosuggestion metrics and add to stats (only for free_form questions)
                        if question_stats[condition].get('question_type') == 'free_form':
                            total_autosuggested = enhanced_df['from_autosuggestion'].sum()
                            total_questions = len(enhanced_df)
                            autosuggestion_rate = total_autosuggested / total_questions if total_questions > 0 else 0
                            
                            question_stats[condition]['total_autosuggested'] = int(total_autosuggested)
                            question_stats[condition]['total_user_generated'] = int(total_questions - total_autosuggested)
                            question_stats[condition]['autosuggestion_rate'] = round(autosuggestion_rate, 3)
                    else:
                        self.logger.warning(f"No question data found for autosuggestion analysis in {condition} condition")
                else:
                    self.logger.warning(f"No suitable column found for question counts in {condition} condition")

        # Save summary to data directory
        if question_stats:
            # Create summary excluding performance_analysis from each condition's data
            summary_data = {}
            for condition, stats in question_stats.items():
                if isinstance(stats, dict):
                    # Include all stats except performance_analysis
                    summary_data[condition] = {k: v for k, v in stats.items() if k != 'performance_analysis'}

            summary_df = pd.DataFrame(summary_data).T
            summary_path = self.data_dir / "question_analysis_summary.csv"
            summary_df.to_csv(summary_path)
            self.logger.info(f"Saved question analysis summary to {summary_path}")

        return question_stats

    def run_exit_questionnaire_analysis(self, data_frames: Dict[str, Dict[str, pd.DataFrame]]):
        """Run exit questionnaire statistical analysis between conditions."""
        try:
            if not data_frames:
                self.logger.warning("No data frames provided for exit questionnaire analysis.")
                return {}

            conditions = self.get_conditions()
            if len(conditions) < 2:
                self.logger.info("Need at least 2 conditions for exit questionnaire comparison")
                return {'exit_questionnaire_analysis': 'skipped - insufficient conditions'}

            self.logger.info("Running exit questionnaire statistical analysis...")

            # Get questionnaire configuration
            questionnaire_analyzer = QuestionnaireAnalyzer()
            prefix_groups = questionnaire_analyzer.configure_exit_questionnaire()

            # Prepare data for statistical analysis
            condition_data_for_stats = {}
            for condition_name, condition_data in data_frames.items():
                if 'users' in condition_data:
                    users_df = condition_data['users']

                    # Debug: Print all columns to see what's available
                    self.logger.debug(f"Columns in {condition_name} users DataFrame: {list(users_df.columns)}")

                    # Check if exit questionnaire item scores exist (look for prefix_score or prefix_mean)
                    exit_item_cols = []
                    for prefix in prefix_groups.keys():
                        score_col = f"{prefix}_score"
                        mean_col = f"{prefix}_mean"
                        if score_col in users_df.columns:
                            exit_item_cols.append(score_col)
                        if mean_col in users_df.columns:
                            exit_item_cols.append(mean_col)

                    if exit_item_cols:
                        condition_data_for_stats[condition_name] = users_df
                        self.logger.debug(
                            f"Found {len(exit_item_cols)} exit questionnaire items for {condition_name}: {exit_item_cols}")
                    else:
                        self.logger.warning(
                            f"No exit questionnaire item scores found for {condition_name}. Available prefixes: {list(prefix_groups.keys())}")

            if len(condition_data_for_stats) < 2:
                self.logger.warning("Insufficient conditions with exit questionnaire data for comparison")
                return {'exit_questionnaire_analysis': 'skipped - no exit data'}

            # Run statistical comparisons for each item/subscale
            from dialogue_analysis_base import SharedStatistics

            results = []
            alpha = 0.05

            for prefix in prefix_groups.keys():
                # Look for either single score or mean score
                score_col = f"{prefix}_score"
                mean_col = f"{prefix}_mean"

                item_col = None
                for condition_name, users_df in condition_data_for_stats.items():
                    if score_col in users_df.columns:
                        item_col = score_col
                        break
                    elif mean_col in users_df.columns:
                        item_col = mean_col
                        break

                if not item_col:
                    continue

                # Get data for each condition
                condition_values = {}
                for condition_name, users_df in condition_data_for_stats.items():
                    if item_col in users_df.columns:
                        values = users_df[item_col].dropna()
                        if len(values) > 2:  # Need at least 3 data points
                            condition_values[condition_name] = values

                if len(condition_values) >= 2:
                    # Perform pairwise comparisons
                    condition_names = list(condition_values.keys())
                    for i in range(len(condition_names)):
                        for j in range(i + 1, len(condition_names)):
                            cond1, cond2 = condition_names[i], condition_names[j]

                            comparison_result = SharedStatistics.perform_mann_whitney_comparison(
                                condition_values[cond1], condition_values[cond2], cond1, cond2
                            )

                            if 'error' not in comparison_result:
                                significance = "Significant" if comparison_result[
                                                                    'p_value'] < alpha else "Not significant"

                                results.append({
                                    "prefix": prefix,
                                    "comparison": f"{cond1} vs {cond2}",
                                    "test_statistic": comparison_result['mann_whitney_u'],
                                    "p": round(comparison_result['p_value'], 3),
                                    f"{cond1}_median": comparison_result['median1'],
                                    f"{cond2}_median": comparison_result['median2'],
                                    "significance": significance,
                                    "higher_group": comparison_result['higher_group'],
                                    "effect_size": round(comparison_result['effect_size'], 3),
                                })

                                self.logger.debug(
                                    f"{prefix}: {comparison_result['higher_group']} > other (p = {comparison_result['p_value']:.3f})")

            # Save results
            if results:
                results_df = pd.DataFrame(results)
                results_path = self.data_dir / "exit_questionnaire_statistical_results.csv"
                results_df.to_csv(results_path, index=False)
                self.logger.info(f"Saved exit questionnaire statistical results to {results_path}")

                # Log summary
                significant_count = sum(1 for r in results if r['significance'] == "Significant")
                self.logger.info(
                    f"Exit questionnaire analysis: {significant_count}/{len(results)} significant comparisons")

            self.logger.info("Exit questionnaire analysis completed")
            return {'exit_questionnaire_analysis': {'total_comparisons': len(results), 'significant_results': sum(
                1 for r in results if r['significance'] == "Significant")}}

        except Exception as e:
            self.logger.error(f"Exit questionnaire analysis failed: {e}")
            return {}

    def run_descriptive_statistics(self, data_frames: Dict[str, Dict[str, pd.DataFrame]]):
        """Run basic descriptive statistics on the groups."""
        try:
            self.logger.info("Running descriptive statistics...")
            conditions = self.get_conditions()
            stats_summary = self._run_descriptive_statistics(data_frames, conditions)

            # Save statistics summary to data directory
            if stats_summary:
                stats_df = pd.DataFrame(stats_summary).T
                stats_path = self.data_dir / "descriptive_statistics.csv"
                stats_df.to_csv(stats_path)
                self.logger.info(f"Saved descriptive statistics to {stats_path}")

            return {'descriptive_statistics': stats_summary}
        except Exception as e:
            self.logger.error(f"Descriptive statistics failed: {e}")
            return {}

    def _run_descriptive_statistics(self, data_frames: Dict[str, Dict[str, pd.DataFrame]], conditions: list):
        """Private method to run descriptive statistics logic."""
        stats_summary = {}
        
        for condition in conditions:
            if condition not in data_frames:
                continue
                
            condition_data = data_frames[condition]
            if 'users' in condition_data:
                users_df = condition_data['users']
                
                # Basic statistics
                condition_stats = {
                    'total_users': len(users_df),
                    'condition_name': condition
                }
                
                # Add numeric column statistics if available
                final_score_col = self.config.config['column_names']['final_score_column']
                if final_score_col in users_df.columns:
                    scores = users_df[final_score_col].dropna()
                    if len(scores) > 0:
                        condition_stats.update({
                            f'{final_score_col}_mean': scores.mean(),
                            f'{final_score_col}_std': scores.std(),
                            f'{final_score_col}_median': scores.median(),
                            f'{final_score_col}_count': len(scores)
                        })
                
                stats_summary[condition] = condition_stats
                self.logger.debug(f"Descriptive stats for {condition}: {condition_stats}")
        
        return stats_summary

    def run_feedback_analysis(self, data_frames: Dict[str, Dict[str, pd.DataFrame]]):
        """Extracts and saves user feedback data."""
        try:
            self.logger.info("Running feedback analysis...")
            results = {}
            for condition, condition_data in data_frames.items():
                if 'users' in condition_data:
                    users_df = condition_data['users']
                    feedback_data = extract_feedback_data(users_df)
                    if feedback_data:
                        feedback_df = pd.DataFrame(feedback_data)
                        output_path = self.data_dir / f"users_feedback_{condition}.csv"
                        feedback_df.to_csv(output_path, index=False)
                        self.logger.info(f"Saved user feedback for {condition} to {output_path}")
                        results[condition] = "completed"
                    else:
                        self.logger.info(f"No feedback data found for {condition}")
                        results[condition] = "no_data"
            return {'feedback_analysis': results}
        except Exception as e:
            self.logger.error(f"Feedback analysis failed: {e}")
            return {}

    def run_likert_scale_analysis(self, data_frames: Dict[str, Dict[str, pd.DataFrame]]):
        """Run Likert scale analysis."""
        try:
            conditions = self.get_conditions()

            # Extract appropriate DataFrames for Likert analysis
            # Look for DataFrames that contain exit questionnaire data
            likert_data_frames = {}
            for condition, condition_data in data_frames.items():
                # Priority order: users (likely has exit data), then other DataFrames
                for df_name in ['users', 'predictions', 'dialogues']:
                    if df_name in condition_data:
                        df = condition_data[df_name]
                        # Check if this DataFrame has exit questionnaire columns
                        if any(col in df.columns for col in
                               ['exit_questions', 'exit_answers', 'exit', 'exit_q', 'exit_q_answers']):
                            likert_data_frames[condition] = df
                            self.logger.debug(f"Using '{df_name}' DataFrame for {condition} Likert analysis")
                            break
                else:
                    # If no DataFrame with exit data found, use users as fallback
                    if 'users' in condition_data:
                        likert_data_frames[condition] = condition_data['users']
                        self.logger.warning(f"No exit questionnaire data found for {condition}, using users DataFrame")

            if not likert_data_frames:
                self.logger.warning("No suitable DataFrames found for Likert scale analysis")
                return {'likert_scale_analysis': 'skipped - no exit data'}

            results = likert_scale_analysis_refactored.main(
                data_frames=likert_data_frames,
                conditions=conditions,
                output_dir=str(self.plots_dir),
                data_dir=str(self.data_dir)
            )

            self.logger.info("Likert scale analysis completed")
            return {'likert_scale_analysis': results}
        except Exception as e:
            self.logger.error(f"Likert scale analysis failed: {e}")
            return {}

    def run_correlation_analysis(self, data_frames: Dict[str, Dict[str, pd.DataFrame]]):
        """Run correlation analysis for each condition."""
        try:
            self.logger.info("Running correlation analysis...")
            analysis_config = self.config.config.get('analyses', {}).get('correlation_analysis', {})
            if not analysis_config.get('enabled', True):
                self.logger.info("Correlation analysis is disabled in the config.")
                return {'correlation_analysis': 'disabled'}

            target_column = analysis_config.get('target_column', self.config.config['column_names']['final_score_column'])
            
            from correlation_analysis_refactored import CorrelationAnalysis
            corr_analyzer = CorrelationAnalysis()
            
            results = {}
            for condition, condition_data in data_frames.items():
                if 'users' in condition_data:
                    users_df = condition_data['users']
                    self.logger.debug(f"Running correlation analysis for condition: {condition}")
                    
                    # The output directory for plots
                    output_dir = self.plots_dir

                    corr_analyzer.run_correlation_analysis(
                        df=users_df,
                        condition_name=condition,
                        target_column=target_column,
                        output_dir=str(output_dir)
                    )
                    results[condition] = "completed"
                else:
                    self.logger.warning(f"No 'users' dataframe for condition {condition}, skipping correlation analysis.")
                    results[condition] = "skipped"

            self.logger.info("Correlation analysis completed.")
            return {'correlation_analysis': results}
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}", exc_info=True)
            return {}

    def _check_normality_shapiro(self, data):
        """Use Shapiro-Wilk test for normality (better power than K-S)."""
        from scipy import stats
        
        if len(data) < 3:
            return False
        
        # Shapiro-Wilk test for normality
        statistic, p_value = stats.shapiro(data)
        return p_value >= 0.05  # Normal if p >= 0.05

    def _perform_unified_comparison(self, group1, group2, score_name):
        """Unified test selection and execution with Shapiro-Wilk normality testing."""
        import pandas as pd
        from scipy.stats import mannwhitneyu, ttest_ind
        
        # Get test direction from config (default to greater: condition_b > condition_a)
        test_direction = self.config.config.get('statistical_tests', {}).get('direction', 'greater')
        
        # Clean data once
        group1_clean = pd.to_numeric(group1, errors='coerce').dropna()
        group2_clean = pd.to_numeric(group2, errors='coerce').dropna()
        
        if len(group1_clean) == 0 or len(group2_clean) == 0:
            return {
                'p_value': 1.0,
                'test_type': 'no_data',
                'group1_clean': group1_clean,
                'group2_clean': group2_clean,
                'normal1': False,
                'normal2': False
            }
        
        # Check normality with Shapiro-Wilk test (better power than K-S)
        normal1 = self._check_normality_shapiro(group1_clean)
        normal2 = self._check_normality_shapiro(group2_clean)
        
        # Single test selection logic
        if normal1 and normal2:
            # Use Welch t-test with specified direction
            # For Welch: positive t means group2 > group1, so we use the direction directly
            stat, p_value = ttest_ind(group1_clean, group2_clean, equal_var=False, alternative=test_direction)
            test_type = 'welch_t'
        else:
            # Use Mann-Whitney U with specified direction
            # For Mann-Whitney: alternative='greater' tests if group2 > group1
            stat, p_value = mannwhitneyu(group1_clean, group2_clean, method="asymptotic", alternative=test_direction)
            test_type = 'mann_whitney'
            
        return {
            'p_value': p_value,
            'test_type': test_type,
            'group1_clean': group1_clean,
            'group2_clean': group2_clean,
            'normal1': normal1,
            'normal2': normal2,
            'statistic': stat
        }


    def run_irt_score_comparison(self, data_frames: Dict[str, Dict[str, pd.DataFrame]]):
        """Run IRT score comparison analysis between conditions."""
        try:
            self.logger.info("Running IRT score comparison analysis...")
            
            # Import required functions
            from statistical_tests import is_t_test_applicable, plot_box_with_significance_bars
            import matplotlib.pyplot as plt
            
            conditions = self.get_conditions()
            if len(conditions) < 2:
                self.logger.info("Need at least 2 conditions for IRT score comparison")
                return {'irt_score_comparison': 'skipped - insufficient conditions'}
            
            # Check if both conditions have users with IRT scores
            users_data = {}
            for condition in conditions:
                if condition in data_frames and 'users' in data_frames[condition]:
                    users_df = data_frames[condition]['users']
                    # Check if IRT score columns exist
                    required_cols = ['intro_irt_score_mean', 'learning_irt_score_mean', 'final_irt_score_mean']
                    if all(col in users_df.columns for col in required_cols):
                        users_data[condition] = users_df
                    else:
                        missing_cols = [col for col in required_cols if col not in users_df.columns]
                        self.logger.warning(f"Missing IRT columns for {condition}: {missing_cols}")
                        return {'irt_score_comparison': f'skipped - missing IRT scores for {condition}'}
                else:
                    self.logger.warning(f"No users data found for condition: {condition}")
                    return {'irt_score_comparison': f'skipped - no users data for {condition}'}
            
            if len(users_data) < 2:
                return {'irt_score_comparison': 'skipped - insufficient conditions with IRT data'}
            
            # Get the two conditions for comparison in config order (condition_a, condition_b)
            conditions = self.get_conditions()
            condition_names = conditions
            df1, df2 = users_data[conditions[0]], users_data[conditions[1]]  # df1=condition_a (chat), df2=condition_b (interactive)
            
            # Add condition labels to dataframes
            df1_labeled = df1.copy().reset_index(drop=True)
            df1_labeled['condition'] = condition_names[0]
            df2_labeled = df2.copy().reset_index(drop=True)
            df2_labeled['condition'] = condition_names[1]
            
            # Define the three IRT scores to compare
            score_comparisons = [
                ('intro_irt_score_mean', 'Initial Understanding'),
                ('learning_irt_score_mean', 'Learning Progress'),
                ('final_irt_score_mean', 'Final Understanding')
            ]
            
            # Unified test selection and execution for all scores
            comparison_results = {}
            from statistical_tests import get_groups_two_dfs
            
            for score_col, title in score_comparisons:
                # Extract groups once
                group1, group2, _, _ = get_groups_two_dfs(df1_labeled, df2_labeled, 'condition', score_col)
                
                # Perform unified comparison with K-S normality testing
                result = self._perform_unified_comparison(group1, group2, score_col)
                result['title'] = title
                comparison_results[score_col] = result
                
                # Log normality results and test direction
                normal_status = f"Group 1: {'Normal' if result['normal1'] else 'Non-normal'}, Group 2: {'Normal' if result['normal2'] else 'Non-normal'}"
                test_direction = self.config.config.get('statistical_tests', {}).get('direction', 'greater')
                
                # Create accurate direction text based on actual test direction
                if test_direction == "less":
                    direction_text = f"{conditions[0]} < {conditions[1]}"
                elif test_direction == "greater": 
                    direction_text = f"{conditions[1]} > {conditions[0]}"
                else:  # two-sided
                    direction_text = f"{conditions[0]} ≠ {conditions[1]}"
                
                self.logger.info(f"{title} - Shapiro-Wilk Normality: {normal_status}, Test: {result['test_type']}, Direction: {direction_text} ({test_direction})")
            
            
            # Create subplot figure and plot with unified results
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharey=True)
            
            for position, (score_col, _) in enumerate(score_comparisons):
                if score_col not in comparison_results:
                    continue
                    
                result = comparison_results[score_col]
                title = result['title']
                
                # Use pre-determined test type for plotting
                use_t_test = result['test_type'] == 'welch_t'
                
                # Get direction from config to determine plot ordering
                direction = self.config.config.get('statistical_tests', {}).get('direction', 'greater')
                
                # Create the boxplot with pre-calculated results
                # Swap dataframes if direction is "less" to put condition_a on the right
                if direction == "less":
                    plot_box_with_significance_bars(
                        df2_labeled, 'condition', score_col, title,
                        ttest=use_t_test, save=False, ax=axes[position],
                        y_label_name='IRT Scores', df2=df1_labeled
                    )
                else:
                    plot_box_with_significance_bars(
                        df1_labeled, 'condition', score_col, title,
                        ttest=use_t_test, save=False, ax=axes[position],
                        y_label_name='IRT Scores', df2=df2_labeled
                    )
                
                self.logger.debug(f"Completed {title} using {result['test_type']} (p={result['p_value']:.4f})")
            
            # Store results for return
            results = {score_col: {
                'test_type': result['test_type'],
                'conditions_compared': condition_names,
                'p_value': result['p_value'],
                'normal1': result['normal1'],
                'normal2': result['normal2']
            } for score_col, result in comparison_results.items()}
            
            # Save the combined figure
            plt.tight_layout()
            output_path = self.plots_dir / f"irt_score_comparison_{condition_names[0]}_vs_{condition_names[1]}.pdf"
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            self.logger.info(f"Saved IRT score comparison plots to {output_path}")
            self.logger.info("IRT score comparison analysis completed")
            
            return {'irt_score_comparison': results}
            
        except Exception as e:
            self.logger.error(f"IRT score comparison analysis failed: {e}", exc_info=True)
            return {}

    def run_all_analyses(self) -> Dict[str, Any]:
        """Run all enabled analyses."""
        self.logger.info("Starting analysis pipeline...")

        # Load all data into DataFrames once
        data_frames = self._load_data_frames()
        if not data_frames:
            self.logger.error("Halting analysis due to data loading failure.")
            return {}

        # Get directories used for metadata reporting
        data_sources = self.config.config['data_sources']
        directories_used = []
        exp_settings = self.config.config['experiment_settings']

        if exp_settings['comparison_mode']:
            directories_used = [data_sources['condition_a_dir'], data_sources['condition_b_dir']]
        else:
            directories_used = [data_sources['condition_a_dir']]

        self.logger.info(f"Using data from directories: {directories_used}")

        all_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_directories_used': directories_used,
                'conditions_loaded': list(data_frames.keys()),
                'dataframes_per_condition': {k: list(v.keys()) for k, v in data_frames.items()},
                'total_rows_per_condition': {k: {df_name: len(df) for df_name, df in v.items()}
                                             for k, v in data_frames.items()}
            }
        }

        analyses_config = self.config.config['analyses']

        # Run enabled analyses with the loaded DataFrames
        if analyses_config.get('correlation_analysis', {}).get('enabled', True):
            all_results.update(self.run_correlation_analysis(data_frames))

        if analyses_config.get('likert_scale_analysis', {}).get('enabled', True):
            all_results.update(self.run_likert_scale_analysis(data_frames))

        if analyses_config.get('questionnaire_analysis', {}).get('enabled', True):
            all_results.update(self.run_exit_questionnaire_analysis(data_frames))

        if analyses_config.get('question_analysis', {}).get('enabled', True):
            all_results.update(self.run_question_analysis(data_frames))

        if analyses_config.get('feedback_analysis', {}).get('enabled', True):
            all_results.update(self.run_feedback_analysis(data_frames))

        if analyses_config.get('descriptive_statistics', {}).get('enabled', True):
            all_results.update(self.run_descriptive_statistics(data_frames))

        if analyses_config.get('irt_score_comparison', {}).get('enabled', True):
            all_results.update(self.run_irt_score_comparison(data_frames))

        self.save_results(all_results)
        self.logger.info("All analyses completed")
        return all_results

    def save_results(self, results: Dict[str, Any]):
        """Save analysis results."""
        formats = self.config.config['output']['format']

        for fmt in formats:
            if fmt == 'json':
                output_file = self.reports_dir / 'results.json'
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                self.logger.info(f"Results saved to {output_file}")

            elif fmt == 'markdown':
                output_file = self.reports_dir / 'report.md'
                with open(output_file, 'w') as f:
                    f.write("# Analysis Report\n\n")
                    f.write(f"**Generated:** {results['metadata']['timestamp']}\n\n")
                    f.write(f"**Data Directories:** {', '.join(results['metadata']['data_directories_used'])}\n\n")
                    f.write(f"**Conditions Loaded:** {', '.join(results['metadata']['conditions_loaded'])}\n\n")
                    f.write("**DataFrames per Condition:**\n")
                    for condition, dataframes in results['metadata']['dataframes_per_condition'].items():
                        f.write(f"- {condition}: {', '.join(dataframes)}\n")
                    f.write("\n**Rows per DataFrame:**\n")
                    for condition, dataframes in results['metadata']['total_rows_per_condition'].items():
                        f.write(f"- {condition}:\n")
                        for df_name, count in dataframes.items():
                            f.write(f"  - {df_name}: {count:,} rows\n")
                    f.write("\n")

                    for key, value in results.items():
                        if key != 'metadata':
                            f.write(f"## {key.replace('_', ' ').title()}\n\n")
                            f.write(f"{value}\n\n")
                self.logger.info(f"Report saved to {output_file}")


def main():
    """Main entry point."""
    import sys
    if len(sys.argv) == 1:
        print("Running analysis pipeline...")

        config_path = 'analysis_config.yaml'

        if not Path(config_path).exists():
            print(f"Creating configuration: {config_path}")
            config = AnalysisConfig(config_path)

        pipeline = UnifiedAnalysisPipeline(config_path)
        exp_info = pipeline.config.config['experiment_settings']
        print(f"Dataset: {exp_info['dataset']}")
        print(f"Mode: {'Comparison' if exp_info['comparison_mode'] else 'Single condition'}")

        results = pipeline.run_all_analyses()
        print(f"Analysis completed! Results in: {pipeline.output_dir}")
        return

    parser = argparse.ArgumentParser(description='Analysis Pipeline')
    parser.add_argument('--config', default='analysis_config.yaml', help='Config file path')
    args = parser.parse_args()

    pipeline = UnifiedAnalysisPipeline(args.config)
    results = pipeline.run_all_analyses()
    print(f"Analysis completed! Results in: {pipeline.output_dir}")


if __name__ == "__main__":
    main()
