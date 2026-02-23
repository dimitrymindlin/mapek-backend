"""Dialogue Analysis Base Module - Unified data loading and analysis utilities."""

import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Any
import yaml
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

# Centralized condition name mapping
CONDITION_DISPLAY_MAPPING = {
    'interactive': 'click',
    'mape_k': 'MAPE-K',
    'mapek': 'MAPE-K',
    'chat': 'chat',
    'mape_k_old': 'MAPE-K old'
}


def get_condition_display_name(condition):
    """Get the display name for a condition."""
    return CONDITION_DISPLAY_MAPPING.get(condition, condition)


class UnifiedDataLoader:
    """Single data loading class for all analysis needs."""

    def __init__(self):
        self.default_separator = ','

    def load_csv(self, file_path: str, separator: str = None) -> pd.DataFrame:
        """Load CSV file with automatic separator detection."""
        if separator is None:
            separator = self.default_separator

        try:
            return pd.read_csv(file_path, sep=separator)
        except Exception:
            try:
                return pd.read_csv(file_path, sep=';')
            except Exception as e:
                raise Exception(f"Failed to load {file_path}: {e}")

    def load_from_directories(self, data_directories: Dict[str, str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all CSV files from specified directories and return dict per condition."""
        condition_dataframes = {}

        for condition_name, directory in data_directories.items():
            dir_path = Path(directory)
            if not dir_path.exists():
                print(f"Warning: Directory {directory} does not exist")
                continue

            csv_files = list(dir_path.glob("*.csv"))
            if not csv_files:
                print(f"Warning: No CSV files found in {directory}")
                continue

            # Create a dictionary for this condition
            condition_dataframes[condition_name] = {}

            for csv_file in csv_files:
                try:
                    df = self.load_csv(str(csv_file))

                    # Create key from filename, removing "filtered" suffix if present
                    file_stem = csv_file.stem  # filename without extension
                    if file_stem.endswith('_filtered'):
                        df_key = file_stem.replace('_filtered', '')
                    else:
                        df_key = file_stem

                    condition_dataframes[condition_name][df_key] = df

                    print(f"Loaded {csv_file.name} as '{condition_name}[{df_key}]': {len(df)} rows")

                except Exception as e:
                    print(f"Warning: Failed to load {csv_file}: {e}")

        return condition_dataframes


class DialogueFilter:
    """Handles filtering of dialogues and users based on various criteria"""

    @staticmethod
    def get_users_with_min_questions(df, min_questions=2, condition=None):
        """Get users who asked at least min_questions."""
        df_filtered = df[df['condition'] == condition] if condition else df
        user_questions = df_filtered[df_filtered['role'] == 'explainee'].groupby('user_id').size()
        return set(user_questions[user_questions > min_questions].index.tolist())

    @staticmethod
    def get_user_dialogues(df, user_id):
        """Get all dialogues for a user, sorted by datapoint_id."""
        user_dialogues = df[df['user_id'] == user_id]
        dialogue_groups = user_dialogues.groupby('datapoint_id').agg({
            'turn_id': 'count',
            'objective_final_score': 'first',
            'condition': 'first'
        }).rename(columns={'turn_id': 'num_turns'})
        dialogue_groups['user_id'] = user_id
        return dialogue_groups.reset_index().sort_values('datapoint_id')

    @staticmethod
    def get_first_last_dialogues(df, users_with_min_questions):
        """Get first and last dialogue for each qualifying user."""
        first_dialogues = []
        last_dialogues = []

        for user_id in users_with_min_questions:
            user_dialogues = DialogueFilter.get_user_dialogues(df, user_id)
            if len(user_dialogues) >= 2:
                first_dialogues.append(user_dialogues.iloc[0])
                last_dialogues.append(user_dialogues.iloc[-1])

        return (pd.DataFrame(first_dialogues) if first_dialogues else pd.DataFrame(),
                pd.DataFrame(last_dialogues) if last_dialogues else pd.DataFrame())

    @staticmethod
    def get_first_last_dialogues_with_min_last_length(df, users_with_min_questions, min_last_length=6):
        """Get first and last dialogues where last dialogue meets minimum length."""
        first_dialogues = []
        last_dialogues = []
        qualifying_users = []

        for user_id in users_with_min_questions:
            user_dialogues = DialogueFilter.get_user_dialogues(df, user_id)
            if len(user_dialogues) >= 2:
                last_dialogue = user_dialogues.iloc[-1]
                if last_dialogue['num_turns'] >= min_last_length:
                    first_dialogues.append(user_dialogues.iloc[0])
                    last_dialogues.append(last_dialogue)
                    qualifying_users.append(user_id)

        return (pd.DataFrame(first_dialogues) if first_dialogues else pd.DataFrame(),
                pd.DataFrame(last_dialogues) if last_dialogues else pd.DataFrame(),
                set(qualifying_users))


class DialogueAnalyzer:
    """Performs statistical analysis on dialogue data"""

    @staticmethod
    def get_conversation_lengths(df, group_by_condition=True):
        """Calculate conversation lengths, optionally grouped by condition."""
        if group_by_condition:
            return df.groupby(['condition', 'user_id', 'datapoint_id']).size().reset_index(name='length')
        else:
            conversation_lengths = df.groupby(['user_id', 'datapoint_id']).size().reset_index(name='length')
            condition_info = df.groupby(['user_id', 'datapoint_id'])['condition'].first().reset_index()
            return conversation_lengths.merge(condition_info, on=['user_id', 'datapoint_id'])

    @staticmethod
    def compare_length_distributions(lengths1, lengths2, label1="Group 1", label2="Group 2"):
        """Compare two distributions of conversation lengths"""
        return {
            'descriptive': {
                label1: {
                    'n': len(lengths1),
                    'mean': lengths1.mean(),
                    'median': lengths1.median(),
                    'std': lengths1.std(),
                    'min': lengths1.min(),
                    'max': lengths1.max()
                },
                label2: {
                    'n': len(lengths2),
                    'mean': lengths2.mean(),
                    'median': lengths2.median(),
                    'std': lengths2.std(),
                    'min': lengths2.min(),
                    'max': lengths2.max()
                }
            }
        }

    @staticmethod
    def analyze_question_irt_correlation(df, conditions=None):
        """Analyze correlation between total questions and model understanding scores."""
        if conditions is None:
            conditions = df['condition'].unique().tolist()

        results = {}
        for condition in conditions:
            condition_df = df[df['condition'] == condition] if 'condition' in df.columns else df
            user_question_counts = condition_df.groupby('user_id').size().reset_index(name='total_questions')

            if 'score_for_user_id' in condition_df.columns:
                user_irt_scores = condition_df.groupby('user_id')['score_for_user_id'].first().reset_index()
                correlation_data = user_question_counts.merge(user_irt_scores, on='user_id')

                if len(correlation_data) > 1:
                    correlation = correlation_data['total_questions'].corr(correlation_data['score_for_user_id'],
                                                                           method='pearson')
                    results[condition] = {
                        'correlation': correlation,
                        'n_users': len(correlation_data),
                        'question_mean': correlation_data['total_questions'].mean(),
                        'question_std': correlation_data['total_questions'].std(),
                        'irt_mean': correlation_data['score_for_user_id'].mean(),
                        'irt_std': correlation_data['score_for_user_id'].std(),
                        'data': correlation_data
                    }
                else:
                    results[condition] = {'error': 'Insufficient data'}
            else:
                results[condition] = {'error': 'score_for_user_id not found'}

        return results


class DialogueVisualizer:
    """Handles visualization of dialogue data"""

    @staticmethod
    def plot_length_comparison(lengths1, lengths2, label1="Group 1", label2="Group 2",
                               title="Conversation Length Comparison", save_path=None):
        """Create comparison plots for conversation lengths."""
        plt.figure(figsize=(15, 5))

        # Histogram comparison
        plt.subplot(1, 3, 1)
        max_length = max(lengths1.max(), lengths2.max())
        bins = range(2, min(21, int(max_length) + 2))

        plt.hist([lengths1, lengths2], bins=bins, alpha=0.7,
                 label=[label1, label2], color=['skyblue', 'lightcoral'])
        plt.xlabel('Conversation Length (turns)')
        plt.ylabel('Number of Conversations')
        plt.title('Distribution of Conversation Lengths')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Box plot comparison
        plt.subplot(1, 3, 2)
        plt.boxplot([lengths1, lengths2], labels=[label1, label2])
        plt.ylabel('Conversation Length (turns)')
        plt.title('Conversation Length Distribution')
        plt.grid(True, alpha=0.3)

        # Cumulative distribution
        plt.subplot(1, 3, 3)
        plt.hist(lengths1, bins=bins, cumulative=True, density=True,
                 alpha=0.7, label=label1, color='skyblue')
        plt.hist(lengths2, bins=bins, cumulative=True, density=True,
                 alpha=0.7, label=label2, color='lightcoral')
        plt.xlabel('Conversation Length (turns)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_question_irt_correlation(correlation_results, conditions=None, save_path=None):
        """Create scatter plots showing correlation between questions and IRT scores."""
        if conditions is None:
            conditions = list(correlation_results.keys())

        valid_conditions = [c for c in conditions if 'data' in correlation_results.get(c, {})]
        if not valid_conditions:
            print("No valid data for correlation plotting")
            return

        n_conditions = len(valid_conditions)
        fig, axes = plt.subplots(1, n_conditions, figsize=(6 * n_conditions, 5))
        if n_conditions == 1:
            axes = [axes]

        for i, condition in enumerate(valid_conditions):
            result = correlation_results[condition]
            data = result['data']
            correlation = result['correlation']

            ax = axes[i]
            x_data = data['total_questions'].values
            y_data = data['score_for_user_id'].values

            ax.scatter(x_data, y_data, alpha=0.7)

            if len(x_data) > 1:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                sorted_indices = np.argsort(x_data)
                ax.plot(x_data[sorted_indices], p(x_data[sorted_indices]), "r--", alpha=0.8)

            condition_display = get_condition_display_name(condition)
            ax.set_xlabel('Total Questions Asked')
            ax.set_ylabel('User Score')
            ax.set_title(f'{condition_display}\nr = {correlation:.3f}, n = {result["n_users"]}')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Correlation: Total Questions vs Model Understanding Score', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class QuestionnaireAnalyzer:
    """Handles questionnaire data analysis with Likert scales."""

    def __init__(self):
        self.question_mappings = {}
        self.reversed_questions = []

    def configure_exit_questionnaire(self):
        """Configure for exit questionnaire analysis."""
        exit_questions = {
            "AL4": "The chatbot is cooperative.",
            "AL2": "I like the chatbot.",
            "coco3": "While explaining, the chatbot met me halfway.",
            "AI3": "The chatbot has no clue of what it is doing.",
            "scaf1": "The chatbot intended to provide me with the opportunity to build an understanding of the topic by asking questions.",
            "scaf2": "The chatbot encouraged me to continuously think about further details of the topic.",
            "UT1": "The chatbot always gives good advice.",
            "UAL4": "The chatbot can collaborate in a productive way.",
            "coco2": "The chatbot considered my understanding.",
            "UT2": "The chatbot acts truthfully.",
            "AC4": "The chatbots appears confused.",
            "moni2": "While explaining, it was important for the chatbot to monitor whether I understood everything.",
            "moni3": "The chatbot responded when I signaled non-understanding.",
            "AS3": "The chatbot interacts socially with me.",
            "coco4": "The chatbot took my statements into account.",
            "AI1": "The chatbot acts intentionally",
            "UAA2": "I can see myself using the chatbot in the future.",
            "scaf3": "When learning about a new topic, it's better to think about details yourself, rather than having everything fully explained.",
            "moni1": "While explaining, it was important for the chatbot to continuously consider whether I understood the explanation.",
            "scaf4": "The chatbot encouraged me to visualize the different processes of the topic.",
            "UT3": "I can rely on the chatbot.",
            "AU1": "The chatbot is easy to use.",
            "coco5": "The explanation was meant to encourage me to question my understanding.",
            "coco1": "The chatbot carefully adapted its utterances to my responses."
        }

        self.question_mappings = exit_questions
        self.reversed_questions = ['AI3', 'AC4']
        return self.get_exit_questionnaire_prefix_groups()

    def get_exit_questionnaire_prefix_groups(self):
        """Get prefix groups for exit questionnaire subscale analysis"""
        import re
        from collections import defaultdict

        prefix_groups = defaultdict(list)
        for code in self.question_mappings:
            match = re.match(r"[A-Za-z]+", code)
            prefix = match.group(0) if match else code
            prefix_groups[prefix].append(code)

        return dict(prefix_groups)

    def extract_exit_questionnaire_data(self, df, question_col='exit', answer_col='exit_answers'):
        """Extract exit questionnaire data from CSV format."""
        import ast

        rows = []
        for idx, row in df.iterrows():
            try:
                questions_list = ast.literal_eval(row[question_col])
                answers_list = ast.literal_eval(row[answer_col])

                reverse_lookup = {v: k for k, v in self.question_mappings.items()}
                code_answer_map = {"user_id": row["user_id"]}

                for q_text, ans in zip(questions_list, answers_list):
                    q_code = reverse_lookup.get(q_text.strip(), None)
                    if q_code:
                        if q_code in self.reversed_questions:
                            code_answer_map[q_code] = ans * -1
                        else:
                            code_answer_map[q_code] = ans

                rows.append(code_answer_map)

            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse questionnaire data for user {row.get('user_id', 'unknown')}: {e}")
                continue

        return pd.DataFrame(rows)


class ExplanationAnalyzer:
    """Analyzer for explanation types and reasoning patterns."""

    def __init__(self):
        self.explanation_types = {
            'Counterfactuals': {'Concept': 0, 'ImpactMultipleFeatures': 0, 'ImpactSingleFeature': 0},
            'FeatureImportances': {'Concept': 0, 'FeatureInfluencesPlot': 0, 'FeaturesInFavourOfOver50k': 0,
                                   'FeaturesInFavourOfUnder50k': 0, 'WhyThisFeatureImportant': 0},
            'AnchorExplanation': {'Concept': 0, 'Anchor': 0},
            'FeatureStatistics': {'Concept': 0, 'Feature Statistics': 0},
            'TextualPartialDependence': {'Concept': 0, 'PDPDescription': 0},
            'PossibleClarifications': {'Concept': 0, 'ClarificationsList': 0},
            'ModelPredictionConfidence': {'Concept': 0, 'Confidence': 0},
            'CeterisParibus': {'Concept': 0, 'PossibleClassFlips': 0, 'ImpossibleClassFlips': 0},
            'ScaffoldingStrategy': {'Reformulating': 0, 'Repeating': 0, 'ElicitingFeedback': 0}
        }

    def count_explanation_types(self, df, reasoning_column='reasoning', role_filter='explainer'):
        """Count occurrences of explanation types in reasoning text."""
        for explanation_type in self.explanation_types:
            for subtype in self.explanation_types[explanation_type]:
                self.explanation_types[explanation_type][subtype] = 0

        filtered_df = df[df["role"] == role_filter] if role_filter else df

        for _, row in filtered_df.iterrows():
            reasoning = str(row[reasoning_column])
            for explanation_type, subtypes in self.explanation_types.items():
                for subtype in subtypes.keys():
                    if subtype in reasoning:
                        self.explanation_types[explanation_type][subtype] += 1

        return self.explanation_types, pd.DataFrame(self.explanation_types).T.fillna(0)


class SharedStatistics:
    """Shared statistical functions across all analyses"""

    @staticmethod
    def perform_mann_whitney_comparison(group1_data, group2_data, group1_name, group2_name):
        """Perform Mann-Whitney U test with effect size calculation."""
        from scipy.stats import mannwhitneyu

        group1_clean = group1_data.dropna()
        group2_clean = group2_data.dropna()

        if len(group1_clean) == 0 or len(group2_clean) == 0:
            return {
                'error': 'Insufficient data for comparison',
                'group1_name': group1_name,
                'group2_name': group2_name
            }

        stat, p_value = mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')

        median1 = np.median(group1_clean)
        median2 = np.median(group2_clean)
        mean1 = np.mean(group1_clean)
        mean2 = np.mean(group2_clean)
        n1 = len(group1_clean)
        n2 = len(group2_clean)

        u_stat = min(stat, n1 * n2 - stat)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)

        higher_group = group1_name if median1 > median2 else group2_name

        return {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'n1': n1,
            'n2': n2,
            'median1': median1,
            'median2': median2,
            'mean1': mean1,
            'mean2': mean2,
            'mann_whitney_u': stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'higher_group': higher_group
        }

    @staticmethod
    def perform_condition_comparison(df, value_column, condition_column='condition', conditions=None):
        """Unified statistical comparison across conditions."""
        if conditions is None:
            conditions = df[condition_column].unique().tolist()

        if len(conditions) < 2:
            return {'error': 'Need at least 2 conditions for comparison'}

        results = {}
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                cond1, cond2 = conditions[i], conditions[j]
                group1_data = df[df[condition_column] == cond1][value_column]
                group2_data = df[df[condition_column] == cond2][value_column]

                comparison_key = f"{cond1} vs {cond2}"
                results[comparison_key] = SharedStatistics.perform_mann_whitney_comparison(
                    group1_data, group2_data, cond1, cond2
                )

        return results


def load_experiment_config(config_path: str = 'analysis_config.yaml') -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
