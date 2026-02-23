"""
Dialogue Analysis - Refactored Version
======================================

This script analyzes dialogue data with a focus on conversation lengths
and experimental condition comparisons. 

Refactored to use the dialogue_analysis_base module for DRY principles.

Key Features:
- Conversation length analysis by condition
- Statis    # Display different types of dialogues
    display_longest_dialogues(df, num_to_show=5, condition="mape_k")
    save_performance_based_dialogues_to_markdown(df, max_datapoints=10)al comparisons between conditions  
- Visualization of conversation patterns
- Display of longest dialogues
- Comprehensive statistical testing

Original functionality preserved while improving code organization.
"""

import os
import yaml
from dialogue_analysis_base import (
    DialogueAnalyzer,
    DialogueVisualizer,
    UnifiedDataLoader,
)


# Helper to ensure parent directory exists before saving a file
def ensure_parent_dir_exists(file_path):
    parent = os.path.dirname(file_path)
    if not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


try:
    from analyse_first_last_dialogues_refactored import display_first_last_comparison
except ImportError:  # Optional dependency
    display_first_last_comparison = None

# Dataset-specific suggestion lists
adult_suggestions = [
    "Explain this prediction to me",
    "Why under 50k?",
    "Why over 50k?",
    "Why do you think so?",
    "What factors matter the most in how the model makes its decision?",
    "Which features play the strongest role in influencing the model's prediction?",
    "What factors have very little effect on the model's decision?",
    "Which features contribute the least to the model's prediction?",
    "How much does each feature influence the prediction for this person?",
    "How strong is the effect of each factor on this person's outcome?",
    "What features would have to change to get a different prediction?",
    "Which factors must be altered for the model to predict something else?",
    "What set of features matters the most for this individual's prediction?",
    "Which collection of factors is most crucial for this person's result?",
    "How sure is the model about its prediction for this individual?",
    "What level of certainty does the model have regarding this person's prediction?",
    "Would the model's prediction change if just the Occupation were different?",
    "If we only alter the Occupation, does the prediction for this person shift?",
    "In general, how does someone's Occupation relate to the model's prediction?",
    "What is the typical relationship between Occupation and the model's decision?",
    "What is the spread of Age values across the dataset?",
    "How do the various Age values compare throughout the dataset?"
]

# Placeholder for diabetes dataset suggestions
diabetes_suggestions = [
    # TODO: Add diabetes-specific suggestions here
]

def get_suggestions(dataset="adult"):
    """Get dataset-specific suggestions list."""
    if dataset == "adult":
        return adult_suggestions
    elif dataset == "diabetes":
        return diabetes_suggestions if diabetes_suggestions else adult_suggestions  # Fallback to adult if empty
    return adult_suggestions  # Default fallback

# Backward compatibility
suggestions = adult_suggestions


def analyze_conversation_lengths_by_condition(df, conditions=None, save_path="conversation_length_analysis.png"):
    """Analyze and display overlapping distribution for conversation lengths by condition"""
    analyzer = DialogueAnalyzer()
    conversation_lengths = analyzer.get_conversation_lengths(df)

    if conditions is None:
        conditions = df['condition'].unique().tolist()

    condition_data = {}
    for condition in conditions:
        if condition in df['condition'].values:
            condition_lengths = conversation_lengths[conversation_lengths['condition'] == condition]['length']
            condition_data[condition] = condition_lengths

    visualizer = DialogueVisualizer()
    if len(condition_data) >= 2:
        condition_list = list(condition_data.keys())
        visualizer.plot_length_comparison(
            condition_data[condition_list[0]],
            condition_data[condition_list[1]],
            condition_list[0], condition_list[1],
            f"Conversation Length Distribution: {condition_list[0]} vs {condition_list[1]}",
            save_path
        )

    return conversation_lengths, condition_data


def print_length_distribution(condition_data, conditions=None):
    """Print distribution of conversation lengths by condition"""
    if conditions is None:
        conditions = list(condition_data.keys())

    if len(conditions) >= 2 and all(cond in condition_data for cond in conditions[:2]):
        cond1, cond2 = conditions[0], conditions[1]
        cond1_lengths = condition_data[cond1]
        cond2_lengths = condition_data[cond2]

        print(f"\nDistribution by Length:")
        print(f"  {'Length':<10} {cond1:<10} {cond2:<12} {cond1 + ' %':<10} {cond2 + ' %':<12}")
        print(f"  {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 12}")

        max_length = max(cond1_lengths.max(), cond2_lengths.max())

        for length in range(2, min(21, int(max_length) + 1)):
            c1_count = (cond1_lengths == length).sum()
            c2_count = (cond2_lengths == length).sum()
            c1_pct = c1_count / len(cond1_lengths) * 100 if len(cond1_lengths) > 0 else 0
            c2_pct = c2_count / len(cond2_lengths) * 100 if len(cond2_lengths) > 0 else 0

            if c1_count > 0 or c2_count > 0:
                print(f"  {length:<10} {c1_count:<10} {c2_count:<12} {c1_pct:<9.1f}% {c2_pct:<11.1f}%")

        long_c1 = (cond1_lengths >= 20).sum()
        long_c2 = (cond2_lengths >= 20).sum()
        if long_c1 > 0 or long_c2 > 0:
            c1_pct = long_c1 / len(cond1_lengths) * 100 if len(cond1_lengths) > 0 else 0
            c2_pct = long_c2 / len(cond2_lengths) * 100 if len(cond2_lengths) > 0 else 0
            print(f"  {'20+ turns':<10} {long_c1:<10} {long_c2:<12} {c1_pct:<9.1f}% {c2_pct:<11.1f}%")


def statistical_comparison_conversation_lengths(df, conditions=None):
    """
    Compare mean and std of conversation lengths per condition per dialogue (datapoint).
    """
    if conditions is None:
        conditions = df['condition'].unique().tolist()

    if len(conditions) < 2:
        return None

    # Get conversation lengths grouped by condition, user_id, and datapoint_id
    conversation_lengths = df.groupby(['condition', 'user_id', 'datapoint_id']).size().reset_index(name='length')

    # Calculate statistics per condition per datapoint
    stats_per_datapoint = conversation_lengths.groupby(['condition', 'datapoint_id'])['length'].agg(
        ['mean', 'std', 'count']).reset_index()

    results = {}
    for condition in conditions:
        cond_data = stats_per_datapoint[stats_per_datapoint['condition'] == condition]
        results[condition] = {
            'mean_per_datapoint': cond_data['mean'].tolist(),
            'std_per_datapoint': cond_data['std'].fillna(0).tolist(),  # Fill NaN with 0 for single observations
            'count_per_datapoint': cond_data['count'].tolist(),
            'datapoints': cond_data['datapoint_id'].tolist(),
            'overall_mean': cond_data['mean'].mean(),
            'overall_std': cond_data['mean'].std()
        }

    # Print comparison
    print("\nCONVERSATION LENGTH STATISTICS PER CONDITION")
    print("=" * 60)
    for condition in conditions:
        print(f"\n{condition.upper()}:")
        print(f"  Overall mean across datapoints: {results[condition]['overall_mean']:.2f}")
        print(f"  Std of means across datapoints: {results[condition]['overall_std']:.2f}")
        print(f"  Datapoints analyzed: {len(results[condition]['datapoints'])}")

    return results


def get_longest_dialogues(df, num_to_show=3, condition=None):
    """Get information about the longest dialogues from the dataset"""
    conversation_lengths = df.groupby(['user_id', 'datapoint_id']).agg({
        'turn_id': 'count',
        'condition': 'first'
    }).rename(columns={'turn_id': 'length'}).reset_index()

    # Filter by condition if specified
    if condition:
        conversation_lengths = conversation_lengths[conversation_lengths['condition'] == condition]

    return conversation_lengths.sort_values('length', ascending=False).head(num_to_show)


def display_longest_dialogues(df, num_to_show=3, condition=None):
    """Display the longest dialogues from the dataset"""
    longest_dialogues = get_longest_dialogues(df, num_to_show, condition)

    print(f"\n{'=' * 60}")
    print(f"LONGEST DIALOGUES{' (' + condition + ')' if condition else ''}")
    print(f"{'=' * 60}")

    for i, (_, dialogue_info) in enumerate(longest_dialogues.iterrows()):
        title = f"#{i + 1} LONGEST DIALOGUE ({dialogue_info['length']} turns)"
        display_dialogue_with_suggestions(
            df,
            dialogue_info['user_id'],
            dialogue_info['datapoint_id'],
            title
        )


def create_visualizations(condition_data, visualizer, conditions=None):
    """Create visualizations for the analysis"""
    if conditions is None:
        conditions = list(condition_data.keys())

    if len(conditions) >= 2 and all(cond in condition_data for cond in conditions[:2]):
        cond1, cond2 = conditions[0], conditions[1]
        visualizer.plot_length_comparison(
            condition_data[cond1],
            condition_data[cond2],
            cond1, cond2,
            f"Conversation Length Comparison: {cond1} vs {cond2}",
            "conversation_length_analysis.png"
        )


def display_first_last_dialogue_comparison(df, conditions=None, min_questions=2):
    """Display comparison between first and last turns across conditions"""
    # Use the refactored function from analyse_first_last_dialogues
    return display_first_last_comparison(df, conditions, min_questions)


def analyze_question_irt_correlation_by_condition(df, conditions=None, save_path="question_irt_correlation.png"):
    """Analyze and visualize correlation between total questions and IRT scores by condition"""
    print("\nQUESTION-IRT CORRELATION ANALYSIS")
    print("=" * 50)

    analyzer = DialogueAnalyzer()
    correlation_results = analyzer.analyze_question_irt_correlation(df, conditions)

    # Print results
    for condition, result in correlation_results.items():
        print(f"\nCondition: {condition}")
        print("-" * 30)

        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Correlation: {result['correlation']:.3f}")
            print(f"  Users analyzed: {result['n_users']}")
            print(f"  Questions - Mean: {result['question_mean']:.1f}, Std: {result['question_std']:.1f}")
            print(f"  IRT Score - Mean: {result['irt_mean']:.3f}, Std: {result['irt_std']:.3f}")

    # Create visualization
    visualizer = DialogueVisualizer()
    visualizer.plot_question_irt_correlation(correlation_results, conditions, save_path)

    return correlation_results


def get_user_performance_ranking(df, score_column='objective_final_score'):
    """
    Get user performance ranking based on scores.
    Returns best, worst, and average performing users.
    """
    analyzer = DialogueAnalyzer()

    # Get unique users with their scores
    user_scores = df.groupby('user_id')[score_column].first().reset_index()
    user_scores = user_scores.sort_values(score_column, ascending=False)

    best_user = user_scores.iloc[0]['user_id']
    worst_user = user_scores.iloc[-1]['user_id']

    # Find user closest to median score
    median_score = user_scores[score_column].median()
    user_scores['distance_from_median'] = abs(user_scores[score_column] - median_score)
    average_user = user_scores.loc[user_scores['distance_from_median'].idxmin(), 'user_id']

    return {
        'best': {'user_id': best_user, 'score': user_scores.iloc[0][score_column]},
        'worst': {'user_id': worst_user, 'score': user_scores.iloc[-1][score_column]},
        'average': {'user_id': average_user,
                    'score': user_scores[user_scores['user_id'] == average_user][score_column].iloc[0]}
    }


def format_dialogue_to_markdown(df, user_id, datapoint_id, title="Dialogue"):
    """Format a single dialogue as markdown text"""
    dialogue_turns = df[
        (df['user_id'] == user_id) &
        (df['datapoint_id'] == datapoint_id)
        ].sort_values('turn_id')

    if dialogue_turns.empty:
        return ""

    condition = dialogue_turns.iloc[0]['condition']
    score = dialogue_turns.iloc[0]['objective_final_score']

    # Count suggested questions
    suggested_count, total_user_questions = count_suggested_questions(dialogue_turns)

    markdown_text = f"\n## {title}\n\n"
    markdown_text += f"- **User ID:** {user_id}\n"
    markdown_text += f"- **Datapoint:** {datapoint_id}\n"
    markdown_text += f"- **Condition:** {condition}\n"
    markdown_text += f"- **Length:** {len(dialogue_turns)} turns\n"
    markdown_text += f"- **Score:** {score:.4f}\n"
    markdown_text += f"- **Questions from suggestions:** {suggested_count}/{total_user_questions}\n\n"

    markdown_text += "### Conversation:\n\n"

    for i, (_, turn) in enumerate(dialogue_turns.iterrows()):
        role = turn['role']
        text = turn['turn_text']

        # Mark suggested questions with *
        marked_text = mark_suggested_questions(text, role)

        role_display = "**USER**" if role == 'explainee' else "**AI**"
        markdown_text += f"{role_display} (Turn {turn['turn_id']}):\n"
        markdown_text += f"> {marked_text}\n\n"

    return markdown_text


def save_user_dialogues_to_markdown(df, user_id, user_type, score, max_datapoints=10, file_handle=None):
    """
    Save all dialogues for a specific user to markdown format.
    """
    user_dialogues = df[df['user_id'] == user_id]
    available_datapoints = sorted(user_dialogues['datapoint_id'].unique())

    markdown_text = f"\n# {user_type.upper()} USER DIALOGUES (Score: {score:.4f})\n\n"
    markdown_text += f"- **User ID:** {user_id}\n"
    markdown_text += f"- **Available datapoints:** {len(available_datapoints)} (showing up to {max_datapoints})\n\n"

    if file_handle:
        file_handle.write(markdown_text)

    for i, datapoint_id in enumerate(available_datapoints[:max_datapoints]):
        title = f"{user_type.upper()} USER - Dialogue {i + 1}/{min(len(available_datapoints), max_datapoints)}"
        dialogue_md = format_dialogue_to_markdown(df, user_id, datapoint_id, title)

        if file_handle:
            file_handle.write(dialogue_md)
        else:
            return markdown_text + dialogue_md


def save_performance_based_dialogues_to_markdown(df, filename="performance_based_dialogues.md", max_datapoints=10,
                                                 condition=None):
    """
    Save dialogues for best, worst, and average performing users to a markdown file.
    Shows all available dialogues (1-10) for each user type.
    """
    # Filter by condition if specified
    if condition:
        df = df[df['condition'] == condition]

    # Get user rankings
    rankings = get_user_performance_ranking(df)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Performance-Based Dialogue Analysis\n\n")
        if condition:
            f.write(f"**Condition:** {condition}\n\n")
        f.write(
            "This document contains dialogues from the best, worst, and average performing users based on their scores.\n\n")
        f.write("**Note:** Questions marked with `*` at the beginning are from the suggested questions list. "
                "Questions without `*` are user-generated.\n\n")
        f.write("---\n\n")

        # Save dialogues for each user type
        for user_type, user_info in rankings.items():
            save_user_dialogues_to_markdown(
                df,
                user_info['user_id'],
                user_type,
                user_info['score'],
                max_datapoints,
                f
            )

    condition_text = f" (condition: {condition})" if condition else ""
    print(f"Performance-based dialogues{condition_text} saved to: {filename}")
    return filename


def main(conditions=None, data_file=None, output_dir="analysis_plots"):
    """Main analysis function
    
    Args:
        conditions: List of conditions to analyze
        data_file: Path to the data file to load
        output_dir: Directory for output files
    """

    if conditions is None:
        conditions = ['mape_k', 'interactive']

    # Load config and determine data file location
    config_path = "1_analysis_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    results_dir = config['output']['results_dir']
    output_dir = results_dir
    data_dir = os.path.join(results_dir, "data")
    conditions = [
        config['experiment_settings']['condition_a'],
        config['experiment_settings']['condition_b'],
    ]

    if data_file is None:
        # Default to meisam_df_chats_mape_k_interactive.csv in the data_dir
        # Use the first condition as default for now
        data_file = os.path.join(data_dir, f"meisam_df_{conditions[0]}.csv")

    # Create a subdirectory in output_dir for dialogue analysis outputs
    dialogue_output_dir = os.path.join(output_dir, "dialogue_analysis")
    os.makedirs(data_dir, exist_ok=True)

    # Load data
    loader = UnifiedDataLoader()
    df = loader.load_csv(data_file)

    # Initialize components
    analyzer = DialogueAnalyzer()
    visualizer = DialogueVisualizer()

    # Analyze conversation lengths and save plot in dialogue_output_dir
    conv_len_plot_path = os.path.join(dialogue_output_dir, "conversation_length_analysis.png")
    # Only create the output directory once
    ensure_parent_dir_exists(conv_len_plot_path)

    conversation_lengths, condition_data = analyze_conversation_lengths_by_condition(
        df, conditions=conditions, save_path=conv_len_plot_path)

    # Save conversation_lengths and condition_data as CSV/JSON
    conversation_lengths_path = os.path.join(dialogue_output_dir, "conversation_lengths.csv")
    conversation_lengths.to_csv(conversation_lengths_path, index=False)

    import json
    condition_data_path = os.path.join(dialogue_output_dir, "condition_data.json")
    with open(condition_data_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in condition_data.items()}, f)

    # Statistical comparison and save as JSON
    statistical_results = statistical_comparison_conversation_lengths(df, conditions=conditions)
    stat_results_path = os.path.join(dialogue_output_dir, "statistical_results.json")
    with open(stat_results_path, 'w') as f:
        json.dump(statistical_results, f, indent=2)

    # First/last dialogue comparison (save output if possible)
    first_last_result = display_first_last_dialogue_comparison(df, conditions)
    if first_last_result is not None:
        first_last_path = os.path.join(dialogue_output_dir, "first_last_dialogue_comparison.json")
        try:
            with open(first_last_path, 'w') as f:
                json.dump(first_last_result, f, indent=2)
        except Exception:
            pass

    # Save performance-based dialogues markdown in dialogue_output_dir (do this first)
    perf_dialogues_path = os.path.join(dialogue_output_dir, "performance_based_dialogues.md")
    save_performance_based_dialogues_to_markdown(df, filename=perf_dialogues_path, max_datapoints=10,
                                                 condition="mape_k_old")

    # Question-IRT correlation analysis and plot
    q_irt_plot_path = os.path.join(dialogue_output_dir, "question_irt_correlation.png")
    correlation_results = analyze_question_irt_correlation_by_condition(df, conditions, save_path=q_irt_plot_path)
    correlation_results_path = os.path.join(dialogue_output_dir, "question_irt_correlation.json")
    with open(correlation_results_path, 'w') as f:
        json.dump(correlation_results, f, indent=2)

    # Plot questions per datapoint for the three performance-based users, save in dialogue_output_dir
    mape_k_df = df[df['condition'] == 'mape_k_old']
    rankings = get_user_performance_ranking(mape_k_df)
    user_ids = [info['user_id'] for info in rankings.values()]
    user_labels = {info['user_id']: f"{user_type.title()} (Score: {info['score']:.3f})"
                   for user_type, info in rankings.items()}
    perf_users_plot_path = os.path.join(dialogue_output_dir, "performance_users_questions.png")
    plot_questions_per_datapoint_for_users(mape_k_df, user_ids, user_labels, save_path=perf_users_plot_path)

    return {
        'data': df,
        'conversation_lengths_csv': conversation_lengths_path,
        'condition_data_json': condition_data_path,
        'statistical_results_json': stat_results_path,
        'first_last_dialogue_comparison_json': first_last_path if first_last_result is not None else None,
        'question_irt_correlation_plot': q_irt_plot_path,
        'question_irt_correlation_json': correlation_results_path,
        'performance_based_dialogues_md': perf_dialogues_path,
        'performance_users_questions_plot': perf_users_plot_path
    }


def plot_conversation_length_comparison(conversation_lengths, conditions=None):
    """Backwards compatibility wrapper"""
    visualizer = DialogueVisualizer()

    if conditions is None:
        conditions = conversation_lengths['condition'].unique().tolist()

    if len(conditions) >= 2:
        cond1, cond2 = conditions[0], conditions[1]
        cond1_lengths = conversation_lengths[conversation_lengths['condition'] == cond1]['length']
        cond2_lengths = conversation_lengths[conversation_lengths['condition'] == cond2]['length']

        visualizer.plot_length_comparison(
            cond1_lengths, cond2_lengths,
            cond1, cond2,
            f"Conversation Length Comparison: {cond1} vs {cond2}",
            "conversation_length_analysis.png"
        )


def display_longest_dialogue(df, nth_longest=0, condition=None):
    """Backwards compatibility wrapper"""

    if condition:
        filtered_df = df[df['condition'] == condition]
        condition_text = f" (condition: {condition})"
    else:
        filtered_df = df
        condition_text = ""

    dialogue_lengths = filtered_df.groupby(['user_id', 'datapoint_id']).size().reset_index(name='length')
    dialogue_lengths = dialogue_lengths.sort_values('length', ascending=False).reset_index(drop=True)

    if nth_longest >= len(dialogue_lengths):
        return

    target_dialogue = dialogue_lengths.iloc[nth_longest]
    title = f"{'LONGEST' if nth_longest == 0 else f'{nth_longest + 1}TH LONGEST'} DIALOGUE{condition_text}"

    display_dialogue_with_suggestions(
        df,
        target_dialogue['user_id'],
        target_dialogue['datapoint_id'],
        title
    )


def analyze_user_performance_dialogues(file_path="meisam_df_chats_mape_k_interactive.csv", max_datapoints=10,
                                       condition=None):
    """
    Standalone function to analyze and save dialogues for best, worst, and average users to markdown.
    Can filter by condition if specified.
    """
    from dialogue_analysis_base import UnifiedDataLoader
    loader = UnifiedDataLoader()
    df = loader.load_csv(file_path)
    if df is None:
        print("Error: Could not load data")
        return None

    save_performance_based_dialogues_to_markdown(df, max_datapoints=max_datapoints, condition=condition)
    return df


def mark_suggested_questions(text, role, dataset="adult"):
    """
    Mark suggested questions with a * if the text matches any suggestion.
    Only applies to user questions (role == 'explainee').
    """
    if role != 'explainee':
        return text

    # Check if the user's question matches any suggestion (case-insensitive, strip whitespace)
    text_clean = text.strip().lower()
    suggestions_list = get_suggestions(dataset)
    for suggestion in suggestions_list:
        if text_clean == suggestion.lower().strip():
            return f"*{text}"

    return text


def display_dialogue_with_suggestions(df, user_id, datapoint_id, title="Dialogue"):
    """Display a single dialogue in chat format with suggested questions marked"""
    dialogue_turns = df[
        (df['user_id'] == user_id) &
        (df['datapoint_id'] == datapoint_id)
        ].sort_values('turn_id')

    if dialogue_turns.empty:
        return

    condition = dialogue_turns.iloc[0]['condition']
    score = dialogue_turns.iloc[0]['score_for_user_id']

    print(f"\n{title}")
    print("=" * 60)
    print(f"User ID: {user_id}")
    print(f"Datapoint: {datapoint_id}")
    print(f"Condition: {condition}")
    print(f"Length: {len(dialogue_turns)} turns")
    print(f"Score: {score:.4f}")

    print(f"\nConversation:")
    print("-" * 60)

    for i, (_, turn) in enumerate(dialogue_turns.iterrows()):
        role = turn['role']
        text = turn['turn_text']

        # Mark suggested questions with *
        marked_text = mark_suggested_questions(text, role)

        role_display = "USER" if role == 'explainee' else "AI"
        indent = "" if role == 'explainee' else "  "

        print(f"\n{role_display} (Turn {turn['turn_id']}):")
        import textwrap
        wrapped_text = textwrap.fill(marked_text, width=80, initial_indent=indent, subsequent_indent=indent)
        print(wrapped_text)

    print("-" * 60)


def count_suggested_questions(dialogue_turns, dataset="adult"):
    """
    Count how many user questions are from the suggestions list.
    Returns (suggested_count, total_user_questions)
    """
    user_turns = dialogue_turns[dialogue_turns['role'] == 'explainee']
    total_user_questions = len(user_turns)
    suggested_count = 0

    suggestions_list = get_suggestions(dataset)
    for _, turn in user_turns.iterrows():
        text_clean = turn['turn_text'].strip().lower()
        for suggestion in suggestions_list:
            if text_clean == suggestion.lower().strip():
                suggested_count += 1
                break

    return suggested_count, total_user_questions

def is_autosuggested_question(text, dataset="adult"):
    """
    Check if a question text matches any autosuggestion.
    Returns True if the text is from suggestions, False otherwise.
    """
    text_clean = text.strip().lower()
    suggestions_list = get_suggestions(dataset)
    for suggestion in suggestions_list:
        if text_clean == suggestion.lower().strip():
            return True
    return False


def plot_questions_per_datapoint_for_users(df, user_ids, user_labels, max_datapoints=10,
                                           save_path="performance_users_questions.png"):
    """
    Plot questions per datapoint for specific users (reusing turn position distribution logic).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Get question counts (user turns) for each datapoint and user
    user_questions = df[df['role'] == 'explainee'].groupby(['user_id', 'datapoint_id']).size().reset_index(
        name='question_count')

    # Filter for specified users
    user_questions = user_questions[user_questions['user_id'].isin(user_ids)]

    # Prepare data for plotting (similar to plot_turn_position_distribution)
    plot_data = []
    datapoint_ids = sorted(user_questions['datapoint_id'].unique())[:max_datapoints]

    for datapoint_id in datapoint_ids:
        for user_id in user_ids:
            user_data = user_questions[
                (user_questions['user_id'] == user_id) &
                (user_questions['datapoint_id'] == datapoint_id)
                ]

            if not user_data.empty:
                count = user_data['question_count'].iloc[0]
                user_label = user_labels.get(user_id, user_id)
                plot_data.append({
                    'datapoint': datapoint_id,
                    'user_type': user_label,
                    'question_count': count
                })

    if not plot_data:
        print("No data found for specified users")
        return

    plot_df = pd.DataFrame(plot_data)

    # Create the plot (reusing visualization style)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert to numpy arrays to avoid pandas indexing issues
    for user_type in plot_df['user_type'].unique():
        user_data = plot_df[plot_df['user_type'] == user_type]
        ax.plot(
            user_data['datapoint'].values,  # Convert to numpy array
            user_data['question_count'].values,  # Convert to numpy array
            marker='o',
            linewidth=2,
            markersize=8,
            label=user_type
        )

    ax.set_xlabel('Datapoint')
    ax.set_ylabel('Number of Questions')
    ax.set_title('Questions per Datapoint: Best vs Worst vs Average Users')
    ax.legend(title='User Performance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Questions per datapoint plot saved to: {save_path}")
    plt.show()


# call main
if __name__ == "__main__":
    main()
