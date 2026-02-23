"""
First and Last Dialogue Analysis - Refactored Version
====================================================

Analyzes first and last dialogues of users with multiple interactions.
Refactored to use dialogue_analysis_base module for DRY principles.
"""

from dialogue_analysis_base import (
    DialogueFilter,
    DialogueAnalyzer,
    DialogueVisualizer,
)


def analyze_first_vs_last_dialogues(df, conditions=None, min_questions=2):
    """Analyze first and last dialogues of users with multiple interactions"""
    if conditions is None:
        conditions = df['condition'].unique().tolist()
    
    users_with_min_questions = DialogueFilter.get_users_with_min_questions(
        df, min_questions
    )
    
    if not users_with_min_questions:
        return None

    first_dialogues_df, last_dialogues_df = DialogueFilter.get_first_last_dialogues(
        df, users_with_min_questions
    )

    if first_dialogues_df.empty or last_dialogues_df.empty:
        return None

    analyzer = DialogueAnalyzer()
    first_lengths = first_dialogues_df['num_turns']
    last_lengths = last_dialogues_df['num_turns']

    results = analyzer.compare_length_distributions(
        first_lengths, last_lengths, "First", "Last"
    )

    analyze_by_condition(first_dialogues_df, last_dialogues_df, conditions)

    visualizer = DialogueVisualizer()
    visualizer.plot_length_comparison(
        first_lengths, last_lengths, "First", "Last",
        "First vs Last Dialogue Comparison", "first_last_comparison.png"
    )

    return {
        'first_dialogues': first_dialogues_df,
        'last_dialogues': last_dialogues_df,
        'statistical_results': results,
        'users_analyzed': users_with_min_questions
    }


def analyze_by_condition(first_dialogues_df, last_dialogues_df, conditions):
    """Analyze first/last differences within each experimental condition"""
    for condition in conditions:
        first_cond = first_dialogues_df[first_dialogues_df['condition'] == condition]
        last_cond = last_dialogues_df[last_dialogues_df['condition'] == condition]

        if first_cond.empty or last_cond.empty:
            continue

        first_lengths_cond = first_cond['num_turns']
        last_lengths_cond = last_cond['num_turns']


def analyze_users_with_long_last_dialogues(df, conditions=None, min_questions=2, min_last_length=6):
    """Analyze users whose last dialogue meets minimum length requirement"""
    if conditions is None:
        conditions = df['condition'].unique().tolist()
    
    users_with_criteria = DialogueFilter.get_users_with_min_last_dialogue_length(
        df, min_questions, min_last_length
    )

    if not users_with_criteria:
        return None

    first_dialogues_df, last_dialogues_df, filtered_users = DialogueFilter.get_first_last_dialogues_with_min_last_length(
        df, users_with_criteria, min_last_length
    )

    if first_dialogues_df.empty or last_dialogues_df.empty:
        return None

    first_lengths = first_dialogues_df['num_turns']
    last_lengths = last_dialogues_df['num_turns']

    visualizer = DialogueVisualizer()
    visualizer.plot_length_comparison(
        first_lengths, last_lengths, "First", "Last",
        f"Long Last Dialogues (≥{min_last_length} turns)", 
        f"long_last_dialogues_min{min_last_length}.png"
    )

    return {
        'first_dialogues': first_dialogues_df,
        'last_dialogues': last_dialogues_df,
        'filtered_users': filtered_users
    }


def display_first_last_comparison(df, conditions=None, min_questions=2):
    """Display comprehensive first vs last dialogue comparison with visualization"""
    results = analyze_first_vs_last_dialogues(df, conditions, min_questions)
    
    if results is None:
        return None
    
    # Additional visualization for first vs last comparison
    visualizer = DialogueVisualizer()
    first_lengths = results['first_dialogues']['num_turns']
    last_lengths = results['last_dialogues']['num_turns']
    
    # Create a specialized plot for first vs last comparison
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Subplot 1: Side-by-side boxplots
    ax1 = axes[0]
    data_for_box = [first_lengths, last_lengths]
    box_plot = ax1.boxplot(data_for_box, labels=['First', 'Last'], patch_artist=True, showfliers=False)
    box_plot['boxes'][0].set_facecolor('lightblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    ax1.set_ylabel('Conversation Length (turns)')
    ax1.set_title('First vs Last Dialogue Lengths')
    ax1.grid(True, alpha=0.3)
    
    # Add significance test
    try:
        stat, p_value = mannwhitneyu(first_lengths, last_lengths, alternative='two-sided')
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        y_max = max(max(first_lengths), max(last_lengths))
        ax1.text(1.5, y_max * 1.1, f'p={p_value:.3f} {significance}', ha='center', fontsize=10)
    except Exception:
        pass
    
    # Subplot 2: Overlapping distributions
    ax2 = axes[1]
    max_length = max(max(first_lengths), max(last_lengths))
    bins = range(1, min(21, int(max_length) + 2))
    
    ax2.hist([first_lengths, last_lengths], bins=bins, alpha=0.7, 
             label=['First', 'Last'], color=['lightblue', 'lightcoral'])
    ax2.set_xlabel('Conversation Length (turns)')
    ax2.set_ylabel('Number of Conversations')
    ax2.set_title('Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Summary statistics
    ax3 = axes[2]
    stats_data = {
        'First': [first_lengths.mean(), first_lengths.median(), first_lengths.std()],
        'Last': [last_lengths.mean(), last_lengths.median(), last_lengths.std()]
    }
    x_pos = [0, 1]
    width = 0.25
    
    ax3.bar([x - width for x in x_pos], [stats_data['First'][0], stats_data['Last'][0]], 
           width, label='Mean', color='skyblue', alpha=0.8)
    ax3.bar(x_pos, [stats_data['First'][1], stats_data['Last'][1]], 
           width, label='Median', color='orange', alpha=0.8)
    ax3.bar([x + width for x in x_pos], [stats_data['First'][2], stats_data['Last'][2]], 
           width, label='Std Dev', color='lightgreen', alpha=0.8)
    
    ax3.set_ylabel('Value')
    ax3.set_title('Statistical Summary')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['First', 'Last'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('First vs Last Dialogue Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('first_last_dialogue_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def main(data_file=None, output_dir="analysis_plots"):
    """Main analysis function for first vs last dialogue comparison
    
    Args:
        data_file: Path to the data file to load
        output_dir: Directory for output files
    """
    import os
    import yaml

    # Load config and determine data file location
    config_path = "1_analysis_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    results_dir = config['output']['results_dir']

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    from dialogue_analysis_base import UnifiedDataLoader
    loader = UnifiedDataLoader()
    df = loader.load_csv(data_file)

    if df is None:
        print(f"Error: Could not load data from {data_file}")
        return None
    
    conditions = df['condition'].unique().tolist()
    
    # Standard first vs last analysis
    result_standard = analyze_first_vs_last_dialogues(df, conditions, min_questions=2)
    
    # Analysis for users with long last dialogues
    result_long = analyze_users_with_long_last_dialogues(df, conditions, min_questions=2, min_last_length=6)
    
    # Display comprehensive comparison
    display_first_last_comparison(df, conditions, min_questions=2)
    
    return {
        'standard_analysis': result_standard,
        'long_dialogue_analysis': result_long
    }


if __name__ == "__main__":
    results = main()
