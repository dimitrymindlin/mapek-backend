"""
Likert Scale Analysis - Refactored
Accepts DataFrames and performs Likert scale analysis for exit questionnaire data.
"""

import os
from collections import defaultdict
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, norm
import ast
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Helpers for proper MWU stats and reporting ---

def _tie_correction(values):
    """Return the tie correction term T = sum(t_i^3 - t_i) over tie groups.
    values: 1D array-like of combined sample values used to detect ties.
    """
    vals, counts = np.unique(values, return_counts=True)
    T = np.sum(counts**3 - counts)
    return T


def _mwu_z_from_u(U, n1, n2, combined_values, continuity=True):
    """Compute a tie-corrected z statistic from a Mann-Whitney U statistic.
    Uses Var(U) = n1*n2*(N+1)/12 - n1*n2*T/(12*N*(N-1)), where
    T = sum(t_i^3 - t_i) over tie groups in the combined sample.
    """
    N = n1 + n2
    mean_U = n1 * n2 / 2.0
    # tie-corrected variance
    T = _tie_correction(combined_values)
    var_U = (n1 * n2 * (N + 1)) / 12.0
    if N > 1 and T > 0:
        var_U -= (n1 * n2 * T) / (12.0 * N * (N - 1))
    sd_U = np.sqrt(max(var_U, 0.0))
    if sd_U == 0:
        return 0.0
    # continuity correction
    if continuity:
        if U > mean_U:
            z = (U - 0.5 - mean_U) / sd_U
        elif U < mean_U:
            z = (U + 0.5 - mean_U) / sd_U
        else:
            z = 0.0
    else:
        z = (U - mean_U) / sd_U
    return z


def _summarize_likert(arr):
    """Return median, IQR (q1, q3), mean, std for a 1D numeric array."""
    a = np.asarray(arr)
    median = float(np.median(a))
    q1 = float(np.percentile(a, 25))
    q3 = float(np.percentile(a, 75))
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return median, (q1, q3), mean, std


def _cliffs_delta_from_U(U, n1, n2):
    """Cliff's delta from U: delta = 2U/(n1*n2) - 1."""
    return (2.0 * U) / (n1 * n2) - 1.0


def _vardelaney_A12_from_U(U, n1, n2):
    """Vargha–Delaney A12 effect size: A12 = U/(n1*n2)."""
    return U / (n1 * n2)



def mann_whitney_test_for_likert(user_df1, user_df2, alpha=0.05):
    """Perform Mann-Whitney U test for Likert scale data between two groups."""
    # Function to extract questions
    def extract_questions(row):
        return row['exit']['questions']

    # Helper function to convert and accumulate data
    def accumulate_feedback(df, column_name=None):
        try:
            column_name = "exit"
            feedback_questions = df[column_name].apply(ast.literal_eval)
        except KeyError:
            column_name = "exit_q"
            feedback_questions = df[column_name].apply(ast.literal_eval)
            feedback_questions = feedback_questions.apply(lambda x: extract_questions(x))

        feedback_answers = df[f'{column_name}_answers'].apply(ast.literal_eval)

        feedback_dict = defaultdict(list)
        for questions, answers in zip(feedback_questions, feedback_answers):
            for question, answer in zip(questions, answers):
                feedback_dict[question].append(answer)
        return feedback_dict

    # Accumulate feedback for both groups
    feedback_dict_chat = accumulate_feedback(user_df1)
    feedback_dict_active_chat = accumulate_feedback(user_df2)

    # Conduct Mann-Whitney U test for each question
    results = []
    all_questions = set(feedback_dict_chat.keys()).union(feedback_dict_active_chat.keys())

    for question in all_questions:
        answers_chat = feedback_dict_chat.get(question, [])
        answers_active_chat = feedback_dict_active_chat.get(question, [])

        # Only perform the test if both groups have data
        if answers_chat and answers_active_chat:
            stat, p_value = mannwhitneyu(answers_chat, answers_active_chat, alternative='two-sided')

            n1, n2 = len(answers_chat), len(answers_active_chat)
            median_chat, (q1_chat, q3_chat), mean_chat, sd_chat = _summarize_likert(answers_chat)
            median_active_chat, (q1_active, q3_active), mean_active_chat, sd_active_chat = _summarize_likert(answers_active_chat)

            # Tie-corrected z and effect sizes
            combined_vals = np.concatenate([answers_chat, answers_active_chat])
            z_score = _mwu_z_from_u(stat, n1, n2, combined_vals, continuity=True)
            effect_size_r = abs(z_score) / np.sqrt(n1 + n2)  # r = |z|/sqrt(N)
            cliffs_delta = _cliffs_delta_from_U(stat, n1, n2)
            a12 = _vardelaney_A12_from_U(stat, n1, n2)

            results.append({
                'Question': question,
                'n1': n1,
                'n2': n2,
                'Median_Group1': median_chat,
                'IQR_Group1': (q1_chat, q3_chat),
                'Mean_Group1': mean_chat,
                'SD_Group1': sd_chat,
                'Median_Group2': median_active_chat,
                'IQR_Group2': (q1_active, q3_active),
                'Mean_Group2': mean_active_chat,
                'SD_Group2': sd_active_chat,
                'Mann-Whitney U': stat,
                'Z': z_score,
                'P-value': p_value,
                'Effect_Size_r': effect_size_r,
                "Cliffs_delta": cliffs_delta,
                "A12": a12,
            })

    # Add significance based on raw p-values
    for r in results:
        r['Significant'] = bool(r['P-value'] < alpha)

    # Convert results to DataFrame for easy viewing
    results_df = pd.DataFrame(results)

    # Display the detailed results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    logger.info("DETAILED RESULTS:")
    display_cols = [
        'Question', 'n1', 'n2', 'Median_Group1', 'IQR_Group1', 'Median_Group2', 'IQR_Group2',
        'Mann-Whitney U', 'Z', 'P-value', 'Effect_Size_r', 'Cliffs_delta', 'A12', 'Significant'
    ]
    existing_cols = [c for c in display_cols if c in results_df.columns]
    logger.info(results_df[existing_cols].to_string())

    return results_df


def detailed_self_assessment_analysis(group1_data, group2_data, group1_name, group2_name, output_dir="analysis_plots", data_dir="data"):
    """
    Perform detailed self-assessment analysis between two specific groups.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Function to safely extract exit questions and answers
    def get_exit_data(df):
        if 'exit_questions' in df.columns and 'exit_answers' in df.columns:
            questions = df['exit_questions'].apply(ast.literal_eval)
            answers = df['exit_answers'].apply(ast.literal_eval)
        elif 'exit_q' in df.columns and 'exit_q_answers' in df.columns:
            exit_data = df['exit_q'].apply(ast.literal_eval)
            questions = exit_data.apply(lambda x: x['exit']['questions'])
            answers = df['exit_q_answers'].apply(ast.literal_eval)
        else:
            return None, None
        return questions, answers

    # Get exit data for both groups
    questions1, answers1 = get_exit_data(group1_data)
    questions2, answers2 = get_exit_data(group2_data)

    if questions1 is None or questions2 is None:
        logger.warning("Cannot perform detailed analysis: Missing exit questionnaire data")
        return {}

    # Accumulate feedback data
    def accumulate_feedback(questions, answers):
        feedback_dict = defaultdict(list)
        for qs, ans in zip(questions, answers):
            for question, answer in zip(qs, ans):
                # Normalize curly quotes and trim
                question = question.replace("’", "'").strip()
                feedback_dict[question].append(answer)
        return feedback_dict

    feedback_dict1 = accumulate_feedback(questions1, answers1)
    feedback_dict2 = accumulate_feedback(questions2, answers2)

    # Get all unique questions
    all_questions = set(feedback_dict1.keys()).union(feedback_dict2.keys())

    mann_whitney_results = {}
    detailed_results = []  # Collect all results for CSV export

    for question in all_questions:
        group1_responses = feedback_dict1.get(question, [])
        group2_responses = feedback_dict2.get(question, [])

        # Only analyze if both groups have responses
        if len(group1_responses) > 0 and len(group2_responses) > 0:
            # Perform Mann-Whitney U test
            stat, p_value = mannwhitneyu(group1_responses, group2_responses, alternative='two-sided')

            # Descriptives
            n1, n2 = len(group1_responses), len(group2_responses)
            median1, (q1_1, q3_1), mean1, sd1 = _summarize_likert(group1_responses)
            median2, (q1_2, q3_2), mean2, sd2 = _summarize_likert(group2_responses)

            # Tie-corrected z and effect sizes
            combined_vals = np.concatenate([group1_responses, group2_responses])
            z_score = _mwu_z_from_u(stat, n1, n2, combined_vals, continuity=True)
            effect_size_r = abs(z_score) / np.sqrt(n1 + n2)
            cliffs_delta = _cliffs_delta_from_U(stat, n1, n2)
            a12 = _vardelaney_A12_from_U(stat, n1, n2)

            # Determine direction using medians then means
            if median1 > median2:
                higher_group = group1_name
                direction = f"{group1_name} > {group2_name}"
            elif median2 > median1:
                higher_group = group2_name
                direction = f"{group2_name} > {group1_name}"
            else:
                if mean1 > mean2:
                    higher_group = group1_name
                    direction = f"{group1_name} ≈ {group2_name} (same median, {group1_name} higher mean)"
                elif mean2 > mean1:
                    higher_group = group2_name
                    direction = f"{group1_name} ≈ {group2_name} (same median, {group2_name} higher mean)"
                else:
                    higher_group = "Equal"
                    direction = f"{group1_name} = {group2_name}"

            comparison_key = f"{group1_name}_vs_{group2_name}"
            question_results = {
                'question': question,
                'group1': group1_name,
                'group2': group2_name,
                'n1': n1,
                'n2': n2,
                'median1': median1,
                'iqr1': (q1_1, q3_1),
                'mean1': mean1,
                'sd1': sd1,
                'median2': median2,
                'iqr2': (q1_2, q3_2),
                'mean2': mean2,
                'sd2': sd2,
                'mann_whitney_u': stat,
                'z': z_score,
                'p_value': p_value,
                'effect_size_r': effect_size_r,
                'cliffs_delta': cliffs_delta,
                'A12': a12,
                'higher_group': higher_group,
                'direction': direction,
                'significant': p_value < 0.05
            }

            # Add to detailed results for CSV export
            detailed_results.append(question_results)

            mann_whitney_results[question] = question_results


    # Save detailed results as CSV
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        csv_path = os.path.join(data_dir, f"likert_analysis_{group1_name}_vs_{group2_name}.csv")
        detailed_df.to_csv(csv_path, index=False)

        # Also save significant results separately
        significant_results = [r for r in detailed_results if r.get('significant', False)]
        if significant_results:
            significant_df = pd.DataFrame(significant_results)
            significant_csv_path = os.path.join(data_dir, f"likert_significant_{group1_name}_vs_{group2_name}.csv")
            significant_df.to_csv(significant_csv_path, index=False)

    return mann_whitney_results


def main(data_frames, conditions, output_dir="analysis_plots", data_dir="data"):
    """Main likert scale analysis function."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    if len(data_frames) < 2:
        logger.warning("Need at least two conditions to compare.")
        return {}

    # Assuming the comparison is between the first two conditions
    condition1, condition2 = conditions[0], conditions[1]
    df1 = data_frames.get(condition1)
    df2 = data_frames.get(condition2)

    if df1 is None or df2 is None:
        logger.warning("Could not load data for one or both conditions. Exiting.")
        return {}

    results = detailed_self_assessment_analysis(df1, df2, condition1, condition2, output_dir, data_dir)
    return results
