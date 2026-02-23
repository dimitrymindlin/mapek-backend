import ast
import json
from collections import defaultdict
import logging

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(context='paper', style='darkgrid', palette='colorblind')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

logger = logging.getLogger(__name__)

# Replace question IDs with question text
question_text = {23: "Most Important Features",
                 27: "Least Important Features",
                 24: "Feature Attributions",
                 7: "Counterfactuals",
                 11: "Anchors",
                 25: "Ceteris Paribus",
                 13: "Feature Ranges"}

colors = {'static': 'darkgrey', 'interactive': 'skyblue'}


def plot_chatbot_feedback(user_df):
    # Convert strings to Python objects outside the loop
    feedback_questions = user_df['exit'].apply(ast.literal_eval)
    feedback_answers = user_df['exit_answers'].apply(ast.literal_eval)

    # Accumulate the data into a dictionary
    feedback_dict = defaultdict(list)

    for questions, answers in zip(feedback_questions, feedback_answers):
        for question, answer in zip(questions, answers):
            question = question.replace("’", "'").strip()  # Clean up the question text
            feedback_dict[question].append(answer)

    # Create a dataframe from the dictionary
    feedback_df = pd.DataFrame(feedback_dict)

    # Sort the ratings by average rating
    feedback_df = feedback_df[feedback_df.mean().sort_values().index]

    # Adjust figure size to increase height
    plt.figure(figsize=(10, 12))  # Increase the second value to heighten the plot

    sns.boxplot(data=feedback_df, orient='h')
    sns.stripplot(data=feedback_df, color=".25", orient='h')

    plt.title("Chatbot Feedback")
    plt.xlabel("Answers")
    plt.ylabel("Questions")

    # Use subplots_adjust to fine-tune the layout, especially the left margin
    plt.subplots_adjust(left=0.3, top=0.9)  # Adjust left as needed to fit the y-axis labels

    plt.show()


def plot_chatbot_feedback_for_2(user_df, user_df2, name1='UserDF1', name2='UserDF2'):
    # Function to extract questions
    def extract_questions(row):
        return row['exit']['questions']

    # Helper function to convert and accumulate data
    def accumulate_feedback(df):
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

    # Accumulate feedback for both dataframes
    feedback_dict1 = accumulate_feedback(user_df)
    feedback_dict2 = accumulate_feedback(user_df2)

    # Combine the feedback data into a single DataFrame
    combined_data = []

    for question in set(feedback_dict1.keys()).union(feedback_dict2.keys()):
        for answer in feedback_dict1.get(question, []):
            combined_data.append((question, answer, name1))
        for answer in feedback_dict2.get(question, []):
            combined_data.append((question, answer, name2))

    combined_df = pd.DataFrame(combined_data, columns=['Question', 'Answer', 'Dataset'])

    # Plotting the feedback
    plt.figure(figsize=(20, 12))  # Increase the size as needed

    sns.boxplot(x='Answer', y='Question', hue='Dataset', data=combined_df, orient='h')

    plt.title("Chatbot Feedback")
    plt.xlabel("Ratings")
    plt.ylabel("Questions")
    plt.legend(title='Dataset')

    plt.tight_layout()
    plt.show()


def handle_duplicates(predictions):
    # Create a dictionary to keep track of the count of each datapoint_count
    count_dict = {}
    duplicates_count = 0

    # Iterate over the sorted predictions
    for x in predictions:
        datapoint_count = x['datapoint_count']

        # If the datapoint_count is already in the dictionary, append the prediction
        if datapoint_count in count_dict:
            count_dict[datapoint_count].append(x)
            duplicates_count += 1
        else:
            # If the datapoint_count is not in the dictionary, add it with the current prediction
            count_dict[datapoint_count] = [x]

    # Iterate over the count_dict
    for datapoint_count, data in count_dict.items():
        # If the count is more than 1, check if the next datapoint_count is not the succeeding count
        if len(data) > 1:
            next_datapoint_count = datapoint_count + 1
            prev_datapoint_count = datapoint_count - 1
            if next_datapoint_count not in count_dict:
                # Increment the current datapoint_count and print the updated data
                data[0]['datapoint_count'] += 1
            elif prev_datapoint_count not in count_dict:
                # Decrement the current datapoint_count and print the updated data
                data[0]['datapoint_count'] -= 1

    # Filter out duplicates, keeping the first occurrence
    seen = set()
    sorted_predictions = [x for x in predictions if
                          x['datapoint_count'] not in seen and not seen.add(x['datapoint_count'])]
    logger.info(f"Found {duplicates_count} duplicates...")
    return sorted_predictions


def get_user_id_predictions_over_time_matrix(user_predictions_over_time_df):
    """
    Create a DataFrame with each row representing a user and columns 'datapoint1' to 'datapoint10'
    representing the first ten datapoints for each user, based on the 'datapoint_count' within the details.

    :param user_predictions_over_time_df: DataFrame with user predictions, where 'details' contains serialized dictionaries
    :return: DataFrame with user_id as index and columns for each of the first ten datapoints
    """
    # Create a new column 'accuracy' to reflect Correct/Wrong
    user_predictions_over_time_df['accuracy'] = user_predictions_over_time_df.apply(
        lambda row: 'Correct' if row['prediction'] == row['true_label'] else 'Wrong', axis=1)

    # Now pivot the table with 'accuracy' as the values
    result_df = user_predictions_over_time_df.pivot_table(
        index='user_id',
        columns='datapoint_count',
        values='accuracy',
        aggfunc=lambda x: x  # you can choose to list, or keep the first, or count the occurrences of 'Correct'/'Wrong'
    )

    # Flattening the multi-index in columns
    result_df.columns = [f'accuracy_{col}' for col in result_df.columns]
    result_df.reset_index(inplace=True)

    # Count rows that do not have 'Correct' or 'Wrong' in 'accuracy' or nan
    def is_single_correct_or_wrong(value):
        try:
            return value in ['Correct', 'Wrong']
        except ValueError:
            return False

    # Select only columns that start with "accuracy_"
    accuracy_columns = [col for col in result_df.columns if col.startswith('accuracy_')]

    # Apply the function to each cell in the selected columns
    mask = result_df[accuracy_columns].applymap(is_single_correct_or_wrong)

    # Get the user_ids where all values in a row are either 'Correct' or 'Wrong'
    remove_ids = result_df[~mask.all(axis=1)]['user_id']
    logger.info(f"Removed {len(remove_ids)} users with missing or invalid data.")
    remove_ids = remove_ids.tolist()
    # get df where user_id = 'e1420868-2337-4935-a730-c77f8275c845'
    dimi_df = result_df[result_df['user_id'].isin(["e1420868-2337-4935-a730-c77f8275c845"])]
    return result_df, remove_ids


def plot_user_predictions(user_accuracy_over_time_df, study_group_name, user_df):
    """
    Plot a matrix where each row represents a user and each column represents the correctness of one of the 5 predictions.
    Optionally includes an end score for each user as the last column.
    """
    # Merge "final score" and "initial score" from user_df to the user_accuracy_over_time_df
    user_accuracy_over_time_df = user_accuracy_over_time_df.merge(
        user_df[['id', 'final_score', 'intro_score', 'study_group']],
        left_on='user_id', right_on='id', how='left'
    )
    user_accuracy_over_time_df.drop(columns='id', inplace=True)
    # Filter by study group and remove col
    user_accuracy_over_time_df = user_accuracy_over_time_df[
        user_accuracy_over_time_df['study_group'] == study_group_name]
    user_accuracy_over_time_df.drop(columns='study_group', inplace=True)

    # Setup color map and apply to accuracy columns
    color_map = {'Correct': 'green', 'Wrong': 'red'}
    accuracy_cols = [col for col in user_accuracy_over_time_df.columns if 'accuracy' in col]
    for col in accuracy_cols:
        user_accuracy_over_time_df[col] = user_accuracy_over_time_df[col].map(color_map)

    # Plotting
    fig, axs = plt.subplots(nrows=len(user_accuracy_over_time_df),
                            figsize=(10, max(2 * len(user_accuracy_over_time_df), 10)), sharex=True)
    if len(user_accuracy_over_time_df) == 1:
        axs = [axs]

    for ax, (_, row) in zip(axs, user_accuracy_over_time_df.iterrows()):
        ax.bar(accuracy_cols, [1] * len(accuracy_cols), color=row[accuracy_cols])
        ax.set_title(f"User: {row['user_id']}")
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        # Plot scores on a secondary y-axis
        score_cols = ['intro_score', 'final_score']
        ax2 = ax.twinx()  # Create a secondary y-axis
        ax2.plot(score_cols, [row[col] for col in score_cols], 'ko-', label='Scores')
        ax2.legend(loc='upper right')

    plt.xticks(ticks=range(len(user_accuracy_over_time_df.columns[1:])), labels=user_accuracy_over_time_df.columns[1:],
               rotation=90)
    plt.tight_layout()
    plt.show()


def plot_understanding_over_time(user_predictions_over_time_list, analysis):
    """
    Plot the proportion of correct and incorrect predictions for each prediction order.
    """
    user_accuracy_over_time_df, exclude_user_ids = get_user_id_predictions_over_time_matrix(
        user_predictions_over_time_list)
    analysis.update_dfs(exclude_user_ids)
    logger.info("Users after removing missing or invalid data:")
    logger.info(analysis.user_df.groupby("study_group").size())
    user_accuracy_over_time_df = user_accuracy_over_time_df[
        ~user_accuracy_over_time_df['user_id'].isin(exclude_user_ids)]

    # Count 'Correct' predictions and save to 'accuracy_over_time'
    user_accuracy_over_time_df['accuracy_over_time'] = user_accuracy_over_time_df.filter(like='accuracy').apply(
        lambda row: sum(row == 'Correct'), axis=1)
    # CHeck for nan values in accuracy_over_time
    if user_accuracy_over_time_df['accuracy_over_time'].isna().sum() > 0:
        logger.warning("Nan values in accuracy_over_time:")
        logger.warning(user_accuracy_over_time_df['accuracy_over_time'].isna().sum())
    # Merge 'accuracy_over_time' with user_df
    analysis.user_df = analysis.user_df.merge(user_accuracy_over_time_df[['user_id', 'accuracy_over_time']],
                                              left_on='id', right_on='user_id', how='left')

    # Check nan in analysis.user_df accuracy_over_time
    if analysis.user_df['accuracy_over_time'].isna().sum() > 0:
        logger.warning("Nan values in accuracy_over_time in user_df:")
        logger.warning(analysis.user_df['accuracy_over_time'].isna().sum())

    """if plot:
        plot_user_predictions(user_accuracy_over_time_df, study_group_name, analysis.user_df)"""

    return user_accuracy_over_time_df


def extract_feedback_data(user_df):
    """Extracts feedback data from the user DataFrame in a more pythonic way."""
    if 'feedback' not in user_df.columns or 'id' not in user_df.columns:
        logger.warning("DataFrame is missing 'feedback' or 'id' column. Cannot extract feedback.")
        return []

    feedback_data = []
    # Define columns to iterate over for efficiency
    cols_to_iterate = ['id', 'feedback']
    if 'study_group' in user_df.columns:
        cols_to_iterate.append('study_group')

    for row in user_df[cols_to_iterate].itertuples(index=False):
        user_id = row.id
        feedback_list = row.feedback

        # 1. Safely evaluate the feedback string
        if isinstance(feedback_list, str):
            try:
                feedback_list = ast.literal_eval(feedback_list)
            except (ValueError, SyntaxError):
                logger.debug(f"Could not parse feedback string for user {user_id}.")
                continue

        # 2. Validate structure and extract JSON string
        if not (isinstance(feedback_list, list) and len(feedback_list) > 1 and isinstance(feedback_list[1], str)):
            continue

        # 3. Parse JSON and create feedback dictionary
        try:
            feedback_dict = json.loads(feedback_list[1])
            if not feedback_dict.get("feedback"):  # Skip if feedback is empty or key is missing
                continue

            # 4. Enhance with user and group info
            feedback_dict['user_id'] = user_id
            if 'study_group' in cols_to_iterate:
                feedback_dict['study_group'] = row.study_group
            
            feedback_data.append(feedback_dict)

        except (json.JSONDecodeError, IndexError):
            logger.warning(f"Could not decode or process feedback JSON for user {user_id}.")
            continue
            
    return feedback_data


def get_user_id_questions_over_time_df(user_questions_df):
    # Create new df with one row per user and datapoint_count as columns and question_ids as values
    summary_questions_df = user_questions_df.pivot_table(
        index='user_id',
        columns='datapoint_count',
        values='question_id',
        aggfunc=lambda x: list(x)
    )

    # Replace NaN values with empty lists
    for col in summary_questions_df.columns:
        summary_questions_df[col] = summary_questions_df[col].apply(
            lambda x: [] if isinstance(x, float) and pd.isna(x) else x)

    # Add column called "total_questions" to user_df
    summary_questions_df["total_questions"] = summary_questions_df.applymap(len).sum(axis=1)
    return summary_questions_df


def plot_user_questions(matrix, user_ids, study_group_name):
    """
    Plots a matrix where each cell contains the list of question IDs asked by each user at each datapoint.

    :param matrix: A nested list containing question IDs for each user at each datapoint.
    :param user_ids: List of user IDs corresponding to the rows in the matrix.
    :param study_group_name: Name of the study group for the plot title.
    """
    # Determine the matrix size
    num_rows = len(matrix)
    num_cols = max(len(row) for row in matrix)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(num_cols * 1.5, num_rows * 0.5))  # Adjust figure size as needed

    # Create an empty matrix for plotting
    plot_matrix = np.full((num_rows, num_cols), "", dtype=object)

    # Fill the plot matrix with question IDs (or a summary)
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            # Here, we join question IDs with a comma, or you can customize this part
            plot_matrix[i, j] = ", ".join(map(str, cell))

    # Use a table to display the matrix since it may contain text
    table = ax.table(cellText=plot_matrix, rowLabels=user_ids, colLabels=[f"DP {i + 1}" for i in range(num_cols)],
                     cellLoc='center', loc='center')

    # Adjust layout
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # Adjust font size as needed
    table.scale(1, 1.5)  # Adjust cell size as needed
    ax.axis('off')

    plt.title(f"{study_group_name} - Questions Asked Per User Per Datapoint")
    plt.show()


def plot_understanding_with_questions(matrix_understanding, matrix_questions, user_ids_u, user_ids_q):
    """
    Plots a matrix where the cell color represents the understanding score (0 or 1),
    and the cell text displays the questions asked.

    :param matrix_understanding: A nested list with understanding scores for each user at each datapoint.
    :param matrix_questions: A nested list with questions IDs for each user at each datapoint.
    :param user_ids_u: List of user IDs corresponding to the rows in the matrix_understanding.
    :param user_ids_q: List of user IDs corresponding to the rows in the matrix_questions.
    """
    assert user_ids_u == user_ids_q, "User IDs must match between understanding and questions matrices"

    num_rows = len(matrix_understanding)
    num_cols = max(max(len(row) for row in matrix_understanding), max(len(row) for row in matrix_questions))

    # Initialize a figure
    fig, ax = plt.subplots(figsize=(num_cols * 1.5, num_rows * 0.5))  # Adjust size as needed

    # Define colors for understanding scores
    colors = {0: "tomato", 1: "lightgreen"}  # Red for 0, Green for 1

    # Plot each cell
    for i, (scores_row, questions_row) in enumerate(zip(matrix_understanding, matrix_questions)):
        for j in range(num_cols):
            score = scores_row[j] if j < len(scores_row) else None  # Handle different row lengths
            questions = ", ".join(map(str, questions_row[j])) if j < len(questions_row) and j < len(scores_row) else ""

            # Set cell color based on understanding score
            cell_color = colors.get(score, "lightgrey")  # Default color for missing scores

            # Create a rectangle as the cell background
            rect = plt.Rectangle((j, num_rows - i - 1), 1, 1, color=cell_color)
            ax.add_patch(rect)

            # Annotate the cell with question IDs
            ax.text(j + 0.5, num_rows - i - 0.5, questions, ha='center', va='center',
                    fontsize=8)  # Adjust text alignment and size as needed

    # Set up the plot axes
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_xticks(np.arange(num_cols) + 0.5)
    ax.set_yticks(np.arange(num_rows) + 0.5)
    ax.set_xticklabels([f"DP {i + 1}" for i in range(num_cols)], rotation=45, ha='right')
    ax.set_yticklabels(reversed(user_ids_u))  # Reverse the order to match the top-to-bottom plotting
    ax.grid(False)  # Turn off the grid

    plt.title("Interactive - Understanding and Questions")
    plt.xlabel("Datapoint")
    plt.ylabel("User ID")

    # Rename last x tick label to "Objective Score"
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[-1] = "Objective Score"
    ax.set_xticklabels(labels)

    plt.tight_layout()  # Adjust layout to fit everything
    plt.show()


def plot_time_boxplots(df):
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    sns.boxplot(x="study_group", y="total_learning_time", data=df, ax=axs[0])
    sns.stripplot(x="study_group", y="total_learning_time", data=df, color=".25", ax=axs[0])
    axs[0].set_title('Boxplot of Time Spent in Learning Phase per Study Group')
    axs[0].set_ylabel('Total Learning Time (minutes)')

    sns.boxplot(x="study_group", y="exp_instruction_time", data=df, ax=axs[1])
    sns.stripplot(x="study_group", y="exp_instruction_time", data=df, color=".25", ax=axs[1])
    axs[1].set_title('Boxplot of Time Spent in Instruction per Study Group')
    axs[1].set_ylabel('Instruction Time (minutes)')

    sns.boxplot(x="study_group", y="total_exp_time", data=df, ax=axs[2])
    sns.stripplot(x="study_group", y="total_exp_time", data=df, color=".25", ax=axs[2])
    axs[2].set_title('Boxplot of Total Time Spent in Experiment per Study Group')
    axs[2].set_ylabel('Total Experiment Time (minutes)')

    plt.tight_layout()
    plt.show()


def plot_questions_tornado(questions_over_time_df,
                           best_ids,
                           wors_ids, save=False,
                           group1_name="highest",
                           group2_name="lowest"):
    assert len(best_ids) == len(wors_ids), "The number of best and worst users should be equal."

    # Filter the DataFrame to include only the best and worst users
    best_users = questions_over_time_df[questions_over_time_df['user_id'].isin(best_ids)]
    worst_users = questions_over_time_df[questions_over_time_df['user_id'].isin(wors_ids)]

    ## Plot tornado plot of questions for interactive group between best and worst users
    # Calculate the counts of questions for each user group
    question_counts_best = best_users.groupby("question_id").size().reset_index(name="count")
    question_counts_worst = worst_users.groupby("question_id").size().reset_index(name="count")
    question_counts_best["group"] = group1_name
    question_counts_worst["group"] = group2_name

    # Merge the counts of questions for the best and worst users
    merged_counts = pd.merge(question_counts_best, question_counts_worst, on='question_id',
                             suffixes=(f'_{group1_name}', f'_{group2_name}'), how='outer').fillna(0)

    # Create a new column 'total_count' as the sum of 'count_best' and 'count_worst'
    merged_counts['total_count'] = merged_counts[f'count_{group1_name}'] + merged_counts[f'count_{group2_name}']

    # Sort the DataFrame based on the 'total_count' in descending order
    merged_counts.sort_values('total_count', ascending=True, inplace=True)

    # replace question_id with question_text
    # merged_counts['question_id'] = merged_counts['question_id'].map(question_text)

    # Set up the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(5, 3))

    # Define the width of the bars
    bar_width = 0.2

    # Generate the positions of the bars
    indices = range(len(merged_counts))

    # Plotting the bars for best and worst counts in opposite directions
    ax.barh(indices, merged_counts[f'count_{group1_name}'], height=bar_width, color='steelblue', label=f'{group1_name}')
    ax.barh(indices, -merged_counts[f'count_{group2_name}'], height=bar_width, color='powderblue',
            label=f'{group2_name}')

    # Set the y-ticks to question_id
    ax.set_yticks(indices)
    ax.set_yticklabels(merged_counts['question_id'])

    # Adding labels and title
    ax.set_xlabel('Count of Question Selections')

    for index, value in enumerate(merged_counts[f'count_{group1_name}']):
        difference = value - merged_counts[f'count_{group2_name}'].iloc[index]
        if difference >= 0:
            plt.text(value, index - bar_width / 2 + 0.1, f"+{difference}", va='center', fontsize=10)
        else:
            plt.text(value, index - bar_width / 2 + 0.1, f"{difference}", va='center', fontsize=10)

    # Draw a vertical line at x=0 to separate the two sides of the plot
    plt.axvline(x=0, color='black', linewidth=0.8)

    # Make the negative numbers on the x axis positive
    ax.set_xticklabels([str(abs(int(x))) for x in ax.get_xticks()])

    ax.set_yticklabels(merged_counts['question_id'], fontsize=13)  # Increased font size
    ax.set_xlabel('Count of Question Selections', fontsize=13)  # Increased font size

    ax.legend()
    plt.tight_layout()
    path = f"CSV ready to analyse/questions_tornado_{group1_name}_vs_{group2_name}.pdf"
    if save:
        plt.savefig(path)
    else:
        plt.show()


def plot_questions_bar(users_dfs, labels, save=False, normalize=True):
    assert len(users_dfs) == len(labels) and len(
        users_dfs) <= 3, "The number of DataFrames and labels should be equal and up to 3."

    merged_counts = None

    # Calculate the counts of questions for each user group
    for i, (users_df, label) in enumerate(zip(users_dfs, labels)):
        # Ensure question_id is a string
        users_df['question_id'] = users_df['question_id'].astype(str)

        question_counts = users_df.groupby("question_id").size().reset_index(name=f"count_{label}")
        question_counts["group"] = label

        if normalize:
            # Calculate the total number of questions for normalization
            total_questions = question_counts[f"count_{label}"].sum()
            question_counts[f"count_{label}"] = question_counts[f"count_{label}"] / total_questions

        if merged_counts is None:
            merged_counts = question_counts
        else:
            # Ensure question_id is a string in merged_counts as well
            merged_counts['question_id'] = merged_counts['question_id'].astype(str)
            merged_counts = pd.merge(merged_counts, question_counts, on='question_id', how='outer').fillna(0)

    # Sort the DataFrame based on the total count of selections
    count_columns = [f"count_{label}" for label in labels]
    merged_counts['total_count'] = merged_counts[count_columns].sum(axis=1)
    merged_counts.sort_values('total_count', ascending=False, inplace=True)

    logger.info("Raw Counts Table:")
    logger.info(merged_counts)

    # Set up the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define the width of the bars
    bar_width = 0.25

    # Generate the positions of the bars
    indices = range(len(merged_counts))
    positions = [indices]
    for i in range(1, len(labels)):
        positions.append([x + bar_width * i for x in indices])

    # Plotting the bars for each group
    for pos, label in zip(positions, labels):
        ax.bar(pos, merged_counts[f"count_{label}"], width=bar_width, label=label)

    # Set the x-ticks to question_id
    ax.set_xticks([i + bar_width for i in indices])
    ax.set_xticklabels(merged_counts['question_id'], rotation=90)

    # Adding labels and title
    ax.set_xlabel('Question ID')
    ax.set_ylabel('Count of Question Selections')
    if normalize:
        ax.set_title("Normalized Bar Plot of Question Selections")
    else:
        ax.set_title('Bar Plot of Question Selections')

    # Adding value labels on the bars
    for pos, label in zip(positions, labels):
        for i, value in enumerate(merged_counts[f"count_{label}"]):
            ax.text(pos[i], value + 0.005, f"{value:.2f}", ha='center', va='bottom', fontsize=9)

    ax.legend()
    plt.tight_layout()
    path = "CSV ready to analyse/questions_bar_plot.pdf"
    if save:
        plt.savefig(path)
    else:
        plt.show()
