"""
Correlation Analysis - Refactored
Accepts a DataFrame and performs correlation analysis.
"""

import ast
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging


class CorrelationAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def expand_subjective_answers(self, df, col_name="subjective_understanding_answer"):
        """Expand subjective answers into separate columns."""
        if col_name not in df.columns:
            return df
            
        df[col_name] = df[col_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        max_len = df[col_name].apply(lambda x: len(x) if isinstance(x, list) else 0).max()

        for i in range(max_len):
            df[f"self_assessment_answer_{i}"] = df[col_name].apply(
                lambda answers: answers[i] if isinstance(answers, list) and len(answers) > i else None
            )
        return df

    def run_correlation_analysis(self, df, condition_name, target_column, output_dir=None):
        """
        Runs correlation analysis for a target column against all other numeric columns.
        Saves the plot if output_dir is provided.
        """
        if df is None or df.empty:
            self.logger.warning(f"No data provided for condition {condition_name}. Skipping correlation analysis.")
            return

        if "subjective_understanding_answer" in df.columns:
            df = self.expand_subjective_answers(df)

        # Ensure the target column exists and is numeric
        if target_column not in df.columns or not pd.api.types.is_numeric_dtype(df[target_column]):
            self.logger.warning(f"Target column '{target_column}' not found or not numeric in {condition_name} data. Skipping.")
            return

        # Find all other numeric columns to correlate against
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != target_column]
        if not numeric_cols:
            self.logger.warning(f"No other numeric columns to correlate with in {condition_name} data. Skipping.")
            return

        # Calculate correlations with the target column
        correlations = df[numeric_cols].corrwith(df[target_column]).dropna()

        if correlations.empty:
            self.logger.warning(f"No valid correlations found for {target_column} in {condition_name} data.")
            return

        # Sort the correlations for better visualization
        correlations = correlations.sort_values(ascending=False)

        plt.figure(figsize=(12, 10))  # Increased figure height to make space for title and labels
        sns.barplot(x=correlations.values, y=correlations.index, palette="coolwarm")
        
        # Add vertical lines for correlation strength thresholds
        plt.axvline(x=0.5, color='r', linestyle='--', linewidth=1)
        plt.axvline(x=-0.5, color='r', linestyle='--', linewidth=1)
        plt.axvline(x=0.3, color='orange', linestyle='--', linewidth=1)
        plt.axvline(x=-0.3, color='orange', linestyle='--', linewidth=1)

        # Add text to indicate strength levels
        plt.text(0.51, plt.gca().get_ylim()[0], 'Strong', color='r', va='bottom', ha='left')
        plt.text(-0.51, plt.gca().get_ylim()[0], 'Strong', color='r', va='bottom', ha='right')
        plt.text(0.31, plt.gca().get_ylim()[0], 'Moderate', color='orange', va='bottom', ha='left')
        plt.text(-0.31, plt.gca().get_ylim()[0], 'Moderate', color='orange', va='bottom', ha='right')
        plt.text(0.1, plt.gca().get_ylim()[0], 'Weak', color='green', va='bottom', ha='left')
        plt.text(-0.1, plt.gca().get_ylim()[0], 'Weak', color='green', va='bottom', ha='right')
        
        plt.title(f"Correlation with {target_column} for {condition_name}")
        plt.xlabel("Pearson Correlation Coefficient")
        plt.ylabel("Features")
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        if output_dir:
            output_path = os.path.join(output_dir, f"correlations_with_{target_column}_{condition_name}.png")
            plt.savefig(output_path)
            self.logger.info(f"Correlation plot saved to {output_path}")
        else:
            plt.show()

        plt.close()
