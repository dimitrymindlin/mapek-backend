"""
Exit Questionnfrom dialogue_analysis_base import (
    UnifiedDataLoader, 
    QuestionnaireAnalyzer, 
    SharedStatistics
)nalysis - Refactored Version
===============================================

Analyzes exit questionnaire Likert scale data with statistical comparisons.
Refactored to use dialogue_analysis_base module for DRY principles.

Key Features:
- Unified data loading with automatic format detection
- Shared statistical testing (Mann-Whitney U with effect size)
- Consistent questionnaire data handling
- Automated subscale analysis by question prefix
- Comprehensive reporting with significance testing

This script replaces the original analyse_exit_questionnaire.py with 
cleaner, more maintainable code that follows DRY principles.
"""

from dialogue_analysis_base import (
    UnifiedDataLoader,
    QuestionnaireAnalyzer,
    SharedStatistics
)
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def analyze_exit_questionnaire_data(data_files, conditions, save_merged_files=True):
    """
    Analyze exit questionnaire data with unified approach
    
    Args:
        data_files: List of file paths for each condition
        conditions: List of condition names corresponding to files
        save_merged_files: Whether to save merged dataframes with exit questionnaire columns
    
    Returns:
        dict: Analysis results including statistical comparisons
    """
    logger.info("EXIT QUESTIONNAIRE ANALYSIS")
    logger.info("=" * 60)
    
    # Initialize unified data loader and questionnaire analyzer
    loader = UnifiedDataLoader()
    analyzer = QuestionnaireAnalyzer()
    
    # Configure for exit questionnaire
    prefix_groups = analyzer.configure_exit_questionnaire()
    
    results = {
        'dataframes': {},
        'exit_questionnaire_data': {},
        'merged_data': {},
        'statistical_results': [],
        'prefix_groups': prefix_groups
    }
    
    # Load and process data for each condition
    for data_file, condition in zip(data_files, conditions):
        logger.info(f"\nProcessing {condition} condition from {data_file}")
        
        # Load data with automatic format detection
        df = loader.load_any_dialogue_data(data_file)
        results['dataframes'][condition] = df
        
        # Extract exit questionnaire data
        exit_df = analyzer.extract_exit_questionnaire_data(df)
        results['exit_questionnaire_data'][condition] = exit_df
        
        # Merge exit questionnaire data back into original dataframe
        merged_df = analyzer.merge_exit_questionnaire_data(df, exit_df)
        results['merged_data'][condition] = merged_df
        
        # Save merged dataframe if requested
        if save_merged_files:
            output_filename = f"data_adult_mapek_{condition}_01_2025_with_irt_exit.csv"
            merged_df.to_csv(output_filename, index=False)
            logger.info(f"  Saved merged data to: {output_filename}")
        
        logger.info(f"  Original shape: {df.shape}")
        logger.info(f"  With exit questionnaire: {merged_df.shape}")
        logger.info(f"  Exit questionnaire columns: {len(exit_df.columns) - 1}")  # -1 for user_id
    
    # Prepare data for statistical analysis
    logger.info(f"\nPreparing data for statistical analysis...")
    
    # Transform to wide format for statistical testing
    wide_dfs = []
    for data_file, condition in zip(data_files, conditions):
        df = results['dataframes'][condition]
        wide_df = analyzer.transform_exit_questionnaire_to_wide(df, condition)
        wide_dfs.append(wide_df)
    
    # Combine into single dataframe for analysis
    combined_df = pd.concat(wide_dfs, ignore_index=True)
    
    # Perform statistical analysis by prefix groups
    logger.info(f"\nPerforming statistical analysis by prefix groups...")
    statistical_results = analyzer.analyze_exit_questionnaire_by_prefix(
        combined_df, conditions, alpha=0.05
    )
    
    results['statistical_results'] = statistical_results
    results['combined_dataframe'] = combined_df
    
    return results


def print_statistical_results(results):
    """Print formatted statistical results"""
    logger.info("\nSTATISTICAL TEST RESULTS PER PREFIX")
    logger.info("=" * 60)
    
    statistical_results = results['statistical_results']
    
    # Sort results by significance first, then by p-value
    results_sorted = sorted(
        statistical_results, 
        key=lambda r: (r['significance'] != "Significant", r['p'])
    )
    
    # Create and display results table
    if results_sorted:
        results_df = pd.DataFrame(results_sorted)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        logger.info(results_df.to_string())
        
        # Summary statistics
        significant_count = sum(1 for r in results_sorted if r['significance'] == "Significant")
        total_count = len(results_sorted)
        
        logger.info(f"\nSUMMARY:")
        logger.info(f"  Total comparisons: {total_count}")
        logger.info(f"  Significant results: {significant_count}")
        logger.info(f"  Non-significant results: {total_count - significant_count}")
        
        if significant_count > 0:
            logger.info(f"\nSIGNIFICANT FINDINGS:")
            for result in results_sorted:
                if result['significance'] == "Significant":
                    logger.info(f"  {result['prefix']}: {result['higher_group']} > other "
                          f"(p = {result['p']}, effect size = {result['effect_size']})")
    else:
        logger.info("No statistical results to display.")


def generate_analysis_report(results, output_file="exit_questionnaire_analysis_report.md"):
    """Generate comprehensive analysis report in markdown format"""
    logger.info(f"\nGenerating analysis report...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Exit Questionnaire Analysis Report\n\n")
        f.write("## Overview\n\n")
        f.write("This report presents the results of exit questionnaire analysis comparing experimental conditions.\n\n")
        
        # Data summary
        f.write("## Data Summary\n\n")
        for condition, df in results['dataframes'].items():
            f.write(f"- **{condition.title()}**: {len(df)} participants\n")
        
        # Questionnaire structure
        f.write("\n## Questionnaire Structure\n\n")
        prefix_groups = results['prefix_groups']
        for prefix, codes in prefix_groups.items():
            f.write(f"- **{prefix}**: {len(codes)} questions ({', '.join(codes)})\n")
        
        # Statistical results
        f.write("\n## Statistical Results\n\n")
        statistical_results = results['statistical_results']
        
        if statistical_results:
            # Significant results
            significant_results = [r for r in statistical_results if r['significance'] == "Significant"]
            
            if significant_results:
                f.write("### Significant Differences\n\n")
                for result in significant_results:
                    f.write(f"**{result['prefix']}**\n")
                    f.write(f"- Higher performing condition: {result['higher_group']}\n")
                    f.write(f"- p-value: {result['p']}\n")
                    f.write(f"- Effect size: {result['effect_size']}\n")
                    f.write(f"- Test statistic: {result['test_statistic']}\n\n")
            else:
                f.write("### No Significant Differences Found\n\n")
                f.write("No statistically significant differences were found between conditions.\n\n")
            
            # All results table
            f.write("### Complete Results Table\n\n")
            results_df = pd.DataFrame(statistical_results)
            f.write(results_df.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("No statistical analyses were performed.\n\n")
        
        # Methodology
        f.write("## Methodology\n\n")
        f.write("- **Statistical Test**: Mann-Whitney U test (non-parametric)\n")
        f.write("- **Effect Size**: Rank-biserial correlation\n")
        f.write("- **Significance Level**: α = 0.05\n")
        f.write("- **Multiple Comparisons**: No correction applied\n")
        f.write("- **Score Reversal**: Applied to negatively worded items (AI3, AC4)\n\n")
    
    logger.info(f"Analysis report saved to: {output_file}")


def main(data_files=None, conditions=None, output_dir="analysis_plots"):
    """
    Main analysis function
    
    Args:
        data_files: List of data file paths to analyze
        conditions: List of condition names corresponding to the data files
        output_dir: Directory to save output files
    """
    import os
    
    # Default configuration for backward compatibility
    if data_files is None:
        data_files = [
            "data_adult_mapek_interactive_01_2025_with_irt.csv",
            "data_adult_mapek_chat_01_2025_with_irt.csv"
        ]
    
    if conditions is None:
        conditions = ["interactive", "chat"]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis
    try:
        results = analyze_exit_questionnaire_data(
            data_files=data_files,
            conditions=conditions,
            save_merged_files=True
        )
        
        # Display results
        print_statistical_results(results)
        
        # Generate report
        generate_analysis_report(results)
        
        logger.info(f"\n✅ Exit questionnaire analysis completed successfully!")
        logger.info(f"📊 Analyzed {len(conditions)} conditions")
        logger.info(f"📈 Found {len(results['statistical_results'])} statistical comparisons")
        
        return results
        
    except FileNotFoundError as e:
        logger.error(f"❌ Error: Data file not found - {e}")
        logger.error("Please ensure the following files exist:")
        for file in data_files:
            logger.error(f"  - {file}")
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        raise


def analyze_custom_questionnaire_data(data_files, conditions, question_mappings, 
                                     reversed_questions=None, save_merged_files=True):
    """
    Analyze custom questionnaire data with user-defined mappings
    
    Args:
        data_files: List of file paths for each condition
        conditions: List of condition names
        question_mappings: Dict mapping question codes to question text
        reversed_questions: List of question codes that need score reversal
        save_merged_files: Whether to save merged files
    
    Returns:
        dict: Analysis results
    """
    logger.info("CUSTOM QUESTIONNAIRE ANALYSIS")
    logger.info("=" * 60)
    
    # Initialize components
    loader = UnifiedDataLoader()
    analyzer = QuestionnaireAnalyzer()
    
    # Set custom questionnaire mapping
    analyzer.set_questionnaire_mapping(question_mappings, reversed_questions)
    
    # Continue with standard analysis...
    # (Implementation similar to analyze_exit_questionnaire_data)
    
    return {"message": "Custom questionnaire analysis - implementation in progress"}


if __name__ == "__main__":
    main()
