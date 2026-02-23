import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib
from pathlib import Path
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.power import TTestIndPower
import matplotlib
from likert_scale_analysis_refactored import _tie_correction, _mwu_z_from_u, _cliffs_delta_from_U, _vardelaney_A12_from_U

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

sns.set_theme(context='paper', style='darkgrid', palette='colorblind')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14


def get_groups_one_df(df, x_label, y_label, groups_to_compare):
    if groups_to_compare is None:
        group_labels = df[x_label].unique()
    else:
        group_labels = groups_to_compare
    group1 = df[df[x_label] == group_labels[0]][y_label]
    group2 = df[df[x_label] == group_labels[1]][y_label]
    print(f"Group 1: {group_labels[0]}: {len(group1)}")
    print(f"Group 2: {group_labels[1]}: {len(group2)}")
    return group1, group2, group_labels[0], group_labels[1]


def get_groups_two_dfs(df1, df2, x_label, y_label):
    group_label1 = df1[x_label].unique()[0]
    group_label2 = df2[x_label].unique()[0]

    group1 = df1[y_label]
    group2 = df2[y_label]

    print(f"Group 1: {group_label1}: {len(group1)}")
    print(f"Group 2: {group_label2}: {len(group2)}")

    return group1, group2, group_label1, group_label2


def perform_ks_test(df, x_label, y_label):
    group1, group2, _, _ = get_groups_one_df(df, x_label, y_label)
    # print name of the groups
    ident_test = stats.ks_2samp(group1, group2)
    print(f"Kolmogorov-Smirnov test: {ident_test}")
    return group1, group2, ident_test



# Welch t-test helper
from scipy import stats

def welch_t_report(x, y, alpha=0.05, n_boot_g=0, random_state=0):
    """Return a detailed Welch t-test report for two independent samples.
    Fields: n1, n2, mean1, mean2, sd1, sd2, mean_diff, se_diff, t, df, p_two_tailed,
            ci95_mean_diff (tuple), cohens_d, hedges_g, hedges_g_ci95 (tuple|None)
    """
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    y = pd.Series(y).dropna().to_numpy(dtype=float)
    n1, n2 = x.size, y.size
    m1, m2 = x.mean(), y.mean()
    s1, s2 = x.std(ddof=1), y.std(ddof=1)

    # Welch t
    se = np.sqrt(s1**2/n1 + s2**2/n2)
    t_stat = (m1 - m2) / se
    df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    p = 2 * stats.t.sf(np.abs(t_stat), df)
    tcrit = stats.t.ppf(1 - alpha/2, df)
    diff = m1 - m2
    ci = (diff - tcrit*se, diff + tcrit*se)

    # Effect sizes
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
    d = (diff) / sp if sp > 0 else np.nan
    J = 1 - 3/(4*(n1+n2) - 9) if (n1+n2) > 2 else 1.0
    g = d * J if np.isfinite(d) else np.nan

    g_ci = (None, None)
    if n_boot_g and n1 > 1 and n2 > 1 and np.isfinite(g):
        rng = np.random.default_rng(random_state)
        boots = []
        for _ in range(n_boot_g):
            xb = rng.choice(x, size=n1, replace=True)
            yb = rng.choice(y, size=n2, replace=True)
            m1b, m2b = xb.mean(), yb.mean()
            s1b, s2b = xb.std(ddof=1), yb.std(ddof=1)
            spb = np.sqrt(((n1-1)*s1b**2 + (n2-1)*s2b**2) / (n1+n2-2))
            db = (m1b - m2b) / spb if spb > 0 else np.nan
            if np.isfinite(db):
                boots.append(db * J)
        if boots:
            g_ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    return {
        "n1": int(n1), "n2": int(n2), "mean1": float(m1), "mean2": float(m2),
        "sd1": float(s1), "sd2": float(s2), "mean_diff": float(diff), "se_diff": float(se),
        "t": float(t_stat), "df": float(df), "p_two_tailed": float(p),
        "ci95_mean_diff": (float(ci[0]), float(ci[1])),
        "cohens_d": float(d) if np.isfinite(d) else np.nan,
        "hedges_g": float(g) if np.isfinite(g) else np.nan,
        "hedges_g_ci95": g_ci
    }

def perform_t_test(group1, group2, alpha=0.05):
    """Welch's t-test wrapper that also returns a rich report dict."""
    report = welch_t_report(group1, group2, alpha=alpha)
    return report["t"], report["p_two_tailed"], report


def plot_qq_plots(df, x_label, y_label):
    plt.figure(figsize=(10, 5))
    for i, group in enumerate(df[x_label].unique()):
        plt.subplot(1, 2, i + 1)
        stats.probplot(df[df[x_label] == group][y_label], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot for {group}")


def perform_shapiro_wilk_test(df, x_label, y_label):
    """
    Perform Shapiro-Wilk test for normality
    """
    for group in df[x_label].unique():
        stat, p_value = stats.shapiro(df[df[x_label] == group][y_label])
        print(f"Shapiro-Wilk test for {group}: Statistic={stat}, P-value={p_value}")


def calculate_effect_size(group1, group2):
    """
    Calculate effect size using Cohen's d
    """
    sd1, sd2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
    effect_size = (np.mean(group1) - np.mean(group2)) / pooled_sd
    return effect_size, pooled_sd


def perform_power_analysis(effect_size=0.5):
    """
    Perform power analysis for t-test. Calculate sample size needed for 0.8 power.
    """
    alpha = 0.05
    for power in [0.8, 0.9, 0.95]:
        power_analysis = TTestIndPower()
        sample_size = power_analysis.solve_power(effect_size=effect_size,
                                                 alpha=alpha,
                                                 power=power,
                                                 ratio=1.0,
                                                 alternative='larger')
        print(f"Sample size for {power} power: {sample_size}")


def check_test_applicability(df, x_label, y_label, df2=None):
    """
    Check which statistical tests are applicable based on data characteristics:
    - T-test (parametric)
    - Mann-Whitney U (non-parametric)
    - Welch's t-test (unequal variances)
    - Bootstrap test (robust alternative)
    - Permutation test (distribution-free)
    """
    if df2 is None:
        group1, group2, _, _ = get_groups_one_df(df, x_label, y_label)
    else:
        group1, group2, _, _ = get_groups_two_dfs(df, df2, x_label, y_label)

    # Remove NaN values
    group1 = group1.dropna()
    group2 = group2.dropna()

    results = {
        'sample_sizes': (len(group1), len(group2)),
        'tests_applicable': {}
    }

    # Check if we have enough data for any test
    if len(group1) < 3 or len(group2) < 3:
        print("Insufficient data for statistical tests")
        return results

    # 1. Check for normality in both groups
    normality_group1 = normality_group2 = True
    for i, group in enumerate([group1, group2], 1):
        if len(group) < 50:
            stat, p_value = stats.shapiro(group)
            test_name = "Shapiro-Wilk"
        else:
            stat, p_value = stats.normaltest(group)
            test_name = "D'Agostino"

        if p_value < 0.05:
            print(f"{test_name} normality test failed for group {i}: p-value={p_value:.4f}")
            if i == 1:
                normality_group1 = False
            else:
                normality_group2 = False

    both_normal = normality_group1 and normality_group2

    # 2. Check for homogeneity of variance
    levene_stat, levene_p = stats.levene(group1, group2)
    equal_variances = levene_p >= 0.05
    if not equal_variances:
        print(f"Levene test for equal variances failed: p-value={levene_p:.4f}")

    # 3. Check for outliers (using IQR method)
    def has_outliers(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((data < lower_bound) | (data > upper_bound)).any()

    outliers_present = has_outliers(group1) or has_outliers(group2)

    # 4. Check sample size adequacy
    min_sample_size = min(len(group1), len(group2))
    large_samples = min_sample_size >= 30

    # Determine applicable tests

    # Independent samples t-test (Student's)
    results['tests_applicable']['independent_t_test'] = {
        'applicable': both_normal and equal_variances and not outliers_present,
        'reasons': []
    }
    if not both_normal:
        results['tests_applicable']['independent_t_test']['reasons'].append('Non-normal distribution(s)')
    if not equal_variances:
        results['tests_applicable']['independent_t_test']['reasons'].append('Unequal variances')
    if outliers_present:
        results['tests_applicable']['independent_t_test']['reasons'].append('Outliers detected')

    # Welch's t-test (unequal variances t-test)
    results['tests_applicable']['welch_t_test'] = {
        'applicable': both_normal and not outliers_present,
        'reasons': []
    }
    if not both_normal:
        results['tests_applicable']['welch_t_test']['reasons'].append('Non-normal distribution(s)')
    if outliers_present:
        results['tests_applicable']['welch_t_test']['reasons'].append('Outliers detected')

    # Mann-Whitney U test (non-parametric)
    results['tests_applicable']['mann_whitney_u'] = {
        'applicable': True,  # Always applicable, but assumptions for interpretation
        'reasons': ['Always applicable - non-parametric test']
    }

    # Bootstrap test
    results['tests_applicable']['bootstrap'] = {
        'applicable': min_sample_size >= 10,  # Need reasonable sample for resampling
        'reasons': ['Robust to distribution assumptions'] if min_sample_size >= 10 else ['Small sample size']
    }

    # Permutation test
    results['tests_applicable']['permutation'] = {
        'applicable': min_sample_size >= 5,  # Very flexible
        'reasons': ['Distribution-free test'] if min_sample_size >= 5 else ['Very small sample size']
    }

    # Robust t-test (e.g., trimmed means)
    results['tests_applicable']['robust_t_test'] = {
        'applicable': min_sample_size >= 10,
        'reasons': ['Robust to outliers and mild non-normality'] if min_sample_size >= 10 else ['Small sample size']
    }

    # Print recommendations
    print("\n=== TEST APPLICABILITY SUMMARY ===")
    print(f"Sample sizes: Group 1 = {len(group1)}, Group 2 = {len(group2)}")
    print(f"Both groups normal: {both_normal}")
    print(f"Equal variances: {equal_variances}")
    print(f"Outliers present: {outliers_present}")
    print(f"Large samples (n≥30): {large_samples}")

    # Print detailed normality test results
    print(f"\nNormality test details:")
    print(f"  Group 1 normal: {normality_group1}")
    print(f"  Group 2 normal: {normality_group2}")
    if levene_p < 0.05:
        print(f"  Levene's test p-value: {levene_p:.4f} (FAILED - unequal variances)")
    else:
        print(f"  Levene's test p-value: {levene_p:.4f} (PASSED - equal variances)")

    print("\nRecommended tests:")
    for test_name, test_info in results['tests_applicable'].items():
        if test_info['applicable']:
            print(f"✓ {test_name.replace('_', ' ').title()}")
        else:
            print(f"✗ {test_name.replace('_', ' ').title()}: {', '.join(test_info['reasons'])}")

    # Add specific warning for t-test
    if results['tests_applicable']['independent_t_test']['applicable']:
        print(f"\n⚠️  WARNING: T-test appears applicable, but consider:")
        print(f"   - Sample sizes: {len(group1)} vs {len(group2)}")
        print(f"   - For Likert scale data, Mann-Whitney U is often more appropriate")
        print(f"   - T-test assumes interval/ratio level data")

    return results


def is_t_test_applicable(df, x_label, y_label, df2=None):
    """
    Legacy function - check if the standard t-test assumptions are met
    """
    results = check_test_applicability(df, x_label, y_label, df2)
    return results['tests_applicable']['independent_t_test']['applicable']


def print_correlation_ranking(df, target_var, group=None, keep_cols=None):
    # Make correlation df for user_df with each column against score_improvement
    if group is not None:
        correlation_df = df[df["study_group"] == group]
    else:
        correlation_df = df
    if keep_cols is None:
        keep_cols = df.columns
        # Remove id column and study_group
        keep_cols = keep_cols[~keep_cols.isin(["id", "study_group"])]
    correlation_df = correlation_df[keep_cols]

    correlation_df = correlation_df.select_dtypes(include=[np.number])
    correlation_df = correlation_df.corr()
    correlation_df = correlation_df[target_var].reset_index()
    correlation_df.columns = ["column", "correlation"]
    correlation_df = correlation_df.sort_values("correlation", ascending=False)
    if group is not None:
        print(f"Correlation ranking for {group}", target_var)
    print(correlation_df)


def one_way_anova_with_checks(df):
    # Check for NaN or infinite values
    if df.isna().any().any() or not np.isfinite(df).all().all():
        # Fill NaN with column mean
        df = df.fillna(df.mean())

    # Check variance of each group
    if any(df.var() == 0):
        raise ValueError("One or more groups have zero variance. Tukey's HSD cannot be performed.")

    # Check the preconditions
    print("Normality check using Shapiro-Wilk test:")
    for column in df.columns:
        stat, p_value = stats.shapiro(df[column])
        print(f"{column}: p-value = {p_value:.4f} ( {'Pass' if p_value > 0.05 else 'Fail'} )")

    levene_stat, levene_p_value = stats.levene(*[df[column] for column in df.columns])
    print(
        f"\nLevene's test for homogeneity of variances:\nLevene's statistic = {levene_stat:.4f}, p-value = {levene_p_value:.4f} ( {'Pass' if levene_p_value > 0.05 else 'Fail'} )")

    melted_df = df.melt(var_name='Group', value_name='Score')
    model = ols('Score ~ C(Group)', data=melted_df).fit()
    anova_table = anova_lm(model, typ=2)

    print("\nOne-way ANOVA results:")
    print(anova_table)

    # Check if ANOVA result is significant
    if anova_table['PR(>F)'][0] < 0.05:
        print("\nSignificant differences found, performing Tukey's HSD post-hoc test:")
        tukey = pairwise_tukeyhsd(endog=melted_df['Score'], groups=melted_df['Group'], alpha=0.05)
        print(tukey)
        return 1
    else:
        print("\nNo significant differences found.")
        return 0


def get_max_y_from_seaborn_boxplot(data, y_label):
    # Calculate the IQR
    Q1 = data[y_label].quantile(0.25)
    Q3 = data[y_label].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the maximum y position in the boxplot
    max_y = Q3 + 1.5 * IQR

    return max_y


def plot_box_with_significance_bars(df,
                                    x_label,
                                    y_label,
                                    title,
                                    ttest=False,
                                    save=True,
                                    ax=None,
                                    y_label_name=None,
                                    df2=None,
                                    groups_to_compare=None):
    """
    Plot boxplot with significance bars based on the result of a statistical test.
    tttest: If True, perform t-test. Otherwise, perform Mann-Whitney U test.
    """

    def perform_mann_whitney_u_test(group1, group2):
        U1, p = mannwhitneyu(group1, group2, method="asymptotic")
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        u_stat = min(U1, n1 * n2 - U1)
        effect_size_r = 1 - (2 * u_stat) / (n1 * n2)
        return U1, p, effect_size_r

    if df2 is None:
        group1, group2, group_1_name, group_2_name = get_groups_one_df(df, x_label, y_label, groups_to_compare)
    else:
        group1, group2, group_1_name, group_2_name = get_groups_two_dfs(df, df2, x_label, y_label)
        # Combine the two dataframes
        df = pd.concat([df, df2], ignore_index=True)

    # Perform statistical tests for each pair of groups
    if ttest:
        stat12, p_value12, rep = perform_t_test(group1, group2)
        effect_size_r = None  # rank-biserial only for MWU
        ci_low, ci_high = rep["ci95_mean_diff"]
        print(
            f"Welch t-test for {y_label}: t(df={rep['df']:.2f})={stat12:.3f}, p={p_value12:.4f}, "
            f"ΔM={rep['mean_diff']:.3f} [95% CI {ci_low:.3f}, {ci_high:.3f}], g={rep['hedges_g']:.3f}"
        )
        # Optionally, adjust the title or store rep if needed
    else:
        # Clean inputs before Mann-Whitney U test
        group1_clean = pd.to_numeric(group1, errors='coerce').dropna()
        group2_clean = pd.to_numeric(group2, errors='coerce').dropna()
        
        if len(group1_clean) == 0 or len(group2_clean) == 0:
            print(f"No valid data for Mann-Whitney U test on {y_label}")
            return
        
        # Perform Mann-Whitney U test with enhanced reporting
        stat12, p_value12, _ = perform_mann_whitney_u_test(group1_clean, group2_clean)
        n1, n2 = len(group1_clean), len(group2_clean)
        
        # Calculate enhanced statistics
        combined_vals = np.concatenate([group1_clean, group2_clean])
        z_score = _mwu_z_from_u(stat12, n1, n2, combined_vals, continuity=True)
        cliffs_delta = _cliffs_delta_from_U(stat12, n1, n2)  
        a12 = _vardelaney_A12_from_U(stat12, n1, n2)
        effect_size_r = cliffs_delta  # Use Cliff's delta as the effect size
        
        print(f"Mann-Whitney U test for {y_label}: U={stat12:.0f}, z={z_score:.3f}, p={p_value12:.4f}, δ={cliffs_delta:.3f}, A12={a12:.3f}")
        # title = "U-Test: " + title

    """if df2 is None:
        # Plot boxplot only for group2
        df = df[df[x_label] == group_2_name]"""

    # Set the categories for the boxplot
    categories = [group_1_name, group_2_name]
    # Convert categories to string if not already
    categories = [str(cat) for cat in categories]

    colors = {
        'static': 'darkgrey',
        'interactive': 'skyblue',
        'chat': 'darkgrey',
        'active_chat': 'darkgrey',
        'conversational': 'lightblue',
        'mapek': 'coral'
    }

    # Begin plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    sns.boxplot(x=x_label, y=y_label, data=df, order=categories, width=0.3, palette=colors, ax=ax,
                showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(y_label_name)
    ax.set_xlabel("")

    # Get the positions of the boxplots
    box_plot_positions = [i for i in range(len(categories))]

    # Determine the highest point on the y-axis for plotting the significance bar
    y_max = get_max_y_from_seaborn_boxplot(df, y_label) + 1
    print(f"y_max: {y_max}")
    h = 0.5  # Increase the height of the significance bar for more space between bars
    col = 'k'  # Color of the significance bar and text
    lw = 1.5  # Increase the linewidth for bigger beginning and end bars

    # Create a list of tuples containing the statistics, p-values, and effect sizes
    stats_and_p_values = [(stat12, p_value12, effect_size_r)]

    for i, (stat, p_value, r_effect) in enumerate(stats_and_p_values):
        if p_value is not None and p_value < 0.05:
            y = y_max + i * h  # Adjust the height based on the loop index
            ax.plot([box_plot_positions[0], box_plot_positions[1]], [y, y], color=col, lw=lw)
            ax.plot([box_plot_positions[0], box_plot_positions[0]], [y - h / 2, y + h / 2], color=col, lw=lw)
            ax.plot([box_plot_positions[1], box_plot_positions[1]], [y - h / 2, y + h / 2], color=col, lw=lw)

            # Create significance text with effect size
            sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'

            # Add effect size for Mann-Whitney U tests or t-test
            if not ttest and r_effect is not None:
                display_text = f'{sig_text}\nδ = {r_effect:.3f}'
            elif ttest and 'rep' in locals():
                g = rep.get('hedges_g', np.nan)
                ci_low, ci_high = rep.get('ci95_mean_diff', (np.nan, np.nan))
                display_text = f"{sig_text}\ng = {g:.2f}\nΔM [{ci_low:.2f},{ci_high:.2f}]"
            else:
                display_text = sig_text

            ax.text((box_plot_positions[0] + box_plot_positions[1]) / 2, y + h / 5, display_text,
                   ha='center', va='bottom', color=col, fontsize=10)

    title = title + "_" + group_1_name + " vs " + group_2_name
    # Set the x-axis tick labels if necessary
    plt.xticks(range(0, len(categories)), categories)
    path = "analysis_plots/" + f"{title}.tex"
    plt.tight_layout()
    csv_path = "analysis_plots/" + f"{title}.csv"
    if save:
        Path("analysis_plots").mkdir(parents=True, exist_ok=True)
        df_to_export = df[[x_label, y_label]].copy()
        df_to_export.to_csv(csv_path, index=False)
        if not ax:
            tikzplotlib.save(path)
