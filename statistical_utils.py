import numpy as np
from scipy import stats
import scikit_posthocs as sp
import pandas as pd # Added for DataFrame operations, especially in post-hoc tests
from typing import List, Dict, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)
ALPHA_LEVEL = 0.05 # Default significance level

# Helper function to prepare data for scikit-posthocs
def _prepare_melted_df_for_posthocs(groups_data: Dict[str, List[float]]) -> pd.DataFrame:
    """Converts a dictionary of group data into a melted DataFrame."""
    data_for_df = []
    for group_name, values in groups_data.items():
        if not values: # Handle empty list for a group
            logger.debug(f"Group '{group_name}' has no data. Will be excluded from melted DataFrame for post-hocs.")
            continue
        for value in values:
            data_for_df.append({'group': group_name, 'value': value})
    if not data_for_df: # If all groups were empty or groups_data was empty
        return pd.DataFrame() # Return empty DataFrame
    return pd.DataFrame(data_for_df)

# 2.a. Normality Test
def check_normality(data: List[float], alpha: float = ALPHA_LEVEL) -> bool:
    """
    Performs Shapiro-Wilk test for normality.
    Returns True if data is likely normal, False otherwise.
    Returns True if len(data) < 3, as Shapiro-Wilk requires at least 3 samples.
    """
    if not isinstance(data, list) or not all(isinstance(x, (int, float)) for x in data):
        logger.warning("Normality check: Input data must be a list of numbers.")
        return False # Or raise error, depending on desired strictness
    
    if len(data) < 3:
        logger.debug(f"Normality check: Data length {len(data)} < 3. Assuming normal by default for Shapiro-Wilk applicability.")
        return True 
    
    # Remove NaNs as stats.shapiro cannot handle them
    data_cleaned = [x for x in data if pd.notna(x)]
    if len(data_cleaned) < 3:
        logger.debug(f"Normality check: Data length after NaN removal {len(data_cleaned)} < 3. Assuming normal.")
        return True

    try:
        stat, p_value = stats.shapiro(np.array(data_cleaned))
        is_normal = p_value > alpha
        logger.debug(f"Shapiro-Wilk on data (len={len(data_cleaned)}): stat={stat:.3f}, p={p_value:.3f}. Normal: {is_normal} (alpha={alpha}).")
        return is_normal
    except Exception as e:
        logger.warning(f"Normality check: Shapiro-Wilk test failed: {e}. Returning False.")
        return False

# 2.b. Variance Homogeneity Test
def check_variance_homogeneity(samples: List[List[float]], alpha: float = ALPHA_LEVEL) -> bool:
    """
    Performs Levene's test for homogeneity of variances.
    Returns True if variances are likely equal, False otherwise.
    Returns True for edge cases (e.g., any sample < 2, or all values identical across all samples).
    """
    if not samples or not isinstance(samples, list) or not all(isinstance(s, list) for s in samples):
        logger.warning("Variance homogeneity: Input 'samples' must be a list of lists.")
        return True # Default to True if input format is wrong to avoid blocking downstream tests inappropriately

    # Filter out empty samples or samples with all NaNs, and clean NaNs within samples
    cleaned_samples = []
    for s_idx, s_original in enumerate(samples):
        s_cleaned = [x for x in s_original if pd.notna(x)] if isinstance(s_original, list) else []
        if s_cleaned: # Only add if there's actual data after cleaning
            cleaned_samples.append(s_cleaned)
        else:
            logger.debug(f"Variance homogeneity: Sample {s_idx} is empty or all NaNs after cleaning. Excluding.")
    
    if len(cleaned_samples) < 2: # Need at least two groups to compare variances
        logger.debug(f"Variance homogeneity: Less than 2 valid samples after cleaning ({len(cleaned_samples)}). Assuming homogeneity.")
        return True

    # Check if any sample has fewer than 2 data points (Levene's needs at least 2 per group for meaningful comparison)
    if any(len(s) < 2 for s in cleaned_samples):
        logger.debug("Variance homogeneity: At least one sample has < 2 data points after cleaning. Assuming homogeneity by default.")
        return True

    # Check if all values in all samples are identical (Levene's test may result in NaN p-value or error)
    try:
        first_val = cleaned_samples[0][0]
        if all(all(val == first_val for val in s) for s in cleaned_samples):
            logger.debug("Variance homogeneity: All values in all samples are identical. Assuming homogeneity.")
            return True
    except IndexError: # Handles case where a cleaned sample might become empty if it had only one value.
        logger.debug("Variance homogeneity: Edge case with sample data (e.g. single value samples). Assuming homogeneity.")
        return True
        
    # Prepare samples for Levene's test (convert to numpy arrays)
    np_samples = [np.array(s) for s in cleaned_samples]

    try:
        stat, p_value = stats.levene(*np_samples)
        is_homogeneous = p_value > alpha
        logger.debug(f"Levene's test on {len(np_samples)} samples: stat={stat:.3f}, p={p_value:.3f}. Homogeneous: {is_homogeneous} (alpha={alpha}).")
        return is_homogeneous
    except Exception as e:
        logger.warning(f"Variance homogeneity: Levene's test failed: {e}. Returning True as a fallback.")
        return True # Fallback to avoid blocking tests, though this assumption might be strong.

# 3.a. Compare Two Groups
def compare_two_groups(group1_data: List[float], group2_data: List[float], 
                       alpha: float = ALPHA_LEVEL, 
                       group1_name: str = "Group1", group2_name: str = "Group2") -> Dict[str, Any]:
    """
    Compares two independent groups using t-test or Mann-Whitney U test based on normality.
    """
    results: Dict[str, Any] = {
        'test_name': 'N/A', 'statistic': np.nan, 'p_value': 1.0, 
        'significant': False, 'group1_name': group1_name, 'group2_name': group2_name,
        'details': ''
    }

    # Clean NaNs from data
    g1_clean = [x for x in group1_data if pd.notna(x)]
    g2_clean = [x for x in group2_data if pd.notna(x)]

    if len(g1_clean) < 3 or len(g2_clean) < 3: # Threshold for meaningful statistical tests
        logger.warning(f"compare_two_groups ({group1_name} vs {group2_name}): Insufficient data after NaN removal (G1: {len(g1_clean)}, G2: {len(g2_clean)}). Min 3 required. Skipping test.")
        results['details'] = f"Insufficient data (G1: {len(g1_clean)}, G2: {len(g2_clean)}). Required >= 3 for tests."
        results['p_value'] = 1.0 # Ensure it's not significant
        results['significant'] = False
        return results

    normality_g1 = check_normality(g1_clean, alpha)
    normality_g2 = check_normality(g2_clean, alpha)
    details_str = f"Normality G1: {normality_g1}, Normality G2: {normality_g2}"

    if normality_g1 and normality_g2:
        equal_vars = check_variance_homogeneity([g1_clean, g2_clean], alpha)
        details_str += f", Equal Variances: {equal_vars}"
        if equal_vars:
            results['test_name'] = "Independent Samples T-test"
            stat, p_val = stats.ttest_ind(np.array(g1_clean), np.array(g2_clean), equal_var=True, nan_policy='omit')
        else:
            results['test_name'] = "Welch's T-test"
            stat, p_val = stats.ttest_ind(np.array(g1_clean), np.array(g2_clean), equal_var=False, nan_policy='omit')
    else:
        details_str += ", Equal Variances: N/A (non-parametric)"
        results['test_name'] = "Mann-Whitney U Test"
        try:
            # Mann-Whitney U needs at least one observation in each group, but more for meaningful results.
            # scipy's default is alternative='two-sided'.
            # Some versions of scipy might warn if data has ties or is all identical.
            if len(np.unique(g1_clean)) == 1 and len(np.unique(g2_clean)) == 1 and np.unique(g1_clean)[0] == np.unique(g2_clean)[0]:
                 logger.warning(f"compare_two_groups ({group1_name} vs {group2_name}): Both groups have identical constant values. Mann-Whitney U may be uninformative.")
                 stat, p_val = np.nan, 1.0 # P-value of 1 as they are identical
            elif len(g1_clean) == 0 or len(g2_clean) == 0: # Should be caught by earlier check, but safeguard
                 logger.warning(f"compare_two_groups ({group1_name} vs {group2_name}): One group empty after cleaning. Mann-Whitney U cannot be run.")
                 stat, p_val = np.nan, 1.0
            else:
                 stat, p_val = stats.mannwhitneyu(np.array(g1_clean), np.array(g2_clean), alternative='two-sided', nan_policy='omit')
        except ValueError as e_mw: # Catch specific errors from mannwhitneyu
            logger.warning(f"compare_two_groups ({group1_name} vs {group2_name}): Mann-Whitney U test failed: {e_mw}. Assuming no significance.")
            stat, p_val = np.nan, 1.0


    results['statistic'] = stat if pd.notna(stat) else np.nan
    results['p_value'] = p_val if pd.notna(p_val) else 1.0
    results['significant'] = results['p_value'] < alpha
    results['details'] = details_str
    
    logger.info(
        f"Comparison ({group1_name} vs {group2_name}): Test={results['test_name']}, "
        f"Stat={results['statistic']:.3f}, P-val={results['p_value']:.3f}, Sig={results['significant']}. "
        f"Details: {results['details']}"
    )
    return results

# 3.b. Compare Multiple Groups
def compare_multiple_groups(groups_data: Dict[str, List[float]], alpha: float = ALPHA_LEVEL) -> Dict[str, Any]:
    """
    Compares multiple groups using ANOVA or Kruskal-Wallis test based on normality and variance homogeneity.
    """
    results: Dict[str, Any] = {
        'test_name': 'N/A', 'statistic': np.nan, 'p_value': 1.0, 
        'significant': False, 'all_normal': False, 'equal_variances': False,
        'post_hoc_required': False
    }

    # Clean data and filter out groups with insufficient data for normality/variance checks
    cleaned_groups_data: Dict[str, List[float]] = {}
    for name, data_list in groups_data.items():
        cleaned_list = [x for x in data_list if pd.notna(x)]
        if len(cleaned_list) >= 3: # Min data for normality and meaningful participation in group tests
            cleaned_groups_data[name] = cleaned_list
        else:
            logger.warning(f"compare_multiple_groups: Group '{name}' has insufficient data ({len(cleaned_list)} after NaN removal). Min 3 required. Excluding from overall test.")
            
    if len(cleaned_groups_data) < 2: # Need at least 2 groups for comparison
        logger.warning(f"compare_multiple_groups: Less than 2 groups remaining after filtering for data sufficiency ({len(cleaned_groups_data)}). Cannot perform test.")
        results['details'] = f"Less than 2 valid groups ({len(cleaned_groups_data)}) for comparison."
        return results

    group_values_list = list(cleaned_groups_data.values()) # List of lists of floats

    # Check normality for all participating groups
    all_groups_normal = all(check_normality(data, alpha) for data in group_values_list)
    results['all_normal'] = all_groups_normal

    # Check variance homogeneity if all groups are normal
    variances_equal = False
    if all_groups_normal:
        variances_equal = check_variance_homogeneity(group_values_list, alpha)
    results['equal_variances'] = variances_equal

    if all_groups_normal and variances_equal:
        results['test_name'] = "One-Way ANOVA"
        try:
            stat, p_val = stats.f_oneway(*[np.array(g) for g in group_values_list])
        except Exception as e_anova:
            logger.warning(f"compare_multiple_groups: ANOVA failed: {e_anova}. Defaulting to non-significant.")
            stat, p_val = np.nan, 1.0
    else:
        results['test_name'] = "Kruskal-Wallis Test"
        try:
            stat, p_val = stats.kruskal(*[np.array(g) for g in group_values_list])
        except Exception as e_kruskal:
            logger.warning(f"compare_multiple_groups: Kruskal-Wallis failed: {e_kruskal}. Defaulting to non-significant.")
            stat, p_val = np.nan, 1.0
            
    results['statistic'] = stat if pd.notna(stat) else np.nan
    results['p_value'] = p_val if pd.notna(p_val) else 1.0
    results['significant'] = results['p_value'] < alpha
    results['post_hoc_required'] = results['significant']

    logger.info(
        f"Comparison of {len(cleaned_groups_data)} groups: Test={results['test_name']}, "
        f"Stat={results['statistic']:.3f}, P-val={results['p_value']:.3f}, Sig={results['significant']}. "
        f"AllNormal: {results['all_normal']}, EqualVars: {results['equal_variances']}. Post-hoc needed: {results['post_hoc_required']}"
    )
    return results

# 3.c. Perform Post-Hoc Tests
def perform_post_hoc_tests(groups_data: Dict[str, List[float]], 
                           parametric: bool, 
                           alpha: float = ALPHA_LEVEL, # alpha is not directly used by scikit-posthocs display but good for context
                           multiple_testing_correction: str = 'bonferroni') -> Union[pd.DataFrame, str]:
    """
    Performs post-hoc tests (Tukey/T-test with Bonferroni for parametric, Dunn for non-parametric).
    """
    # Prepare data in melted DataFrame format for scikit-posthocs
    # Filter out groups with insufficient data before melting
    valid_groups_data = {name: data for name, data in groups_data.items() if data and len([x for x in data if pd.notna(x)]) > 1}
    if len(valid_groups_data) < 2:
        return "Post-hoc tests require at least two groups with sufficient data."

    df_melted = _prepare_melted_df_for_posthocs(valid_groups_data)
    if df_melted.empty or df_melted['group'].nunique() < 2 :
        logger.warning("Post-hoc: DataFrame is empty or has less than 2 unique groups after preparation. Cannot perform post-hoc tests.")
        return "Post-hoc: Not enough data or groups for comparison after preparation."

    try:
        if parametric:
            logger.info("Performing parametric post-hoc (Tukey HSD or T-test with Bonferroni).")
            # Tukey HSD is often preferred but posthoc_tukey might be tricky with direct dict input or unequal sample sizes.
            # Using posthoc_ttest with Bonferroni is a robust alternative.
            post_hoc_results_df = sp.posthoc_ttest(df_melted, val_col='value', group_col='group', p_adjust='bonferroni', equal_var=True) # Assuming equal_var for simplicity after ANOVA
            # If Welch's ANOVA was hypothetically used (not standard f_oneway), then equal_var=False here.
            # For posthoc_tukey, it typically expects equal sample sizes.
            # post_hoc_results_df = sp.posthoc_tukey(df_melted, val_col='value', group_col='group')
        else:
            logger.info(f"Performing non-parametric post-hoc (Dunn's test with {multiple_testing_correction} correction).")
            post_hoc_results_df = sp.posthoc_dunn(df_melted, val_col='value', group_col='group', p_adjust=multiple_testing_correction)
        
        logger.info(f"Post-hoc test completed. Resulting DataFrame shape: {post_hoc_results_df.shape}")
        # Displaying the table of p-values (scikit-posthocs returns a DataFrame of p-values)
        # The significance (True/False based on alpha) can be determined by comparing elements to alpha.
        # Example: significant_pairs_df = post_hoc_results_df < alpha
        return post_hoc_results_df
    except Exception as e:
        logger.error(f"Post-hoc test failed: {e}", exc_info=True)
        return f"Error during post-hoc test execution: {e}"

# 4.a. Friedman Test and Posthoc
def perform_friedman_test_and_posthoc(
    results_df: pd.DataFrame, 
    metric_col: str = 'BestFitness', 
    config_col: str = 'Configuration', 
    run_col: str = 'Run', 
    alpha: float = ALPHA_LEVEL
) -> Optional[Tuple[float, float, Optional[pd.DataFrame], Optional[pd.Series]]]:
    """
    Performs Friedman test and Nemenyi post-hoc if significant.
    Calculates average ranks for configurations.
    Assumes results_df contains one row per run per configuration.
    Returns: (friedman_statistic, friedman_p_value, nemenyi_results_df, avg_ranks)
             Returns None if the test cannot be run.
    """
    if results_df.empty or not {metric_col, config_col, run_col}.issubset(results_df.columns):
        logger.warning("Friedman: DataFrame empty or missing required columns. Cannot perform test.")
        return None

    avg_ranks: Optional[pd.Series] = None
    pivoted_df_cleaned: Optional[pd.DataFrame] = None

    try:
        pivoted_df = results_df.dropna(subset=[metric_col])
        pivoted_df = pivoted_df.pivot_table(index=run_col, columns=config_col, values=metric_col)
        
        pivoted_df_cleaned = pivoted_df.dropna(axis=1) 

        if pivoted_df_cleaned.shape[0] < 2 or pivoted_df_cleaned.shape[1] < 2:
            logger.warning(f"Friedman: Not enough complete data after pivoting (Configs: {pivoted_df_cleaned.shape[1]}, Runs: {pivoted_df_cleaned.shape[0]}). Min 2x2 required. Test skipped.")
            return None # Cannot calculate ranks or run test
            
        # Calculate average ranks: Rank configurations within each run (axis=1), then average these ranks.
        # Lower metric value should result in a lower (better) rank.
        avg_ranks = pivoted_df_cleaned.rank(axis=1, method='average', ascending=True).mean(axis=0)
        
    except Exception as e_pivot:
        logger.error(f"Friedman: Error pivoting data or calculating ranks: {e_pivot}", exc_info=True)
        return None # Pivoting or rank calculation failed

    if pivoted_df_cleaned is None: # Should not happen if previous error handling is correct
        logger.error("Friedman: pivoted_df_cleaned is unexpectedly None. Aborting.")
        return None

    data_for_friedman = [pivoted_df_cleaned[col].values for col in pivoted_df_cleaned.columns]

    if not data_for_friedman or len(data_for_friedman) < 2:
        logger.warning("Friedman: Not enough groups for Friedman test after data preparation. Test skipped.")
        # Still return avg_ranks if they were calculable
        return None, None, None, avg_ranks if avg_ranks is not None else None


    try:
        friedman_stat, friedman_p_val = stats.friedmanchisquare(*data_for_friedman)
        logger.info(f"Friedman Test: Statistic={friedman_stat:.3f}, P-value={friedman_p_val:.3f}")
    except Exception as e_friedman:
        logger.error(f"Friedman: Friedman test execution failed: {e_friedman}", exc_info=True)
        # Return avg_ranks even if Friedman test itself fails
        return None, None, None, avg_ranks

    nemenyi_results_df: Optional[pd.DataFrame] = None
    if friedman_p_val is not None and friedman_p_val < alpha:
        logger.info("Friedman test significant. Performing Nemenyi post-hoc test.")
        try:
            nemenyi_results_df = sp.posthoc_nemenyi_friedman(pivoted_df_cleaned) # Use the cleaned pivoted data
            logger.info("Nemenyi post-hoc test completed.")
        except Exception as e_nemenyi:
            logger.error(f"Friedman: Nemenyi post-hoc test failed: {e_nemenyi}", exc_info=True)
            # Nemenyi failed, but Friedman results and ranks are still valid
    else:
        logger.info("Friedman test not significant (or p-value is None). Skipping Nemenyi post-hoc test.")

    return friedman_stat, friedman_p_val, nemenyi_results_df, avg_ranks

# Placeholder for Cohen's d if needed later
# def calculate_cohens_d(sample1: List[float], sample2: List[float], pooled_std: bool = True) -> float:
#     s1, s2 = np.array(sample1), np.array(sample2)
#     n1, n2 = len(s1), len(s2)
#     if n1 == 0 or n2 == 0: return np.nan
#     mean1, mean2 = np.mean(s1), np.mean(s2)
#     if pooled_std:
#         std1, std2 = np.std(s1, ddof=1), np.std(s2, ddof=1)
#         s_pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
#         if s_pooled == 0: return np.nan # Avoid division by zero
#         d = (mean1 - mean2) / s_pooled
#     else: # Use std of control group or average std if no clear control
#         std_control = np.std(s1, ddof=1) # Assuming s1 is control or baseline
#         if std_control == 0: return np.nan
#         d = (mean1 - mean2) / std_control
#     return d

def generate_statistical_test_selection_report(
    groups_data: Dict[str, List[float]], 
    metric_name: str = "Metric", 
    alpha_level: float = ALPHA_LEVEL
) -> str:
    """
    Generates a textual report explaining the choice of statistical tests 
    for comparing groups based on normality and variance homogeneity.
    """
    report_lines = [
        f"Statistical Test Selection Report for '{metric_name}'",
        "-" * (len(f"Statistical Test Selection Report for '{metric_name}'") + 2),
    ]

    original_group_count = len(groups_data)
    group_names_original = list(groups_data.keys())
    report_lines.append(f"Original number of groups: {original_group_count} ({', '.join(group_names_original)})")
    report_lines.append(f"Significance level (alpha): {alpha_level}")
    report_lines.append("\nData Preprocessing:")

    cleaned_groups_data: Dict[str, List[float]] = {}
    preprocessing_details = []
    for name, data_list in groups_data.items():
        if not isinstance(data_list, list):
            preprocessing_details.append(f"- {name}: Invalid data format (not a list), excluded.")
            continue
        
        original_len = len(data_list)
        cleaned_list = [x for x in data_list if pd.notna(x)]
        removed_count = original_len - len(cleaned_list)
        
        if len(cleaned_list) < 3:
            preprocessing_details.append(f"- {name}: {len(cleaned_list)} data points after removing {removed_count} NaN(s). Insufficient data (min 3 required), excluded from analysis.")
        else:
            preprocessing_details.append(f"- {name}: {len(cleaned_list)} data points after removing {removed_count} NaN(s).")
            cleaned_groups_data[name] = cleaned_list
            
    report_lines.extend(preprocessing_details)

    num_valid_groups = len(cleaned_groups_data)
    valid_group_names = list(cleaned_groups_data.keys())
    report_lines.append(f"\nNumber of groups with sufficient data for analysis: {num_valid_groups} ({', '.join(valid_group_names)})")

    if num_valid_groups < 2:
        report_lines.append("\nComparison Path: Less than 2 groups with sufficient data. No statistical comparison performed.")
        return "\n".join(report_lines)

    # --- Two Groups Path ---
    if num_valid_groups == 2:
        report_lines.append(f"\nComparison Path ({num_valid_groups} groups remaining: {', '.join(valid_group_names)}):")
        
        group1_name = valid_group_names[0]
        group2_name = valid_group_names[1]
        group1_data = cleaned_groups_data[group1_name]
        group2_data = cleaned_groups_data[group2_name]

        # Store p-values for detailed reporting
        normality_p_values: Dict[str, Optional[float]] = {name: None for name in valid_group_names}

        try:
            _, p1 = stats.shapiro(np.array(group1_data)) 
            normality_p_values[group1_name] = p1
        except Exception: p1 = -1 # Error in test
        
        try:
            _, p2 = stats.shapiro(np.array(group2_data))
            normality_p_values[group2_name] = p2
        except Exception: p2 = -1 # Error in test
            
        normality_g1 = p1 > alpha_level if p1 != -1 else False
        normality_g2 = p2 > alpha_level if p2 != -1 else False
        
        report_lines.append(f"Normality Tests (alpha = {alpha_level}):")
        report_lines.append(f"- {group1_name}: Data appears {'Normal' if normality_g1 else 'Not Normal'} (Shapiro-Wilk p-value = {normality_p_values[group1_name]:.4f if normality_p_values[group1_name] is not None else 'Test Failed'})")
        report_lines.append(f"- {group2_name}: Data appears {'Normal' if normality_g2 else 'Not Normal'} (Shapiro-Wilk p-value = {normality_p_values[group2_name]:.4f if normality_p_values[group2_name] is not None else 'Test Failed'})")

        if normality_g1 and normality_g2:
            report_lines.append("Conclusion: Both groups appear Normal.")
            # Check variance homogeneity
            levene_stat, levene_p_val = stats.levene(np.array(group1_data), np.array(group2_data))
            equal_vars = levene_p_val > alpha_level
            report_lines.append(f"Variance Homogeneity Test (Levene's, alpha = {alpha_level}):")
            report_lines.append(f"- Variances appear {'Equal' if equal_vars else 'Unequal'} (p-value = {levene_p_val:.4f})")
            if equal_vars:
                report_lines.append("Recommended Test: Independent Samples T-test.")
            else:
                report_lines.append("Recommended Test: Welch's T-test (for unequal variances).")
        else:
            report_lines.append("Conclusion: Since not all group data are Normal, a non-parametric test is chosen.")
            report_lines.append("Recommended Test: Mann-Whitney U Test.")
        report_lines.append("Recommended Post-Hoc (if applicable): Not directly applicable for two groups, but consider effect size (e.g., Cohen's d for T-tests, Rank Biserial Correlation for Mann-Whitney U).")

    # --- Multiple Groups Path ---
    elif num_valid_groups > 2:
        report_lines.append(f"\nComparison Path ({num_valid_groups} groups remaining: {', '.join(valid_group_names)}):")
        
        normality_results_detail = []
        all_groups_normal = True
        normality_p_values_multi: Dict[str, Optional[float]] = {name: None for name in valid_group_names}

        for name, data in cleaned_groups_data.items():
            try:
                _, p_val_shapiro = stats.shapiro(np.array(data))
                normality_p_values_multi[name] = p_val_shapiro
                is_normal = p_val_shapiro > alpha_level
                normality_results_detail.append(f"- {name}: Data appears {'Normal' if is_normal else 'Not Normal'} (Shapiro-Wilk p-value = {p_val_shapiro:.4f})")
                if not is_normal:
                    all_groups_normal = False
            except Exception as e_shapiro:
                 normality_results_detail.append(f"- {name}: Normality test failed ({e_shapiro}). Assuming Not Normal.")
                 all_groups_normal = False


        report_lines.append(f"Normality Tests (alpha = {alpha_level}):")
        report_lines.extend(normality_results_detail)

        equal_variances = False
        if all_groups_normal:
            report_lines.append("Conclusion: All groups appear Normal based on individual tests.")
            # Check variance homogeneity
            try:
                levene_stat, levene_p_val = stats.levene(*[np.array(d) for d in cleaned_groups_data.values()])
                equal_variances = levene_p_val > alpha_level
                report_lines.append(f"Variance Homogeneity Test (Levene's, alpha = {alpha_level}):")
                report_lines.append(f"- Variances across groups appear {'Equal' if equal_variances else 'Unequal'} (p-value = {levene_p_val:.4f})")
            except Exception as e_levene:
                report_lines.append(f"Variance Homogeneity Test (Levene's, alpha = {alpha_level}): Failed ({e_levene}). Assuming Unequal for safety.")
                equal_variances = False # Assume unequal if test fails
        else:
            report_lines.append("Conclusion: Not all group data are Normal.")
            # No need to report Levene's test for Kruskal-Wallis, as it doesn't assume variance homogeneity.
            # However, very different variances can affect Kruskal-Wallis interpretation.
            # For simplicity of the flowchart, we only strictly require Levene's for ANOVA.
            report_lines.append("Variance Homogeneity Test: Not strictly required for non-parametric multi-group test, but heterogeneity can affect interpretation.")


        if all_groups_normal and equal_variances:
            report_lines.append("Recommended Overall Test: One-Way ANOVA.")
            report_lines.append("Recommended Post-Hoc: Parametric (e.g., Tukey HSD, Bonferroni t-test).")
        else:
            if not all_groups_normal:
                 report_lines.append("Reason for non-parametric: Not all groups passed normality tests.")
            elif not equal_variances: # Implies all_groups_normal was True
                 report_lines.append("Reason for non-parametric: Groups passed normality tests, but variances are unequal (consider Welch's ANOVA if available and appropriate, or Kruskal-Wallis as robust alternative).")
            report_lines.append("Recommended Overall Test: Kruskal-Wallis Test.")
            report_lines.append("Recommended Post-Hoc: Non-parametric (e.g., Dunn's test with Bonferroni or other appropriate p-value adjustment).")

    report_lines.append("\nEffect Size Note:")
    report_lines.append("After significance testing, calculating appropriate effect sizes is recommended to understand the magnitude of any observed differences (e.g., Cohen's d, Eta-squared, Rank Biserial Correlation).")
    
    return "\n".join(report_lines)

logger.info("statistical_utils.py loaded with statistical test functions.")
