import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple # Added Tuple

# Import LeagueSolution for type hinting if used explicitly, or rely on duck typing.
# from solution import LeagueSolution
# Import recreate_solution_from_data from the new serialization_utils module
from serialization_utils import recreate_solution_from_data
import statistical_utils as su # For statistical tests
import scikit_posthocs as sp # For plotting CD diagram or sign_plot
import matplotlib.pyplot as plt # Already imported but ensure for clarity

# For notebook detection and rich display
from experiment_utils import is_notebook
try:
    from IPython.display import display, Markdown, HTML # Added HTML
    ipython_available = True
except ImportError:
    ipython_available = False

from scipy import stats as scipy_stats # For confidence intervals
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates

logger = logging.getLogger(__name__)

def display_plot_info(title: str, what_it_shows: str, how_to_interpret: str, purpose: str, verbose: bool = True):
    """
    Displays information about a plot, adapting output for notebooks or standard consoles.
    Controlled by a verbose flag.
    """
    if not verbose:
        return

    if is_notebook() and ipython_available:
        md_string = f"""### Plot: {title}
**What it shows:**
{what_it_shows}

**How to interpret it:**
{how_to_interpret}

**Purpose:**
{purpose}
---"""
        display(Markdown(md_string))
    else:
        print("\n==================================================")
        print(f"PLOT INFO: {title}")
        print("==================================================")
        print("WHAT IT SHOWS:")
        print(what_it_shows)
        print("\nHOW TO INTERPRET IT:")
        print(how_to_interpret)
        print("\nPURPOSE:")
        print(purpose)
        print("--------------------------------------------------\n")

def plot_summary_statistics_bars(final_results_df: pd.DataFrame, verbose: bool = True):
    if final_results_df.empty:
        if verbose:
            logger.info("No results for summary statistics bars.")
        return

    if verbose:
        logger.info("\n--- Summary Statistics & Bar Charts ---")
    metrics = ["BestFitness", "Iterations", "FunctionEvaluations", "RuntimeSeconds"]

    for metric_col in metrics:
        if metric_col not in final_results_df.columns:
            logger.warning(f"Metric '{metric_col}' not found for summary. Skipping.")
            continue

        valid_data = final_results_df[
            final_results_df[metric_col].notna()
            & np.isfinite(final_results_df[metric_col])
            & (~final_results_df["AlgorithmName"].str.contains("ERROR", na=False))
        ]
        if valid_data.empty:
            if verbose:
                logger.info(f"No valid data for metric '{metric_col}'. Skipping.")
            continue

        summary_stats = valid_data.groupby("Configuration")[metric_col].agg(
            ["mean", "std", "min", "max"]
        )
        if verbose:
            logger.info(f"\n📊 Statistics for: {metric_col}")

        fmt = "{:.4f}" if metric_col in ["BestFitness", "RuntimeSeconds"] else "{:.1f}"
        formatters = {
            col: (lambda x, f=fmt: f.format(x)) for col in summary_stats.columns
        }
        try:
            from IPython.display import display # type: ignore
            if verbose: # Assuming display is for verbose output
                display(summary_stats.style.format(formatters))
        except ImportError:
            if verbose:
                print(summary_stats.to_string(formatters=formatters))

        plt.figure(figsize=(10, 6)) # Set default figure size
        ax = summary_stats["mean"].plot(
            kind="bar",
            yerr=summary_stats["std"],
            capsize=4,
            color=sns.color_palette("viridis", len(summary_stats)),
            edgecolor="grey",
        )
        title_suffix = "(Lower is Better)"
        if metric_col == "Iterations":
            title_suffix = "(Context Dependent)"
        plt.title(f"Average {metric_col} by Algorithm {title_suffix}", fontsize=14)
        plt.ylabel(f"Average {metric_col}", fontsize=11)
        plt.xlabel("Configuration", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle=":", alpha=0.7)
        for i, val_mean in enumerate(summary_stats["mean"]):
            if np.isfinite(val_mean):
                txt_val = fmt.format(val_mean)
                std_err = summary_stats["std"].iloc[i]
                offset_y = (std_err * 0.1 if pd.notna(std_err) else 0) + (
                    val_mean * 0.02
                )
                ax.text(
                    i,
                    val_mean + offset_y,
                    txt_val,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        plt.tight_layout()
        plt.show()

def display_summary_statistics_table(
    final_results_df: pd.DataFrame, 
    metrics_to_display: Optional[List[str]] = None, 
    alpha_level: float = 0.05, # For 1-alpha confidence level
    verbose: bool = True
):
    """
    Displays a detailed summary statistics table for specified metrics,
    adapting output for notebooks (HTML/styled) or standard consoles.
    Includes count, mean, median, std dev, min, max, and confidence intervals.
    """
    if not verbose:
        return

    if metrics_to_display is None:
        metrics_to_display = ["BestFitness", "RuntimeSeconds", "FunctionEvaluations"]
    
    if not metrics_to_display:
        logger.info("No metrics specified for summary statistics table.")
        return

    if verbose:
        logger.info("\n--- Detailed Summary Statistics Tables ---")
    
    processed_any_metric = False
    for metric_col in metrics_to_display:
        if verbose:
            logger.info(f"Processing summary statistics for metric: {metric_col}")

        if metric_col not in final_results_df.columns:
            logger.warning(f"Metric '{metric_col}' not found in results DataFrame. Skipping.")
            continue

        valid_data_metric = final_results_df[
            final_results_df[metric_col].notna() & 
            np.isfinite(final_results_df[metric_col]) &
            (~final_results_df["AlgorithmName"].str.contains("ERROR", na=False))
        ]

        if valid_data_metric.empty:
            if verbose:
                logger.info(f"No valid data for metric '{metric_col}' after filtering. Skipping table for this metric.")
            continue
        
        processed_any_metric = True
        grouped = valid_data_metric.groupby("Configuration")[metric_col]
        
        summary_aggs = grouped.agg(
            Count='count',
            Mean='mean',
            Median='median',
            StdDev='std',
            Min='min',
            Max='max'
        ).reset_index() # Reset index to make 'Configuration' a column for easier manipulation initially

        # Calculate Confidence Interval
        # scipy_stats.t.interval(confidence, df, loc=mean, scale=sem)
        # sem = std / sqrt(count)
        summary_aggs['SEM'] = summary_aggs['StdDev'] / np.sqrt(summary_aggs['Count'])
        
        # Initialize CI columns with NaNs
        ci_col_lower_name = f'{100*(1-alpha_level):.0f}% CI Lower'
        ci_col_upper_name = f'{100*(1-alpha_level):.0f}% CI Upper'
        summary_aggs[ci_col_lower_name] = np.nan
        summary_aggs[ci_col_upper_name] = np.nan

        for index, row in summary_aggs.iterrows():
            if row['Count'] >= 2 and row['SEM'] > 0: # CI is meaningful for count >= 2 and non-zero SEM
                ci = scipy_stats.t.interval(
                    confidence=(1 - alpha_level), 
                    df=row['Count'] - 1, 
                    loc=row['Mean'], 
                    scale=row['SEM']
                )
                summary_aggs.loc[index, ci_col_lower_name] = ci[0]
                summary_aggs.loc[index, ci_col_upper_name] = ci[1]
            elif row['Count'] == 1: # SEM might be 0 or NaN if std is 0 for a single point
                 summary_aggs.loc[index, ci_col_lower_name] = row['Mean'] # CI is just the point itself
                 summary_aggs.loc[index, ci_col_upper_name] = row['Mean']


        summary_df = summary_aggs.set_index("Configuration") # Set 'Configuration' as index for final display
        summary_df = summary_df[['Count', 'Mean', 'Median', 'StdDev', ci_col_lower_name, ci_col_upper_name, 'Min', 'Max']] # Reorder and select columns


        # Display logic
        title_str = f"Summary Statistics for: {metric_col}"
        float_format_cols = {
            'Mean': '{:.3f}', 'Median': '{:.3f}', 'StdDev': '{:.3f}',
            ci_col_lower_name: '{:.3f}', ci_col_upper_name: '{:.3f}',
            'Min': '{:.3f}', 'Max': '{:.3f}'
        }
        # Adjust for metrics that might not need so many decimal places
        if metric_col in ["Iterations", "FunctionEvaluations", "Count"]:
            float_format_cols = {k: '{:.1f}' if v == '{:.3f}' else v for k,v in float_format_cols.items()}
            float_format_cols['Count'] = '{:.0f}' # Ensure count is integer
        
        if is_notebook() and ipython_available:
            display(Markdown(f"#### {title_str}"))
            styled_df = summary_df.style.format(float_format_cols)
            
            if metric_col in ["BestFitness", "RuntimeSeconds"]:
                styled_df = styled_df.highlight_min(subset=['Mean'], color='lightgreen')
            # Example for "higher is better" if such a metric exists
            # elif metric_col == "SomeScoreMetricWhereHigherIsBetter":
            #    styled_df = styled_df.highlight_max(subset=['Mean'], color='lightgreen')
            
            display(styled_df)
        else:
            print(f"\n--- {title_str} ---")
            # For console, apply formatting directly if possible or just print
            try:
                 print(summary_df.to_string(formatters={col: (lambda x, fmt_str=fmt: fmt_str.format(x) if pd.notna(x) else 'NaN') 
                                                       for col, fmt in float_format_cols.items()}))
            except Exception: # Fallback if direct formatting fails
                print(summary_df.to_string())
        
    if not processed_any_metric and verbose:
        logger.info("No metrics were processed or had valid data for summary statistics tables.")


# This is the new, enhanced version of the function. 
# The old, simpler version that caused the IndentationError has been removed.
def plot_metric_distributions_boxplots(
    final_results_df: pd.DataFrame, 
    plot_type: str = 'box', 
    show_points: bool = True, 
    show_significance: bool = True, 
    alpha_level: float = 0.05,
    annotation_mode: str = 'all', # 'all', 'vs_baseline', 'selected_pairs', 'none'
    baseline_config_name: Optional[str] = None,
    selected_pairs_for_annotation: Optional[List[Tuple[str, str]]] = None,
    verbose: bool = True
):
    if final_results_df.empty:
        if verbose:
            logger.info("Plotting distributions: DataFrame is empty.")
        return
        
    if verbose:
        logger.info(f"\n--- Enhanced Metric Distributions ({plot_type.capitalize()} Plots) ---")
    metrics_to_plot = ["BestFitness", "RuntimeSeconds", "FunctionEvaluations"] # Can be extended

    for metric_col in metrics_to_plot:
        if metric_col not in final_results_df.columns:
            logger.warning(f"Metric '{metric_col}' not found in results. Skipping its distribution plot.")
            continue

        # Filter data for valid, finite values and non-error algorithm runs
        valid_data_for_metric = final_results_df[
            final_results_df[metric_col].notna() & 
            np.isfinite(final_results_df[metric_col]) &
            (~final_results_df["AlgorithmName"].str.contains("ERROR", na=False))
        ]

        if valid_data_for_metric.empty:
            if verbose:
                logger.info(f"No valid data for metric '{metric_col}' after filtering. Skipping plot.")
            continue
            
        # Prepare data for statistical tests: Dict[str, List[float]]
        groups_data_for_stat_tests: Dict[str, List[float]] = valid_data_for_metric.groupby("Configuration")[metric_col].apply(list).to_dict()
        
        # Filter out groups with too few data points for reliable stats/plotting after grouping
        # (e.g. less than 2-3 points per group might not be meaningful for box/violin or stats)
        min_points_per_group = 3
        groups_data_filtered = {
            name: data for name, data in groups_data_for_stat_tests.items() if len(data) >= min_points_per_group
        }
        
        if len(groups_data_filtered) < 2: # Need at least two groups for comparison
            if verbose:
                logger.info(f"Metric '{metric_col}': Less than 2 configurations with sufficient data ({min_points_per_group} points) after filtering. Skipping significance testing and plot.")
            # Still, we can plot if there's at least one group, just no comparisons.
            if not groups_data_filtered: continue # No groups at all
        
        # Rebuild valid_data_for_metric DataFrame based on filtered groups to ensure plot matches stats
        configs_with_enough_data = list(groups_data_filtered.keys())
        plottable_data = valid_data_for_metric[valid_data_for_metric["Configuration"].isin(configs_with_enough_data)]

        if plottable_data.empty:
            if verbose:
                logger.info(f"Metric '{metric_col}': No plottable data after filtering groups by size. Skipping plot.")
            continue

        plt.figure(figsize=(max(10, len(configs_with_enough_data) * 1.5), 7)) # Dynamic width
        
        # Main plot: Box or Violin
        if plot_type == 'violin':
            ax = sns.violinplot(x="Configuration", y=metric_col, data=plottable_data, palette="pastel", hue="Configuration", legend=False, cut=0) # cut=0 limits violin range to data
        else: # Default to box plot
            ax = sns.boxplot(x="Configuration", y=metric_col, data=plottable_data, palette="pastel", hue="Configuration", legend=False)

        # Overlay jittered points if requested
        if show_points:
            sns.stripplot(x="Configuration", y=metric_col, data=plottable_data, jitter=True, size=4, color=".3", dodge=True, ax=ax, legend=False)

        ax.set_title(f"Distribution of {metric_col} by Algorithm ({plot_type.capitalize()} Plot)", fontsize=15)
        ax.set_ylabel(metric_col, fontsize=12)
        ax.set_xlabel("Configuration", fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10) # Removed ha="right"
        plt.xticks(rotation=45, ha="right") # Ensure ha is applied via plt.xticks if needed globally for the figure/subplot x-axis
        ax.grid(axis='y', linestyle=':', alpha=0.7)

        # Significance Annotation
        if show_significance and len(groups_data_filtered) >= 2:
            if verbose:
                logger.info(f"Calculating significance for {metric_col} across {len(groups_data_filtered)} configurations...")
                logger.info(f"Annotation mode: {annotation_mode}")

            overall_stat_result = su.compare_multiple_groups(groups_data_filtered, alpha=alpha_level)
            significant_pairs_details = [] # To store details of all significant pairs for logging

            if overall_stat_result and overall_stat_result['significant']:
                if verbose:
                    logger.info(f"Overall test ({overall_stat_result['test_name']}) for '{metric_col}' is significant (p={overall_stat_result['p_value']:.4f}). Performing post-hoc tests.")
                
                is_parametric_posthoc = overall_stat_result.get('all_normal', False) and overall_stat_result.get('equal_variances', False)
                post_hoc_results = su.perform_post_hoc_tests(groups_data_filtered, parametric=is_parametric_posthoc, alpha=alpha_level)
                
                if isinstance(post_hoc_results, pd.DataFrame) and annotation_mode != 'none':
                    config_names = list(groups_data_filtered.keys())
                    y_max = plottable_data[metric_col].max()
                    y_offset_increment = (y_max - plottable_data[metric_col].min()) * 0.08
                    current_y_offset = y_max + y_offset_increment * 0.5
                    num_significant_annotated = 0
                    max_annotations_for_all_mode = 5 # Clutter limit for 'all' mode

                    # Validate baseline_config_name for 'vs_baseline' mode
                    valid_baseline_for_mode = True
                    if annotation_mode == 'vs_baseline':
                        if not baseline_config_name or baseline_config_name not in config_names:
                            if verbose:
                                logger.warning(f"Baseline config '{baseline_config_name}' not provided or not in available data for metric '{metric_col}'. Skipping visual annotations.")
                            valid_baseline_for_mode = False
                    
                    # Validate selected_pairs_for_annotation for 'selected_pairs' mode
                    valid_selected_pairs_for_mode = True
                    if annotation_mode == 'selected_pairs':
                        if not selected_pairs_for_annotation:
                            if verbose:
                                logger.warning(f"Annotation mode is 'selected_pairs' but no pairs provided. Skipping visual annotations for metric '{metric_col}'.")
                            valid_selected_pairs_for_mode = False

                    for i in range(len(config_names)):
                        for j in range(i + 1, len(config_names)):
                            group1_name = config_names[i]
                            group2_name = config_names[j]
                            
                            p_val_pair = None
                            if group1_name in post_hoc_results.index and group2_name in post_hoc_results.columns:
                                p_val_pair = post_hoc_results.loc[group1_name, group2_name]
                            elif group2_name in post_hoc_results.index and group1_name in post_hoc_results.columns:
                                p_val_pair = post_hoc_results.loc[group2_name, group1_name]

                            if p_val_pair is not None and p_val_pair < alpha_level:
                                significant_pairs_details.append(f"{group1_name} vs {group2_name}: p={p_val_pair:.4f}")
                                
                                # Determine if this pair should be visually annotated
                                should_annotate_pair = False
                                if annotation_mode == 'all':
                                    if num_significant_annotated < max_annotations_for_all_mode:
                                        should_annotate_pair = True
                                    elif num_significant_annotated == max_annotations_for_all_mode and verbose:
                                        logger.info(f"Reached max ({max_annotations_for_all_mode}) visual annotations for '{metric_col}' in 'all' mode to avoid clutter.")
                                        num_significant_annotated += 1 # Increment to stop further logging of this message
                                
                                elif annotation_mode == 'vs_baseline' and valid_baseline_for_mode:
                                    if group1_name == baseline_config_name or group2_name == baseline_config_name:
                                        should_annotate_pair = True
                                
                                elif annotation_mode == 'selected_pairs' and valid_selected_pairs_for_mode and selected_pairs_for_annotation:
                                    pair_tuple = tuple(sorted((group1_name, group2_name)))
                                    if any(tuple(sorted(p)) == pair_tuple for p in selected_pairs_for_annotation):
                                        should_annotate_pair = True
                                
                                if should_annotate_pair:
                                    try:
                                        x1 = configs_with_enough_data.index(group1_name)
                                        x2 = configs_with_enough_data.index(group2_name)
                                    except ValueError:
                                        if verbose: logger.warning(f"Could not find {group1_name} or {group2_name} in plotted categories for annotation. Skipping this pair.")
                                        continue

                                    line_x = [x1, x1, x2, x2]
                                    line_y = [current_y_offset, current_y_offset + y_offset_increment*0.2, 
                                              current_y_offset + y_offset_increment*0.2, current_y_offset]
                                    ax.plot(line_x, line_y, lw=1.2, color='dimgray')
                                    
                                    p_text = f"p={p_val_pair:.3f}"
                                    if p_val_pair < 0.001: p_text = "***"
                                    elif p_val_pair < 0.01: p_text = "**"
                                    elif p_val_pair < 0.05: p_text = "*"
                                        
                                    ax.text((x1 + x2) / 2, current_y_offset + y_offset_increment*0.25, p_text, 
                                            ha='center', va='bottom', color='black', fontsize=9)
                                    current_y_offset += y_offset_increment 
                                    if annotation_mode == 'all': # Only increment counter for 'all' mode's limit
                                        num_significant_annotated +=1
                    
                    # Log all significant pairs found, regardless of visual annotation (if verbose)
                    if significant_pairs_details and verbose:
                        logger.info(f"All significant pairwise differences for '{metric_col}' (alpha={alpha_level}):")
                        for detail in significant_pairs_details: logger.info(f"  - {detail}")
                    elif verbose:
                        logger.info(f"No significant pairwise differences found after post-hoc for '{metric_col}'.")

                elif verbose: # If post_hoc_results is not a DataFrame or annotation_mode is 'none'
                    if annotation_mode == 'none':
                        logger.info(f"Visual annotations skipped for '{metric_col}' due to annotation_mode='none'.")
                    elif not isinstance(post_hoc_results, pd.DataFrame):
                        logger.warning(f"Post-hoc results for '{metric_col}' not in expected DataFrame format. Skipping visual annotations.")

            elif overall_stat_result and verbose: # Overall test was not significant
                logger.info(f"Overall test ({overall_stat_result['test_name']}) for '{metric_col}' is not significant (p={overall_stat_result['p_value']:.4f}). No post-hoc tests or visual annotations performed.")
            elif verbose: # overall_stat_result was None or other issue
                 logger.warning(f"Could not perform overall statistical test for '{metric_col}'. Skipping significance annotations.")

        plt.tight_layout()
        plt.show()


def plot_convergence_curves(final_history_map: dict):
    if not final_history_map:
        return
from typing import Dict, List, Any, Optional # Ensure this is imported

def plot_convergence_curves(
    final_history_map: Dict[str, Dict[int, List[Dict[str, Any]]]],
    # metric_key_override: Optional[str] = None # This was in plan, but fitness key detection is better
    verbose: bool = True
):
    if not final_history_map:
        if verbose:
            logger.info("No history data provided for convergence plots.")
        return

    if verbose:
        logger.info("\n--- Aggregated Convergence Plots (Mean +/- 1 Std Dev) ---")
    
    # Filter out configurations that might have empty run data
    valid_configs_data = {
        cfg_name: runs for cfg_name, runs in final_history_map.items() if runs and isinstance(runs, dict)
    }
    if not valid_configs_data:
        if verbose:
            logger.info("No valid configuration data found in history_map for convergence plots.")
        return

    num_configs = len(valid_configs_data)
    # Adjust layout: Aim for a single plot if few configs, otherwise subplots.
    # For now, let's stick to subplots per config as it's more robust to many configs.
    # If a single plot is desired later, this logic would change.
    rows = (num_configs + 1) // 2  # Adjust as needed, e.g., (num_configs + 2) // 3 for 3 columns
    cols = 2                     # Adjust as needed
    if num_configs == 1:
        rows, cols = 1, 1
    elif num_configs == 0:
        if verbose:
            logger.info("No configurations with data to plot.")
        return

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
    ax_flat = axes.flatten()
    plot_idx = 0

    for config_name, runs_data in valid_configs_data.items():
        if plot_idx >= len(ax_flat): # Should not happen if rows/cols are calculated well
            logger.warning("Ran out of subplot axes for configurations. Some plots might be missing.")
            break 
        
        ax = ax_flat[plot_idx]
        all_runs_fitness_trajectories = []
        max_len = 0 # To store the maximum length of any run's history for this config

        if not runs_data or not isinstance(runs_data, dict):
            logger.warning(f"Skipping config '{config_name}': No run data or invalid format.")
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Convergence: {config_name} (No Data)", fontsize=12)
            plot_idx += 1
            continue

        for run_id, history_list_of_dicts in runs_data.items():
            if not history_list_of_dicts or not isinstance(history_list_of_dicts, list):
                # This is a debug message, typically not controlled by main verbose, but can be if desired.
                # For now, let's assume debug messages are less critical for user-facing verbosity.
                logger.debug(f"Convergence Plot: Config '{config_name}', Run {run_id}: History is empty or not a list. Skipping this run.")
                continue

            current_run_trajectory = []
            if history_list_of_dicts: # Check if list is not empty
                first_item = history_list_of_dicts[0]
                if isinstance(first_item, dict):
                    fitness_key = None
                    if 'global_best_fitness' in first_item:
                        fitness_key = 'global_best_fitness'
                    elif 'best_fitness' in first_item:
                        fitness_key = 'best_fitness'
                    
                    if fitness_key:
                        current_run_trajectory = [item.get(fitness_key, np.nan) for item in history_list_of_dicts]
                    else:
                        logger.warning(f"Convergence Plot: Config '{config_name}', Run {run_id}: Could not determine fitness key from dict history items like {first_item}. Skipping this run.")
                        continue
                elif isinstance(first_item, (int, float)): # History is a list of numbers (HC, SA)
                    current_run_trajectory = history_list_of_dicts
                else:
                    logger.warning(f"Convergence Plot: Config '{config_name}', Run {run_id}: Unknown history item format {type(first_item)}. Skipping this run.")
                    continue
            
            # Filter out NaNs that might result from missing keys or non-numeric items
            current_run_trajectory = [f for f in current_run_trajectory if isinstance(f, (int, float)) and not np.isnan(f)]

            if current_run_trajectory:
                all_runs_fitness_trajectories.append(current_run_trajectory)
                if len(current_run_trajectory) > max_len:
                    max_len = len(current_run_trajectory)
            else:
                # Debug message, not typically controlled by verbose.
                logger.debug(f"Convergence Plot: Config '{config_name}', Run {run_id}: Trajectory is empty after processing. Skipping this run.")


        if not all_runs_fitness_trajectories:
            if verbose:
                logger.info(f"Config '{config_name}': No valid fitness trajectories found across all runs.")
            ax.text(0.5, 0.5, "No Valid Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Convergence: {config_name} (No Valid Data)", fontsize=12)
            plot_idx += 1
            continue

        # Pad trajectories to max_len using their last valid value
        padded_trajectories = []
        for traj in all_runs_fitness_trajectories:
            if not traj: continue # Should be filtered by now, but as a safeguard
            last_val = traj[-1]
            padding = [last_val] * (max_len - len(traj))
            padded_trajectories.append(traj + padding)
        
        if not padded_trajectories: # If after padding, still no data (e.g. all original trajectories were empty)
            if verbose:
                logger.info(f"Config '{config_name}': No data after padding trajectories.")
            ax.text(0.5, 0.5, "No Data After Padding", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Convergence: {config_name} (No Data After Padding)", fontsize=12)
            plot_idx += 1
            continue

        all_runs_fitness_np = np.array(padded_trajectories, dtype=float) # Ensure float for NaN compatibility if any slip through

        mean_fitness = np.nanmean(all_runs_fitness_np, axis=0) # Use nanmean
        std_fitness = np.nanstd(all_runs_fitness_np, axis=0)   # Use nanstd
        
        generations = np.arange(max_len)

        ax.plot(generations, mean_fitness, label=f"Mean Fitness ({len(padded_trajectories)} runs)", color='blue', lw=2)
        ax.fill_between(
            generations,
            mean_fitness - std_fitness,
            mean_fitness + std_fitness,
            color='blue',
            alpha=0.2,
            label="Mean +/- 1 Std Dev"
        )
        
        ax.set_title(f"Aggregated Convergence: {config_name}", fontsize=13)
        ax.set_xlabel("Generation / Iteration", fontsize=10)
        ax.set_ylabel("Fitness Value", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(fontsize=9)
        
        plot_idx += 1

    # Hide any unused subplots
    for i in range(plot_idx, len(ax_flat)):
        fig.delaxes(ax_flat[i])

    fig.tight_layout(pad=2.5)
    plt.show()


def plot_final_fitness_distributions(
    final_results_df: pd.DataFrame, 
    metric_col: str = 'BestFitness', 
    config_col: str = 'Configuration', 
    plot_type: str = 'hist_kde',
    verbose: bool = True
):
    """
    Visualizes the distribution of final 'BestFitness' values for each algorithm configuration.
    """
    if verbose:
        logger.info(f"\n--- Final '{metric_col}' Distributions ({plot_type}) ---")

    if final_results_df.empty:
        logger.warning(f"Plotting '{metric_col}' distributions: final_results_df is empty. Skipping.")
        return

    # Ensure metric_col and config_col exist
    if metric_col not in final_results_df.columns:
        logger.error(f"Plotting '{metric_col}' distributions: Metric column '{metric_col}' not found in DataFrame. Skipping.")
        return
    if config_col not in final_results_df.columns:
        logger.error(f"Plotting '{metric_col}' distributions: Configuration column '{config_col}' not found in DataFrame. Skipping.")
        return

    # Filter for valid data
    algorithm_name_col = "AlgorithmName" # Standard column name for algorithm errors
    if algorithm_name_col not in final_results_df.columns:
        # If AlgorithmName column doesn't exist, we can't filter by it. Log a debug message.
        logger.debug(f"Column '{algorithm_name_col}' not found. Proceeding without filtering error runs by this column.")
        # Create a temporary valid_data without this filter if the column is missing
        valid_data = final_results_df[
            final_results_df[metric_col].notna() & np.isfinite(final_results_df[metric_col])
        ]
    else:
        valid_data = final_results_df[
            final_results_df[metric_col].notna() & 
            np.isfinite(final_results_df[metric_col]) &
            (~final_results_df[algorithm_name_col].str.contains("ERROR", na=False))
        ]

    if valid_data.empty:
        logger.warning(f"Plotting '{metric_col}' distributions: No valid data after filtering. Skipping.")
        return

    unique_configs = valid_data[config_col].unique()
    if len(unique_configs) == 0:
        logger.warning(f"Plotting '{metric_col}' distributions: No unique configurations found after filtering. Skipping.")
        return

    num_configs = len(unique_configs)
    rows = (num_configs + 1) // 2 
    cols = 2
    if num_configs == 1: rows, cols = 1,1

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    ax_flat = axes.flatten()
    plot_idx = 0

    for i, config_name in enumerate(unique_configs):
        if plot_idx >= len(ax_flat): break # Should not happen if layout is correct

        ax = ax_flat[i]
        config_specific_fitness_values = valid_data[valid_data[config_col] == config_name][metric_col]

        if config_specific_fitness_values.empty:
                if verbose:
                    logger.info(f"No valid '{metric_col}' data for configuration '{config_name}'. Plotting empty subplot.")
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{config_name}\n(No Data)", fontsize=12)
            plot_idx += 1
            continue

        if plot_type == 'hist' or plot_type == 'hist_kde':
            use_kde = plot_type == 'hist_kde'
            stat_type = 'density' if use_kde else 'count'
            sns.histplot(data=config_specific_fitness_values, ax=ax, kde=use_kde, stat=stat_type, element="step", fill=True, alpha=0.7)
            ax.set_ylabel(stat_type.capitalize() if use_kde else "Frequency", fontsize=10)
        elif plot_type == 'kde':
            sns.kdeplot(data=config_specific_fitness_values, ax=ax, fill=True, alpha=0.7)
            ax.set_ylabel("Density", fontsize=10)
        else:
            logger.warning(f"Invalid plot_type '{plot_type}' specified. Defaulting to 'hist_kde'.")
            sns.histplot(data=config_specific_fitness_values, ax=ax, kde=True, stat='density', element="step", fill=True, alpha=0.7)
            ax.set_ylabel("Density", fontsize=10)
            
        ax.set_title(f"Distribution of Final {metric_col}\n{config_name}", fontsize=13)
        ax.set_xlabel(f"Final {metric_col}", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        plot_idx += 1
    
    # Hide any unused subplots
    for j in range(plot_idx, len(ax_flat)):
        fig.delaxes(ax_flat[j])

    fig.tight_layout(pad=2.0)
    plt.show()


def plot_time_to_target(
    final_history_map: Dict[str, Dict[int, List[Dict[str, Any]]]], 
    target_fitness_values: List[float], 
    max_evaluations: Optional[int] = None, 
    # config_col: str = 'Configuration' # Not directly needed as final_history_map keys are configs
    verbose: bool = True
):
    """
    Generates Time-to-Target (Attainment) plots.
    Shows the proportion of runs that reached a target fitness by a certain evaluation/generation.
    """
    if verbose:
        logger.info("\n--- Time-to-Target (Attainment) Plots ---")

    if not final_history_map:
        logger.warning("Time-to-Target: final_history_map is empty. Skipping plot.")
        return
    if not target_fitness_values:
        logger.warning("Time-to-Target: target_fitness_values list is empty. Skipping plot.")
        return
        
    # Sort targets for consistent plotting order, typically from harder (lower fitness) to easier
    # Assuming lower fitness is better. If higher is better, reverse sort or adjust logic.
    sorted_target_fitness_values = sorted(target_fitness_values) 

    # Determine max_evaluations dynamically if not provided
    determined_max_evals = 0
    if max_evaluations is None:
        for config_name, runs_data in final_history_map.items():
            if runs_data and isinstance(runs_data, dict):
                for run_id, history_list in runs_data.items():
                    if history_list and isinstance(history_list, list):
                        determined_max_evals = max(determined_max_evals, len(history_list))
        if determined_max_evals == 0:
            logger.warning("Time-to-Target: Could not determine max_evaluations from history data (all histories might be empty). Skipping plot.")
            return
        max_evaluations_to_use = determined_max_evals
        if verbose:
            logger.info(f"Time-to-Target: max_evaluations determined dynamically as {max_evaluations_to_use}.")
    else:
        max_evaluations_to_use = max_evaluations
        if verbose:
            logger.info(f"Time-to-Target: Using provided max_evaluations: {max_evaluations_to_use}.")


    plt.figure(figsize=(12, 7))
    
    # Use a color palette for configurations and line styles for targets
    config_palette = sns.color_palette("viridis", n_colors=len(final_history_map))
    target_linestyles = ['-', '--', ':', '-.'] # Cycle through these for different targets

    config_idx = 0
    for config_name, runs_data in final_history_map.items():
        if not runs_data or not isinstance(runs_data, dict):
            logger.debug(f"Time-to-Target: Skipping config '{config_name}' due to empty or invalid runs_data.")
            continue

        num_runs_for_config = len(runs_data)
        if num_runs_for_config == 0:
            logger.debug(f"Time-to-Target: Skipping config '{config_name}' as it has no runs.")
            continue
            
        config_color = config_palette[config_idx % len(config_palette)]

        for target_idx, target_fitness in enumerate(sorted_target_fitness_values):
            reached_at_eval_counts = np.zeros(max_evaluations_to_use, dtype=int)
            
            for run_id, history_list_generic in runs_data.items():
                if not history_list_generic or not isinstance(history_list_generic, list):
                    logger.debug(f"Time-to-Target: Config '{config_name}', Run {run_id}: History is empty or not a list.")
                    continue

                current_run_trajectory = []
                # Determine history format and extract fitness values
                if history_list_generic:
                    first_item = history_list_generic[0]
                    if isinstance(first_item, dict):
                        fitness_key = None
                        if 'global_best_fitness' in first_item: fitness_key = 'global_best_fitness'
                        elif 'best_fitness' in first_item: fitness_key = 'best_fitness'
                        
                        if fitness_key:
                            current_run_trajectory = [item.get(fitness_key, float('inf')) for item in history_list_generic]
                        else:
                            logger.warning(f"Time-to-Target: Config '{config_name}', Run {run_id}: Could not determine fitness key from dict history. Skipping run for target {target_fitness}.")
                            continue
                    elif isinstance(first_item, (int, float)): # List of numbers
                        current_run_trajectory = history_list_generic
                    else:
                        logger.warning(f"Time-to-Target: Config '{config_name}', Run {run_id}: Unknown history item format {type(first_item)}. Skipping run for target {target_fitness}.")
                        continue
                
                # Clean NaNs and ensure numeric
                cleaned_trajectory = [f for f in current_run_trajectory if isinstance(f, (int,float)) and pd.notna(f)]

                run_reached_target_at_step = -1
                for eval_step, current_fitness in enumerate(cleaned_trajectory):
                    if eval_step >= max_evaluations_to_use: break
                    if current_fitness <= target_fitness:
                        run_reached_target_at_step = eval_step
                        break
                
                if run_reached_target_at_step != -1:
                    reached_at_eval_counts[run_reached_target_at_step:] += 1
            
            # Convert counts to proportions
            proportion_reached = reached_at_eval_counts / num_runs_for_config
            
            # Plotting for this config and this target
            linestyle = target_linestyles[target_idx % len(target_linestyles)]
            plt.plot(
                np.arange(max_evaluations_to_use), 
                proportion_reached, 
                label=f"{config_name} (Target: {target_fitness:.1f})", 
                color=config_color,
                linestyle=linestyle,
                lw=1.5 + (0.5 * (len(sorted_target_fitness_values) - 1 - target_idx)) # Thicker lines for "harder" targets
            )
        config_idx += 1

    plt.title("Time-to-Target (Attainment) Plot", fontsize=15)
    plt.xlabel("Evaluations / Generations", fontsize=12)
    plt.ylabel("Proportion of Runs Reaching Target Fitness", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5)) # Place legend outside
    plt.xlim(0, max_evaluations_to_use -1 if max_evaluations_to_use > 1 else 1) # Adjust xlim
    plt.ylim(0, 1.05) # Proportion from 0 to 1
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside
    plt.show()


def plot_island_model_diagnostics(
    final_history_map: Dict[str, Dict[int, List[Dict[str, Any]]]],
    # config_col: str = 'Configuration' # Implicitly handled by final_history_map keys
    verbose: bool = True
):
    """
    Visualizes per-island convergence and diversity metrics for Island Model GA configurations.
    """
    if verbose:
        logger.info("\n--- Island Model GA Diagnostics ---")

    if not final_history_map:
        logger.warning("Island Model Plot: final_history_map is empty. Skipping.")
        return

    island_model_configs_found = False

    for config_name, runs_data in final_history_map.items():
        if not runs_data or not isinstance(runs_data, dict) or not runs_data.values():
            logger.debug(f"Island Model Plot: Config '{config_name}' has no runs_data or it's invalid. Skipping.")
            continue

        # Check if this config is an Island Model by inspecting history structure
        # Take the first run's history as representative to determine structure
        first_run_id = next(iter(runs_data.keys()), None)
        if first_run_id is None: continue # No runs in this config
        first_run_history = runs_data[first_run_id]
        
        if not first_run_history or not isinstance(first_run_history, list) or not first_run_history:
            logger.debug(f"Island Model Plot: Config '{config_name}' - first run history is empty/invalid. Skipping.")
            continue
        
        first_history_item = first_run_history[0]
        is_island_model_history = (
            isinstance(first_history_item, dict) and
            'islands_data' in first_history_item and 
            isinstance(first_history_item['islands_data'], list) and
            'global_best_fitness' in first_history_item 
        )

        if not is_island_model_history:
            logger.debug(f"Island Model Plot: Config '{config_name}' does not appear to be an Island Model. Skipping.") # Debug, not verbose controlled
            continue
        
        island_model_configs_found = True
        if verbose:
            logger.info(f"Processing Island Model diagnostics for configuration: {config_name}")

        num_islands = 0
        if first_history_item['islands_data']: # Check if islands_data is not empty
             num_islands = len(first_history_item['islands_data'])
        
        if num_islands == 0:
            logger.warning(f"Island Model Plot: Config '{config_name}' - 'islands_data' is empty or num_islands is 0. Cannot determine number of islands. Skipping.")
            continue

        # Data Extraction for this Configuration
        all_runs_global_best_fitness_traj: List[List[float]] = []
        # Correct initialization: List of lists of lists. Outer: island_idx, Middle: run_idx, Inner: trajectory values for that run for that island
        all_runs_island_best_fitness_traj: List[List[List[float]]] = [[] for _ in range(num_islands)]
        all_runs_island_std_fitness_traj: List[List[List[float]]] = [[] for _ in range(num_islands)]
        all_runs_island_geno_diversity_traj: List[List[List[float]]] = [[] for _ in range(num_islands)]
        max_len = 0

        for run_idx, (run_id, history_list_of_dicts) in enumerate(runs_data.items()):
            if not history_list_of_dicts or not isinstance(history_list_of_dicts, list):
                logger.debug(f"Island Model Plot: Config '{config_name}', Run {run_id}: History is empty/invalid. Skipping run.")
                continue
            
            current_run_length = len(history_list_of_dicts)
            max_len = max(max_len, current_run_length)
            
            global_best_run_traj = [item.get('global_best_fitness', np.nan) for item in history_list_of_dicts]
            all_runs_global_best_fitness_traj.append([x for x in global_best_run_traj if pd.notna(x)])

            # Initialize lists for this run's island trajectories
            current_run_island_best_fitness_temp: List[List[float]] = [[] for _ in range(num_islands)]
            current_run_island_std_fitness_temp: List[List[float]] = [[] for _ in range(num_islands)]
            current_run_island_geno_diversity_temp: List[List[float]] = [[] for _ in range(num_islands)]

            for gen_idx, gen_data in enumerate(history_list_of_dicts):
                if 'islands_data' in gen_data and isinstance(gen_data['islands_data'], list) and len(gen_data['islands_data']) == num_islands:
                    for island_idx, island_data in enumerate(gen_data['islands_data']):
                        if isinstance(island_data, dict):
                            current_run_island_best_fitness_temp[island_idx].append(island_data.get('best_fitness', np.nan))
                            current_run_island_std_fitness_temp[island_idx].append(island_data.get('std_fitness', np.nan))
                            current_run_island_geno_diversity_temp[island_idx].append(island_data.get('geno_diversity', np.nan))
                        else: # Fill with NaN if island_data is not a dict for this generation
                            current_run_island_best_fitness_temp[island_idx].append(np.nan)
                            current_run_island_std_fitness_temp[island_idx].append(np.nan)
                            current_run_island_geno_diversity_temp[island_idx].append(np.nan)
                else: # Fill with NaN if islands_data is missing or malformed for this generation
                     for island_idx in range(num_islands):
                        current_run_island_best_fitness_temp[island_idx].append(np.nan)
                        current_run_island_std_fitness_temp[island_idx].append(np.nan)
                        current_run_island_geno_diversity_temp[island_idx].append(np.nan)
            
            # Append this run's trajectories to the main list for each island
            for i in range(num_islands):
                all_runs_island_best_fitness_traj[i].append([x for x in current_run_island_best_fitness_temp[i] if pd.notna(x)])
                all_runs_island_std_fitness_traj[i].append([x for x in current_run_island_std_fitness_temp[i] if pd.notna(x)])
                all_runs_island_geno_diversity_traj[i].append([x for x in current_run_island_geno_diversity_temp[i] if pd.notna(x)])

        if max_len == 0:
            logger.warning(f"Island Model Plot: Config '{config_name}' - max_len is 0. No plottable data. Skipping config.")
            continue

        def pad_and_average_trajectories(list_of_run_trajectories: List[List[float]], current_max_len: int):
            padded_trajs = []
            for traj in list_of_run_trajectories:
                if not traj: continue 
                clean_traj = [x for x in traj if pd.notna(x)] 
                if not clean_traj: continue # Skip if trajectory becomes empty after NaN cleaning
                
                last_val = clean_traj[-1]
                padding_len = current_max_len - len(clean_traj)
                
                if padding_len < 0: # Trajectory is longer than max_len, truncate
                    final_traj = clean_traj[:current_max_len]
                else:
                    final_traj = clean_traj + [last_val] * padding_len
                
                if len(final_traj) == current_max_len : # Ensure correct length before appending
                    padded_trajs.append(final_traj)
                else:
                    logger.debug(f"Trajectory for {config_name} had unexpected length after padding attempt. Original len: {len(clean_traj)}, target_len: {current_max_len}, final_len: {len(final_traj)}. Skipping this trajectory.")

            if not padded_trajs: return np.array([]), np.array([]) 
            
            try:
                np_trajs = np.array(padded_trajs, dtype=float)
                # Check again if it's empty after array conversion (e.g. if all were skipped)
                if np_trajs.size == 0: return np.array([]), np.array([])
            except ValueError as e: 
                logger.error(f"Padding error for {config_name}: {e}. Trajectories might have inconsistent lengths despite padding attempts. Padded_trajs example: {padded_trajs[0] if padded_trajs else 'empty'}")
                return np.array([]), np.array([])

            return np.nanmean(np_trajs, axis=0), np.nanstd(np_trajs, axis=0)

        mean_global_best_fitness, std_global_best_fitness = pad_and_average_trajectories(all_runs_global_best_fitness_traj, max_len)
        
        island_metrics_processed = []
        for i in range(num_islands):
            mean_island_best, std_island_best = pad_and_average_trajectories(all_runs_island_best_fitness_traj[i], max_len)
            mean_island_std_fit, std_island_std_fit = pad_and_average_trajectories(all_runs_island_std_fitness_traj[i], max_len)
            mean_island_geno_div, std_island_geno_div = pad_and_average_trajectories(all_runs_island_geno_diversity_traj[i], max_len)
            island_metrics_processed.append({
                'best_fitness': (mean_island_best, std_island_best),
                'std_fitness': (mean_island_std_fit, std_island_std_fit),
                'geno_diversity': (mean_island_geno_div, std_island_geno_div),
            })

        # Plotting Per-Island and Global Convergence
        plt.figure(figsize=(12, 7)) 
        generations = np.arange(max_len)
        
        if mean_global_best_fitness.size > 0:
            plt.plot(generations, mean_global_best_fitness, label="Global Best Fitness (Mean)", color='black', lw=2.5, zorder=num_islands + 2)
            plt.fill_between(generations, mean_global_best_fitness - std_global_best_fitness, mean_global_best_fitness + std_global_best_fitness, color='black', alpha=0.2, zorder=num_islands+1)
        
        island_colors = sns.color_palette("husl", num_islands)
        for i in range(num_islands):
            mean_bf, std_bf = island_metrics_processed[i]['best_fitness']
            if mean_bf.size > 0:
                plt.plot(generations, mean_bf, label=f"Island {i} Best Fitness (Mean)", color=island_colors[i], lw=1.5, linestyle='--')

        plt.title(f"Per-Island & Global Convergence: {config_name}", fontsize=14)
        plt.xlabel("Generation", fontsize=11)
        plt.ylabel("Best Fitness", fontsize=11)
        plt.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5))
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        plt.show()

        # Plotting Per-Island Diversity Metrics
        if num_islands > 0:
            div_rows = (num_islands + 1) // 2 
            div_cols = 2 if num_islands > 1 else 1 
            
            fig_div, axes_div = plt.subplots(div_rows, div_cols, figsize=(7 * div_cols, 5 * div_rows), squeeze=False)
            ax_div_flat = axes_div.flatten()
            
            for i in range(num_islands):
                if i >= len(ax_div_flat): break 
                ax_d = ax_div_flat[i]
                lines_div = []

                mean_std_fit, std_std_fit = island_metrics_processed[i]['std_fitness']
                if mean_std_fit.size > 0:
                    line1, = ax_d.plot(generations, mean_std_fit, label="Phenotypic Diversity (Fitness StdDev)", color='dodgerblue', lw=2)
                    ax_d.fill_between(generations, mean_std_fit - std_std_fit, mean_std_fit + std_std_fit, color='dodgerblue', alpha=0.2)
                    lines_div.append(line1)
                ax_d.set_xlabel("Generation", fontsize=10)
                ax_d.set_ylabel("Phenotypic Diversity", fontsize=10, color='dodgerblue')
                ax_d.tick_params(axis='y', labelcolor='dodgerblue')

                ax_d_twin = ax_d.twinx()
                mean_geno_div, std_geno_div = island_metrics_processed[i]['geno_diversity']
                if mean_geno_div.size > 0:
                    line2, = ax_d_twin.plot(generations, mean_geno_div, label="Genotypic Diversity (Unique Solutions)", color='seagreen', lw=2, linestyle='--')
                    ax_d_twin.fill_between(generations, mean_geno_div - std_geno_div, mean_geno_div + std_geno_div, color='seagreen', alpha=0.2)
                    lines_div.append(line2)
                ax_d_twin.set_ylabel("Genotypic Diversity", fontsize=10, color='seagreen')
                ax_d_twin.tick_params(axis='y', labelcolor='seagreen')
                
                ax_d.set_title(f"Island {i} Diversity: {config_name}", fontsize=12)
                ax_d.grid(True, linestyle=":", alpha=0.5, axis='x')
                if lines_div:
                     ax_d.legend(handles=lines_div, loc='best', fontsize=8)

            for j in range(num_islands, len(ax_div_flat)): 
                fig_div.delaxes(ax_div_flat[j])
            
            fig_div.suptitle(f"Per-Island Diversity Metrics: {config_name}", fontsize=16, y=1.02 if div_rows > 1 else 1.04)
            fig_div.tight_layout(rect=[0, 0, 1, 0.97 if div_rows > 1 else 0.95]) 
            plt.show()

    if not island_model_configs_found:
        if verbose:
            logger.info("Island Model Plot: No Island Model configurations found in the provided history data.")


def plot_ga_diversity_metrics(
    final_history_map: Dict[str, Dict[int, List[Dict[str, Any]]]],
    # config_col: str = 'Configuration' # Not directly needed as keys are configs
    verbose: bool = True
):
    """
    Visualizes population diversity metrics (phenotypic and genotypic) over generations 
    for Genetic Algorithm configurations.
    """
    if verbose:
        logger.info("\n--- GA Population Diversity Metrics Over Generations ---")

    if not final_history_map:
        logger.warning("GA Diversity Plot: final_history_map is empty. Skipping plot.")
        return

    ga_configs_data = {} # To store processed data for GA configs only

    for config_name, runs_data in final_history_map.items():
        if not runs_data or not isinstance(runs_data, dict) or not runs_data.values():
            logger.debug(f"GA Diversity Plot: Config '{config_name}' has no runs_data or it's invalid. Skipping.") # Debug
            continue

        # Check if this config is likely a standard GA by inspecting the structure of its history
        # Take the first run's history as representative
        first_run_history = next(iter(runs_data.values()), None)
        if not first_run_history or not isinstance(first_run_history, list) or not first_run_history:
            logger.debug(f"GA Diversity Plot: Config '{config_name}' - first run history is empty/invalid. Skipping.") # Debug
            continue
        
        first_history_item = first_run_history[0]
        is_standard_ga_history = (
            isinstance(first_history_item, dict) and
            'avg_fitness' in first_history_item and # 'avg_fitness' is a key in standard GA history
            'std_fitness' in first_history_item and
            'geno_diversity' in first_history_item
        )

        if not is_standard_ga_history:
            logger.debug(f"GA Diversity Plot: Config '{config_name}' does not appear to be a standard GA. Skipping.") # Debug
            continue

        # This is likely a GA config, process its runs for diversity metrics
        all_runs_std_fitness_trajectories = []
        all_runs_geno_diversity_trajectories = []
        max_len = 0

        for run_id, history_list_of_dicts in runs_data.items():
            if not history_list_of_dicts or not isinstance(history_list_of_dicts, list):
                logger.debug(f"GA Diversity Plot: Config '{config_name}', Run {run_id}: History is empty/invalid. Skipping run.")
                continue

            current_run_std_fitness = [item.get('std_fitness', np.nan) for item in history_list_of_dicts]
            current_run_geno_diversity = [item.get('geno_diversity', np.nan) for item in history_list_of_dicts]
            
            # Filter out initial NaNs if any (e.g. if first history item was problematic, though .get handles it)
            current_run_std_fitness = [x for x in current_run_std_fitness if pd.notna(x)]
            current_run_geno_diversity = [x for x in current_run_geno_diversity if pd.notna(x)]

            if current_run_std_fitness: # Check if list is not empty after potential NaN removal
                all_runs_std_fitness_trajectories.append(current_run_std_fitness)
                max_len = max(max_len, len(current_run_std_fitness))
            if current_run_geno_diversity:
                all_runs_geno_diversity_trajectories.append(current_run_geno_diversity)
                max_len = max(max_len, len(current_run_geno_diversity)) # max_len should be consistent

        if not all_runs_std_fitness_trajectories and not all_runs_geno_diversity_trajectories:
            if verbose:
                logger.info(f"GA Diversity Plot: Config '{config_name}' - no valid diversity trajectories found.")
            continue # Skip this config if no data

        # Pad trajectories
        padded_std_fitness = []
        for traj in all_runs_std_fitness_trajectories:
            last_val = traj[-1] if traj else np.nan
            padding = [last_val] * (max_len - len(traj))
            padded_std_fitness.append(traj + padding)

        padded_geno_diversity = []
        for traj in all_runs_geno_diversity_trajectories:
            last_val = traj[-1] if traj else np.nan
            padding = [last_val] * (max_len - len(traj))
            padded_geno_diversity.append(traj + padding)
            
        ga_configs_data[config_name] = {
            'std_fitness_np': np.array(padded_std_fitness, dtype=float) if padded_std_fitness else np.array([]),
            'geno_diversity_np': np.array(padded_geno_diversity, dtype=float) if padded_geno_diversity else np.array([]),
            'max_len': max_len,
            'num_runs': len(runs_data) # Number of runs that contributed some data
        }

    if not ga_configs_data:
        if verbose:
            logger.info("GA Diversity Plot: No GA configurations with valid diversity data found. Skipping plot generation.")
        return

    num_ga_configs = len(ga_configs_data)
    rows = (num_ga_configs + 1) // 2 
    cols = 2
    if num_ga_configs == 1: rows, cols = 1,1
    
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
    ax_flat = axes.flatten()
    plot_idx = 0

    for config_name, data in ga_configs_data.items():
        if plot_idx >= len(ax_flat): break
        ax = ax_flat[plot_idx]
        
        max_len = data['max_len']
        if max_len == 0: # Should have been caught by earlier check, but safeguard
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Diversity: {config_name}\n(No Data)", fontsize=12)
            plot_idx += 1
            continue
            
        generations = np.arange(max_len)
        lines = [] # For collecting lines for the legend
        
        # Plot Phenotypic Diversity (Fitness StdDev)
        if data['std_fitness_np'].size > 0:
            mean_std_fitness = np.nanmean(data['std_fitness_np'], axis=0)
            std_dev_std_fitness = np.nanstd(data['std_fitness_np'], axis=0)
            line1, = ax.plot(generations, mean_std_fitness, label=f"Phenotypic Diversity (Mean Fitness StdDev, {data['num_runs']} runs)", color='dodgerblue', lw=2)
            ax.fill_between(generations, mean_std_fitness - std_dev_std_fitness, mean_std_fitness + std_dev_std_fitness, color='dodgerblue', alpha=0.2)
            lines.append(line1)
        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel("Phenotypic Diversity (Fitness StdDev)", fontsize=10, color='dodgerblue')
        ax.tick_params(axis='y', labelcolor='dodgerblue')

        # Create a twin Y-axis for Genotypic Diversity
        ax_twin = ax.twinx()
        if data['geno_diversity_np'].size > 0:
            mean_geno_diversity = np.nanmean(data['geno_diversity_np'], axis=0)
            std_dev_geno_diversity = np.nanstd(data['geno_diversity_np'], axis=0)
            line2, = ax_twin.plot(generations, mean_geno_diversity, label=f"Genotypic Diversity (Mean Unique Solutions, {data['num_runs']} runs)", color='seagreen', lw=2, linestyle='--')
            ax_twin.fill_between(generations, mean_geno_diversity - std_dev_geno_diversity, mean_geno_diversity + std_dev_geno_diversity, color='seagreen', alpha=0.2)
            lines.append(line2)
        ax_twin.set_ylabel("Genotypic Diversity (Unique Solutions)", fontsize=10, color='seagreen')
        ax_twin.tick_params(axis='y', labelcolor='seagreen')

        ax.set_title(f"Diversity Over Generations: {config_name}", fontsize=13)
        ax.grid(True, linestyle=":", alpha=0.5, axis='x') # Grid for x-axis from primary
        
        # Combined legend
        if lines:
            # ax.legend(handles=lines, loc='best', fontsize=9) # This places legend inside
            # For twin axes, fig.legend() is often better if placing outside or in a shared spot
             fig.legend(lines, [l.get_label() for l in lines], loc='upper center', 
                        bbox_to_anchor=(0.5, 0.02 + (0.98 * (1-(plot_idx//cols)/rows))), # Adjust y based on row to avoid overlap
                        ncol=2, fontsize=9)


        plot_idx += 1

    # Hide any unused subplots
    for i in range(plot_idx, len(ax_flat)):
        fig.delaxes(ax_flat[i])

    fig.suptitle("GA Population Diversity Metrics", fontsize=16, y=1.03 if rows >1 else 1.0) # Adjust y for suptitle
    fig.tight_layout(rect=[0, 0.05, 1, 0.98 if rows > 1 else 0.95]) # Adjust rect for suptitle and fig.legend
    plt.show()


def plot_critical_difference_diagram(
    final_results_df: pd.DataFrame, 
    metric_col: str = 'BestFitness', 
    config_col: str = 'Configuration', 
    run_col: str = 'Run', 
    alpha_level: float = 0.05,
    verbose: bool = True
):
    """
    Generates and displays a Critical Difference (CD) diagram if results are significant.
    Uses Friedman test and Nemenyi post-hoc results.
    """
    if verbose:
        logger.info(f"\n--- Critical Difference Analysis for Metric: {metric_col} ---")

    if final_results_df.empty:
        logger.warning("CD Diagram: final_results_df is empty. Skipping.")
        return

    friedman_results = su.perform_friedman_test_and_posthoc(
        final_results_df, 
        metric_col=metric_col, 
        config_col=config_col, 
        run_col=run_col, 
        alpha=alpha_level
    )

    if friedman_results is None:
        logger.error("CD Diagram: Friedman test and rank calculation failed or returned None. Cannot proceed.")
        return

    friedman_stat, friedman_p_value, nemenyi_results_df, avg_ranks = friedman_results

    if friedman_stat is None or friedman_p_value is None:
        logger.warning(f"CD Diagram: Friedman test did not yield valid statistic or p-value (Stat: {friedman_stat}, P-val: {friedman_p_value}). Average ranks might still be available but CD plot cannot be generated meaningfully without a test outcome.")
        # The following log about avg_ranks is already conditional on verbose from previous changes.
        if avg_ranks is not None and verbose:
            logger.info(f"Average Ranks for {metric_col}:\n{avg_ranks.sort_values()}")
        return

    if verbose: # This is already correctly conditional
        logger.info(f"Friedman Test for {metric_col}: Statistic={friedman_stat:.3f}, P-value={friedman_p_value:.4f}")

    if avg_ranks is None or avg_ranks.empty: # This is a warning, should be unconditional
        logger.warning("CD Diagram: Average ranks are not available. Cannot plot CD diagram.")
        return
        
    # Sort average ranks for better visualization in CD diagram (lower rank is better)
    # scikit-posthocs CD diagram typically expects ranks where lower is better.
    # Our avg_ranks from Friedman are already like this (lower metric value -> lower rank number).
    sorted_avg_ranks = avg_ranks.sort_values(ascending=True)

    if friedman_p_value >= alpha_level:
        if verbose: # This block is correctly conditional
            logger.info(f"Friedman test for {metric_col} is not significant (p={friedman_p_value:.4f} >= {alpha_level}). "
                        "A CD diagram is not typically shown as no overall statistical difference was detected.")
            logger.info(f"Average Ranks for {metric_col} (lower is better):\n{sorted_avg_ranks}")
        # Optionally, still show a plot of average ranks without significance bars if desired
        # For now, returning as per instructions if not significant.
        return

    if nemenyi_results_df is None or nemenyi_results_df.empty: # This is a warning, should be unconditional
        logger.warning(f"CD Diagram: Friedman test was significant for {metric_col}, but Nemenyi post-hoc results are unavailable or empty. "
                       "Cannot plot CD diagram. Check Nemenyi test execution.")
        if verbose: # This part is correctly conditional
            logger.info(f"Average Ranks for {metric_col} (lower is better):\n{sorted_avg_ranks}")
        return

    # Attempt to plot Critical Difference Diagram using scikit-posthocs
    # scikit-posthocs' critical_difference_diagram function is not standard.
    # The Nemenyi test itself gives p-values. We need to find a way to plot these.
    # A common way is to use the Orange library's CD diagram plotting or a custom implementation.
    # scikit-posthocs' `sign_plot` is a more readily available alternative for visualizing Nemenyi results.

    try:
        if verbose:
            logger.info(f"Attempting to generate significance plot (e.g., sign_plot) for {metric_col} based on Nemenyi results.")
        
        # The sign_plot typically takes the p-value matrix (nemenyi_results_df)
        # and average ranks (sorted_avg_ranks) can be used to order/interpret it.
        
        # Check if nemenyi_results_df is a square matrix of p-values as expected by sign_plot
        if not (isinstance(nemenyi_results_df, pd.DataFrame) and nemenyi_results_df.shape[0] == nemenyi_results_df.shape[1]):
             logger.error(f"CD Diagram: Nemenyi results are not in the expected square DataFrame format. Shape: {nemenyi_results_df.shape if isinstance(nemenyi_results_df, pd.DataFrame) else 'Not a DataFrame'}. Cannot use sign_plot.")
             logger.info(f"Average Ranks for {metric_col}:\n{sorted_avg_ranks}")
             logger.info(f"Nemenyi p-values (raw table):\n{nemenyi_results_df}")
             return

        # Sort avg_ranks and reorder Nemenyi table columns/index to match for sign_plot if necessary
        # sign_plot itself might handle ordering if ranks are passed, or it uses the order from the p-value df.
        # Let's ensure the Nemenyi table is ordered by the average ranks for consistent plotting.
        ordered_config_names = sorted_avg_ranks.index.tolist()
        nemenyi_ordered = nemenyi_results_df.loc[ordered_config_names, ordered_config_names]

        plt.figure(figsize=(max(8, len(sorted_avg_ranks) * 0.8), max(6, len(sorted_avg_ranks) * 0.5)))
        
        # Using sp.sign_plot as the primary visualization for Nemenyi results
        # This function creates a "significance plot" or "heatmap" of p-values.
        # It does not draw the traditional CD diagram with a number line and bars.
        sp.sign_plot(nemenyi_ordered, standardized=False) # standardized=False shows p-values
        
        # Adaptive title font size
        title_font_size = 14
        num_configs = len(sorted_avg_ranks)
        if num_configs > 15: # Example threshold: more than 15 configs
            title_font_size = 11
        elif num_configs > 10: # Example threshold: more than 10 configs
            title_font_size = 12
            
        plt.title(f"Pairwise Significance (Nemenyi) for {metric_col}\n(Lower ranks are better; cell values are p-values)", fontsize=title_font_size)
        # Rotate x-axis labels for better readability if many configs
        plt.xticks(rotation=45, ha='right') 
        plt.yticks(rotation=0) # Ensure y-axis labels are horizontal
        plt.tight_layout()
        plt.show()
        
        if verbose: # These logs are now conditional
            logger.info(f"Displayed Nemenyi significance plot (sign_plot) for {metric_col}.")
            logger.info(f"Average Ranks for {metric_col} (lower is better):\n{sorted_avg_ranks}")
        
        # Note: A true CD diagram often requires a specific CD value and plots ranks on a line.
        # `scikit-posthocs` does not have a direct `critical_difference_diagram` function like some other libraries (e.g., Orange).
        # The `sign_plot` is the closest utility within `scikit-posthocs` for visualizing these pairwise comparisons.
        # For a traditional CD diagram, one might need to:
        # 1. Calculate the Critical Difference value (e.g., using q_alpha for Nemenyi).
        # 2. Plot average ranks on a number line.
        # 3. Draw bars connecting groups whose difference in average rank is less than the CD.
        # This manual construction is beyond the direct capabilities of `scikit-posthocs` plotting utilities.

    except Exception as e_plot: # This is an error, should be unconditional
        logger.error(f"CD Diagram: Failed to generate significance plot for {metric_col}: {e_plot}", exc_info=True)
        if verbose: # Fallback logging is conditional
            logger.info("Fallback: Logging average ranks and Nemenyi p-values table.")
            if avg_ranks is not None: # avg_ranks check is good, verbose already applied
                logger.info(f"Average Ranks for {metric_col} (lower is better):\n{avg_ranks.sort_values()}")
            if nemenyi_results_df is not None: # nemenyi_results_df check is good, verbose already applied
                logger.info(f"Nemenyi Post-Hoc p-values for {metric_col}:\n{nemenyi_results_df}")



def rank_algorithms_custom(results_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame or None:
    if results_df.empty:
        logger.warning("Ranking: Input DataFrame empty.")
        return None
    if verbose:
        logger.info("\n--- Algorithm Performance Ranking ---")

    valid_df = results_df[  # Ensure we only use valid, finite results for ranking
        results_df["BestFitness"].notna()
        & np.isfinite(results_df["BestFitness"])
        & results_df["RuntimeSeconds"].notna()
        & np.isfinite(results_df["RuntimeSeconds"])
        & results_df["FunctionEvaluations"].notna()
        & np.isfinite(results_df["FunctionEvaluations"])
        & (~results_df["AlgorithmName"].str.contains("ERROR", na=False))
    ]
    if valid_df.empty:
        logger.warning("Ranking: No valid data after filtering.")
        return None

    means = valid_df.groupby("Configuration").agg(
        MeanBestFitness=("BestFitness", "mean"),
        MeanRuntime=("RuntimeSeconds", "mean"),
        MeanEvals=("FunctionEvaluations", "mean"),
    )
    stds = (
        valid_df.groupby("Configuration")["BestFitness"]
        .std()
        .rename("StdBestFitness")
        .fillna(0.0)
    )
    metrics = pd.concat([means, stds], axis=1)
    metrics["ConsistencyScore"] = 1 / (metrics["StdBestFitness"] + 1e-9)

    ranks = pd.DataFrame(index=metrics.index)
    ranks["FitnessRank"] = metrics["MeanBestFitness"].rank(method="min", ascending=True)
    ranks["RuntimeRank"] = metrics["MeanRuntime"].rank(method="min", ascending=True)
    ranks["EvalsRank"] = metrics["MeanEvals"].rank(method="min", ascending=True)
    ranks["ConsistencyRank"] = metrics["ConsistencyScore"].rank(
        method="min", ascending=False
    )

    rank_cols = ["FitnessRank", "RuntimeRank", "EvalsRank", "ConsistencyRank"]
    ranks["OverallRankScore"] = ranks[rank_cols].sum(axis=1)
    ranks["OverallRank"] = ranks["OverallRankScore"].rank(method="min", ascending=True)

    final_ranks = pd.concat([ranks, metrics], axis=1).sort_values("OverallRank")
    if verbose:
        logger.info("\nAlgorithm Ranking Table (Lower OverallRank is Better):")

    display_cols = (
        ["OverallRank"]
        + rank_cols
        + [
            "MeanBestFitness",
            "StdBestFitness",
            "MeanRuntime",
            "MeanEvals",
            "ConsistencyScore",
        ]
    )
    existing_cols = [c for c in display_cols if c in final_ranks.columns]

    fmts = {c: "{:.1f}" for c in rank_cols + ["OverallRank", "OverallRankScore"]}
    fmts.update(
        {
            "MeanBestFitness": "{:.4f}",
            "StdBestFitness": "{:.4f}",
            "MeanRuntime": "{:.3f}",
            "MeanEvals": "{:.1f}",
            "ConsistencyScore": "{:.2f}",
        }
    )
    active_fmts = {k: v for k, v in fmts.items() if k in existing_cols}

    try:
        from IPython.display import display # type: ignore
        if verbose: # Assuming display is for verbose output
            display(final_ranks[existing_cols].style.format(active_fmts))
    except ImportError:
        if verbose:
            call_fmts = {k: (lambda x, f=v: f.format(x)) for k, v in active_fmts.items()}
            print(final_ranks[existing_cols].to_string(formatters=call_fmts))

    if "OverallRank" in final_ranks.columns:
        num_configs = len(final_ranks)
        plt.figure(figsize=(max(10, num_configs * 0.5), 6)) # Dynamic figure size
        final_ranks["OverallRank"].sort_values().plot(
            kind="bar", color=sns.color_palette("YlOrRd_r", len(final_ranks))
        )
        plt.title("Overall Algorithm Ranking", fontsize=14)
        plt.ylabel("Overall Rank", fontsize=11)
        plt.xlabel("Configuration", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    return final_ranks


def display_single_best_solution_details(
    best_sol_data: dict,
    config_name: str,
    fitness: float,
    run_num: int,
    players_list: list,
    problem_def: dict,
    sol_module,  # Pass the imported 'solution' module (e.g., import solution as sol_module)
    verbose: bool = True
):
    """
    Displays detailed statistics and visualizations for the single overall best solution.
    This function is intended to be called from the main experiment script.
    """
    if not best_sol_data:
        if verbose:
            logger.info(
                "No best solution data provided to display_single_best_solution_details."
            )
        return

    if verbose:
        logger.info(
            f"\n--- Detailed Analysis: Best Solution from '{config_name}', Run {run_num} (Fitness: {fitness:.4f}) ---"
        )

    # Recreate the solution object using the utility function (also in this file or imported)
    # and the passed solution module reference.
    best_obj = recreate_solution_from_data(  # Assumes recreate_solution_from_data is defined in this utils file
        best_sol_data, players_list, problem_def, solution_classes_module=sol_module
    )

    if not best_obj:
        logger.error( # Error, not verbose controlled
            "Could not recreate the best solution object for detailed display. Aborting details."
        )
        return

    # Log validity and re-calculated fitness of the recreated object
    is_recreated_valid = best_obj.is_valid()  # This call uses the original .is_valid()
    recalculated_fitness_val = (
        best_obj.fitness()
    )  # This call uses the original .fitness()
    if verbose: # This is informational about the recreated object
        logger.info(
            f"Recreated Best Solution: Is Valid = {is_recreated_valid}, Recalculated Fitness = {recalculated_fitness_val:.4f}"
        )
    if not is_recreated_valid: # This is a warning, always show
        logger.warning(
            "The recreated overall best solution is flagged as INVALID. Displayed statistics might be misleading or incomplete."
        )

    # Get detailed team statistics from the solution object
    team_stats_list = best_obj.get_team_stats()
    if not team_stats_list:
        logger.warning( # Warning, always show
            "No team statistics could be generated for the overall best solution."
        )
        return

    if verbose:
        logger.info("\n--- Detailed Team Breakdown for Overall Best Solution ---")
        for team_stat_item in team_stats_list:
            # Using print for potentially better multi-line formatting in console for these details
            print(f"\nTeam {team_stat_item.get('team_id', 'N/A')}:")
            print(
                f"  Players ({team_stat_item.get('num_players',0)}): {team_stat_item.get('players_names', [])}"
            )
            print(f"  Avg Skill: {team_stat_item.get('avg_skill', 0.0):.2f}")
            print(
                f"  Total Salary: {team_stat_item.get('total_salary', 0.0):.2f} (Budget Limit: {problem_def.get('max_budget', 'N/A')})"
            )
            print(
                f"  Actual Positions: {team_stat_item.get('positions', {})} (Required: {problem_def.get('position_requirements', {})})"
            )

            # Inline validity checks for quick visual reference during display
            if team_stat_item.get("total_salary", 0.0) > problem_def.get(
                "max_budget", float("inf")
            ):
                print("    🔴 WARNING: TEAM BUDGET EXCEEDED!") # These warnings within verbose block are fine as they relate to printed details
            if team_stat_item.get("positions", {}) != problem_def.get(
                "position_requirements", {}
            ):
                print("    🔴 WARNING: TEAM POSITIONAL MISMATCH!")
            if team_stat_item.get("num_players", 0) != problem_def.get(
                "team_size", -1
            ):  # Compare with problem_def team_size
                print(
                    f"    🔴 WARNING: TEAM SIZE IS {team_stat_item.get('num_players',0)}, "
                    f"EXPECTED {problem_def.get('team_size',-1)}!"
                )

    # Create a DataFrame from the team statistics for easier plotting
    team_stats_df_best_plot = pd.DataFrame(team_stats_list)
    if team_stats_df_best_plot.empty:
        logger.warning(
            "Team statistics DataFrame for the best solution is empty. Skipping detailed visualizations."
        )
        return

    # Ensure 'team_id' is numeric for plotting to avoid categorical unit warnings from Matplotlib
    if "team_id" in team_stats_df_best_plot.columns:
        team_stats_df_best_plot["team_id"] = pd.to_numeric(
            team_stats_df_best_plot["team_id"], errors="coerce"
        ).fillna(
            -1
        )  # Coerce and fill to handle issues

    # --- Visualize Team Skill and Salary Balance for the Best Solution ---
    required_balance_cols = {"avg_skill", "total_salary", "team_id"}
    if required_balance_cols.issubset(team_stats_df_best_plot.columns):
        fig_balance_best, (ax_skill_plot, ax_salary_plot) = plt.subplots(
            1, 2, figsize=(16, 5.5)
        )  # Adjusted size

        # Average Skill by Team Plot
        sns.barplot(
            x="team_id",
            y="avg_skill",
            data=team_stats_df_best_plot,
            ax=ax_skill_plot,
            palette="coolwarm_r",
            hue="team_id",
            legend=False,
        )  # Added hue and legend=False
        ax_skill_plot.set_title("Average Skill by Team", fontsize=13)
        ax_skill_plot.set_xlabel("Team ID", fontsize=11)
        ax_skill_plot.set_ylabel("Average Skill", fontsize=11)
        ax_skill_plot.grid(axis="y", linestyle=":", alpha=0.7)
        ax_skill_plot.tick_params(axis="both", which="major", labelsize=9)

        # Total Salary by Team Plot
        sns.barplot(
            x="team_id",
            y="total_salary",
            data=team_stats_df_best_plot,
            ax=ax_salary_plot,
            palette="viridis_r",
            hue="team_id",
            legend=False,
        )  # Added hue and legend=False
        # Use label for axhline for the legend
        ax_salary_plot.axhline(
            problem_def.get("max_budget", float("inf")),
            color="red",
            linestyle="--",
            label=f"Max Budget ({problem_def.get('max_budget', 'N/A'):.0f})",
        )
        ax_salary_plot.legend(fontsize=9)
        ax_salary_plot.set_title("Total Salary by Team", fontsize=13)
        ax_salary_plot.set_xlabel("Team ID", fontsize=11)
        ax_salary_plot.set_ylabel("Total Salary", fontsize=11)
        ax_salary_plot.grid(axis="y", linestyle=":", alpha=0.7)
        ax_salary_plot.tick_params(axis="both", which="major", labelsize=9)

        fig_balance_best.suptitle(
            f"Team Balance Analysis for Best Solution ({config_name} - Fitness: {fitness:.4f})",
            fontsize=16,
        )
        fig_balance_best.tight_layout(
            rect=[0, 0.03, 1, 0.94]
        )  # Adjust rect for suptitle
        plt.show()
    else:
        logger.warning(
            "Missing required columns ('team_id', 'avg_skill', 'total_salary') "
            "for skill/salary balance plot of the best solution."
        )

    # --- Visualize Position Distribution for the Best Solution (Corrected Melt Logic) ---
    position_keys_expected = list(problem_def.get("position_requirements", {}).keys())

    if (
        "positions" in team_stats_df_best_plot.columns
        and not team_stats_df_best_plot["positions"].empty
    ):
        try:
            # Expand the 'positions' dictionary column (which contains dicts) into separate columns
            positions_as_df_expanded = pd.json_normalize(
                team_stats_df_best_plot["positions"]
            )

            # Ensure 'team_id' column exists for merging/melting.
            if "team_id" not in team_stats_df_best_plot.columns:
                logger.error(
                    "CRITICAL for position plot: 'team_id' column is missing in team_stats_df_best_plot."
                )
                raise KeyError("'team_id' not found in DataFrame for position plot.")

            # Concatenate the new position columns with 'team_id' from the original DataFrame
            # Ensure indices are aligned for a clean concat if they were somehow misaligned.
            df_ready_for_position_melt = pd.concat(
                [
                    team_stats_df_best_plot[["team_id"]].reset_index(drop=True),
                    positions_as_df_expanded.reset_index(drop=True),
                ],
                axis=1,
            )

            # Determine which of the required position columns actually exist after normalization
            # (in case player data or position_requirements led to some positions not appearing)
            actual_position_columns_present = [
                col
                for col in position_keys_expected
                if col in df_ready_for_position_melt.columns
            ]

            if not actual_position_columns_present:
                logger.warning(
                    f"No required position columns ({position_keys_expected}) found in DataFrame "
                    f"after normalization. Skipping position distribution plot."
                )
            else:
                if len(actual_position_columns_present) < len(position_keys_expected):
                    logger.warning(
                        f"Only found columns {actual_position_columns_present} for position plot, "
                        f"expected all of {position_keys_expected}. Plotting with available columns."
                    )

                melted_position_data_best = df_ready_for_position_melt.melt(
                    id_vars=["team_id"],
                    value_vars=actual_position_columns_present,  # Use only columns that actually exist
                    var_name="PositionType",  # Renamed from 'Position' to avoid potential conflict
                    value_name="PlayerCount",
                )

                plt.figure(figsize=(12, 6.5))  # Adjusted size
                sns.barplot(
                    x="team_id",
                    y="PlayerCount",
                    hue="PositionType",
                    data=melted_position_data_best,
                    palette="Set2",
                )
                plt.title(
                    f"Position Distribution by Team ({config_name} - Best Solution)",
                    fontsize=14,
                )
                plt.xlabel("Team ID", fontsize=11)
                plt.ylabel("Number of Players", fontsize=11)
                plt.legend(title="Position", fontsize=9, title_fontsize="10")
                plt.grid(axis="y", linestyle=":", alpha=0.6)
                plt.tight_layout()
                plt.show()
        except KeyError as e_key_error_melt:
            logger.error(
                f"KeyError during position distribution plot generation (melt step): {e_key_error_melt}. "
                "This might be due to 'team_id' or expected position columns being missing after normalization."
            )
        except (
            Exception
        ) as e_position_plot_general:  # Catch any other unexpected errors
            logger.error(
                f"An unexpected error occurred during position distribution plot generation: {e_position_plot_general}",
                exc_info=True,
            )  # Log with traceback for more details
    else:
        logger.warning(
            "Could not generate position distribution plot: 'positions' data column missing or empty in team_stats_df_best_plot."
        )

        if verbose: # End of analysis message
            logger.info(
                f"--- End of Detailed Analysis for Overall Best Solution ({config_name}) ---"
            )


def display_best_solution_per_configuration_table(
    best_solutions_map: Dict[str, Dict[str, Any]],
    master_players_list: List[Dict],
    problem_def: Dict[str, Any],
    sol_module: Any, # The imported solution module
    top_n: Optional[int] = None,
    verbose: bool = True
) -> None:
    """
    Displays a table of the best solution found by each algorithm configuration.
    """
    if verbose:
        logger.info("\n--- Best Solution Found by Each Algorithm Configuration ---")

    if not best_solutions_map:
        if verbose: # Only log this if verbose, it's not a warning, just info.
            logger.info("No best solutions data available to display.")
        return

    table_data = []
    for config_name, data_item in best_solutions_map.items():
        solution_data = data_item.get('solution_data')
        fitness = data_item.get('fitness', float('inf'))
        run_number = data_item.get('run_number', -1)
        solution_repr_str = "N/A"

        if solution_data:
            solution_obj = recreate_solution_from_data(
                stored_solution_data=solution_data,
                master_players_list_for_recreation=master_players_list,
                problem_definition_for_recreation=problem_def,
                solution_classes_module=sol_module
            )
            if solution_obj:
                try:
                    # Attempt to use a more sophisticated representation if available
                    solution_repr_str = solution_obj.repr_to_string(max_teams_to_display=1)
                except AttributeError:
                    # Fallback to simpler representation
                    num_teams_to_show = 1
                    players_per_team = problem_def.get('team_size', 5) # Default to 5 if not in problem_def
                    max_players_preview = num_teams_to_show * players_per_team
                    
                    if hasattr(solution_obj, 'repr') and isinstance(solution_obj.repr, list):
                        solution_repr_str = str(solution_obj.repr[:max_players_preview])
                        if len(solution_obj.repr) > max_players_preview:
                            solution_repr_str += "..."
                    else:
                        solution_repr_str = "Representation not available"
                except TypeError: # If repr_to_string exists but does not take max_teams_to_display
                    try:
                        solution_repr_str = solution_obj.repr_to_string() # Try calling without argument
                        # Optionally truncate this string if it's too long
                        if len(solution_repr_str) > 100: # Example length limit
                             solution_repr_str = solution_repr_str[:97] + "..."
                    except Exception: # Fallback if repr_to_string fails without arg either
                        num_teams_to_show = 1
                        players_per_team = problem_def.get('team_size', 5) 
                        max_players_preview = num_teams_to_show * players_per_team
                        if hasattr(solution_obj, 'repr') and isinstance(solution_obj.repr, list):
                            solution_repr_str = str(solution_obj.repr[:max_players_preview])
                            if len(solution_obj.repr) > max_players_preview:
                                solution_repr_str += "..."
                        else:
                            solution_repr_str = "Representation not available"


            else:
                logger.warning(f"Could not recreate solution object for configuration '{config_name}'.")
                solution_repr_str = "Recreation Failed"
        else:
            logger.warning(f"No solution data found for configuration '{config_name}'.")

        table_data.append({
            'Configuration': config_name,
            'Best Fitness': fitness,
            'Run Number': run_number,
            'Solution Representation (Preview)': solution_repr_str
        })

    if not table_data:
        if verbose: # Informational
            logger.info("No solution data processed for the table.")
        return

    df = pd.DataFrame(table_data)
    df.sort_values(by='Best Fitness', ascending=True, inplace=True)

    if top_n is not None:
        df = df.head(top_n)

    # Configure pandas display options
    original_width = pd.get_option('display.width')
    original_max_colwidth = pd.get_option('display.max_colwidth')
    
    pd.set_option('display.width', 1000) # Adjust as needed, might be too wide for some consoles
    pd.set_option('display.max_colwidth', 200) # Show more of the solution representation

    if verbose:
        logger.info("Best solutions per configuration table:")
        if is_notebook() and ipython_available:
            # escape=True is important for security if data might contain HTML/JS
            display(HTML(df.to_html(index=False, escape=True)))
        else:
            # Use print for better table formatting in typical console outputs than logger.info
            print(df.to_string(index=False))

    # Reset pandas display options to original
    pd.set_option('display.width', original_width)
    pd.set_option('display.max_colwidth', original_max_colwidth)


def plot_overall_best_fitness_convergence(
    history_data_dict: Dict[str, Dict[int, List[Dict[str, Any]]]],
    metric_to_plot_vs: str = 'Evaluations', # 'Evaluations' or 'RuntimeSeconds'
    verbose: bool = True
) -> None:
    """
    Plots the overall best fitness convergence for each configuration, 
    showing the best fitness found up to a certain point (evaluations, runtime, or step).
    This plot aggregates data from all runs of a configuration.
    """
    if not verbose:
        # This function relies on display_plot_info being called by the caller
        # for verbose descriptions. Direct calls might not have that context.
        # logger.info("Plotting overall best fitness convergence (verbose mode off).")
        pass

    if not history_data_dict:
        if verbose:
            logger.warning("Overall Best Fitness Convergence Plot: History data dictionary is empty. Skipping plot.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Define potential keys for fitness and x-axis metrics
    fitness_keys = ['global_best_fitness', 'best_fitness']
    eval_keys = ['cumulative_evaluations', 'FunctionEvaluations', 'evaluations']
    runtime_keys = ['cumulative_runtime', 'RuntimeSeconds', 'runtime']

    actual_x_axis_label = "Step/Generation" # Default
    using_index_fallback = False

    for config_name, runs_data in history_data_dict.items():
        if not runs_data or not isinstance(runs_data, dict):
            if verbose:
                logger.debug(f"Overall Best Fitness Plot: Config '{config_name}' has no runs_data or it's invalid. Skipping.")
            continue

        all_points_for_config: List[Tuple[float, float]] = []

        for run_id, history_list in runs_data.items():
            if not history_list or not isinstance(history_list, list):
                if verbose:
                    logger.debug(f"Overall Best Fitness Plot: Config '{config_name}', Run {run_id}: History is empty/invalid. Skipping run.")
                continue

            cumulative_x_offset = 0 # Used if actual x-values are per-step and need accumulation (less common for this plot type)
                                    # Primarily, we expect cumulative values directly in history or use index.

            for idx, step_data in enumerate(history_list):
                if not isinstance(step_data, dict):
                    if verbose:
                        logger.warning(f"Overall Best Fitness Plot: Config '{config_name}', Run {run_id}, Step {idx}: step_data is not a dict. Skipping step.")
                    continue

                fitness_value = None
                for fk in fitness_keys:
                    if fk in step_data and pd.notna(step_data[fk]):
                        fitness_value = step_data[fk]
                        break
                
                if fitness_value is None:
                    if verbose:
                         logger.debug(f"Overall Best Fitness Plot: Config '{config_name}', Run {run_id}, Step {idx}: No valid fitness key found. Skipping step.")
                    continue

                x_value = float(idx) # Default to index
                current_x_axis_label_candidate = "Step/Generation"
                
                found_specific_x_key = False
                if metric_to_plot_vs == 'Evaluations':
                    current_x_axis_label_candidate = "Evaluations"
                    for xk in eval_keys:
                        if xk in step_data and pd.notna(step_data[xk]):
                            x_value = float(step_data[xk])
                            found_specific_x_key = True
                            break
                elif metric_to_plot_vs == 'RuntimeSeconds':
                    current_x_axis_label_candidate = "Runtime (seconds)"
                    for xk in runtime_keys:
                        if xk in step_data and pd.notna(step_data[xk]):
                            x_value = float(step_data[xk])
                            found_specific_x_key = True
                            break
                
                if idx == 0 and not using_index_fallback: # Set label based on first valid data point
                    actual_x_axis_label = current_x_axis_label_candidate

                if metric_to_plot_vs in ['Evaluations', 'RuntimeSeconds'] and not found_specific_x_key:
                    if idx == 0 and run_id == next(iter(runs_data.keys())): # Log warning only once per config
                         if verbose:
                            logger.warning(f"Overall Best Fitness Plot: Config '{config_name}': X-axis key for '{metric_to_plot_vs}' not found in history items. Defaulting to step index for x-axis.")
                    using_index_fallback = True # If any config defaults to index, label should reflect that
                    actual_x_axis_label = "Step/Generation" # Global fallback label if any config uses index
                
                all_points_for_config.append((x_value, float(fitness_value)))

        if not all_points_for_config:
            if verbose:
                logger.info(f"Overall Best Fitness Plot: Config '{config_name}': No plottable points after processing all runs.")
            continue

        # Sort all collected points by x_value
        all_points_for_config.sort(key=lambda p: p[0])

        # Compute the "best-so-far" trajectory from the aggregated points
        best_so_far_trajectory: List[Tuple[float, float]] = []
        current_min_fitness = float('inf')
        if all_points_for_config:
            # Add a point at x=0 if not present, using the first known best fitness
            # This ensures plots start from a common visual origin if x-values don't start at 0.
            # However, for 'steps-post', the visual start is handled by the first point itself.
            # If x-values can be non-zero and we want to show a line from 0, more complex logic is needed.
            # For now, assume x-values capture the start correctly or index starts at 0.

            for x_val, fit_val in all_points_for_config:
                current_min_fitness = min(current_min_fitness, fit_val)
                # Avoid adding duplicate x-values with only updated y if the true x hasn't changed
                # This is important for 'steps-post' to correctly represent plateaus.
                if not best_so_far_trajectory or best_so_far_trajectory[-1][0] != x_val:
                    best_so_far_trajectory.append((x_val, current_min_fitness))
                elif best_so_far_trajectory[-1][1] > current_min_fitness : # x_val is same, but fitness improved
                    best_so_far_trajectory[-1] = (x_val, current_min_fitness)


        if not best_so_far_trajectory:
            if verbose:
                logger.info(f"Overall Best Fitness Plot: Config '{config_name}': Best-so-far trajectory is empty. Skipping plot for this config.")
            continue
            
        x_plot_values = [p[0] for p in best_so_far_trajectory]
        y_plot_values = [p[1] for p in best_so_far_trajectory]
        
        # Ensure the plot extends to the last x-value if using steps-post
        if x_plot_values and y_plot_values and len(x_plot_values) == len(y_plot_values):
             if len(x_plot_values) > 1 and x_plot_values[-1] > x_plot_values[-2]: # Check if there's a final segment to draw
                # Add a duplicate of the last point to make 'steps-post' draw the last segment correctly to its x-value
                # This is often needed if the "next" x value isn't explicitly in the data.
                # However, if data already includes all points up to a max_x, this might not be needed.
                # Let's test without first, as drawstyle='steps-post' should handle it.
                pass


        plt.plot(x_plot_values, y_plot_values, label=config_name, drawstyle='steps-post', alpha=0.8, lw=1.5)

    plt.title(f"Overall Best Fitness Convergence vs. {actual_x_axis_label}", fontsize=15)
    plt.xlabel(actual_x_axis_label, fontsize=12)
    plt.ylabel("Best Fitness Found So Far (Lower is Better)", fontsize=12)
    
    if len(history_data_dict) > 6: # Conditional legend placement
        plt.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
    else:
        plt.legend(fontsize=9, loc='best')
        plt.tight_layout()

    plt.grid(True, linestyle=":", alpha=0.7)
    plt.show()


def plot_scatter_plot_matrix(
    final_results_df: pd.DataFrame, 
    metrics_to_plot: Optional[List[str]] = None, 
    config_col: str = 'Configuration', 
    diag_kind: str = 'auto', 
    verbose: bool = True
):
    """
    Generates a scatter plot matrix (SPLOM) to visualize pairwise relationships 
    between different performance metrics for various algorithm configurations.
    """
    if not verbose:
        return

    if metrics_to_plot is None:
        metrics_to_plot = ['BestFitness', 'RuntimeSeconds', 'FunctionEvaluations']

    if not metrics_to_plot or len(metrics_to_plot) < 2:
        if verbose: logger.info("Scatter Plot Matrix: At least two metrics are required to plot. Skipping.")
        return

    if verbose:
        logger.info("\n--- Scatter Plot Matrix (SPLOM) for Performance Metrics ---")
        logger.info(f"Plotting for metrics: {metrics_to_plot} with hue: {config_col}")

    # Data Preparation
    if "AlgorithmName" not in final_results_df.columns:
        logger.warning("Scatter Plot Matrix: 'AlgorithmName' column not found. Cannot filter error runs. Proceeding with all data.")
        plot_df = final_results_df.copy()
    else:
        plot_df = final_results_df[~final_results_df["AlgorithmName"].str.contains("ERROR", na=False)].copy()
    
    # Ensure all metrics_to_plot exist and are numeric, then drop NaNs
    valid_metrics_to_plot = []
    for metric in metrics_to_plot:
        if metric not in plot_df.columns:
            logger.warning(f"Scatter Plot Matrix: Metric '{metric}' not found in DataFrame. Skipping this metric.")
            continue
        if not pd.api.types.is_numeric_dtype(plot_df[metric]):
            logger.warning(f"Scatter Plot Matrix: Metric '{metric}' is not numeric. Attempting conversion or skipping.")
            try:
                plot_df[metric] = pd.to_numeric(plot_df[metric], errors='coerce')
                if plot_df[metric].isnull().all(): # If conversion results in all NaNs
                     logger.warning(f"Scatter Plot Matrix: Metric '{metric}' became all NaNs after numeric conversion. Skipping.")
                     continue
            except Exception as e:
                logger.error(f"Scatter Plot Matrix: Error converting metric '{metric}' to numeric: {e}. Skipping.")
                continue
        valid_metrics_to_plot.append(metric)
    
    if len(valid_metrics_to_plot) < 2:
        if verbose: logger.info("Scatter Plot Matrix: Fewer than two valid numeric metrics available. Skipping plot.")
        return
        
    plot_df.dropna(subset=valid_metrics_to_plot, inplace=True)

    if plot_df.empty:
        if verbose: logger.info("Scatter Plot Matrix: No data remains after filtering errors and NaNs. Skipping plot.")
        return
        
    if config_col not in plot_df.columns:
        logger.error(f"Scatter Plot Matrix: Configuration column '{config_col}' not found in DataFrame. Skipping plot.")
        return

    if plot_df[config_col].nunique() < 1: # Need at least one group to hue by, though more is typical
        if verbose: logger.info(f"Scatter Plot Matrix: Less than 1 unique configuration found in '{config_col}'. Skipping plot.")
        return

    # Plotting
    try:
        g = sns.pairplot(
            plot_df, 
            vars=valid_metrics_to_plot, 
            hue=config_col, 
            diag_kind=diag_kind, 
            palette='viridis', # Using a consistent palette
            corner=False # Full matrix
        )
        g.fig.suptitle('Scatter Plot Matrix (SPLOM) of Performance Metrics', y=1.02, fontsize=16)
        g.fig.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to prevent suptitle overlap
        plt.show()
    except Exception as e_splom:
        logger.error(f"Scatter Plot Matrix: Error during plotting: {e_splom}", exc_info=True)
        if 'g' in locals() and hasattr(g.fig, 'clf'): # Check if figure exists to close
            plt.close(g.fig)
        elif plt.gcf(): # Check if there's any current figure
            plt.close(plt.gcf())



def plot_parallel_coordinates(
    final_results_df: pd.DataFrame, 
    metrics_to_plot: Optional[List[str]] = None, 
    config_col: str = 'Configuration', 
    lower_is_better_metrics: Optional[List[str]] = None, 
    verbose: bool = True
):
    """
    Generates a parallel coordinates plot to visualize and compare the performance
    of different algorithm configurations across multiple normalized metrics.
    """
    if not verbose:
        return

    # Default metrics if not provided
    if metrics_to_plot is None:
        metrics_to_plot = ['BestFitness', 'RuntimeSeconds', 'FunctionEvaluations']
    if lower_is_better_metrics is None:
        lower_is_better_metrics = ['BestFitness', 'RuntimeSeconds', 'FunctionEvaluations'] # Assume all default metrics are lower-is-better

    if not metrics_to_plot:
        if verbose: logger.info("Parallel Coordinates: No metrics specified to plot.")
        return

    if verbose:
        logger.info("\n--- Parallel Coordinates Plot for Normalized Performance ---")
        logger.info(f"Plotting for metrics: {metrics_to_plot}")
        logger.info(f"Metrics where lower is better (will be inverted): {lower_is_better_metrics}")

    # Data Preparation
    if "AlgorithmName" not in final_results_df.columns:
        logger.warning("Parallel Coordinates: 'AlgorithmName' column not found. Cannot filter error runs. Proceeding with all data.")
        valid_data = final_results_df.copy()
    else:
        valid_data = final_results_df[~final_results_df["AlgorithmName"].str.contains("ERROR", na=False)].copy()

    if valid_data.empty:
        if verbose: logger.info("Parallel Coordinates: No valid data after filtering error runs. Skipping plot.")
        return
        
    # Check if all metrics_to_plot are present in the DataFrame
    missing_metrics = [m for m in metrics_to_plot if m not in valid_data.columns]
    if missing_metrics:
        logger.warning(f"Parallel Coordinates: The following metrics are missing from the DataFrame and will be skipped: {missing_metrics}")
        metrics_to_plot = [m for m in metrics_to_plot if m in valid_data.columns]
        if not metrics_to_plot:
            if verbose: logger.info("Parallel Coordinates: No plottable metrics remain after checking columns. Skipping plot.")
            return
            
    # Group by configuration and calculate mean performance
    try:
        mean_perf_df = valid_data.groupby(config_col)[metrics_to_plot].mean().reset_index()
    except KeyError as e:
        logger.error(f"Parallel Coordinates: Grouping by '{config_col}' failed. Ensure it's a valid column. Error: {e}")
        return
        
    if mean_perf_df.empty or len(mean_perf_df) < 1: # Allow plotting even for a single configuration
        if verbose: logger.info("Parallel Coordinates: Not enough data or configurations after grouping. Skipping plot.")
        return

    # Normalization
    normalized_df = mean_perf_df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))

    for metric in metrics_to_plot:
        if metric not in normalized_df.columns: # Should be caught by earlier check, but safeguard
            logger.warning(f"Parallel Coordinates: Metric '{metric}' not found during normalization. Skipping this metric.")
            continue
        
        # Handle potential non-numeric or all-NaN columns before scaling
        if not pd.api.types.is_numeric_dtype(normalized_df[metric]):
            logger.warning(f"Parallel Coordinates: Metric '{metric}' is not numeric. Skipping normalization for this metric.")
            normalized_df[metric] = np.nan # Set to NaN to avoid errors in plotting or indicate issue
            continue
        
        if normalized_df[metric].isnull().all():
            logger.warning(f"Parallel Coordinates: Metric '{metric}' contains all NaN values. Skipping normalization for this metric.")
            continue # Will plot as NaNs if parallel_coordinates handles it, or may need explicit drop

        # Ensure column is 2D for scaler
        metric_values = normalized_df[metric].values.reshape(-1, 1)
        scaled_values = scaler.fit_transform(metric_values)
        
        if metric in lower_is_better_metrics:
            scaled_values = 1 - scaled_values # Invert scale
            
        normalized_df[metric] = scaled_values.flatten()

    # Plotting
    if normalized_df.empty or config_col not in normalized_df.columns:
         if verbose: logger.info("Parallel Coordinates: DataFrame became empty or config_col is missing before plotting. Skipping.")
         return

    plt.figure(figsize=(max(10, len(metrics_to_plot) * 2.5), 7)) # Adjusted width factor to 2.5
    
    try:
        parallel_coordinates(
            normalized_df, 
            class_column=config_col, 
            colormap='viridis', 
            linewidth=1.5, 
            alpha=0.8
        )
    except Exception as e_pc_plot:
        logger.error(f"Parallel Coordinates: Error during plotting: {e_pc_plot}", exc_info=True)
        plt.close() # Close the figure if plotting failed
        return

    plt.title('Parallel Coordinates Plot of Normalized Algorithm Performance', fontsize=15)
    plt.ylabel('Normalized Metric Value (0=Worst, 1=Best)', fontsize=12) # Clarified Y-axis
    plt.xlabel('Performance Metrics', fontsize=12)
    
    if len(metrics_to_plot) > 5:
        plt.xticks(rotation=30, ha="right")
    
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    
    num_unique_configs = len(mean_perf_df[config_col].unique())
    if num_unique_configs > 6:
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title=config_col, fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for external legend
    else:
        plt.legend(title=config_col, fontsize=9, loc='best') # 'best' or 'upper right' might be fine
        plt.tight_layout()
        
    plt.show()
