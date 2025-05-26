import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import LeagueSolution for type hinting if used explicitly, or rely on duck typing.
# from solution import LeagueSolution
# Import recreate_solution_from_data from the new serialization_utils module
from serialization_utils import recreate_solution_from_data

logger = logging.getLogger(__name__)

def plot_summary_statistics_bars(final_results_df: pd.DataFrame):
    if final_results_df.empty:
        logger.info("No results for summary statistics bars.")
        return

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
            logger.info(f"No valid data for metric '{metric_col}'. Skipping.")
            continue

        summary_stats = valid_data.groupby("Configuration")[metric_col].agg(
            ["mean", "std", "min", "max"]
        )
        logger.info(f"\n📊 Statistics for: {metric_col}")

        fmt = "{:.4f}" if metric_col in ["BestFitness", "RuntimeSeconds"] else "{:.1f}"
        formatters = {
            col: (lambda x, f=fmt: f.format(x)) for col in summary_stats.columns
        }
        try:
            from IPython.display import display # type: ignore

            display(summary_stats.style.format(formatters))
        except ImportError:
            print(summary_stats.to_string(formatters=formatters))

        plt.figure()
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


def plot_metric_distributions_boxplots(final_results_df: pd.DataFrame):
    if final_results_df.empty:
        return
    logger.info("\n--- Box Plots for Metric Distributions ---")
    metrics = ["BestFitness", "RuntimeSeconds", "FunctionEvaluations"]
    for metric_col in metrics:
        if metric_col not in final_results_df.columns:
            continue
        valid_data = final_results_df[
            final_results_df[metric_col].notna()
            & np.isfinite(final_results_df[metric_col])
            & (~final_results_df["AlgorithmName"].str.contains("ERROR", na=False))
        ]
        if not valid_data.empty:
            plt.figure()
            sns.boxplot(
                x="Configuration",
                y=metric_col,
                data=valid_data,
                palette="pastel",
                hue="Configuration",
                legend=False,
            )
            plt.title(f"Distribution of {metric_col} by Algorithm", fontsize=14)
            plt.ylabel(metric_col, fontsize=11)
            plt.xlabel("Configuration", fontsize=11)
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.show()
        else:
            logger.info(f"No valid data for box plot of '{metric_col}'.")


def plot_convergence_curves(final_history_map: dict):
    if not final_history_map:
        return
    logger.info("\n--- Convergence Plots ---")
    valid_configs = {k: v for k, v in final_history_map.items() if v}
    num_configs = len(valid_configs)
    if num_configs == 0:
        logger.info("No history data for convergence plots.")
        return

    rows = (num_configs + 1) // 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows), squeeze=False)
    ax_flat = axes.flatten()
    plot_idx = 0
    for config_name, runs_data in valid_configs.items():
        if plot_idx >= len(ax_flat):
            break
        ax = ax_flat[plot_idx]
        has_data = False
        num_labeled = 0
        for run_id, hist_list in runs_data.items():
            if hist_list:
                lbl = f"Run {run_id+1}" if num_labeled < 5 else None
                ax.plot(hist_list, label=lbl, alpha=0.7, lw=1.2)
                has_data = True
                if lbl:
                    num_labeled += 1
        if has_data:
            ax.set_title(f"Convergence: {config_name}", fontsize=12)
            ax.set_xlabel("Iteration/Generation", fontsize=10)
            ax.set_ylabel("Best Fitness", fontsize=10)
            ax.grid(True, ls=":", alpha=0.5)
            if num_labeled > 0:
                ax.legend(fontsize=8)
        else:
            ax.text(
                0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"Convergence: {config_name} (No Data)", fontsize=12)
        plot_idx += 1
    for i in range(plot_idx, len(ax_flat)):
        fig.delaxes(ax_flat[i])
    fig.tight_layout(pad=2.0)
    plt.show()


def rank_algorithms_custom(results_df: pd.DataFrame) -> pd.DataFrame or None:
    if results_df.empty:
        logger.warning("Ranking: Input DataFrame empty.")
        return None
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
        display(final_ranks[existing_cols].style.format(active_fmts))
    except ImportError:
        call_fmts = {k: (lambda x, f=v: f.format(x)) for k, v in active_fmts.items()}
        print(final_ranks[existing_cols].to_string(formatters=call_fmts))

    if "OverallRank" in final_ranks.columns:
        plt.figure(figsize=(10, 5))
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
):
    """
    Displays detailed statistics and visualizations for the single overall best solution.
    This function is intended to be called from the main experiment script.
    """
    if not best_sol_data:
        logger.info(
            "No best solution data provided to display_single_best_solution_details."
        )
        return

    logger.info(
        f"\n--- Detailed Analysis: Best Solution from '{config_name}', Run {run_num} (Fitness: {fitness:.4f}) ---"
    )

    # Recreate the solution object using the utility function (also in this file or imported)
    # and the passed solution module reference.
    best_obj = recreate_solution_from_data(  # Assumes recreate_solution_from_data is defined in this utils file
        best_sol_data, players_list, problem_def, solution_classes_module=sol_module
    )

    if not best_obj:
        logger.error(
            "Could not recreate the best solution object for detailed display. Aborting details."
        )
        return

    # Log validity and re-calculated fitness of the recreated object
    is_recreated_valid = best_obj.is_valid()  # This call uses the original .is_valid()
    recalculated_fitness_val = (
        best_obj.fitness()
    )  # This call uses the original .fitness()
    logger.info(
        f"Recreated Best Solution: Is Valid = {is_recreated_valid}, Recalculated Fitness = {recalculated_fitness_val:.4f}"
    )
    if not is_recreated_valid:
        logger.warning(
            "The recreated overall best solution is flagged as INVALID. Displayed statistics might be misleading or incomplete."
        )

    # Get detailed team statistics from the solution object
    team_stats_list = best_obj.get_team_stats()
    if not team_stats_list:
        logger.warning(
            "No team statistics could be generated for the overall best solution."
        )
        return

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
            print("    🔴 WARNING: TEAM BUDGET EXCEEDED!")
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

    logger.info(
        f"--- End of Detailed Analysis for Overall Best Solution ({config_name}) ---"
    )
