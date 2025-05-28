import logging
from solution import LeagueSolution, LeagueHillClimbingSolution
from evolution import hill_climbing

logger = logging.getLogger(__name__)

def hc_wrapper_for_ga(
    current_ga_solution,
    max_iterations=20,
    max_no_improvement=10,
    verbose=False,
    **kwargs_for_hc,
):
    """
    Wrapper to use hill_climbing as a local search method within GA.
    It converts a GA's LeagueSolution to a LeagueHillClimbingSolution, applies HC,
    and returns the (potentially improved) LeagueHillClimbingSolution object.
    """
    if not isinstance(current_ga_solution, LeagueSolution):
        logger.warning(
            "hc_wrapper_for_ga: Received non-LeagueSolution object. Cannot apply HC."
        )
        return current_ga_solution

    # Convert GA's LeagueSolution to LeagueHillClimbingSolution for HC
    hc_solution_instance = LeagueHillClimbingSolution(
        repr=list(current_ga_solution.repr),  # Use a copy of the repr
        num_teams=current_ga_solution.num_teams,
        team_size=current_ga_solution.team_size,
        max_budget=current_ga_solution.max_budget,
        players=current_ga_solution.players,  # players list is shared by reference
        position_requirements=current_ga_solution.position_requirements,  # shared by reference
    )

    # It's good practice to ensure the solution is valid before applying HC,
    # though GA usually maintains valid solutions.
    if not hc_solution_instance.is_valid():
        logger.warning(
            f"hc_wrapper_for_ga: Converted solution for HC is invalid (Fitness: {hc_solution_instance.fitness()}). Skipping HC application."
        )
        return current_ga_solution  # Return original GA solution if conversion leads to invalid

    logger.debug(
        f"hc_wrapper_for_ga: Applying HC (max_iter={max_iterations}, max_no_imp={max_no_improvement}) to solution with initial fitness {hc_solution_instance.fitness()}"
    )

    # Apply hill_climbing
    improved_hc_solution, _, _ = hill_climbing(
        initial_solution=hc_solution_instance,
        max_iterations=max_iterations,
        max_no_improvement=max_no_improvement,
        verbose=verbose,
        **kwargs_for_hc,  # Pass any other specific kwargs for HC or its get_neighbors
    )

    logger.debug(
        f"hc_wrapper_for_ga: HC application finished. Fitness before: {current_ga_solution.fitness()} (approx), Fitness after: {improved_hc_solution.fitness()}"
    )
    # The genetic_algorithm function will take this improved_hc_solution's repr
    # and create a new LeagueSolution object for its population.
    return improved_hc_solution
