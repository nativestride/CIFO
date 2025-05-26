import logging
import random # Added for parent selection fallback
from copy import deepcopy

from solution import LeagueSolution, InsufficientPlayersForPositionError
from experiment_utils import safe_exp # For default safe_exp in GA params

# Import from new operator modules
from mutation_operators import (
    mutate_swap,  # Example, add others if used as defaults
)
from crossover_operators import (
    crossover_one_point,  # Example, add others if used as defaults
)
from selection_operators import (
    selection_tournament,  # Example, add others if used as defaults
    selection_boltzmann, # For type hinting or default
)
from ga_utilities import hc_wrapper_for_ga # If used as a local search default

# Import List and Callable for type hinting
from typing import List, Callable, Dict, Any 

logger = logging.getLogger(__name__)

def _evolve_one_ga_generation(
    current_population: List[LeagueSolution], 
    population_size: int, 
    elitism_size: int, 
    selection_operator: Callable, 
    selection_kwargs: Dict[str, Any], 
    crossover_operator: Callable, 
    crossover_rate: float, 
    mutation_operator: Callable, 
    prob_apply_mutation: float, 
    logger_instance: logging.Logger
) -> List[LeagueSolution]:
    """
    Evolves a population for a single GA generation.
    Handles elitism, selection, crossover, and mutation.
    """
    current_population.sort(key=lambda x: x.fitness()) # Sort by fitness (lower is better)

    new_population: List[LeagueSolution] = []
    if elitism_size > 0:
        actual_elitism_size = min(elitism_size, len(current_population))
        new_population.extend(deepcopy(current_population[:actual_elitism_size]))

    while len(new_population) < population_size:
        if not current_population: 
            logger_instance.warning("_evolve_one_ga_generation: Parent population unexpectedly depleted. Loop will break.")
            break

        parent1 = selection_operator(current_population, **selection_kwargs)
        parent2 = selection_operator(current_population, **selection_kwargs)

        if parent1 is None or parent2 is None:
            logger_instance.warning(f"_evolve_one_ga_generation: Parent selection failed (pop size: {len(current_population)}). Using random fallback if possible.")
            if current_population : 
                parent1 = deepcopy(random.choice(current_population)) if parent1 is None and current_population else parent1
                parent2 = deepcopy(random.choice(current_population)) if parent2 is None and current_population else parent2
            else:
                logger_instance.error("_evolve_one_ga_generation: Population empty, cannot select parents. Breaking breeding.")
                break 
            if parent1 is None or parent2 is None: 
                logger_instance.error("_evolve_one_ga_generation: Fallback parent selection also failed. Breaking breeding.")
                break 

        child1 = None
        if random.random() < crossover_rate:
            child1 = crossover_operator(parent1, parent2)
        else: 
            child1 = deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)

        if random.random() < prob_apply_mutation:
            if child1 is not None:
                child1 = mutation_operator(child1)
            else: 
                logger_instance.warning("_evolve_one_ga_generation: child1 is None before mutation. Using parent1 as fallback.")
                child1 = deepcopy(parent1) 

        if child1 is not None and child1.is_valid():
            new_population.append(child1)
        else:
            logger_instance.debug("_evolve_one_ga_generation: Generated child was invalid or None. Not added.")
    
    return new_population


def generate_population(
    players_data: list,
    population_size: int,
    num_teams: int,
    team_size: int,
    max_budget: float,
    position_requirements: dict,
    logger_instance: logging.Logger, # Changed from logger to logger_instance for clarity
    max_initial_solution_attempts: int = 50,
) -> list:
    """
    Generates an initial population of valid LeagueSolution instances.
    Tries to meet positional quotas and team sizes. Budget is checked by is_valid().
    """
    population = []
    total_attempts_allowed = population_size * max_initial_solution_attempts * 2
    current_total_attempts = 0
    solutions_generated = 0

    logger_instance.info(
        f"Attempting to generate initial population of size {population_size} (max attempts per solution: {max_initial_solution_attempts})..."
    )

    while (
        solutions_generated < population_size
        and current_total_attempts < total_attempts_allowed
    ):
        current_total_attempts += 1
        attempts_for_this_solution = 0

        current_solution = None
        while (
            attempts_for_this_solution < max_initial_solution_attempts
        ):
            attempts_for_this_solution += 1
            try:
                solution = LeagueSolution(
                    num_teams=num_teams,
                    team_size=team_size,
                    max_budget=max_budget,
                    players=deepcopy(players_data), # Ensure players_data is deepcopied for each solution
                    position_requirements=position_requirements,
                )
                if solution.is_valid():
                    current_solution = solution
                    break
            except InsufficientPlayersForPositionError as e_pos:
                if attempts_for_this_solution % 10 == 0:
                    logger_instance.warning(
                        f"generate_population: Attempt {attempts_for_this_solution}/{max_initial_solution_attempts} for sol {solutions_generated + 1} - {e_pos}"
                    )
            except ValueError as e_val:
                if attempts_for_this_solution % 10 == 0:
                    logger_instance.warning(
                        f"generate_population: Attempt {attempts_for_this_solution}/{max_initial_solution_attempts} for sol {solutions_generated + 1} - Value error: {e_val}"
                    )
            except Exception as e_unexpected: # Catch any other unexpected error during solution creation
                logger_instance.error(
                    f"generate_population: Unexpected error on attempt {attempts_for_this_solution} for sol {solutions_generated + 1}: {e_unexpected}",
                    exc_info=True,
                )

        if current_solution:
            population.append(current_solution)
            solutions_generated += 1
            if solutions_generated % (population_size // 10 or 1) == 0: # Log progress every 10% or so
                logger_instance.info(
                    f"  Generated {solutions_generated}/{population_size} valid initial solutions..."
                )
        else:
            logger_instance.warning(
                f"generate_population: Failed to generate a valid solution for individual {solutions_generated + 1} "
                f"after {max_initial_solution_attempts} attempts. Continuing..."
            )

    if solutions_generated < population_size:
        logger_instance.warning(
            f"generate_population: Only generated {solutions_generated}/{population_size} valid solutions "
            f"after {current_total_attempts} total attempts. Population will be smaller than requested."
        )

    if not population and population_size > 0: # If no solutions were generated at all
        logger_instance.error(
            "generate_population: Failed to generate ANY valid solutions for the initial population."
        )
        # Depending on desired behavior, you might raise an error here:
        # raise RuntimeError("Failed to generate any valid solutions for the initial GA population.")

    logger_instance.info(
        f"generate_population: Finished. Generated {len(population)} solutions."
    )
    return population


def genetic_algorithm(
    players_data: list,
    problem_params: dict,
    ga_params: dict,
) -> tuple:
    """
    Genetic Algorithm implementation.
    """
    # Logger for this specific GA run, can be the main module logger or a passed one.
    # Using module logger here for simplicity.
    # global logger # Not needed if logger is defined at module scope as logger = logging.getLogger(__name__)

    num_teams = problem_params["num_teams"]
    team_size = problem_params["team_size"]
    max_budget = problem_params["max_budget"]
    position_requirements = problem_params["position_requirements"]

    population_size = ga_params["population_size"]
    max_generations = ga_params["max_generations"]

    selection_operator = ga_params["selection_operator"]
    selection_kwargs = ga_params.get("selection_params", {}).copy()
    if (
        selection_operator.__name__ == "selection_boltzmann"
        and "safe_exp_func" not in selection_kwargs
    ):
        selection_kwargs["safe_exp_func"] = ga_params.get("safe_exp_func", safe_exp)

    crossover_operator = ga_params["crossover_operator"]
    crossover_rate = ga_params["crossover_rate"]

    mutation_operator = ga_params["mutation_operator"]
    prob_apply_mutation = ga_params.get("mutation_rate", 0.1) # Default mutation rate

    elitism_size = ga_params.get("elitism_size", 0)
    # Handle boolean elitism flag for backward compatibility or simpler config
    if 'elitism' in ga_params and isinstance(ga_params['elitism'], bool):
        if ga_params['elitism'] and elitism_size == 0: elitism_size = 1 # Default to 1 if elitism is true
        elif not ga_params['elitism']: elitism_size = 0


    local_search_func = ga_params.get("local_search_func", None)
    local_search_config = ga_params.get("local_search_params", {})

    verbose = ga_params.get("verbose", False)
    max_init_attempts_for_population = ga_params.get("max_initial_solution_attempts", 50)

    population = generate_population(
        players_data,
        population_size,
        num_teams,
        team_size,
        max_budget,
        position_requirements,
        logger_instance=logger, # Pass the logger instance
        max_initial_solution_attempts=max_init_attempts_for_population,
    )
    if not population: # generate_population should log this, but good to handle
        logger.error("GA Error: Failed to initialize the population. Aborting.")
        # Return a tuple indicating failure, matching the expected return type
        return None, float('inf'), []


    try:
        best_solution_overall = min(population, key=lambda s: s.fitness())
        best_fitness_overall = best_solution_overall.fitness()
    except ValueError: # Should not happen if population is not empty
        logger.error("GA Error: Initial population is empty after generation. Cannot determine initial best.")
        return None, float('inf'), [] # Return a failure tuple

    history = [best_fitness_overall]

    if verbose:
        logger.info(f"Initial best fitness: {best_fitness_overall:.4f}")

    for generation in range(max_generations):
        # Call the new helper function to evolve one generation
        new_population = _evolve_one_ga_generation(
            current_population=population,
            population_size=population_size,
            elitism_size=elitism_size,
            selection_operator=selection_operator,
            selection_kwargs=selection_kwargs,
            crossover_operator=crossover_operator,
            crossover_rate=crossover_rate,
            mutation_operator=mutation_operator,
            prob_apply_mutation=prob_apply_mutation,
            logger_instance=logger
        )

        if not new_population: 
            logger.error(f"GA Gen {generation + 1}: _evolve_one_ga_generation returned an empty new_population. Stopping.")
            break 

        population = new_population[:population_size] 

        # Fill population if it's under target size (e.g. if many invalid children were skipped)
        # This loop tries to fill with copies from the new_population (which contains elites and valid children)
        idx_to_fill_from_current = 0
        while len(population) < population_size and new_population : # Check new_population is not empty
            population.append(deepcopy(new_population[idx_to_fill_from_current % len(new_population)]))
            idx_to_fill_from_current +=1
            # Safety break for the fill loop
            if idx_to_fill_from_current > len(new_population)*2 and len(population) < population_size :
                 logger.warning(f"GA Gen {generation + 1}: Could not fill population to target size {population_size} despite fallbacks (current size: {len(population)}).")
                 break


        if not population: # If population somehow became critically empty
            logger.critical(f"GA Gen {generation + 1}: Population became critically empty. Stopping algorithm.")
            break

        # --- LOCAL SEARCH APPLICATION ---
        if local_search_func and (generation + 1) % local_search_config.get("frequency", 5) == 0:
            num_to_improve = local_search_config.get("num_to_improve", max(1, int(population_size * 0.1)))
            population.sort(key=lambda x: x.fitness()) # Sort before picking best for LS

            logger.info(f"GA Gen {generation + 1}: Applying local search to {min(num_to_improve, len(population))} individuals.")

            for i in range(min(num_to_improve, len(population))):
                solution_to_improve = population[i]
                original_fitness = solution_to_improve.fitness()

                # Prepare params for local_search_func, removing LS-specific control params
                ls_call_params = local_search_config.copy()
                ls_call_params.pop("frequency", None)
                ls_call_params.pop("num_to_improve", None)
                ls_call_params.setdefault('verbose', verbose) # Pass GA's verbosity to LS if not set

                logger.debug(f"  Applying LS to individual {i} (fitness: {original_fitness:.4f}) with params: {ls_call_params}")

                # Assuming local_search_func (like hc_wrapper_for_ga) returns a solution object
                improved_solution_object_from_ls = local_search_func(
                    solution_to_improve,
                    **ls_call_params
                )

                if improved_solution_object_from_ls:
                    improved_fit = improved_solution_object_from_ls.fitness()
                    # Replace if LS found a better solution
                    if improved_fit < original_fitness:
                        # Important: Create a new LeagueSolution if LS returns a different type (e.g. LeagueHillClimbingSolution)
                        # or ensure LS returns the correct type. hc_wrapper_for_ga returns the HC solution type.
                        # The GA should manage LeagueSolution instances.
                        population[i] = LeagueSolution(
                            repr=list(improved_solution_object_from_ls.repr),
                            num_teams=problem_params['num_teams'],
                            team_size=problem_params['team_size'],
                            max_budget=problem_params['max_budget'],
                            players=players_data, # Use the original players_data reference
                            position_requirements=problem_params['position_requirements']
                        )
                        # Re-check fitness of the new LeagueSolution object, as it might differ if fitness caching or other state is involved
                        logger.info(f"    LS improved individual {i}. Old fitness: {original_fitness:.4f}, New fitness: {population[i].fitness():.4f}")
                    else:
                        logger.debug(f"    LS did not improve individual {i}. Original: {original_fitness:.4f}, LS attempt: {improved_fit:.4f}")
                else:
                    logger.warning(f"    Local search function returned None for individual {i} (original fitness: {original_fitness:.4f}).")
        # --- END OF LOCAL SEARCH ---

        try:
            current_gen_best_sol = min(population, key=lambda s: s.fitness())
            current_gen_best_fit = current_gen_best_sol.fitness()
        except ValueError: # Population might be empty if all LS attempts failed or other issues
            logger.error(f"GA Gen {generation + 1}: Population empty at end of generation. Using overall best for history.")
            current_gen_best_fit = best_fitness_overall # Use last known best

        if current_gen_best_fit < best_fitness_overall:
            best_solution_overall = deepcopy(current_gen_best_sol)
            best_fitness_overall = current_gen_best_fit
            if verbose:
                logger.info(f"GA Gen {generation + 1}: New best fitness: {best_fitness_overall:.4f}")
        elif verbose and (generation + 1) % 5 == 0: # Log progress periodically
            logger.info(f"GA Gen {generation + 1}: Current gen best: {current_gen_best_fit:.4f} (Overall best: {best_fitness_overall:.4f})")

        history.append(best_fitness_overall)

    if verbose:
        logger.info(f"Genetic Algorithm finished. Final best fitness: {best_fitness_overall:.4f}")
    return best_solution_overall, best_fitness_overall, history
