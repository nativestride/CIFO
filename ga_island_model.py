import logging
import random
from copy import deepcopy
import numpy as np # Required by one of the functions, ensure it's here.

from solution import LeagueSolution, InsufficientPlayersForPositionError
from genetic_algorithms import generate_population # For re-populating islands
from selection_operators import selection_boltzmann # As a potential default
from experiment_utils import safe_exp # As a potential default for selection_boltzmann

# It's assumed that specific crossover, mutation, and selection operators
# will be passed via ga_params_per_island within island_model_ga_params.
# If _evolve_one_generation needs to call them directly with defaults,
# they would need to be imported here as well.
# For now, assuming they are passed in.

# Import List and Callable for type hinting
from typing import List, Callable, Dict, Any, Optional # Added Optional

logger = logging.getLogger(__name__)

# Import the function to be delegated to, and generate_population for recovery
from genetic_algorithms import _evolve_one_ga_generation, generate_population


# Refactored Migration Helper Functions
def _perform_ring_migration(
    islands_list: List[List[LeagueSolution]], 
    num_migrants: int, 
    island_population_size: int, 
    logger_instance: logging.Logger
):
    """Handles ring migration logic."""
    num_islands = len(islands_list)
    if num_islands <= 1:
        logger_instance.debug("Ring migration skipped: Not enough islands.")
        return

    migrants_to_send: List[List[LeagueSolution]] = [[] for _ in range(num_islands)]
    for i in range(num_islands):
        if not islands_list[i]:
            logger_instance.warning(f"Migration (Ring): Island {i+1} is empty, cannot select migrants.")
            continue
        islands_list[i].sort(key=lambda s: s.fitness())
        actual_num_migrants = min(num_migrants, len(islands_list[i]))
        if actual_num_migrants > 0:
            migrants_to_send[i] = deepcopy(islands_list[i][:actual_num_migrants])
        logger_instance.debug(f"Island {i+1} selected {len(migrants_to_send[i])} migrants for ring migration.")

    for i_target in range(num_islands):
        source_island_idx = (i_target - 1 + num_islands) % num_islands
        current_migrants = migrants_to_send[source_island_idx]
        if not current_migrants: 
            continue

        if not islands_list[i_target]: 
            islands_list[i_target] = current_migrants[:island_population_size] 
            logger_instance.info(f"Island {i_target+1} (empty) populated by {len(islands_list[i_target])} migrants from island {source_island_idx+1}.")
        else:
            islands_list[i_target].sort(key=lambda s: s.fitness(), reverse=True) 
            num_to_replace = min(len(current_migrants), len(islands_list[i_target]))
            
            temp_destination_pop = islands_list[i_target][num_to_replace:]
            temp_destination_pop.extend(current_migrants) 
            islands_list[i_target] = sorted(temp_destination_pop, key=lambda s: s.fitness())[:island_population_size]
            logger_instance.info(f"Island {i_target+1} received {len(current_migrants)} migrants from island {source_island_idx+1}. Pop size: {len(islands_list[i_target])}")

def _perform_random_pair_migration(
    islands_list: List[List[LeagueSolution]], 
    num_migrants: int, 
    island_population_size: int, 
    logger_instance: logging.Logger
):
    """Handles random pair migration logic, utilizing _migrate_between_two."""
    num_islands = len(islands_list)
    if num_islands < 2:
        logger_instance.debug("Random pair migration skipped: Need at least 2 islands.")
        return
    
    idx1, idx2 = random.sample(range(num_islands), 2)
    logger_instance.debug(f"Migration (Random Pair): Exchanging between Island {idx1+1} and Island {idx2+1}.")
    _migrate_between_two(idx1, idx2, islands_list, num_migrants, island_population_size, logger_instance, log_prefix="  ")
    _migrate_between_two(idx2, idx1, islands_list, num_migrants, island_population_size, logger_instance, log_prefix="  ")

def _perform_broadcast_migration(
    islands_list: List[List[LeagueSolution]], 
    global_best_solution: Optional[LeagueSolution], 
    island_population_size: int, 
    logger_instance: logging.Logger, 
    verbose: bool
):
    """Handles broadcast best migration logic."""
    num_islands = len(islands_list)
    if not global_best_solution:
        logger_instance.warning("Migration (Broadcast): No global best solution available to broadcast.")
        return

    # The global_best_solution passed in should be the one identified from all islands *before* migration phase.
    # No need to re-scan here.
    logger_instance.debug(f"Migration (Broadcast): Best (fitness {global_best_solution.fitness():.4f}) to all islands.")
    for i_target_island in range(num_islands):
        if not islands_list[i_target_island]: # If target island is empty
            islands_list[i_target_island] = [deepcopy(global_best_solution)]
            logger_instance.info(f"  Island {i_target_island+1} (empty) received broadcast best.")
            continue
        
        # Avoid adding if identical representation already exists
        already_present = any(sol.repr == global_best_solution.repr for sol in islands_list[i_target_island])
        if not already_present:
            islands_list[i_target_island].sort(key=lambda s: s.fitness(), reverse=True) # Sort worst first
            if len(islands_list[i_target_island]) < island_population_size:
                islands_list[i_target_island].append(deepcopy(global_best_solution))
            elif islands_list[i_target_island]: # Replace worst if full
                islands_list[i_target_island][0] = deepcopy(global_best_solution)
            
            islands_list[i_target_island].sort(key=lambda s: s.fitness()) # Re-sort
            islands_list[i_target_island] = islands_list[i_target_island][:island_population_size] # Ensure size
            logger_instance.info(f"  Island {i_target_island+1} received broadcast best. Pop size: {len(islands_list[i_target_island])}")
        elif verbose: # Only log if verbose and it was already present
            logger_instance.debug(f"  Island {i_target_island+1} already contains broadcast best. Not adding clone.")


# Helper function for Island Model GA
def _evolve_one_generation( # This function is now a wrapper
    current_population: list,
    problem_params: dict, 
    single_gen_ga_params: dict,
    players_data: list,
    logger_instance: logging.Logger,
) -> list:
    """
    Evolves a population for a single generation using specified GA operators.
    This function now delegates to _evolve_one_ga_generation from genetic_algorithms.py,
    but preserves the empty population recovery mechanism.
    """
    population_size = single_gen_ga_params["population_size"] # Needed for recovery

    if not current_population:
        logger_instance.warning(
            "_evolve_one_generation: Received empty current_population. Attempting recovery."
        )
        # Try to use generate_population from genetic_algorithms.py for recovery
        try:
            current_population = generate_population( # Assign to current_population
                players_data=players_data,
                population_size=population_size,
                num_teams=problem_params["num_teams"],
                team_size=problem_params["team_size"],
                max_budget=problem_params["max_budget"],
                position_requirements=problem_params["position_requirements"],
                logger_instance=logger_instance, 
                max_initial_solution_attempts=single_gen_ga_params.get('max_initial_solution_attempts', 20)
            )
            if current_population: # Check if recovery was successful
                 logger_instance.info(f"_evolve_one_generation: Recovered with {len(current_population)} new solutions.")
            else: # Recovery failed
                 logger_instance.error("_evolve_one_generation: Recovery failed, no new solutions generated.")
                 return [] # Return empty list if recovery fails
        except Exception as e_rec:
            logger_instance.error(f"_evolve_one_generation: Exception during recovery population generation: {e_rec}", exc_info=True)
            return [] # Return empty list on exception during recovery

    # If population is now valid (either initially or after recovery), proceed with evolution
    # Call the imported _evolve_one_ga_generation function
    # The logger_instance for _evolve_one_ga_generation will be the one from its own module (genetic_algorithms.py)
    # This wrapper's logger_instance is the one from ga_island_model.
    
    # Prepare selection_kwargs, specifically for Boltzmann selection's safe_exp_func
    selection_kwargs = single_gen_ga_params.get("selection_params", {}).copy()
    if (
        single_gen_ga_params["selection_operator"].__name__ == "selection_boltzmann" # Check by name
        and "safe_exp_func" not in selection_kwargs
    ):
        selection_kwargs["safe_exp_func"] = single_gen_ga_params.get(
            "safe_exp_func", safe_exp 
        )

    return _evolve_one_ga_generation( 
        current_population=current_population,
        population_size=single_gen_ga_params["population_size"],
        elitism_size=single_gen_ga_params.get("elitism_size", 0),
        selection_operator=single_gen_ga_params["selection_operator"],
        selection_kwargs=selection_kwargs, # Pass the prepared selection_kwargs
        crossover_operator=single_gen_ga_params["crossover_operator"],
        crossover_rate=single_gen_ga_params["crossover_rate"],
        mutation_operator=single_gen_ga_params["mutation_operator"],
        prob_apply_mutation=single_gen_ga_params.get("mutation_rate", 0.1),
        logger_instance=logger # Pass the module-level logger of ga_island_model.py
    )


# Helper for random pair migration
def _migrate_between_two( # This function remains the same
    source_idx,
    target_idx,
    islands_list,
    num_m,
    pop_size,
    logger_instance, # Changed from logger to logger_instance
    log_prefix="",
):
    if not islands_list[source_idx] or len(islands_list[source_idx]) == 0:
        logger_instance.debug(
            f"{log_prefix}Source island {source_idx+1} is empty. Cannot migrate."
        )
        return

    actual_num_migrants = min(num_m, len(islands_list[source_idx]))
    if actual_num_migrants == 0:
        logger_instance.debug(
            f"{log_prefix}Not enough individuals in source island {source_idx+1} to migrate ({len(islands_list[source_idx])} < {num_m} requested, or num_m is 0)."
        )
        return

    islands_list[source_idx].sort(key=lambda s: s.fitness())
    migrants = deepcopy(islands_list[source_idx][:actual_num_migrants])

    if not islands_list[target_idx]:  # Target empty
        islands_list[target_idx] = migrants[:pop_size]  # Ensure not oversized
        logger_instance.info(
            f"{log_prefix}Migrated {len(islands_list[target_idx])} from Island {source_idx+1} to empty Island {target_idx+1}."
        )
    else:
        islands_list[target_idx].sort(
            key=lambda s: s.fitness(), reverse=True
        )  # Worst first
        num_to_replace_in_target = min(len(migrants), len(islands_list[target_idx]))

        temp_pop = islands_list[target_idx][
            num_to_replace_in_target:
        ]  # Keep the better ones
        temp_pop.extend(migrants)  # Add all selected migrants
        islands_list[target_idx] = sorted(temp_pop, key=lambda s: s.fitness())[
            :pop_size
        ]  # Keep best, ensure size
        logger_instance.info(
            f"{log_prefix}Migrated {len(migrants)} from Island {source_idx+1} to Island {target_idx+1}. Target pop size: {len(islands_list[target_idx])}"
        )


def genetic_algorithm_island_model(
    players_data: list, problem_params: dict, island_model_ga_params: dict
) -> tuple[LeagueSolution | None, float, list]:
    """
    Implements a Genetic Algorithm with an Island Model.
    """
    # Use module-level logger
    # global logger # Not needed if logger is defined at module scope
    
    logger.info(
        f"Starting Genetic Algorithm - Island Model with params: {island_model_ga_params}"
    )
    num_islands = island_model_ga_params["num_islands"]
    island_population_size = island_model_ga_params["island_population_size"]
    max_generations_total = island_model_ga_params["max_generations_total"]
    migration_frequency = island_model_ga_params["migration_frequency"]
    num_migrants = island_model_ga_params["num_migrants"]
    migration_topology = island_model_ga_params.get("migration_topology", "ring")
    ga_params_per_island = island_model_ga_params[
        "ga_params_per_island"
    ].copy()
    verbose = island_model_ga_params.get("verbose", False)

    if ga_params_per_island.get("population_size") != island_population_size:
        logger.warning(
            f"Adjusting ga_params_per_island['population_size'] to {island_population_size} "
            f"to match island_population_size for _evolve_one_generation."
        )
        ga_params_per_island["population_size"] = island_population_size

    max_init_attempts_for_island_pop = ga_params_per_island.get(
        "max_initial_solution_attempts", 50 # Default from original GA
    )

    islands = []
    logger.info(
        f"Initializing {num_islands} islands, each with population size {island_population_size}."
    )
    for i in range(num_islands):
        if verbose:
            logger.info(f"Initializing island {i+1}/{num_islands}...")
        try:
            # generate_population is now in genetic_algorithms.py
            island_pop = generate_population(
                players_data,
                island_population_size,
                problem_params["num_teams"],
                problem_params["team_size"],
                problem_params["max_budget"],
                problem_params["position_requirements"],
                logger_instance=logger,  # Pass the module-level logger
                max_initial_solution_attempts=max_init_attempts_for_island_pop,
            )
            if not island_pop:
                logger.error(
                    f"Island {i+1} failed to generate any initial valid solutions. This is critical."
                )
                raise RuntimeError(
                    f"Island {i+1} failed to generate any initial valid solutions."
                )
            islands.append(island_pop)
            logger.debug(
                f"Island {i+1} initialized with {len(island_pop)} individuals."
            )
        except RuntimeError as e:
            logger.critical(
                f"Fatal Error: Could not initialize island {i+1}. Error: {e}"
            )
            raise

    if not islands or any(not island_pop for island_pop in islands):
        logger.critical(
            "One or more islands could not be initialized with a population. Stopping Island GA."
        )
        return None, float("inf"), []

    global_best_solution: LeagueSolution | None = None
    global_best_fitness = float("inf")

    for island_idx, island_pop in enumerate(islands):
        if not island_pop:
            logger.warning(
                f"Island {island_idx+1} is empty after initialization. Skipping for initial best."
            )
            continue
        try:
            current_island_best = min(island_pop, key=lambda s: s.fitness())
            current_island_best_fitness = current_island_best.fitness()
            if current_island_best_fitness < global_best_fitness:
                global_best_fitness = current_island_best_fitness
                global_best_solution = deepcopy(current_island_best)
        except ValueError:
            logger.error(
                f"Error finding best in initialized island {island_idx+1} (empty despite checks)."
            )
        except Exception as e_fit:
            logger.error(
                f"Error calculating fitness in island {island_idx+1} during initial scan: {e_fit}",
                exc_info=True,
            )

    history_of_global_best_fitness = [global_best_fitness]
    if global_best_solution:
        logger.info(f"Initial global best fitness: {global_best_fitness:.4f}")
    else: # Should be rare if initialization succeeded for at least one island
        logger.warning(
             f"Initial global best fitness: {global_best_fitness:.4f} (No valid initial solution found to set as best_solution_overall)"
        )


    logger.info(f"Starting evolution for {max_generations_total} generations.")
    for current_generation in range(max_generations_total):
        log_level_gen = logging.DEBUG if not verbose else logging.INFO
        if (
            (current_generation + 1) % 10 == 0
            or current_generation == max_generations_total - 1
            or verbose
        ):
            logger.log(
                log_level_gen,
                f"Island GA - Gen {current_generation + 1}/{max_generations_total}. Global Best: {global_best_fitness:.4f}",
            )

        for i_island in range(num_islands):
            if not islands[i_island]:
                logger.warning(
                    f"Island {i_island+1} is empty at start of gen {current_generation+1}. Skipping evolution."
                )
                continue
            islands[i_island] = _evolve_one_generation(
                current_population=islands[i_island],
                problem_params=problem_params,
                single_gen_ga_params=ga_params_per_island,
                players_data=players_data,
                logger_instance=logger, # Pass the module-level logger
            )
            if not islands[i_island]: # Check if it became empty
                 logger.warning(f"Island {i_island+1} became empty after evolution in gen {current_generation+1}.")


        for island_idx, island_pop_after_evol in enumerate(islands):
            if not island_pop_after_evol:
                continue
            try:
                current_island_best_sol = min(
                    island_pop_after_evol, key=lambda s: s.fitness()
                )
                ind_fitness = current_island_best_sol.fitness()
                if ind_fitness < global_best_fitness:
                    global_best_fitness = ind_fitness
                    global_best_solution = deepcopy(current_island_best_sol)
                    if verbose:
                        logger.info(
                            f"Gen {current_generation + 1}: New global best from island {island_idx+1}! Fitness: {global_best_fitness:.4f}"
                        )
            except ValueError:
                 logger.warning(f"Island {island_idx+1} empty after evolution. Cannot update global best from it.")
            except Exception as e_fit_update: # Catch any other error during fitness calculation
                 logger.error(f"Error updating global best from island {island_idx+1}: {e_fit_update}", exc_info=True)


        history_of_global_best_fitness.append(global_best_fitness)

        if (
            (current_generation + 1) % migration_frequency == 0
            and num_migrants > 0
            and num_islands > 1
        ):
            logger.info(
                f"--- Migration event at generation {current_generation + 1} using {migration_topology} topology ---"
            )
            if migration_topology == "ring":
                _perform_ring_migration(islands, num_migrants, island_population_size, logger)
            elif migration_topology == "random_pair":
                _perform_random_pair_migration(islands, num_migrants, island_population_size, logger)
            elif migration_topology == "broadcast_best_to_all":
                # For broadcast, we need the current global_best_solution
                # This should be updated *before* this migration call if it's based on pre-migration fitness.
                # The existing logic updates global_best_solution after evolution and before migration.
                _perform_broadcast_migration(islands, global_best_solution, island_population_size, logger, verbose)
            else:
                logger.warning(f"Unknown migration topology: {migration_topology}. Skipping migration.")
        elif num_islands <= 1 and (current_generation + 1) % migration_frequency == 0 :
            logger.debug(f"Migration skipped: Not enough islands ({num_islands}) for migration.")

    logger.info(
        f"Genetic Algorithm - Island Model finished. Final Global Best Fitness: {global_best_fitness:.4f}"
    )
    return global_best_solution, global_best_fitness, history_of_global_best_fitness
