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
from typing import List, Callable, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import necessary functions and modules
from genetic_algorithms import _evolve_one_ga_generation, generate_population, calculate_genotypic_diversity
import numpy as np # Ensure numpy is imported

# Helper function to calculate metrics for a single island
def _calculate_island_metrics(island_population: List[LeagueSolution]) -> Dict[str, float]:
    """
    Calculates best_fitness, avg_fitness, std_fitness, and geno_diversity for an island.
    """
    if not island_population:
        return {
            'best_fitness': float('inf'),
            'avg_fitness': float('inf'),
            'std_fitness': 0.0,
            'geno_diversity': 0.0
        }

    fitness_values = [sol.fitness() for sol in island_population if sol] # Ensure sol is not None
    
    if not fitness_values: # If all solutions were None or list became empty
        return {
            'best_fitness': float('inf'),
            'avg_fitness': float('inf'),
            'std_fitness': 0.0,
            'geno_diversity': 0.0 # Assuming diversity is 0 if no valid fitness
        }

    best_fitness = min(fitness_values)
    avg_fitness = np.mean(fitness_values)
    std_fitness = np.std(fitness_values)
    # Assuming calculate_genotypic_diversity handles empty list gracefully or we ensure island_population is not empty here
    geno_diversity = calculate_genotypic_diversity(island_population) 

    return {
        'best_fitness': best_fitness,
        'avg_fitness': avg_fitness,
        'std_fitness': std_fitness,
        'geno_diversity': geno_diversity
    }

# Refactored Migration Helper Functions
def _perform_ring_migration(
    islands_list: List[List[LeagueSolution]], 
    num_migrants: int, 
    island_population_size: int, 
    logger_instance: logging.Logger
) -> List[Dict[str, Any]]:
    """Handles ring migration logic and returns a list of migration events."""
    migration_events: List[Dict[str, Any]] = []
    num_islands = len(islands_list)
    if num_islands <= 1:
        logger_instance.debug("Ring migration skipped: Not enough islands.")
        return migration_events

    migrants_to_send_per_island: List[Optional[List[LeagueSolution]]] = [None] * num_islands # Store actual migrants
    
    for i in range(num_islands):
        if not islands_list[i]:
            logger_instance.warning(f"Migration (Ring): Island {i+1} is empty, cannot select migrants.")
            continue
        islands_list[i].sort(key=lambda s: s.fitness())
        actual_num_migrants_to_select = min(num_migrants, len(islands_list[i]))
        if actual_num_migrants_to_select > 0:
            migrants_to_send_per_island[i] = deepcopy(islands_list[i][:actual_num_migrants_to_select])
        logger_instance.debug(f"Island {i+1} selected {len(migrants_to_send_per_island[i]) if migrants_to_send_per_island[i] else 0} migrants for ring migration.")

    for i_target in range(num_islands):
        source_island_idx = (i_target - 1 + num_islands) % num_islands
        
        # Use the actual selected migrants from migrants_to_send_per_island
        migrants_from_source = migrants_to_send_per_island[source_island_idx]

        if not migrants_from_source: 
            logger_instance.debug(f"Ring migration: No migrants from island {source_island_idx+1} to island {i_target+1}.")
            continue

        migrant_fitnesses = [sol.fitness() for sol in migrants_from_source]
        num_actual_migrants = len(migrants_from_source)

        event = {
            'source_island_idx': source_island_idx,
            'target_island_idx': i_target,
            'num_migrants': num_actual_migrants,
            'migrants_fitnesses': migrant_fitnesses
        }
        migration_events.append(event)

        if not islands_list[i_target]: 
            islands_list[i_target] = migrants_from_source[:island_population_size] 
            logger_instance.info(f"Island {i_target+1} (empty) populated by {len(islands_list[i_target])} migrants from island {source_island_idx+1}.")
        else:
            islands_list[i_target].sort(key=lambda s: s.fitness(), reverse=True) 
            num_to_replace = min(num_actual_migrants, len(islands_list[i_target]))
            
            temp_destination_pop = islands_list[i_target][num_to_replace:]
            temp_destination_pop.extend(migrants_from_source) 
            islands_list[i_target] = sorted(temp_destination_pop, key=lambda s: s.fitness())[:island_population_size]
            logger_instance.info(f"Island {i_target+1} received {num_actual_migrants} migrants from island {source_island_idx+1}. Pop size: {len(islands_list[i_target])}")
    
    return migration_events

def _perform_random_pair_migration(
    islands_list: List[List[LeagueSolution]], 
    num_migrants: int, 
    island_population_size: int, 
    logger_instance: logging.Logger
) -> List[Dict[str, Any]]:
    """Handles random pair migration logic and returns a list of migration events."""
    migration_events: List[Dict[str, Any]] = []
    num_islands = len(islands_list)
    if num_islands < 2:
        logger_instance.debug("Random pair migration skipped: Need at least 2 islands.")
        return migration_events
    
    idx1, idx2 = random.sample(range(num_islands), 2)
    logger_instance.debug(f"Migration (Random Pair): Attempting exchange between Island {idx1+1} and Island {idx2+1}.")

    # Migration from idx1 to idx2
    event1 = _migrate_between_two(idx1, idx2, islands_list, num_migrants, island_population_size, logger_instance, log_prefix="  ")
    if event1:
        migration_events.append(event1)
    
    # Migration from idx2 to idx1
    event2 = _migrate_between_two(idx2, idx1, islands_list, num_migrants, island_population_size, logger_instance, log_prefix="  ")
    if event2:
        migration_events.append(event2)
        
    return migration_events

def _perform_broadcast_migration(
    islands_list: List[List[LeagueSolution]], 
    global_best_solution: Optional[LeagueSolution], 
    island_population_size: int, 
    logger_instance: logging.Logger, 
    verbose: bool
) -> List[Dict[str, Any]]:
    """Handles broadcast best migration logic and returns a list of migration events."""
    migration_events: List[Dict[str, Any]] = []
    num_islands = len(islands_list)
    if not global_best_solution:
        logger_instance.warning("Migration (Broadcast): No global best solution available to broadcast.")
        return migration_events

    global_best_fitness = global_best_solution.fitness()
    logger_instance.debug(f"Migration (Broadcast): Best (fitness {global_best_fitness:.4f}) to all islands.")

    for i_target_island in range(num_islands):
        event_details_for_island = {
            'source_island_idx': -1, # Special value for global best
            'target_island_idx': i_target_island,
            'num_migrants': 0, # Will be 1 if migration happens
            'migrants_fitnesses': [] # Will contain global_best_fitness if migration happens
        }
        
        migrated_this_island = False
        if not islands_list[i_target_island]: 
            islands_list[i_target_island] = [deepcopy(global_best_solution)]
            logger_instance.info(f"  Island {i_target_island+1} (empty) received broadcast best.")
            migrated_this_island = True
        else:
            already_present = any(sol.repr == global_best_solution.repr for sol in islands_list[i_target_island])
            if not already_present:
                islands_list[i_target_island].sort(key=lambda s: s.fitness(), reverse=True) 
                if len(islands_list[i_target_island]) < island_population_size:
                    islands_list[i_target_island].append(deepcopy(global_best_solution))
                elif islands_list[i_target_island]: 
                    islands_list[i_target_island][0] = deepcopy(global_best_solution)
                
                islands_list[i_target_island].sort(key=lambda s: s.fitness()) 
                islands_list[i_target_island] = islands_list[i_target_island][:island_population_size] 
                logger_instance.info(f"  Island {i_target_island+1} received broadcast best. Pop size: {len(islands_list[i_target_island])}")
                migrated_this_island = True
            elif verbose: 
                logger_instance.debug(f"  Island {i_target_island+1} already contains broadcast best. Not adding clone.")

        if migrated_this_island:
            event_details_for_island['num_migrants'] = 1
            event_details_for_island['migrants_fitnesses'] = [global_best_fitness]
            migration_events.append(event_details_for_island)
            
    return migration_events


# Helper function for Island Model GA
def _evolve_one_generation( # This function is now a wrapper
    current_population: list,
    problem_params: dict, 
    single_gen_ga_params: dict,
    players_data: list,
    logger_instance: logging.Logger,
) -> tuple[List[LeagueSolution], float, float, float]: # Modified return type
    """
    Evolves a population for a single generation using specified GA operators.
    Delegates to _evolve_one_ga_generation from genetic_algorithms.py.
    Captures pre-evolution metrics for the island.
    Returns the evolved population and its pre-evolution metrics.
    """
    population_size = single_gen_ga_params["population_size"] # Needed for recovery

    # Calculate pre-evolution metrics for the current island's population
    # These are metrics *before* calling the evolution step for this generation
    if current_population:
        pre_evol_metrics = _calculate_island_metrics(current_population)
        pre_evol_avg_fitness = pre_evol_metrics['avg_fitness']
        pre_evol_std_fitness = pre_evol_metrics['std_fitness']
        pre_evol_geno_diversity = pre_evol_metrics['geno_diversity']
        # pre_evol_best_fitness is also available in pre_evol_metrics if needed later for island-specific pre-evolution best
    else:
        pre_evol_avg_fitness = float('inf')
        pre_evol_std_fitness = 0.0
        pre_evol_geno_diversity = 0.0
        logger_instance.warning(
            "_evolve_one_generation: current_population is empty before calculating pre-evolution metrics."
        )

    if not current_population: # This block handles recovery if population is empty
        logger_instance.warning(
            "_evolve_one_generation: Received empty current_population. Attempting recovery."
        )
        try:
            current_population = generate_population(
                players_data=players_data,
                population_size=population_size,
                num_teams=problem_params["num_teams"],
                team_size=problem_params["team_size"],
                max_budget=problem_params["max_budget"],
                position_requirements=problem_params["position_requirements"],
                logger_instance=logger_instance,
                max_initial_solution_attempts=single_gen_ga_params.get('max_initial_solution_attempts', 20)
            )
            if current_population:
                logger_instance.info(f"_evolve_one_generation: Recovered with {len(current_population)} new solutions.")
                # Recalculate pre-evolution metrics if recovery happened
                recovered_metrics = _calculate_island_metrics(current_population)
                pre_evol_avg_fitness = recovered_metrics['avg_fitness']
                pre_evol_std_fitness = recovered_metrics['std_fitness']
                pre_evol_geno_diversity = recovered_metrics['geno_diversity']
            else:
                logger_instance.error("_evolve_one_generation: Recovery failed, no new solutions generated.")
                # Return empty list and the (likely poor) pre-evol metrics if recovery fails
                return [], pre_evol_avg_fitness, pre_evol_std_fitness, pre_evol_geno_diversity
        except Exception as e_rec:
            logger_instance.error(f"_evolve_one_generation: Exception during recovery population generation: {e_rec}", exc_info=True)
            return [], pre_evol_avg_fitness, pre_evol_std_fitness, pre_evol_geno_diversity

    selection_kwargs = single_gen_ga_params.get("selection_params", {}).copy()
    if (
        single_gen_ga_params["selection_operator"].__name__ == "selection_boltzmann"
        and "safe_exp_func" not in selection_kwargs
    ):
        selection_kwargs["safe_exp_func"] = single_gen_ga_params.get("safe_exp_func", safe_exp)

    # Call the imported _evolve_one_ga_generation from genetic_algorithms.py
    # This function now returns (new_population, avg_fitness, std_fitness, geno_diversity)
    # The metrics it returns are based on its *input* population (which is current_population here)
    # These are essentially the same pre-evolution metrics we just calculated, so we can ignore them from its return if desired,
    # or use them for verification. The subtask asks to capture them *before* this call.
    evolved_population_from_ga, _, _, _ = _evolve_one_ga_generation(
        current_population=current_population, # This is the island's population
        population_size=single_gen_ga_params["population_size"], # Target size for this island
        elitism_size=single_gen_ga_params.get("elitism_size", 0),
        selection_operator=single_gen_ga_params["selection_operator"],
        selection_kwargs=selection_kwargs,
        crossover_operator=single_gen_ga_params["crossover_operator"],
        crossover_rate=single_gen_ga_params["crossover_rate"],
        mutation_operator=single_gen_ga_params["mutation_operator"],
        prob_apply_mutation=single_gen_ga_params.get("mutation_rate", 0.1),
        # logger_instance for _evolve_one_ga_generation is its own module's logger (genetic_algorithms.logger)
        logger_instance=logging.getLogger('genetic_algorithms') # Be explicit if passing a logger
    )
    
    # Return the newly evolved population for this island, along with the pre-evolution metrics captured earlier
    return evolved_population_from_ga, pre_evol_avg_fitness, pre_evol_std_fitness, pre_evol_geno_diversity


# Helper for random pair migration
def _migrate_between_two( # This function remains the same
    source_idx,
    target_idx,
    islands_list,
    num_m,
    pop_size,
    logger_instance, 
    log_prefix="",
) -> Optional[Dict[str, Any]]: # Modified to return an event dictionary or None
    if not islands_list[source_idx] or len(islands_list[source_idx]) == 0:
        logger_instance.debug(
            f"{log_prefix}Source island {source_idx+1} is empty. Cannot migrate."
        )
        return None

    actual_num_migrants_to_select = min(num_m, len(islands_list[source_idx]))
    if actual_num_migrants_to_select == 0:
        logger_instance.debug(
            f"{log_prefix}Not enough individuals in source island {source_idx+1} to migrate "
            f"({len(islands_list[source_idx])} vs {num_m} requested, or num_m is 0)."
        )
        return None

    islands_list[source_idx].sort(key=lambda s: s.fitness())
    migrants_selected = deepcopy(islands_list[source_idx][:actual_num_migrants_to_select])
    
    migrant_fitnesses = [sol.fitness() for sol in migrants_selected]
    num_actual_migrants = len(migrants_selected)

    event = {
        'source_island_idx': source_idx,
        'target_island_idx': target_idx,
        'num_migrants': num_actual_migrants,
        'migrants_fitnesses': migrant_fitnesses
    }

    if not islands_list[target_idx]:
        islands_list[target_idx] = migrants_selected[:pop_size]
        logger_instance.info(
            f"{log_prefix}Migrated {len(islands_list[target_idx])} from Island {source_idx+1} to empty Island {target_idx+1}."
        )
    else:
        islands_list[target_idx].sort(key=lambda s: s.fitness(), reverse=True)
        num_to_replace_in_target = min(num_actual_migrants, len(islands_list[target_idx]))
        
        temp_pop = islands_list[target_idx][num_to_replace_in_target:]
        temp_pop.extend(migrants_selected)
        islands_list[target_idx] = sorted(temp_pop, key=lambda s: s.fitness())[:pop_size]
        logger_instance.info(
            f"{log_prefix}Migrated {num_actual_migrants} from Island {source_idx+1} to Island {target_idx+1}. Target pop size: {len(islands_list[target_idx])}"
        )
    return event


def genetic_algorithm_island_model(
    players_data: list, problem_params: dict, island_model_ga_params: dict
) -> tuple[LeagueSolution | None, float, list]:
    """
    Implements a Genetic Algorithm with an Island Model.
    """
    # Use module-level logger
    # global logger # Not needed if logger is defined at module scope
    
    logger.info(f"Starting Genetic Algorithm - Island Model with params: {island_model_ga_params}")
    num_islands = island_model_ga_params["num_islands"]
    island_population_size = island_model_ga_params["island_population_size"]
    max_generations_total = island_model_ga_params["max_generations_total"]
    migration_frequency = island_model_ga_params["migration_frequency"]
    num_migrants = island_model_ga_params["num_migrants"]
    migration_topology = island_model_ga_params.get("migration_topology", "ring")
    ga_params_per_island = island_model_ga_params["ga_params_per_island"].copy()
    verbose = island_model_ga_params.get("verbose", False)

    if ga_params_per_island.get("population_size") != island_population_size:
        logger.warning(f"Adjusting ga_params_per_island['population_size'] to {island_population_size} for consistency.")
        ga_params_per_island["population_size"] = island_population_size

    max_init_attempts_for_island_pop = ga_params_per_island.get("max_initial_solution_attempts", 50)

    islands: List[List[LeagueSolution]] = []
    initial_islands_data: List[Dict[str, Any]] = []
    logger.info(f"Initializing {num_islands} islands, each with population size {island_population_size}.")

    for i in range(num_islands):
        if verbose: logger.info(f"Initializing island {i+1}/{num_islands}...")
        try:
            island_pop = generate_population(
                players_data, island_population_size,
                problem_params["num_teams"], problem_params["team_size"], problem_params["max_budget"],
                problem_params["position_requirements"], logger, max_init_attempts_for_island_pop
            )
            if not island_pop:
                logger.error(f"Island {i+1} failed to generate initial population. Critical error.")
                raise RuntimeError(f"Island {i+1} failed to generate initial population.")
            islands.append(island_pop)
            island_metrics = _calculate_island_metrics(island_pop)
            initial_islands_data.append({
                'island_id': i,
                'best_fitness': island_metrics['best_fitness'],
                'avg_fitness': island_metrics['avg_fitness'],
                'std_fitness': island_metrics['std_fitness'],
                'geno_diversity': island_metrics['geno_diversity']
            })
            logger.debug(f"Island {i+1} initialized with {len(island_pop)} individuals. Metrics: {island_metrics}")
        except RuntimeError as e:
            logger.critical(f"Fatal Error: Could not initialize island {i+1}. Error: {e}")
            raise

    if not islands or any(not island_pop for island_pop in islands):
        logger.critical("One or more islands empty after initialization. Stopping.")
        return None, float("inf"), []

    global_best_solution: Optional[LeagueSolution] = None
    global_best_fitness = float("inf")

    for island_idx, island_metrics in enumerate(initial_islands_data):
        if island_metrics['best_fitness'] < global_best_fitness:
            global_best_fitness = island_metrics['best_fitness']
            # Find the best solution in that island to set as global_best_solution
            if islands[island_idx]: # Should not be empty if best_fitness is not inf
                 try:
                    global_best_solution = deepcopy(min(islands[island_idx], key=lambda s: s.fitness()))
                 except ValueError: # Should not happen if island_metrics['best_fitness'] was updated
                    logger.error(f"Error retrieving best solution from island {island_idx} despite valid metrics.")


    history_log: List[Dict[str, Any]] = [{
        'generation': 0,
        'global_best_fitness': global_best_fitness,
        'islands_data': initial_islands_data,
        'migration_events': [] # No migration at generation 0
    }]

    if global_best_solution: logger.info(f"Initial global best fitness: {global_best_fitness:.4f}")
    else: logger.warning(f"Initial global best: {global_best_fitness:.4f} (No valid solution found)")

    logger.info(f"Starting evolution for {max_generations_total} generations.")
    for current_generation in range(max_generations_total):
        current_gen_island_data_list: List[Dict[str, Any]] = []
        
        log_level_gen = logging.DEBUG if not verbose else logging.INFO
        if ((current_generation + 1) % 10 == 0 or current_generation == max_generations_total - 1 or verbose):
            logger.log(log_level_gen, f"Island GA - Gen {current_generation + 1}/{max_generations_total}. Current Global Best: {global_best_fitness:.4f}")

        for i_island in range(num_islands):
            if not islands[i_island]:
                logger.warning(f"Island {i_island+1} is empty at start of gen {current_generation+1}. Logging empty metrics.")
                # Log empty/default metrics for this island for this generation
                current_gen_island_data_list.append({
                    'island_id': i_island, 'best_fitness': float('inf'), 'avg_fitness': float('inf'),
                    'std_fitness': 0.0, 'geno_diversity': 0.0
                })
                continue

            # _evolve_one_generation now returns: evolved_pop, pre_avg, pre_std, pre_div
            evolved_pop, pre_avg, pre_std, pre_div = _evolve_one_generation(
                current_population=islands[i_island],
                problem_params=problem_params,
                single_gen_ga_params=ga_params_per_island,
                players_data=players_data,
                logger_instance=logger,
            )
            islands[i_island] = evolved_pop # Update island with the new population
            
            # Capture pre-evolution best fitness for this island
            pre_evol_island_best_fitness = float('inf')
            if history_log and current_generation < len(history_log) and \
               history_log[current_generation]['islands_data'] and \
               i_island < len(history_log[current_generation]['islands_data']):
                pre_evol_island_best_fitness = history_log[current_generation]['islands_data'][i_island].get('best_fitness', float('inf'))
            elif current_generation == 0 and initial_islands_data and i_island < len(initial_islands_data):
                 pre_evol_island_best_fitness = initial_islands_data[i_island].get('best_fitness', float('inf'))


            current_gen_island_data_list.append({
                'island_id': i_island,
                'best_fitness': pre_evol_island_best_fitness, # Best fitness *before* this generation's evolution
                'avg_fitness': pre_avg,
                'std_fitness': pre_std,
                'geno_diversity': pre_div
            })
            if not islands[i_island]:
                 logger.warning(f"Island {i_island+1} became empty after evolution in gen {current_generation+1}.")


        # Update global best based on post-evolution populations
        for island_idx, island_pop_after_evol in enumerate(islands):
            if not island_pop_after_evol: continue
            try:
                current_island_best_sol = min(island_pop_after_evol, key=lambda s: s.fitness())
                ind_fitness = current_island_best_sol.fitness()
                if ind_fitness < global_best_fitness:
                    global_best_fitness = ind_fitness
                    global_best_solution = deepcopy(current_island_best_sol)
                    if verbose: logger.info(f"Gen {current_generation + 1}: New global best from island {island_idx+1}! Fitness: {global_best_fitness:.4f}")
            except ValueError: logger.warning(f"Island {island_idx+1} empty after evolution. Cannot update global best.")
            except Exception as e_fit_update: logger.error(f"Error updating global best from island {island_idx+1}: {e_fit_update}", exc_info=True)

        migration_events_this_generation: List[Dict[str, Any]] = []
        if (current_generation + 1) % migration_frequency == 0 and num_migrants > 0 and num_islands > 1:
            logger.info(f"--- Migration event at generation {current_generation + 1} using {migration_topology} topology ---")
            if migration_topology == "ring":
                migration_events_this_generation = _perform_ring_migration(islands, num_migrants, island_population_size, logger)
            elif migration_topology == "random_pair":
                migration_events_this_generation = _perform_random_pair_migration(islands, num_migrants, island_population_size, logger)
            elif migration_topology == "broadcast_best_to_all":
                migration_events_this_generation = _perform_broadcast_migration(islands, global_best_solution, island_population_size, logger, verbose)
            else:
                logger.warning(f"Unknown migration topology: {migration_topology}. Skipping migration.")
        elif num_islands <= 1 and (current_generation + 1) % migration_frequency == 0:
            logger.debug(f"Migration skipped: Not enough islands ({num_islands}) for migration.")

        history_log.append({
            'generation': current_generation + 1,
            'global_best_fitness': global_best_fitness,
            'islands_data': current_gen_island_data_list,
            'migration_events': migration_events_this_generation
        })

    logger.info(f"Genetic Algorithm - Island Model finished. Final Global Best Fitness: {global_best_fitness:.4f}")
    return global_best_solution, global_best_fitness, history_log
