"""
Island Genetic Algorithm adapter module for Fantasy League Optimization.

This module provides the island_genetic_algorithm function that was missing from the original codebase.
"""

import sys
import logging
from copy import deepcopy

# Add project root to path
sys.path.append('/home/ubuntu/upload')

# Import from ga_island_model.py
from ga_island_model import _evolve_one_generation, _perform_ring_migration, _perform_random_pair_migration, _perform_broadcast_migration

# Configure logger
logger = logging.getLogger(__name__)

def island_genetic_algorithm(players_data, problem_params, island_ga_params):
    """
    Island Model Genetic Algorithm implementation.
    
    Args:
        players_data: List of player data
        problem_params: Dictionary of problem parameters
        island_ga_params: Dictionary of Island GA parameters
        
    Returns:
        Tuple of (best_solution, best_fitness, history)
    """
    from genetic_algorithms import generate_population
    
    # Extract parameters
    num_islands = island_ga_params.get("num_islands", 4)
    island_population_sizes = island_ga_params.get("island_population_sizes", [50] * num_islands)
    if isinstance(island_population_sizes, int):
        island_population_sizes = [island_population_sizes] * num_islands
    
    max_generations = island_ga_params.get("max_generations", 100)
    
    selection_operator = island_ga_params.get("selection_operator")
    selection_params = island_ga_params.get("selection_params", {})
    
    crossover_operator = island_ga_params.get("crossover_operator")
    crossover_rate = island_ga_params.get("crossover_rate", 0.8)
    
    mutation_operator = island_ga_params.get("mutation_operator")
    mutation_rate = island_ga_params.get("mutation_rate", 0.1)
    
    elitism_size = island_ga_params.get("elitism_size", 1)
    
    migration_frequency = island_ga_params.get("migration_frequency", 10)
    migration_rate = island_ga_params.get("migration_rate", 0.2)
    migration_topology = island_ga_params.get("migration_topology", "ring")
    
    verbose = island_ga_params.get("verbose", False)
    
    # Initialize islands
    islands = []
    for i in range(num_islands):
        island_population = generate_population(
            players_data=players_data,
            population_size=island_population_sizes[i],
            num_teams=problem_params["num_teams"],
            team_size=problem_params["team_size"],
            max_budget=problem_params["max_budget"],
            position_requirements=problem_params["position_requirements"],
            logger_instance=logger,
            max_initial_solution_attempts=island_ga_params.get("max_initial_solution_attempts", 50)
        )
        islands.append(island_population)
    
    # Initialize global best solution
    global_best_solution = None
    global_best_fitness = float('inf')
    
    # Initialize history log
    history_log = []
    
    # Run evolution for each generation
    for current_generation in range(max_generations):
        # Evolve each island
        current_gen_island_data_list = []
        migration_events_this_generation = []
        
        for i, island_population in enumerate(islands):
            # Configure GA parameters for this island
            single_gen_ga_params = {
                "population_size": island_population_sizes[i],
                "selection_operator": selection_operator,
                "selection_params": selection_params,
                "crossover_operator": crossover_operator,
                "crossover_rate": crossover_rate,
                "mutation_operator": mutation_operator,
                "mutation_rate": mutation_rate,
                "elitism_size": elitism_size,
                "safe_exp_func": island_ga_params.get("safe_exp_func")
            }
            
            # Evolve island population
            new_island_population, avg_fitness, std_fitness, geno_diversity = _evolve_one_generation(
                current_population=island_population,
                problem_params=problem_params,
                single_gen_ga_params=single_gen_ga_params,
                players_data=players_data,
                logger_instance=logger
            )
            
            # Update island population
            islands[i] = new_island_population
            
            # Calculate island metrics
            island_best_solution = None
            island_best_fitness = float('inf')
            
            if new_island_population:
                island_best_solution = min(new_island_population, key=lambda x: x.fitness())
                island_best_fitness = island_best_solution.fitness()
                
                # Update global best solution
                if island_best_fitness < global_best_fitness:
                    global_best_solution = deepcopy(island_best_solution)
                    global_best_fitness = island_best_fitness
            
            # Record island data
            island_data = {
                "island_idx": i,
                "population_size": len(new_island_population),
                "best_fitness": island_best_fitness,
                "avg_fitness": avg_fitness,
                "std_fitness": std_fitness,
                "geno_diversity": geno_diversity
            }
            current_gen_island_data_list.append(island_data)
        
        # Perform migration if needed
        if migration_frequency > 0 and (current_generation + 1) % migration_frequency == 0 and num_islands > 1:
            # Calculate number of migrants
            num_migrants = max(1, int(min(island_population_sizes) * migration_rate))
            
            # Perform migration based on topology
            if migration_topology == "ring":
                migration_events_this_generation = _perform_ring_migration(islands, num_migrants, min(island_population_sizes), logger)
            elif migration_topology == "random_pair":
                migration_events_this_generation = _perform_random_pair_migration(islands, num_migrants, min(island_population_sizes), logger)
            elif migration_topology == "broadcast_best_to_all":
                migration_events_this_generation = _perform_broadcast_migration(islands, global_best_solution, min(island_population_sizes), logger, verbose)
            else:
                logger.warning(f"Unknown migration topology: {migration_topology}. Skipping migration.")
        elif num_islands <= 1 and (current_generation + 1) % migration_frequency == 0:
            logger.debug(f"Migration skipped: Not enough islands ({num_islands}) for migration.")
        
        # Record history
        history_log.append({
            'generation': current_generation + 1,
            'global_best_fitness': global_best_fitness,
            'islands_data': current_gen_island_data_list,
            'migration_events': migration_events_this_generation
        })
    
    logger.info(f"Genetic Algorithm - Island Model finished. Final Global Best Fitness: {global_best_fitness:.4f}")
    return global_best_solution, global_best_fitness, history_log
