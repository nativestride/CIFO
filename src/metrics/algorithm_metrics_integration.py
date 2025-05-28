"""
Integration module for connecting optimization algorithms with metric collection.

This module provides wrapper functions and decorators to integrate the metric collection
framework with the existing Fantasy League optimization algorithms.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import numpy as np

from metrics.metric_collector import (
    MetricCollector,
    HillClimbingMetricCollector,
    SimulatedAnnealingMetricCollector,
    GeneticAlgorithmMetricCollector,
    IslandGAMetricCollector,
    create_metric_collector
)

logger = logging.getLogger(__name__)

# Hill Climbing Integration
def hill_climbing_with_metrics(
    initial_solution,
    max_iterations=1000,
    max_no_improvement=100,
    verbose=False,
    metric_collector: Optional[HillClimbingMetricCollector] = None,
    config_name: str = "HillClimbing_Custom",
    **kwargs
):
    """
    Hill Climbing with integrated metric collection.
    
    Args:
        initial_solution: Initial solution object
        max_iterations: Maximum number of iterations
        max_no_improvement: Maximum iterations without improvement
        verbose: Whether to print progress
        metric_collector: Optional pre-configured metric collector
        config_name: Configuration name for metrics
        **kwargs: Additional arguments for hill climbing
        
    Returns:
        Tuple of (best_solution, best_fitness, history, metrics)
    """
    # Create metric collector if not provided
    if metric_collector is None:
        metric_collector = HillClimbingMetricCollector(config_name)
    
    # Set parameters
    metric_collector.set_parameters({
        "max_iterations": max_iterations,
        "max_no_improvement": max_no_improvement,
        "verbose": verbose,
        **kwargs
    })
    
    # Start timer
    metric_collector.start_timer()
    
    # Initialize
    current_sol = initial_solution
    current_fitness = current_sol.fitness()
    metric_collector.record_iteration(current_fitness)
    
    history = [current_fitness]
    iterations_without_improvement = 0
    
    for iteration in range(max_iterations):
        # Get neighbors
        neighbors = current_sol.get_neighbors()
        metric_collector.record_neighbor_generation(len(neighbors))
        
        if not neighbors:
            if verbose:
                print(f"Hill Climbing: No valid neighbors found at iteration {iteration}. Stopping.")
            metric_collector.set_early_termination("No valid neighbors")
            break
        
        # Evaluate neighbors
        metric_collector.record_neighbor_evaluation(len(neighbors))
        
        # Find best neighbor
        best_neighbor = min(neighbors, key=lambda x: x.fitness())
        best_neighbor_fitness = best_neighbor.fitness()
        
        if best_neighbor_fitness < current_fitness:
            current_sol = best_neighbor
            current_fitness = best_neighbor_fitness
            history.append(current_fitness)
            
            # Record improvement
            metric_collector.record_iteration(current_fitness, True)
            
            iterations_without_improvement = 0
            if verbose:
                print(f"Hill Climbing: Iteration {iteration}, Fitness = {current_fitness:.6f}")
        else:
            iterations_without_improvement += 1
            
            # Record non-improvement
            metric_collector.record_iteration(current_fitness, False)
            
            if iterations_without_improvement >= max_no_improvement:
                if verbose:
                    print(f"Hill Climbing: Stopping after {max_no_improvement} iterations without improvement (Iter {iteration}).")
                metric_collector.set_early_termination(f"Max no improvement ({max_no_improvement})")
                metric_collector.record_plateau(iterations_without_improvement)
                break
    
    # Record local optimum if terminated due to no improvement
    if iterations_without_improvement >= max_no_improvement:
        metric_collector.record_local_optimum()
    
    # Stop timer
    metric_collector.stop_timer()
    
    # Calculate derived metrics
    metric_collector.calculate_derived_metrics()
    
    return current_sol, current_fitness, history, metric_collector.get_metrics()


# Hill Climbing with Random Restart Integration
def hill_climbing_random_restart_with_metrics(
    initial_solution_params: dict,
    solution_class_for_hc: type,
    num_restarts: int,
    max_iterations_per_hc: int,
    max_no_improvement_per_hc: int,
    verbose: bool = False,
    metric_collector: Optional[HillClimbingMetricCollector] = None,
    config_name: str = "HillClimbing_RandomRestart_Custom",
    hc_specific_kwargs: dict = None,
):
    """
    Hill Climbing with Random Restart with integrated metric collection.
    
    Args:
        initial_solution_params: Parameters for creating initial solutions
        solution_class_for_hc: Solution class to use
        num_restarts: Number of restarts
        max_iterations_per_hc: Maximum iterations per hill climbing run
        max_no_improvement_per_hc: Maximum iterations without improvement per run
        verbose: Whether to print progress
        metric_collector: Optional pre-configured metric collector
        config_name: Configuration name for metrics
        hc_specific_kwargs: Additional arguments for hill climbing
        
    Returns:
        Tuple of (best_solution, best_fitness, history, metrics)
    """
    if hc_specific_kwargs is None:
        hc_specific_kwargs = {}
    
    # Create metric collector if not provided
    if metric_collector is None:
        metric_collector = HillClimbingMetricCollector(config_name)
    
    # Set parameters
    metric_collector.set_parameters({
        "num_restarts": num_restarts,
        "max_iterations_per_hc": max_iterations_per_hc,
        "max_no_improvement_per_hc": max_no_improvement_per_hc,
        "verbose": verbose,
        **hc_specific_kwargs
    })
    
    # Start timer
    metric_collector.start_timer()
    
    logger.info(f"Starting Hill Climbing with Random Restart: {num_restarts} restarts planned.")
    
    best_solution_overall = None
    best_fitness_overall = float("inf")
    history_of_best_fitness_per_restart = []
    
    for i_restart in range(num_restarts):
        logger.info(f"HCRR Restart {i_restart + 1}/{num_restarts} starting...")
        
        # Record restart
        metric_collector.record_restart()
        
        current_initial_sol = None
        MAX_INIT_ATTEMPTS = 10
        
        # Create valid initial solution
        for attempt in range(MAX_INIT_ATTEMPTS):
            try:
                temp_sol = solution_class_for_hc(**initial_solution_params)
                if temp_sol.is_valid():
                    current_initial_sol = temp_sol
                    logger.info(f"HCRR Restart {i_restart + 1}: Valid initial solution created on attempt {attempt + 1} (Fitness: {current_initial_sol.fitness():.4f}).")
                    break
                else:
                    logger.debug(f"HCRR Restart {i_restart + 1}, Attempt {attempt + 1}: Initial solution invalid.")
                    metric_collector.record_constraint_violation()
            except Exception as e:
                logger.warning(f"HCRR Restart {i_restart + 1}, Attempt {attempt + 1}: Error creating initial solution - {e}")
                metric_collector.record_constraint_violation()
        
        if current_initial_sol is None:
            logger.warning(f"HCRR Restart {i_restart + 1}: Failed to create a valid initial solution after {MAX_INIT_ATTEMPTS} attempts. Skipping this restart.")
            history_of_best_fitness_per_restart.append(float("inf"))
            continue
        
        # Run hill climbing for this restart
        inner_hc_verbose = hc_specific_kwargs.get("verbose", False)
        hc_run_params = hc_specific_kwargs.copy()
        
        # Create a sub-collector for this HC run
        sub_collector = HillClimbingMetricCollector(f"{config_name}_Restart_{i_restart+1}")
        
        best_sol_this_restart, best_fit_this_restart, _, sub_metrics = hill_climbing_with_metrics(
            initial_solution=current_initial_sol,
            max_iterations=max_iterations_per_hc,
            max_no_improvement=max_no_improvement_per_hc,
            verbose=inner_hc_verbose,
            metric_collector=sub_collector,
            **hc_run_params
        )
        
        # Aggregate metrics from sub-collector
        metric_collector.metrics["function_evaluations"] += sub_collector.metrics["function_evaluations"]
        metric_collector.metrics["neighbors_generated"] += sub_collector.metrics["neighbors_generated"]
        metric_collector.metrics["neighbors_evaluated"] += sub_collector.metrics["neighbors_evaluated"]
        metric_collector.metrics["local_optima_count"] += sub_collector.metrics["local_optima_count"]
        
        logger.info(f"HCRR Restart {i_restart + 1} completed. Best fitness for this restart: {best_fit_this_restart:.4f}")
        history_of_best_fitness_per_restart.append(best_fit_this_restart)
        
        if best_fit_this_restart < best_fitness_overall:
            best_fitness_overall = best_fit_this_restart
            best_solution_overall = best_sol_this_restart
            
            # Record new best solution
            metric_collector.metrics["best_fitness"] = best_fitness_overall
            metric_collector.metrics["iterations_to_best"] = metric_collector.metrics["iterations"]
            
            logger.info(f"HCRR Restart {i_restart + 1}: New GLOBAL best solution found! Fitness: {best_fitness_overall:.4f}")
    
    # Record final iteration
    metric_collector.record_iteration(best_fitness_overall)
    
    # Stop timer
    metric_collector.stop_timer()
    
    # Calculate derived metrics
    metric_collector.calculate_derived_metrics()
    
    logger.info(f"Hill Climbing with Random Restart finished. Overall best fitness: {best_fitness_overall:.4f}")
    return best_solution_overall, best_fitness_overall, history_of_best_fitness_per_restart, metric_collector.get_metrics()


# Simulated Annealing Integration
def simulated_annealing_with_metrics(
    initial_solution,
    initial_temperature=100.0,
    cooling_rate=0.95,
    min_temperature=0.1,
    iterations_per_temp=20,
    maximization=False,
    verbose=False,
    safe_exp_func=None,
    metric_collector: Optional[SimulatedAnnealingMetricCollector] = None,
    config_name: str = "SimulatedAnnealing_Custom",
    **kwargs
):
    """
    Simulated Annealing with integrated metric collection.
    
    Args:
        initial_solution: Initial solution object
        initial_temperature: Initial temperature
        cooling_rate: Cooling rate
        min_temperature: Minimum temperature
        iterations_per_temp: Iterations per temperature
        maximization: Whether to maximize (True) or minimize (False)
        verbose: Whether to print progress
        safe_exp_func: Function for safe exponential calculation
        metric_collector: Optional pre-configured metric collector
        config_name: Configuration name for metrics
        **kwargs: Additional arguments for simulated annealing
        
    Returns:
        Tuple of (best_solution, best_fitness, history, metrics)
    """
    import random
    from copy import deepcopy
    
    # Create metric collector if not provided
    if metric_collector is None:
        metric_collector = SimulatedAnnealingMetricCollector(config_name)
    
    # Set parameters
    metric_collector.set_parameters({
        "initial_temperature": initial_temperature,
        "cooling_rate": cooling_rate,
        "min_temperature": min_temperature,
        "iterations_per_temp": iterations_per_temp,
        "maximization": maximization,
        "verbose": verbose,
        **kwargs
    })
    
    # Start timer
    metric_collector.start_timer()
    
    # Initialize
    current_solution = initial_solution
    current_fitness = current_solution.fitness()
    
    # Record initial temperature
    metric_collector.record_temperature(initial_temperature)
    
    # Record initial iteration
    metric_collector.record_iteration(current_fitness)
    
    best_solution = deepcopy(current_solution)
    best_fitness = current_fitness
    
    temperature = initial_temperature
    history = [best_fitness]
    
    if verbose:
        print(f"SA Initial: Temp={temperature:.2f}, Fitness={current_fitness:.6f}, Best Fitness={best_fitness:.6f}")
    
    while temperature > min_temperature:
        for i_iter in range(iterations_per_temp):
            # Get random neighbor
            neighbor_solution = current_solution.get_random_neighbor()
            neighbor_fitness = neighbor_solution.fitness()
            
            delta_fitness = neighbor_fitness - current_fitness
            
            # Determine acceptance
            accepted_move = False
            probability = None
            
            if (maximization and delta_fitness > 0) or (not maximization and delta_fitness < 0):
                accepted_move = True  # Accept better solution
            else:
                # Calculate acceptance probability for worse solution
                probability = safe_exp_func(-abs(delta_fitness) / temperature)
                if random.random() < probability:
                    accepted_move = True
            
            # Record solution evaluation
            metric_collector.record_solution_evaluation(
                current_fitness, 
                neighbor_fitness, 
                accepted_move, 
                temperature, 
                probability
            )
            
            if accepted_move:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                
                # Update overall best if current accepted solution is better
                if (maximization and current_fitness > best_fitness) or (not maximization and current_fitness < best_fitness):
                    best_solution = deepcopy(current_solution)
                    best_fitness = current_fitness
                    
                    # Record new best
                    metric_collector.metrics["best_fitness"] = best_fitness
                    
                    if verbose and i_iter % 10 == 0:
                        print(f"  SA New Best: Temp={temperature:.2f}, Iter={i_iter}, Best Fitness={best_fitness:.6f}")
            
            # Record iteration
            metric_collector.record_iteration(best_fitness)
            history.append(best_fitness)
        
        # Cool down
        temperature *= cooling_rate
        
        # Record new temperature
        metric_collector.record_temperature(temperature)
        
        if verbose:
            print(f"SA Cooled: Temp={temperature:.2f}, Current Fitness={current_fitness:.6f}, Best Fitness={best_fitness:.6f}")
    
    # Stop timer
    metric_collector.stop_timer()
    
    # Calculate derived metrics
    metric_collector.calculate_derived_metrics()
    
    if verbose:
        print(f"SA Final Best Fitness: {best_fitness:.6f}")
    
    return best_solution, best_fitness, history, metric_collector.get_metrics()


# Genetic Algorithm Integration
def genetic_algorithm_with_metrics(
    players_data: list,
    problem_params: dict,
    ga_params: dict,
    metric_collector: Optional[GeneticAlgorithmMetricCollector] = None,
    config_name: str = "GeneticAlgorithm_Custom"
):
    """
    Genetic Algorithm with integrated metric collection.
    
    Args:
        players_data: List of player data
        problem_params: Problem parameters
        ga_params: Genetic algorithm parameters
        metric_collector: Optional pre-configured metric collector
        config_name: Configuration name for metrics
        
    Returns:
        Tuple of (best_solution, best_fitness, history, metrics)
    """
    import random
    from copy import deepcopy
    
    # Create metric collector if not provided
    if metric_collector is None:
        metric_collector = GeneticAlgorithmMetricCollector(config_name)
    
    # Set parameters
    metric_collector.set_parameters(ga_params)
    
    # Start timer
    metric_collector.start_timer()
    
    # Extract parameters
    population_size = ga_params["population_size"]
    max_generations = ga_params["max_generations"]
    selection_operator = ga_params["selection_operator"]
    selection_kwargs = ga_params.get("selection_params", {}).copy()
    crossover_operator = ga_params["crossover_operator"]
    crossover_rate = ga_params["crossover_rate"]
    mutation_operator = ga_params["mutation_operator"]
    prob_apply_mutation = ga_params.get("mutation_rate", 0.1)
    elitism_size = ga_params.get("elitism_size", 0)
    verbose = ga_params.get("verbose", False)
    
    # Generate initial population
    from genetic_algorithms import generate_population
    population = generate_population(
        players_data,
        population_size,
        problem_params["num_teams"],
        problem_params["team_size"],
        problem_params["max_budget"],
        problem_params["position_requirements"],
        logger,
        ga_params.get("max_initial_solution_attempts", 50)
    )
    
    if not population:
        logger.error("GA Error: Failed to initialize the population. Aborting.")
        return None, float('inf'), [], metric_collector.get_metrics()
    
    # Find initial best solution
    best_solution_overall = min(population, key=lambda s: s.fitness())
    best_fitness_overall = best_solution_overall.fitness()
    
    # Calculate initial population metrics
    initial_fitness_values = [sol.fitness() for sol in population]
    avg_pop_fitness = np.mean(initial_fitness_values)
    std_pop_fitness = np.std(initial_fitness_values)
    
    # Calculate genotypic diversity
    from genetic_algorithms import calculate_genotypic_diversity
    geno_pop_diversity = calculate_genotypic_diversity(population)
    
    # Record initial generation
    metric_collector.record_generation(
        0, 
        population, 
        best_fitness_overall, 
        avg_pop_fitness, 
        std_pop_fitness, 
        geno_pop_diversity
    )
    
    history = [{
        'best_fitness': best_fitness_overall,
        'avg_fitness': avg_pop_fitness,
        'std_fitness': std_pop_fitness,
        'geno_diversity': geno_pop_diversity
    }]
    
    if verbose:
        logger.info(
            f"Initial best fitness: {best_fitness_overall:.4f}, Avg fitness: {avg_pop_fitness:.4f}, "
            f"Std fitness: {std_pop_fitness:.4f}, Geno diversity: {geno_pop_diversity:.1f}"
        )
    
    # Main GA loop
    for generation in range(max_generations):
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness())
        
        # Apply elitism
        new_population = []
        if elitism_size > 0:
            actual_elitism_size = min(elitism_size, len(population))
            new_population.extend(deepcopy(population[:actual_elitism_size]))
        
        # Breeding loop
        while len(new_population) < population_size:
            # Selection
            parent1 = selection_operator(population, **selection_kwargs)
            parent2 = selection_operator(population, **selection_kwargs)
            
            if parent1 is None or parent2 is None:
                logger.warning(f"GA Gen {generation + 1}: Parent selection failed. Using random fallback.")
                if population:
                    parent1 = deepcopy(random.choice(population)) if parent1 is None else parent1
                    parent2 = deepcopy(random.choice(population)) if parent2 is None else parent2
                else:
                    logger.error(f"GA Gen {generation + 1}: Population empty, cannot select parents.")
                    break
            
            # Record selection
            if parent1 is not None:
                metric_collector.record_selection(id(parent1))
            if parent2 is not None:
                metric_collector.record_selection(id(parent2))
            
            # Crossover
            child1 = None
            if random.random() < crossover_rate and parent1 is not None and parent2 is not None:
                pre_crossover_fitness1 = parent1.fitness()
                pre_crossover_fitness2 = parent2.fitness()
                
                child1 = crossover_operator(parent1, parent2)
                
                if child1 is not None:
                    post_crossover_fitness = child1.fitness()
                    metric_collector.record_crossover(
                        pre_crossover_fitness1, 
                        pre_crossover_fitness2, 
                        post_crossover_fitness, 
                        True
                    )
                else:
                    # Crossover failed
                    metric_collector.record_crossover(
                        pre_crossover_fitness1, 
                        pre_crossover_fitness2, 
                        float('inf'), 
                        False
                    )
                    # Use better parent as fallback
                    child1 = deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)
            else:
                # No crossover, use better parent
                if parent1 is not None and parent2 is not None:
                    child1 = deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)
                elif parent1 is not None:
                    child1 = deepcopy(parent1)
                elif parent2 is not None:
                    child1 = deepcopy(parent2)
            
            # Mutation
            if random.random() < prob_apply_mutation and child1 is not None:
                pre_mutation_fitness = child1.fitness()
                
                mutated_child = mutation_operator(child1)
                
                if mutated_child is not None and mutated_child.is_valid():
                    post_mutation_fitness = mutated_child.fitness()
                    metric_collector.record_mutation(pre_mutation_fitness, post_mutation_fitness, True)
                    child1 = mutated_child
                else:
                    # Mutation failed or produced invalid solution
                    metric_collector.record_mutation(pre_mutation_fitness, float('inf'), False)
            
            # Add to new population if valid
            if child1 is not None and child1.is_valid():
                new_population.append(child1)
            else:
                logger.debug(f"GA Gen {generation + 1}: Generated child was invalid or None. Not added.")
        
        # Update population
        population = new_population[:population_size]
        
        # Calculate population metrics
        fitness_values = [sol.fitness() for sol in population]
        best_fitness = min(fitness_values)
        avg_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        geno_diversity = calculate_genotypic_diversity(population)
        
        # Record generation
        metric_collector.record_generation(
            generation + 1, 
            population, 
            best_fitness, 
            avg_fitness, 
            std_fitness, 
            geno_diversity
        )
        
        # Update best solution
        current_best = min(population, key=lambda s: s.fitness())
        current_best_fitness = current_best.fitness()
        
        if current_best_fitness < best_fitness_overall:
            best_solution_overall = deepcopy(current_best)
            best_fitness_overall = current_best_fitness
        
        # Record history
        history.append({
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'std_fitness': std_fitness,
            'geno_diversity': geno_diversity
        })
        
        if verbose and (generation + 1) % 10 == 0:
            logger.info(
                f"GA Gen {generation + 1}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, "
                f"Std={std_fitness:.4f}, Diversity={geno_diversity:.1f}"
            )
    
    # Stop timer
    metric_collector.stop_timer()
    
    # Calculate derived metrics
    metric_collector.calculate_derived_metrics()
    
    return best_solution_overall, best_fitness_overall, history, metric_collector.get_metrics()


# Island Genetic Algorithm Integration
def island_ga_with_metrics(
    players_data: list,
    problem_params: dict,
    island_model_ga_params: dict,
    metric_collector: Optional[IslandGAMetricCollector] = None,
    config_name: str = "IslandGA_Custom"
):
    """
    Island Genetic Algorithm with integrated metric collection.
    
    Args:
        players_data: List of player data
        problem_params: Problem parameters
        island_model_ga_params: Island GA parameters
        metric_collector: Optional pre-configured metric collector
        config_name: Configuration name for metrics
        
    Returns:
        Tuple of (best_solution, best_fitness, history, metrics)
    """
    import random
    from copy import deepcopy
    
    # Create metric collector if not provided
    if metric_collector is None:
        metric_collector = IslandGAMetricCollector(config_name)
    
    # Set parameters
    metric_collector.set_parameters(island_model_ga_params)
    
    # Start timer
    metric_collector.start_timer()
    
    # Extract parameters
    num_islands = island_model_ga_params["num_islands"]
    island_population_size = island_model_ga_params["island_population_size"]
    max_generations_total = island_model_ga_params.get("max_generations_total", 100)
    migration_frequency = island_model_ga_params["migration_frequency"]
    num_migrants = island_model_ga_params["num_migrants"]
    migration_topology = island_model_ga_params["migration_topology"]
    verbose = island_model_ga_params.get("verbose", False)
    
    # Set number of islands in metric collector
    metric_collector.set_num_islands(num_islands)
    
    # Initialize islands
    from genetic_algorithms import generate_population
    islands = []
    
    for i in range(num_islands):
        island_pop = generate_population(
            players_data,
            island_population_size,
            problem_params["num_teams"],
            problem_params["team_size"],
            problem_params["max_budget"],
            problem_params["position_requirements"],
            logger,
            island_model_ga_params.get("max_initial_solution_attempts", 50)
        )
        islands.append(island_pop)
    
    # Find initial global best
    global_best_solution = None
    global_best_fitness = float('inf')
    
    for island_idx, island_pop in enumerate(islands):
        if island_pop:
            island_best = min(island_pop, key=lambda s: s.fitness())
            island_best_fitness = island_best.fitness()
            
            # Calculate island metrics
            island_fitness_values = [sol.fitness() for sol in island_pop]
            island_avg_fitness = np.mean(island_fitness_values)
            island_diversity = calculate_genotypic_diversity(island_pop)
            
            # Record island metrics
            metric_collector.record_island_metrics(
                island_idx,
                island_best_fitness,
                island_avg_fitness,
                island_diversity
            )
            
            if island_best_fitness < global_best_fitness:
                global_best_solution = deepcopy(island_best)
                global_best_fitness = island_best_fitness
    
    # Initialize history
    history = [{
        'global_best_fitness': global_best_fitness,
        'island_best_fitness': {i: min([sol.fitness() for sol in island_pop]) if island_pop else float('inf') 
                               for i, island_pop in enumerate(islands)}
    }]
    
    # Main loop
    for generation in range(max_generations_total):
        # Evolve each island for one generation
        for island_idx, island_pop in enumerate(islands):
            if not island_pop:
                logger.warning(f"Island {island_idx} is empty at generation {generation}. Skipping evolution.")
                continue
            
            # Get GA parameters for this island
            island_ga_params = island_model_ga_params["ga_params_per_island"].copy()
            island_ga_params["population_size"] = len(island_pop)
            island_ga_params["max_generations"] = 1  # Just one generation per island per outer loop
            
            # Create a sub-collector for this island's evolution
            sub_collector = GeneticAlgorithmMetricCollector(f"{config_name}_Island_{island_idx}")
            
            # Evolve island for one generation
            from genetic_algorithms import _evolve_one_ga_generation
            
            # Calculate pre-evolution metrics
            pre_evol_fitness_values = [sol.fitness() for sol in island_pop]
            pre_evol_best_fitness = min(pre_evol_fitness_values)
            pre_evol_avg_fitness = np.mean(pre_evol_fitness_values)
            pre_evol_std_fitness = np.std(pre_evol_fitness_values)
            pre_evol_diversity = calculate_genotypic_diversity(island_pop)
            
            # Record pre-evolution metrics
            metric_collector.record_island_metrics(
                island_idx,
                pre_evol_best_fitness,
                pre_evol_avg_fitness,
                pre_evol_diversity
            )
            
            # Evolve island
            new_island_pop, _, _, _ = _evolve_one_ga_generation(
                current_population=island_pop,
                population_size=island_population_size,
                elitism_size=island_ga_params.get("elitism_size", 0),
                selection_operator=island_ga_params["selection_operator"],
                selection_kwargs=island_ga_params.get("selection_params", {}),
                crossover_operator=island_ga_params["crossover_operator"],
                crossover_rate=island_ga_params["crossover_rate"],
                mutation_operator=island_ga_params["mutation_operator"],
                prob_apply_mutation=island_ga_params.get("mutation_rate", 0.1),
                logger_instance=logger
            )
            
            # Update island population
            islands[island_idx] = new_island_pop
            
            # Calculate post-evolution metrics
            post_evol_fitness_values = [sol.fitness() for sol in new_island_pop]
            post_evol_best_fitness = min(post_evol_fitness_values)
            post_evol_avg_fitness = np.mean(post_evol_fitness_values)
            post_evol_diversity = calculate_genotypic_diversity(new_island_pop)
            
            # Update global best if improved
            if post_evol_best_fitness < global_best_fitness:
                island_best = min(new_island_pop, key=lambda s: s.fitness())
                global_best_solution = deepcopy(island_best)
                global_best_fitness = post_evol_best_fitness
                
                # Record new global best
                metric_collector.metrics["best_fitness"] = global_best_fitness
            
            # Aggregate metrics from sub-collector
            metric_collector.metrics["function_evaluations"] += sub_collector.metrics["function_evaluations"]
        
        # Record iteration
        metric_collector.record_iteration(global_best_fitness)
        
        # Migration phase (if it's time to migrate)
        if (generation + 1) % migration_frequency == 0:
            # Calculate inter-island diversity before migration
            all_individuals = []
            for island_pop in islands:
                all_individuals.extend(island_pop)
            
            inter_island_diversity = calculate_genotypic_diversity(all_individuals)
            metric_collector.record_inter_island_diversity(inter_island_diversity)
            
            # Perform migration based on topology
            if migration_topology == "ring":
                from ga_island_model import _perform_ring_migration
                
                # Record pre-migration best fitness for each island
                pre_migration_best = {i: min([sol.fitness() for sol in island_pop]) if island_pop else float('inf') 
                                     for i, island_pop in enumerate(islands)}
                
                # Perform migration
                migration_events = _perform_ring_migration(
                    islands_list=islands,
                    num_migrants=num_migrants,
                    island_population_size=island_population_size,
                    logger_instance=logger
                )
                
                # Record post-migration best fitness for each island
                post_migration_best = {i: min([sol.fitness() for sol in island_pop]) if island_pop else float('inf') 
                                      for i, island_pop in enumerate(islands)}
                
                # Record migration events
                for event in migration_events:
                    source_island = event["source_island_idx"]
                    target_island = event["target_island_idx"]
                    migrants_fitness = event["migrants_fitnesses"]
                    
                    metric_collector.record_migration(
                        source_island,
                        target_island,
                        migrants_fitness,
                        True,  # Assuming migration was successful if event was recorded
                        pre_migration_best[target_island],
                        post_migration_best[target_island]
                    )
            
            elif migration_topology == "random_pair":
                from ga_island_model import _perform_random_pair_migration
                
                # Record pre-migration best fitness for each island
                pre_migration_best = {i: min([sol.fitness() for sol in island_pop]) if island_pop else float('inf') 
                                     for i, island_pop in enumerate(islands)}
                
                # Perform migration
                migration_events = _perform_random_pair_migration(
                    islands_list=islands,
                    num_migrants=num_migrants,
                    island_population_size=island_population_size,
                    logger_instance=logger
                )
                
                # Record post-migration best fitness for each island
                post_migration_best = {i: min([sol.fitness() for sol in island_pop]) if island_pop else float('inf') 
                                      for i, island_pop in enumerate(islands)}
                
                # Record migration events
                for event in migration_events:
                    source_island = event["source_island_idx"]
                    target_island = event["target_island_idx"]
                    migrants_fitness = event["migrants_fitnesses"]
                    
                    metric_collector.record_migration(
                        source_island,
                        target_island,
                        migrants_fitness,
                        True,  # Assuming migration was successful if event was recorded
                        pre_migration_best[target_island],
                        post_migration_best[target_island]
                    )
            
            elif migration_topology == "broadcast_best":
                from ga_island_model import _perform_broadcast_migration
                
                # Record pre-migration best fitness for each island
                pre_migration_best = {i: min([sol.fitness() for sol in island_pop]) if island_pop else float('inf') 
                                     for i, island_pop in enumerate(islands)}
                
                # Perform migration
                migration_events = _perform_broadcast_migration(
                    islands_list=islands,
                    global_best_solution=global_best_solution,
                    island_population_size=island_population_size,
                    logger_instance=logger,
                    verbose=verbose
                )
                
                # Record post-migration best fitness for each island
                post_migration_best = {i: min([sol.fitness() for sol in island_pop]) if island_pop else float('inf') 
                                      for i, island_pop in enumerate(islands)}
                
                # Record migration events
                for event in migration_events:
                    source_island = event["source_island_idx"]  # -1 for global best
                    target_island = event["target_island_idx"]
                    migrants_fitness = event["migrants_fitnesses"]
                    
                    metric_collector.record_migration(
                        source_island,
                        target_island,
                        migrants_fitness,
                        True,  # Assuming migration was successful if event was recorded
                        pre_migration_best[target_island],
                        post_migration_best[target_island]
                    )
        
        # Update history
        history.append({
            'global_best_fitness': global_best_fitness,
            'island_best_fitness': {i: min([sol.fitness() for sol in island_pop]) if island_pop else float('inf') 
                                   for i, island_pop in enumerate(islands)}
        })
        
        if verbose and (generation + 1) % 10 == 0:
            logger.info(f"Island GA Gen {generation + 1}: Global Best Fitness = {global_best_fitness:.4f}")
    
    # Stop timer
    metric_collector.stop_timer()
    
    # Calculate derived metrics
    metric_collector.calculate_derived_metrics()
    
    return global_best_solution, global_best_fitness, history, metric_collector.get_metrics()
