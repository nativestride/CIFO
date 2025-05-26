import random
import numpy as np
from copy import deepcopy

# Solution classes are used as type hints and for instantiation if needed,
# but no direct dependency on operators.py anymore.
from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution, InsufficientPlayersForPositionError
import logging # Added for logging

# Get a logger for this module
logger = logging.getLogger(__name__)

# Default safe_exp function if not provided (can be overridden by passing a different one)
# Copied here for self-containment if not using a central utils.py
# Ideally, this would be imported from a common utility module or passed.
def default_safe_exp(x, max_value=700):
    return np.exp(np.clip(x, -max_value, max_value))

# Hill Climbing algorithm
def hill_climbing(initial_solution, max_iterations=1000, max_no_improvement=100, verbose=False, **kwargs): # Added **kwargs for flexibility
    """
    Implementation of the Hill Climbing optimization algorithm.
    initial_solution should be of a type that has a get_neighbors method (e.g., LeagueHillClimbingSolution).
    """
    # Ensure initial_solution is of the correct type or has get_neighbors
    if not hasattr(initial_solution, 'get_neighbors'):
        raise TypeError("initial_solution for hill_climbing must have a 'get_neighbors' method.")

    current_sol = initial_solution # No deepcopy needed, HC modifies based on this
    current_fitness = current_sol.fitness()
    
    # The best solution found by HC is simply the 'current_sol' at the end.
    # If initial_solution was part of a larger population, the caller should handle copying if needed.
    # For the GA local search, a new solution is made from the result.
    
    history = [current_fitness]
    iterations_without_improvement = 0
    
    for iteration in range(max_iterations):
        neighbors = current_sol.get_neighbors() # This method should return new solution objects
        
        if not neighbors:
            if verbose: print(f"Hill Climbing: No valid neighbors found at iteration {iteration}. Stopping.")
            break
        
        # Find best neighbor
        best_neighbor = min(neighbors, key=lambda x: x.fitness())
        best_neighbor_fitness = best_neighbor.fitness() # Could cache this if fitness is expensive
        
        if best_neighbor_fitness < current_fitness:
            current_sol = best_neighbor # best_neighbor is already a new instance from get_neighbors
            current_fitness = best_neighbor_fitness
            history.append(current_fitness)
            iterations_without_improvement = 0
            if verbose: print(f"Hill Climbing: Iteration {iteration}, Fitness = {current_fitness:.6f}")
        else:
            iterations_without_improvement += 1
            if iterations_without_improvement >= max_no_improvement:
                if verbose: print(f"Hill Climbing: Stopping after {max_no_improvement} iterations without improvement (Iter {iteration}).")
                break
    
    return current_sol, current_fitness, history


# Simulated Annealing algorithm
def simulated_annealing(
    initial_solution,
    initial_temperature=100.0,
    cooling_rate=0.95,
    min_temperature=0.1,
    iterations_per_temp=20,
    maximization=False, # Though problem is minimization
    verbose=False,
    safe_exp_func=None, # Parameter for the safe exponential function
    **kwargs # For other potential params through config
):
    """
    Implementation of the Simulated Annealing optimization algorithm.
    initial_solution should have get_random_neighbor method (e.g. LeagueSASolution)
    """
    if not hasattr(initial_solution, 'get_random_neighbor'):
        raise TypeError("initial_solution for simulated_annealing must have a 'get_random_neighbor' method.")

    if safe_exp_func is None:
        safe_exp_func = default_safe_exp # Use default if not provided

    current_solution = initial_solution # SA modifies its current solution path
    current_fitness = current_solution.fitness()
    
    # best_solution tracks the overall best state found
    best_solution = deepcopy(current_solution) # Deepcopy to store the best state independently
    best_fitness = current_fitness
    
    temperature = initial_temperature
    history = [best_fitness] # Track overall best fitness history
    
    if verbose: print(f"SA Initial: Temp={temperature:.2f}, Fitness={current_fitness:.6f}, Best Fitness={best_fitness:.6f}")
    
    while temperature > min_temperature:
        for i_iter in range(iterations_per_temp):
            neighbor_solution = current_solution.get_random_neighbor() # This returns a NEW solution instance
            neighbor_fitness = neighbor_solution.fitness()
            
            delta_fitness = neighbor_fitness - current_fitness
            
            accepted_move = False
            if (maximization and delta_fitness > 0) or (not maximization and delta_fitness < 0):
                accepted_move = True # Accept better solution
            else: # Worse or equal solution
                # For minimization, delta_fitness >= 0. abs(delta_fitness) = delta_fitness
                # For maximization, delta_fitness <= 0. abs(delta_fitness) = -delta_fitness
                # Standard Metropolis: exp(-change_in_energy / T), where positive change is "bad"
                # For minimization, "bad" is positive delta_fitness. So, exp(-delta_fitness / T)
                # For maximization, "bad" is negative delta_fitness. So, exp(delta_fitness / T)
                # The use of abs(delta_fitness) in the original was for "distance",
                # but the sign of delta_fitness determines if it's "uphill" or "downhill".
                
                # Corrected acceptance for worse solutions:
                # P = exp(- (E_new - E_old) / T) for minimization if E_new > E_old
                # P = exp(- (E_old - E_new) / T) for maximization if E_new < E_old
                # Which is exp(-abs(delta_fitness)/T) if delta_fitness is defined as E_new - E_old
                
                # The original logic: np.exp(-abs(delta_fitness) / temperature) is standard and correct.
                probability = safe_exp_func(-abs(delta_fitness) / temperature)
                if random.random() < probability:
                    accepted_move = True
            
            if accepted_move:
                current_solution = neighbor_solution # Assign new instance, NO deepcopy needed here
                current_fitness = neighbor_fitness
                
                # Update overall best if current accepted solution is better
                if (maximization and current_fitness > best_fitness) or \
                   (not maximization and current_fitness < best_fitness):
                    best_solution = deepcopy(current_solution) # Deepcopy for independent best record
                    best_fitness = current_fitness
                    if verbose and i_iter % 10 == 0 : # Print less frequently for new best
                         print(f"  SA New Best: Temp={temperature:.2f}, Iter={i_iter}, Best Fitness={best_fitness:.6f}")
            
            history.append(best_fitness) # Log overall best fitness at each step
            
        temperature *= cooling_rate
        if verbose:
            print(f"SA Cooled: Temp={temperature:.2f}, Current Fitness={current_fitness:.6f}, Best Fitness={best_fitness:.6f}")
            
    if verbose: print(f"SA Final Best Fitness: {best_fitness:.6f}")
    return best_solution, best_fitness, history

# Hill Climbing with Random Restart
def hill_climbing_random_restart(
    initial_solution_params: dict, # Contains players, num_teams, etc.
    solution_class_for_hc: type,   # e.g., LeagueHillClimbingSolution
    num_restarts: int,
    max_iterations_per_hc: int,
    max_no_improvement_per_hc: int,
    verbose: bool = False,
    hc_specific_kwargs: dict = None # For any other params to pass to the base hill_climbing
) -> tuple: # Returns: best_solution_overall, best_fitness_overall, history_of_best_fitness_per_restart
    """
    Implements Hill Climbing with Random Restarts.
    Performs standard Hill Climbing multiple times from different random initial solutions.
    """
    if hc_specific_kwargs is None:
        hc_specific_kwargs = {}

    logger.info(f"Starting Hill Climbing with Random Restart: {num_restarts} restarts planned.")
    
    best_solution_overall = None
    best_fitness_overall = float('inf')
    history_of_best_fitness_per_restart = [] # Tracks the best fitness achieved in each restart

    for i_restart in range(num_restarts):
        logger.info(f"HCRR Restart {i_restart + 1}/{num_restarts} starting...")
        current_initial_sol = None
        MAX_INIT_ATTEMPTS = 10 # Number of tries to get a valid starting solution

        for attempt in range(MAX_INIT_ATTEMPTS):
            try:
                # Create new instance for each attempt
                temp_sol = solution_class_for_hc(**initial_solution_params)
                if temp_sol.is_valid():
                    current_initial_sol = temp_sol
                    logger.info(f"HCRR Restart {i_restart + 1}: Valid initial solution created on attempt {attempt + 1} (Fitness: {current_initial_sol.fitness():.4f}).")
                    break 
                else:
                    logger.debug(f"HCRR Restart {i_restart + 1}, Attempt {attempt + 1}: Initial solution invalid.")
            except InsufficientPlayersForPositionError as e_pos:
                logger.warning(f"HCRR Restart {i_restart + 1}, Attempt {attempt + 1}: Failed to create initial solution due to position constraints - {e_pos}")
            except Exception as e_init:
                logger.error(f"HCRR Restart {i_restart + 1}, Attempt {attempt + 1}: Unexpected error during initial solution creation - {e_init}", exc_info=True)
        if current_initial_sol is None:
            logger.warning(f"HCRR Restart {i_restart + 1}: Failed to create a valid initial solution after {MAX_INIT_ATTEMPTS} attempts. Skipping this restart.")
            history_of_best_fitness_per_restart.append(float('inf')) # Record failure for this restart
            continue # Skip to the next restart if no valid initial solution

        # Proceed to run standard Hill Climbing
        inner_hc_verbose = hc_specific_kwargs.get("verbose", False)
        
        # Ensure kwargs passed to hill_climbing do not conflict with its explicit params
        # For example, hill_climbing already defines max_iterations, max_no_improvement, verbose.
        # hc_specific_kwargs should ideally contain other params specific to the get_neighbors method, etc.
        # We will pass max_iterations_per_hc etc. directly.
        
        hc_run_params = hc_specific_kwargs.copy() # Start with a copy of hc_specific_kwargs
        # Ensure direct parameters for hill_climbing are not duplicated or overridden unintentionally
        # by just **hc_specific_kwargs if they share names.
        # The explicit parameters will take precedence.
        # No explicit conflicting keys seen in typical hc_specific_kwargs example.

        best_sol_this_restart, best_fit_this_restart, _ = hill_climbing(
            initial_solution=current_initial_sol, 
            max_iterations=max_iterations_per_hc,
            max_no_improvement=max_no_improvement_per_hc,
            verbose=inner_hc_verbose,
            **hc_run_params # Pass other relevant kwargs for HC or its solution's get_neighbors
        )

        logger.info(f"HCRR Restart {i_restart + 1} completed. Best fitness for this restart: {best_fit_this_restart:.4f}")
        history_of_best_fitness_per_restart.append(best_fit_this_restart)

        if best_fit_this_restart < best_fitness_overall:
            best_fitness_overall = best_fit_this_restart
            best_solution_overall = deepcopy(best_sol_this_restart)
            logger.info(f"HCRR Restart {i_restart + 1}: New GLOBAL best solution found! Fitness: {best_fitness_overall:.4f}")

    if best_solution_overall is None and num_restarts > 0:
        logger.warning("HCRR completed, but no valid solution was found that could be optimized across all restarts.")
    
    logger.info(f"Hill Climbing with Random Restart finished. Overall best fitness: {best_fitness_overall:.4f}")
    return best_solution_overall, best_fitness_overall, history_of_best_fitness_per_restart


