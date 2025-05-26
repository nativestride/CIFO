# config.py
# This file will contain the TypedDict definitions and the all_configs dictionary.
from typing import TypedDict, Callable, List, Dict, Optional, Any

# Import solution classes
from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution

# Import algorithm functions
from evolution import hill_climbing, simulated_annealing, hill_climbing_random_restart
from genetic_algorithms import genetic_algorithm
from ga_island_model import genetic_algorithm_island_model

# Import operator functions
from selection_operators import selection_tournament_variable_k, selection_ranking, selection_boltzmann
from crossover_operators import crossover_one_point_prefer_valid, crossover_two_point_prefer_valid, crossover_uniform_prefer_valid
from mutation_operators import mutate_targeted_player_exchange, mutate_swap_constrained, mutate_team_shift, mutate_swap, mutate_shuffle_within_team_constrained
from ga_utilities import hc_wrapper_for_ga
from experiment_utils import safe_exp # For SA default safe_exp_func and GA Boltzmann


class HCParams(TypedDict, total=False): # Using total=False for flexibility with HCRR params
    max_iterations: int
    max_no_improvement: int
    verbose: bool
    # For HCRR
    solution_class_for_hc: Optional[Callable] 
    num_restarts: Optional[int]
    max_iterations_per_hc: Optional[int]
    max_no_improvement_per_hc: Optional[int]
    hc_specific_kwargs: Optional[Dict[str, Any]]
    initial_solution_params: Optional[Dict[str, Any]]


class SAParams(TypedDict):
    initial_temperature: float
    cooling_rate: float
    min_temperature: float
    iterations_per_temp: int
    verbose: bool
    safe_exp_func: Optional[Callable]


class LSParams(TypedDict, total=False):
    frequency: int
    max_iterations: int
    max_no_improvement: int
    verbose: bool


# TypedDict for the 'ga_params_per_island' nested dictionary
class IslandGAParams(TypedDict, total=False):
    population_size: Optional[int] # Should match island_population_size, but good for structure
    max_generations: Optional[int] # Typically 1 for island model's per-generation evolution step
    elitism_size: int
    selection_operator: Callable
    selection_params: Dict[str, Any]
    crossover_operator: Callable
    crossover_rate: float
    mutation_operator: Callable
    mutation_rate: float
    verbose: Optional[bool] # Usually False for inner islands
    safe_exp_func: Optional[Callable]
    local_search_func: Optional[Callable]
    local_search_params: Optional[LSParams]
    max_initial_solution_attempts: Optional[int] # If generate_population is called per island


class GASpecificParams(TypedDict, total=False):
    population_size: int
    max_generations: int
    elitism_size: int
    selection_operator: Callable
    selection_params: Dict[str, Any]
    crossover_operator: Callable
    crossover_rate: float
    mutation_operator: Callable
    mutation_rate: float
    verbose: bool
    safe_exp_func: Optional[Callable] 
    local_search_func: Optional[Callable]
    local_search_params: Optional[LSParams]
    max_initial_solution_attempts: Optional[int]

    # For Island Model (nested within ga_specific_params as per current structure)
    # These are specific to the island model itself, not the GA running on each island.
    num_islands: Optional[int]
    island_population_size: Optional[int]
    # max_generations_total: Optional[int] # This is 'max_generations' for the island model call
    migration_frequency: Optional[int]
    num_migrants: Optional[int]
    migration_topology: Optional[str]
    ga_params_per_island: Optional[IslandGAParams]


class AlgorithmConfig(TypedDict):
    algorithm_func: Callable
    solution_class: Callable # e.g., LeagueSolution, LeagueHillClimbingSolution
    params: Optional[HCParams | SAParams] # For HC, SA, HCRR (HCRR uses HCParams with more fields)
    ga_specific_params: Optional[GASpecificParams] # For GAs (standard and island)

AllConfigs = Dict[str, AlgorithmConfig]

all_configs: AllConfigs = {
    # --- Hill Climbing Based (use 'params') ---
    "HillClimbing_Std": {
        "algorithm_func": hill_climbing,
        "solution_class": LeagueHillClimbingSolution,
        "params": {"max_iterations": 500, "max_no_improvement": 100, "verbose": False},
        "ga_specific_params": None,
    },
    "HillClimbing_Intensive": {
        "algorithm_func": hill_climbing,
        "solution_class": LeagueHillClimbingSolution,
        "params": {"max_iterations": 1000, "max_no_improvement": 200, "verbose": False},
        "ga_specific_params": None,
    },
    "HillClimbing_RandomRestart_Test": {
        "algorithm_func": hill_climbing_random_restart,
        "solution_class": LeagueHillClimbingSolution,
        "params": {
            "num_restarts": 10,
            "max_iterations_per_hc": 200,
            "max_no_improvement_per_hc": 50,
            "verbose": False,
            "hc_specific_kwargs": {},
            # solution_class_for_hc and initial_solution_params are injected by run_single_experiment_instance
        },
        "ga_specific_params": None,
    },
    # --- Simulated Annealing Based (use 'params') ---
    "SimulatedAnnealing_Std": {
        "algorithm_func": simulated_annealing,
        "solution_class": LeagueSASolution,
        "params": {
            "initial_temperature": 200.0,
            "cooling_rate": 0.95,
            "min_temperature": 1e-5,
            "iterations_per_temp": 20,
            "verbose": False,
            # safe_exp_func is injected by run_single_experiment_instance
        },
        "ga_specific_params": None,
    },
    "SimulatedAnnealing_Enhanced": {
        "algorithm_func": simulated_annealing,
        "solution_class": LeagueSASolution,
        "params": {
            "initial_temperature": 300.0,
            "cooling_rate": 0.97,
            "min_temperature": 1e-6,
            "iterations_per_temp": 30,
            "verbose": False,
        },
        "ga_specific_params": None,
    },
    # --- Genetic Algorithm Based (use 'ga_specific_params') ---
    "GA_Tournament_OnePoint": {
        "algorithm_func": genetic_algorithm,
        "solution_class": LeagueSolution,
        "params": None,
        "ga_specific_params": {
            "population_size": 100,
            "max_generations": 50,
            "elitism_size": 2,
            "selection_operator": selection_tournament_variable_k,
            "selection_params": {"k_percentage": 0.03},
            "crossover_operator": crossover_one_point_prefer_valid,
            "crossover_rate": 0.8,
            "mutation_operator": mutate_targeted_player_exchange,
            "mutation_rate": 0.1,
            "verbose": False,
            "local_search_func": None,
            "local_search_params": {},
        },
    },
    "GA_Ranking_Uniform": {
        "algorithm_func": genetic_algorithm,
        "solution_class": LeagueSolution,
        "params": None,
        "ga_specific_params": {
            "population_size": 100,
            "max_generations": 50,
            "elitism_size": 2,
            "selection_operator": selection_ranking,
            "selection_params": {},
            "crossover_operator": crossover_uniform_prefer_valid,
            "crossover_rate": 0.8,
            "mutation_operator": mutate_swap_constrained,
            "mutation_rate": 0.1,
            "verbose": False,
        },
    },
    "GA_TwoPointCrossover_Test": {
        "algorithm_func": genetic_algorithm,
        "solution_class": LeagueSolution,
        "params": None,
        "ga_specific_params": {
            "population_size": 100,
            "max_generations": 50,
            "elitism_size": 2,
            "selection_operator": selection_tournament_variable_k,
            "selection_params": {"k_percentage": 0.03},
            "crossover_operator": crossover_two_point_prefer_valid,
            "crossover_rate": 0.8,
            "mutation_operator": mutate_targeted_player_exchange,
            "mutation_rate": 0.1,
            "verbose": False,
        },
    },
    "GA_Hybrid_Optimized": {
        "algorithm_func": genetic_algorithm,
        "solution_class": LeagueSolution,
        "params": None,
        "ga_specific_params": {
            "population_size": 100,
            "max_generations": 50,
            "elitism_size": 3,
            "selection_operator": selection_tournament_variable_k,
            "selection_params": {"k_percentage": 0.05},
            "crossover_operator": crossover_uniform_prefer_valid,
            "crossover_rate": 0.9,
            "mutation_operator": mutate_targeted_player_exchange,
            "mutation_rate": 0.2,
            "verbose": False,
            "local_search_func": hc_wrapper_for_ga,
            "local_search_params": {
                "frequency": 3,
                "max_iterations": 75,
                "max_no_improvement": 30,
                "verbose": False,
            },
        },
    },
    "GA_Island_Model_Test": {
        "algorithm_func": genetic_algorithm_island_model,
        "solution_class": LeagueSolution,
        "params": None,
        "ga_specific_params": { # This is for the island_model_ga_params argument
            "num_islands": 3,
            "island_population_size": 30,
            "max_generations_total": 100, # Using max_generations_total as expected by the island model code
            "migration_frequency": 10,
            "num_migrants": 3,
            "migration_topology": "ring",
            "verbose": False,
            "ga_params_per_island": { # This is the nested dict for each island's GA
                "elitism_size": 2,
                "selection_operator": selection_tournament_variable_k,
                "selection_params": {"k_percentage": 0.15},
                "crossover_operator": crossover_one_point_prefer_valid,
                "crossover_rate": 0.8,
                "mutation_operator": mutate_swap_constrained,
                "mutation_rate": 0.15,
                # population_size and max_generations for island are implicit/handled by island model
            },
        },
    },
}
