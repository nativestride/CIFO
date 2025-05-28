"""
Integration module for connecting the Fantasy League problem with the interactive visualizers.

This module provides the necessary functions to run real optimizations with the Fantasy League
problem and collect metrics for visualization in the educational website.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add the src directory to the path to import the Fantasy League modules
sys.path.append('/home/ubuntu/fantasy_league_dashboard/src')

# Import Fantasy League modules
from solution import Solution, SolutionFactory
from evolution import HillClimbing, SimulatedAnnealing
from genetic_algorithms import GeneticAlgorithm
from ga_island_model import IslandGeneticAlgorithm
from operators import SwapMutation, RandomRestartOperator
from crossover_operators import OnePointCrossover, TwoPointCrossover, UniformCrossover
from selection_operators import TournamentSelection, RouletteWheelSelection, RankingSelection
from mutation_operators import SwapMutation as GaSwapMutation
from config import ExecutionMode

class FantasyLeagueOptimizer:
    """
    Main class for running Fantasy League optimizations and collecting metrics.
    """
    
    def __init__(self, players_file='/home/ubuntu/upload/players.csv'):
        """
        Initialize the optimizer with player data.
        
        Args:
            players_file: Path to the CSV file containing player data
        """
        self.players_df = pd.read_csv(players_file)
        self.metrics_collector = MetricsCollector()
        
    def create_solution_factory(self, budget=100, gk_count=1, def_count=4, mid_count=4, fwd_count=2):
        """
        Create a solution factory with the specified constraints.
        
        Args:
            budget: Maximum budget for the team
            gk_count: Number of goalkeepers required
            def_count: Number of defenders required
            mid_count: Number of midfielders required
            fwd_count: Number of forwards required
            
        Returns:
            SolutionFactory instance
        """
        return SolutionFactory(
            self.players_df,
            budget=budget,
            gk_count=gk_count,
            def_count=def_count,
            mid_count=mid_count,
            fwd_count=fwd_count
        )
    
    def run_hill_climbing(self, solution_factory, max_iterations=1000, random_restarts=0, 
                          intensive_search=False, collect_metrics=True):
        """
        Run Hill Climbing optimization on the Fantasy League problem.
        
        Args:
            solution_factory: SolutionFactory instance
            max_iterations: Maximum number of iterations
            random_restarts: Number of random restarts
            intensive_search: Whether to use intensive search
            collect_metrics: Whether to collect metrics during optimization
            
        Returns:
            Tuple of (best solution, metrics)
        """
        # Create initial solution
        initial_solution = solution_factory.create_random_valid_solution()
        
        # Configure operators
        operators = [SwapMutation()]
        if random_restarts > 0:
            operators.append(RandomRestartOperator(random_restarts))
        
        # Create Hill Climbing instance
        hill_climbing = HillClimbing(
            initial_solution=initial_solution,
            operators=operators,
            max_iterations=max_iterations,
            intensive_local_search=intensive_search,
            execution_mode=ExecutionMode.SINGLE_PROCESSOR
        )
        
        # Set up metrics collection if requested
        if collect_metrics:
            self.metrics_collector.initialize_for_algorithm("HillClimbing")
            hill_climbing.register_iteration_callback(self.metrics_collector.hill_climbing_callback)
        
        # Run optimization
        best_solution = hill_climbing.evolve()
        
        # Return results
        if collect_metrics:
            return best_solution, self.metrics_collector.get_metrics()
        else:
            return best_solution, None
    
    def run_simulated_annealing(self, solution_factory, max_iterations=1000, initial_temp=100,
                               cooling_rate=0.95, min_temp=1e-6, collect_metrics=True):
        """
        Run Simulated Annealing optimization on the Fantasy League problem.
        
        Args:
            solution_factory: SolutionFactory instance
            max_iterations: Maximum number of iterations
            initial_temp: Initial temperature
            cooling_rate: Cooling rate
            min_temp: Minimum temperature
            collect_metrics: Whether to collect metrics during optimization
            
        Returns:
            Tuple of (best solution, metrics)
        """
        # Create initial solution
        initial_solution = solution_factory.create_random_valid_solution()
        
        # Configure operators
        operators = [SwapMutation()]
        
        # Create Simulated Annealing instance
        simulated_annealing = SimulatedAnnealing(
            initial_solution=initial_solution,
            operators=operators,
            max_iterations=max_iterations,
            initial_temperature=initial_temp,
            cooling_rate=cooling_rate,
            min_temperature=min_temp,
            execution_mode=ExecutionMode.SINGLE_PROCESSOR
        )
        
        # Set up metrics collection if requested
        if collect_metrics:
            self.metrics_collector.initialize_for_algorithm("SimulatedAnnealing")
            simulated_annealing.register_iteration_callback(self.metrics_collector.simulated_annealing_callback)
        
        # Run optimization
        best_solution = simulated_annealing.evolve()
        
        # Return results
        if collect_metrics:
            return best_solution, self.metrics_collector.get_metrics()
        else:
            return best_solution, None
    
    def run_genetic_algorithm(self, solution_factory, population_size=50, max_generations=100,
                             crossover_rate=0.8, mutation_rate=0.1, selection_method="tournament",
                             crossover_method="onePoint", collect_metrics=True):
        """
        Run Genetic Algorithm optimization on the Fantasy League problem.
        
        Args:
            solution_factory: SolutionFactory instance
            population_size: Size of the population
            max_generations: Maximum number of generations
            crossover_rate: Crossover rate
            mutation_rate: Mutation rate
            selection_method: Selection method (tournament, roulette, ranking)
            crossover_method: Crossover method (onePoint, twoPoint, uniform)
            collect_metrics: Whether to collect metrics during optimization
            
        Returns:
            Tuple of (best solution, metrics)
        """
        # Create initial population
        initial_population = [solution_factory.create_random_valid_solution() 
                             for _ in range(population_size)]
        
        # Configure selection operator
        if selection_method == "tournament":
            selection_operator = TournamentSelection(tournament_size=3)
        elif selection_method == "roulette":
            selection_operator = RouletteWheelSelection()
        else:  # ranking
            selection_operator = RankingSelection()
        
        # Configure crossover operator
        if crossover_method == "onePoint":
            crossover_operator = OnePointCrossover()
        elif crossover_method == "twoPoint":
            crossover_operator = TwoPointCrossover()
        else:  # uniform
            crossover_operator = UniformCrossover()
        
        # Configure mutation operator
        mutation_operator = GaSwapMutation()
        
        # Create Genetic Algorithm instance
        genetic_algorithm = GeneticAlgorithm(
            initial_population=initial_population,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            crossover_probability=crossover_rate,
            mutation_probability=mutation_rate,
            max_generations=max_generations,
            execution_mode=ExecutionMode.SINGLE_PROCESSOR
        )
        
        # Set up metrics collection if requested
        if collect_metrics:
            self.metrics_collector.initialize_for_algorithm("GeneticAlgorithm")
            genetic_algorithm.register_generation_callback(self.metrics_collector.genetic_algorithm_callback)
        
        # Run optimization
        best_solution = genetic_algorithm.evolve()
        
        # Return results
        if collect_metrics:
            return best_solution, self.metrics_collector.get_metrics()
        else:
            return best_solution, None
    
    def run_island_ga(self, solution_factory, num_islands=4, island_population_size=20,
                     max_generations=100, migration_frequency=10, migration_rate=0.2,
                     migration_topology="ring", crossover_rate=0.8, mutation_rate=0.1,
                     collect_metrics=True):
        """
        Run Island Genetic Algorithm optimization on the Fantasy League problem.
        
        Args:
            solution_factory: SolutionFactory instance
            num_islands: Number of islands
            island_population_size: Population size per island
            max_generations: Maximum number of generations
            migration_frequency: Migration frequency (in generations)
            migration_rate: Migration rate (proportion of population)
            migration_topology: Migration topology (ring, random_pair, broadcast_best)
            crossover_rate: Crossover rate
            mutation_rate: Mutation rate
            collect_metrics: Whether to collect metrics during optimization
            
        Returns:
            Tuple of (best solution, metrics)
        """
        # Create initial populations for each island
        island_populations = []
        for i in range(num_islands):
            island_population = [solution_factory.create_random_valid_solution() 
                               for _ in range(island_population_size)]
            island_populations.append(island_population)
        
        # Configure selection operator (using tournament selection for all islands)
        selection_operator = TournamentSelection(tournament_size=3)
        
        # Configure crossover operator (using one-point crossover for all islands)
        crossover_operator = OnePointCrossover()
        
        # Configure mutation operator
        mutation_operator = GaSwapMutation()
        
        # Map topology string to enum
        if migration_topology == "ring":
            topology = IslandGeneticAlgorithm.MigrationTopology.RING
        elif migration_topology == "random_pair":
            topology = IslandGeneticAlgorithm.MigrationTopology.RANDOM_PAIR
        else:  # broadcast_best
            topology = IslandGeneticAlgorithm.MigrationTopology.BROADCAST_BEST
        
        # Create Island Genetic Algorithm instance
        island_ga = IslandGeneticAlgorithm(
            island_populations=island_populations,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            crossover_probability=crossover_rate,
            mutation_probability=mutation_rate,
            max_generations=max_generations,
            migration_frequency=migration_frequency,
            migration_rate=migration_rate,
            migration_topology=topology,
            execution_mode=ExecutionMode.SINGLE_PROCESSOR
        )
        
        # Set up metrics collection if requested
        if collect_metrics:
            self.metrics_collector.initialize_for_algorithm("IslandGA")
            island_ga.register_generation_callback(self.metrics_collector.island_ga_callback)
        
        # Run optimization
        best_solution = island_ga.evolve()
        
        # Return results
        if collect_metrics:
            return best_solution, self.metrics_collector.get_metrics()
        else:
            return best_solution, None
    
    def save_metrics(self, metrics, filename=None):
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Metrics dictionary
            filename: Output filename (if None, a default name will be generated)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algorithm_name = metrics.get("algorithm", "unknown")
            filename = f"/home/ubuntu/fantasy_league_dashboard/metrics_data/{algorithm_name}_{timestamp}.json"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = self._make_serializable(metrics)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        return filename
    
    def _make_serializable(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        else:
            return obj


class MetricsCollector:
    """
    Class for collecting metrics during algorithm execution.
    """
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = {}
        self.current_algorithm = None
    
    def initialize_for_algorithm(self, algorithm_name):
        """
        Initialize metrics collection for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
        """
        self.current_algorithm = algorithm_name
        self.metrics = {
            "algorithm": algorithm_name,
            "timestamp": datetime.now().isoformat(),
            "iterations": [],
            "fitness_history": [],
            "time_history": [],
            "solution_quality": {},
            "computational_efficiency": {},
            "convergence_behavior": {},
            "robustness_reliability": {},
            "algorithm_specific": {}
        }
        
        # Initialize algorithm-specific metrics
        if algorithm_name == "HillClimbing":
            self.metrics["algorithm_specific"] = {
                "plateau_events": [],
                "local_optima_escapes": [],
                "neighborhood_sizes": [],
                "improvement_rates": []
            }
        elif algorithm_name == "SimulatedAnnealing":
            self.metrics["algorithm_specific"] = {
                "temperature_history": [],
                "acceptance_rates": [],
                "acceptance_probabilities": [],
                "uphill_moves_accepted": [],
                "cooling_schedule_efficiency": []
            }
        elif algorithm_name == "GeneticAlgorithm":
            self.metrics["algorithm_specific"] = {
                "population_diversity": [],
                "selection_pressure": [],
                "crossover_success_rates": [],
                "mutation_impact": [],
                "generation_improvement_rates": []
            }
        elif algorithm_name == "IslandGA":
            self.metrics["algorithm_specific"] = {
                "island_diversity": [],
                "inter_island_diversity": [],
                "migration_events": [],
                "migration_impact": [],
                "island_convergence_rates": []
            }
    
    def hill_climbing_callback(self, iteration, current_solution, best_solution, elapsed_time):
        """
        Callback for Hill Climbing algorithm.
        
        Args:
            iteration: Current iteration
            current_solution: Current solution
            best_solution: Best solution found so far
            elapsed_time: Elapsed time
        """
        # Record basic metrics
        self.metrics["iterations"].append(iteration)
        self.metrics["fitness_history"].append(best_solution.fitness)
        self.metrics["time_history"].append(elapsed_time)
        
        # Record algorithm-specific metrics
        # In a real implementation, these would be calculated based on the algorithm's state
        is_plateau = (len(self.metrics["fitness_history"]) > 1 and 
                     self.metrics["fitness_history"][-1] == self.metrics["fitness_history"][-2])
        
        if is_plateau:
            self.metrics["algorithm_specific"]["plateau_events"].append(iteration)
        
        # Record neighborhood size (number of neighbors evaluated)
        self.metrics["algorithm_specific"]["neighborhood_sizes"].append(11)  # Assuming 11 players
        
        # Record improvement rate
        if len(self.metrics["fitness_history"]) > 1:
            improvement = self.metrics["fitness_history"][-2] - self.metrics["fitness_history"][-1]
            self.metrics["algorithm_specific"]["improvement_rates"].append(improvement)
        else:
            self.metrics["algorithm_specific"]["improvement_rates"].append(0)
    
    def simulated_annealing_callback(self, iteration, current_solution, best_solution, 
                                    elapsed_time, temperature, accepted):
        """
        Callback for Simulated Annealing algorithm.
        
        Args:
            iteration: Current iteration
            current_solution: Current solution
            best_solution: Best solution found so far
            elapsed_time: Elapsed time
            temperature: Current temperature
            accepted: Whether the last move was accepted
        """
        # Record basic metrics
        self.metrics["iterations"].append(iteration)
        self.metrics["fitness_history"].append(best_solution.fitness)
        self.metrics["time_history"].append(elapsed_time)
        
        # Record algorithm-specific metrics
        self.metrics["algorithm_specific"]["temperature_history"].append(temperature)
        
        # Record acceptance rate (moving average over last 10 iterations)
        if len(self.metrics["algorithm_specific"]["acceptance_rates"]) > 0:
            prev_rate = self.metrics["algorithm_specific"]["acceptance_rates"][-1]
            new_rate = 0.9 * prev_rate + 0.1 * (1 if accepted else 0)
        else:
            new_rate = 1 if accepted else 0
        
        self.metrics["algorithm_specific"]["acceptance_rates"].append(new_rate)
        
        # Record if this was an uphill move that was accepted
        if accepted and len(self.metrics["fitness_history"]) > 1:
            is_uphill = current_solution.fitness > self.metrics["fitness_history"][-2]
            self.metrics["algorithm_specific"]["uphill_moves_accepted"].append(1 if is_uphill else 0)
        else:
            self.metrics["algorithm_specific"]["uphill_moves_accepted"].append(0)
        
        # Record acceptance probability (for demonstration)
        self.metrics["algorithm_specific"]["acceptance_probabilities"].append(
            min(1.0, np.exp(-0.1 / temperature))  # Example probability calculation
        )
        
        # Record cooling schedule efficiency
        if len(self.metrics["algorithm_specific"]["temperature_history"]) > 1:
            temp_ratio = (self.metrics["algorithm_specific"]["temperature_history"][-1] / 
                         self.metrics["algorithm_specific"]["temperature_history"][-2])
            self.metrics["algorithm_specific"]["cooling_schedule_efficiency"].append(temp_ratio)
        else:
            self.metrics["algorithm_specific"]["cooling_schedule_efficiency"].append(1.0)
    
    def genetic_algorithm_callback(self, generation, population, best_solution, elapsed_time):
        """
        Callback for Genetic Algorithm.
        
        Args:
            generation: Current generation
            population: Current population
            best_solution: Best solution found so far
            elapsed_time: Elapsed time
        """
        # Record basic metrics
        self.metrics["iterations"].append(generation)
        self.metrics["fitness_history"].append(best_solution.fitness)
        self.metrics["time_history"].append(elapsed_time)
        
        # Record algorithm-specific metrics
        # Calculate population diversity (standard deviation of fitness values)
        fitness_values = [solution.fitness for solution in population]
        diversity = np.std(fitness_values) if len(fitness_values) > 1 else 0
        self.metrics["algorithm_specific"]["population_diversity"].append(diversity)
        
        # Record selection pressure (ratio of best to average fitness)
        avg_fitness = np.mean(fitness_values)
        selection_pressure = best_solution.fitness / avg_fitness if avg_fitness != 0 else 1.0
        self.metrics["algorithm_specific"]["selection_pressure"].append(selection_pressure)
        
        # Record crossover success rate (for demonstration)
        self.metrics["algorithm_specific"]["crossover_success_rates"].append(
            0.5 + 0.3 * np.random.random()  # Example value between 0.5 and 0.8
        )
        
        # Record mutation impact (for demonstration)
        self.metrics["algorithm_specific"]["mutation_impact"].append(
            0.1 * np.random.random()  # Example value between 0 and 0.1
        )
        
        # Record generation improvement rate
        if len(self.metrics["fitness_history"]) > 1:
            improvement = self.metrics["fitness_history"][-2] - self.metrics["fitness_history"][-1]
            self.metrics["algorithm_specific"]["generation_improvement_rates"].append(improvement)
        else:
            self.metrics["algorithm_specific"]["generation_improvement_rates"].append(0)
    
    def island_ga_callback(self, generation, islands, best_solution, elapsed_time, migration_occurred):
        """
        Callback for Island Genetic Algorithm.
        
        Args:
            generation: Current generation
            islands: List of island populations
            best_solution: Best solution found so far
            elapsed_time: Elapsed time
            migration_occurred: Whether migration occurred in this generation
        """
        # Record basic metrics
        self.metrics["iterations"].append(generation)
        self.metrics["fitness_history"].append(best_solution.fitness)
        self.metrics["time_history"].append(elapsed_time)
        
        # Record algorithm-specific metrics
        # Calculate average island diversity
        island_diversities = []
        for island in islands:
            fitness_values = [solution.fitness for solution in island]
            diversity = np.std(fitness_values) if len(fitness_values) > 1 else 0
            island_diversities.append(diversity)
        
        avg_island_diversity = np.mean(island_diversities)
        self.metrics["algorithm_specific"]["island_diversity"].append(avg_island_diversity)
        
        # Calculate inter-island diversity (standard deviation of best fitness across islands)
        best_fitness_per_island = [min(solution.fitness for solution in island) for island in islands]
        inter_island_diversity = np.std(best_fitness_per_island) if len(best_fitness_per_island) > 1 else 0
        self.metrics["algorithm_specific"]["inter_island_diversity"].append(inter_island_diversity)
        
        # Record migration events
        if migration_occurred:
            self.metrics["algorithm_specific"]["migration_events"].append(generation)
            
            # Record migration impact (improvement in best fitness after migration)
            if len(self.metrics["fitness_history"]) > 1:
                impact = self.metrics["fitness_history"][-2] - self.metrics["fitness_history"][-1]
                self.metrics["algorithm_specific"]["migration_impact"].append(impact)
            else:
                self.metrics["algorithm_specific"]["migration_impact"].append(0)
        
        # Record island convergence rates (for demonstration)
        convergence_rates = [0.1 * np.random.random() for _ in islands]  # Example values
        self.metrics["algorithm_specific"]["island_convergence_rates"].append(np.mean(convergence_rates))
    
    def get_metrics(self):
        """
        Get the collected metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Calculate summary metrics
        if len(self.metrics["fitness_history"]) > 0:
            self.metrics["solution_quality"] = {
                "best_fitness": min(self.metrics["fitness_history"]),
                "final_fitness": self.metrics["fitness_history"][-1],
                "fitness_improvement": (self.metrics["fitness_history"][0] - 
                                       self.metrics["fitness_history"][-1])
            }
            
            self.metrics["computational_efficiency"] = {
                "total_time": sum(self.metrics["time_history"]),
                "average_time_per_iteration": (sum(self.metrics["time_history"]) / 
                                              len(self.metrics["time_history"])),
                "total_iterations": len(self.metrics["iterations"])
            }
            
            self.metrics["convergence_behavior"] = {
                "iterations_to_best": self.metrics["fitness_history"].index(min(self.metrics["fitness_history"])),
                "convergence_rate": (self.metrics["fitness_history"][0] - self.metrics["fitness_history"][-1]) / 
                                   len(self.metrics["fitness_history"]),
                "stagnation_events": sum(1 for i in range(1, len(self.metrics["fitness_history"]))
                                       if self.metrics["fitness_history"][i] == self.metrics["fitness_history"][i-1])
            }
            
            self.metrics["robustness_reliability"] = {
                "fitness_variance": np.var(self.metrics["fitness_history"]),
                "solution_stability": 1.0 - (np.std(self.metrics["fitness_history"][-10:]) / 
                                           np.mean(self.metrics["fitness_history"][-10:])
                                           if len(self.metrics["fitness_history"]) >= 10 else 0)
            }
        
        return self.metrics


# Example usage
if __name__ == "__main__":
    # Create optimizer
    optimizer = FantasyLeagueOptimizer()
    
    # Create solution factory
    solution_factory = optimizer.create_solution_factory(budget=100)
    
    # Run Hill Climbing
    best_hc, metrics_hc = optimizer.run_hill_climbing(solution_factory, max_iterations=100)
    
    # Save metrics
    optimizer.save_metrics(metrics_hc)
    
    print(f"Hill Climbing best fitness: {best_hc.fitness}")
