"""
Comprehensive metric collection framework for optimization algorithms.

This module provides a unified interface for collecting, storing, and analyzing
performance metrics across different optimization algorithms (Hill Climbing,
Simulated Annealing, Genetic Algorithm, and Island Genetic Algorithm).
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class MetricCollector:
    """Base class for collecting performance metrics across optimization algorithms."""
    
    def __init__(self, algorithm_name: str, config_name: str, run_id: Optional[str] = None):
        """
        Initialize the metric collector.
        
        Args:
            algorithm_name: Name of the algorithm (e.g., "HillClimbing", "SimulatedAnnealing")
            config_name: Configuration name (e.g., "HillClimbing_Std", "GA_Tournament_OnePoint")
            run_id: Optional unique identifier for this run
        """
        self.algorithm_name = algorithm_name
        self.config_name = config_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Common metrics for all algorithms
        self.metrics = {
            # Solution quality metrics
            "best_fitness": float('inf'),
            "best_fitness_history": [],
            "final_solution_quality": None,
            "constraint_violations": 0,
            
            # Computational efficiency metrics
            "start_time": time.time(),
            "end_time": None,
            "runtime_seconds": None,
            "iterations": 0,
            "function_evaluations": 0,
            "iterations_to_best": None,
            "evaluations_to_best": None,
            
            # Convergence behavior metrics
            "convergence_rate": None,
            "plateau_detection": False,
            "early_termination_reason": None,
            
            # Robustness and reliability metrics
            "success_rate": None,
            "solution_stability": None,
            
            # Algorithm-specific metrics will be added by subclasses
        }
        
        # Parameters used for this run
        self.parameters = {}
        
        # Raw data for detailed analysis
        self.raw_data = {
            "fitness_history": [],
            "time_history": [],
            "iteration_history": [],
            "evaluation_history": []
        }
    
    def start_timer(self):
        """Start the execution timer."""
        self.metrics["start_time"] = time.time()
    
    def stop_timer(self):
        """Stop the execution timer and calculate runtime."""
        self.metrics["end_time"] = time.time()
        self.metrics["runtime_seconds"] = self.metrics["end_time"] - self.metrics["start_time"]
    
    def record_iteration(self, current_fitness: float, is_improvement: bool = False):
        """
        Record metrics for a single iteration.
        
        Args:
            current_fitness: Current fitness value
            is_improvement: Whether this iteration improved the best fitness
        """
        self.metrics["iterations"] += 1
        self.metrics["function_evaluations"] += 1
        
        # Record raw data
        current_time = time.time() - self.metrics["start_time"]
        self.raw_data["fitness_history"].append(current_fitness)
        self.raw_data["time_history"].append(current_time)
        self.raw_data["iteration_history"].append(self.metrics["iterations"])
        self.raw_data["evaluation_history"].append(self.metrics["function_evaluations"])
        
        # Update best fitness if improved
        if current_fitness < self.metrics["best_fitness"]:
            self.metrics["best_fitness"] = current_fitness
            self.metrics["iterations_to_best"] = self.metrics["iterations"]
            self.metrics["evaluations_to_best"] = self.metrics["function_evaluations"]
        
        # Record best fitness history (for convergence plots)
        if not self.metrics["best_fitness_history"] or current_fitness < self.metrics["best_fitness_history"][-1]:
            self.metrics["best_fitness_history"].append(current_fitness)
        else:
            self.metrics["best_fitness_history"].append(self.metrics["best_fitness_history"][-1])
    
    def record_function_evaluation(self, fitness: float):
        """
        Record a function evaluation without incrementing iteration counter.
        
        Args:
            fitness: Fitness value from this evaluation
        """
        self.metrics["function_evaluations"] += 1
        
        # Update best fitness if improved
        if fitness < self.metrics["best_fitness"]:
            self.metrics["best_fitness"] = fitness
            self.metrics["evaluations_to_best"] = self.metrics["function_evaluations"]
    
    def record_constraint_violation(self):
        """Record a constraint violation."""
        self.metrics["constraint_violations"] += 1
    
    def set_parameters(self, params: Dict[str, Any]):
        """
        Set algorithm parameters for this run.
        
        Args:
            params: Dictionary of parameter names and values
        """
        self.parameters = params
    
    def set_early_termination(self, reason: str):
        """
        Record early termination of the algorithm.
        
        Args:
            reason: Reason for early termination
        """
        self.metrics["early_termination_reason"] = reason
        
        # Check for plateau
        if "max_no_improvement" in reason.lower() or "plateau" in reason.lower():
            self.metrics["plateau_detection"] = True
    
    def calculate_derived_metrics(self):
        """Calculate metrics derived from raw data after algorithm completion."""
        if not self.raw_data["fitness_history"]:
            logger.warning("No fitness history available to calculate derived metrics")
            return
        
        # Calculate convergence rate (if enough data points)
        if len(self.raw_data["fitness_history"]) > 10:
            # Simple convergence rate: improvement per iteration
            initial_fitness = self.raw_data["fitness_history"][0]
            final_fitness = self.raw_data["fitness_history"][-1]
            total_iterations = len(self.raw_data["fitness_history"])
            
            if initial_fitness != final_fitness and total_iterations > 1:
                self.metrics["convergence_rate"] = (initial_fitness - final_fitness) / (total_iterations - 1)
        
        # Set final solution quality
        self.metrics["final_solution_quality"] = self.metrics["best_fitness"]
        
        # Calculate solution stability (standard deviation of last 10% of fitness values)
        if len(self.raw_data["fitness_history"]) >= 10:
            last_n = max(int(len(self.raw_data["fitness_history"]) * 0.1), 5)
            last_fitness_values = self.raw_data["fitness_history"][-last_n:]
            self.metrics["solution_stability"] = np.std(last_fitness_values)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.
        
        Returns:
            Dictionary of all metrics
        """
        # Calculate derived metrics if not already done
        if self.metrics["convergence_rate"] is None:
            self.calculate_derived_metrics()
        
        return {
            "algorithm_name": self.algorithm_name,
            "config_name": self.config_name,
            "run_id": self.run_id,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "raw_data": self.raw_data
        }
    
    def save_metrics(self, file_path: str):
        """
        Save metrics to a JSON file.
        
        Args:
            file_path: Path to save the metrics
        """
        metrics_data = self.get_metrics()
        
        # Convert numpy types to native Python types for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            else:
                return obj
        
        serializable_data = make_serializable(metrics_data)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Metrics saved to {file_path}")


class HillClimbingMetricCollector(MetricCollector):
    """Metric collector for Hill Climbing algorithm."""
    
    def __init__(self, config_name: str, run_id: Optional[str] = None):
        """
        Initialize the Hill Climbing metric collector.
        
        Args:
            config_name: Configuration name
            run_id: Optional unique identifier for this run
        """
        super().__init__("HillClimbing", config_name, run_id)
        
        # Hill Climbing specific metrics
        self.metrics.update({
            "neighbors_generated": 0,
            "neighbors_evaluated": 0,
            "local_optima_count": 0,
            "restarts": 0,  # For random restart hill climbing
            "plateau_length": 0,
            "improvement_rate": None,
        })
    
    def record_neighbor_generation(self, num_neighbors: int = 1):
        """
        Record neighbor generation.
        
        Args:
            num_neighbors: Number of neighbors generated
        """
        self.metrics["neighbors_generated"] += num_neighbors
    
    def record_neighbor_evaluation(self, num_neighbors: int = 1):
        """
        Record neighbor evaluation.
        
        Args:
            num_neighbors: Number of neighbors evaluated
        """
        self.metrics["neighbors_evaluated"] += num_neighbors
        self.metrics["function_evaluations"] += num_neighbors
    
    def record_local_optimum(self):
        """Record reaching a local optimum."""
        self.metrics["local_optima_count"] += 1
    
    def record_restart(self):
        """Record a restart (for random restart hill climbing)."""
        self.metrics["restarts"] += 1
    
    def record_plateau(self, length: int):
        """
        Record a plateau in the fitness landscape.
        
        Args:
            length: Length of the plateau (iterations without improvement)
        """
        self.metrics["plateau_length"] = max(self.metrics["plateau_length"], length)
        if length > 0:
            self.metrics["plateau_detection"] = True
    
    def calculate_derived_metrics(self):
        """Calculate Hill Climbing specific derived metrics."""
        super().calculate_derived_metrics()
        
        # Calculate improvement rate (improvements per neighbor evaluated)
        if self.metrics["neighbors_evaluated"] > 0:
            # Assuming best_fitness_history contains one entry per improvement
            improvements = len(self.metrics["best_fitness_history"])
            self.metrics["improvement_rate"] = improvements / self.metrics["neighbors_evaluated"]


class SimulatedAnnealingMetricCollector(MetricCollector):
    """Metric collector for Simulated Annealing algorithm."""
    
    def __init__(self, config_name: str, run_id: Optional[str] = None):
        """
        Initialize the Simulated Annealing metric collector.
        
        Args:
            config_name: Configuration name
            run_id: Optional unique identifier for this run
        """
        super().__init__("SimulatedAnnealing", config_name, run_id)
        
        # Simulated Annealing specific metrics
        self.metrics.update({
            "temperature_history": [],
            "acceptance_rate": None,
            "worse_solutions_accepted": 0,
            "total_solutions_evaluated": 0,
            "cooling_efficiency": None,
            "temperature_impact": None,
            "reheating_events": 0,
        })
        
        # Raw data for SA-specific analysis
        self.raw_data.update({
            "accepted_solutions": [],
            "rejected_solutions": [],
            "delta_fitness_history": [],
            "acceptance_probability_history": [],
        })
    
    def record_temperature(self, temperature: float):
        """
        Record current temperature.
        
        Args:
            temperature: Current temperature value
        """
        self.metrics["temperature_history"].append(temperature)
    
    def record_solution_evaluation(self, current_fitness: float, neighbor_fitness: float, 
                                  accepted: bool, temperature: float, probability: Optional[float] = None):
        """
        Record a solution evaluation in Simulated Annealing.
        
        Args:
            current_fitness: Current solution fitness
            neighbor_fitness: Neighbor solution fitness
            accepted: Whether the neighbor was accepted
            temperature: Current temperature
            probability: Acceptance probability (for worse solutions)
        """
        self.metrics["total_solutions_evaluated"] += 1
        self.metrics["function_evaluations"] += 1
        
        delta_fitness = neighbor_fitness - current_fitness
        
        # Record raw data
        self.raw_data["delta_fitness_history"].append(delta_fitness)
        
        if probability is not None:
            self.raw_data["acceptance_probability_history"].append(probability)
        
        if accepted:
            self.raw_data["accepted_solutions"].append(neighbor_fitness)
            if delta_fitness > 0:  # Worse solution accepted
                self.metrics["worse_solutions_accepted"] += 1
        else:
            self.raw_data["rejected_solutions"].append(neighbor_fitness)
        
        # Update best fitness if improved
        if neighbor_fitness < self.metrics["best_fitness"] and accepted:
            self.metrics["best_fitness"] = neighbor_fitness
            self.metrics["iterations_to_best"] = self.metrics["iterations"]
            self.metrics["evaluations_to_best"] = self.metrics["function_evaluations"]
    
    def record_reheating(self):
        """Record a reheating event."""
        self.metrics["reheating_events"] += 1
    
    def calculate_derived_metrics(self):
        """Calculate Simulated Annealing specific derived metrics."""
        super().calculate_derived_metrics()
        
        # Calculate acceptance rate
        if self.metrics["total_solutions_evaluated"] > 0:
            accepted_count = len(self.raw_data["accepted_solutions"])
            self.metrics["acceptance_rate"] = accepted_count / self.metrics["total_solutions_evaluated"]
        
        # Calculate cooling efficiency (correlation between temperature and acceptance rate)
        if len(self.metrics["temperature_history"]) > 10 and len(self.raw_data["accepted_solutions"]) > 0:
            # Group acceptance by temperature ranges
            temp_ranges = np.linspace(min(self.metrics["temperature_history"]), 
                                     max(self.metrics["temperature_history"]), 10)
            acceptance_by_temp = [0] * 9
            total_by_temp = [0] * 9
            
            for i, temp in enumerate(self.metrics["temperature_history"]):
                if i >= len(self.raw_data["delta_fitness_history"]):
                    continue
                    
                # Find temperature bin
                bin_idx = np.digitize(temp, temp_ranges) - 1
                if bin_idx >= 9:
                    bin_idx = 8
                
                total_by_temp[bin_idx] += 1
                
                # Check if solution was accepted at this step
                if i < len(self.raw_data["accepted_solutions"]):
                    acceptance_by_temp[bin_idx] += 1
            
            # Calculate acceptance rate per temperature bin
            acceptance_rates = []
            for accepted, total in zip(acceptance_by_temp, total_by_temp):
                if total > 0:
                    acceptance_rates.append(accepted / total)
                else:
                    acceptance_rates.append(0)
            
            # Calculate correlation between temperature and acceptance rate
            if len(acceptance_rates) > 1 and sum(acceptance_rates) > 0:
                temp_midpoints = [(temp_ranges[i] + temp_ranges[i+1])/2 for i in range(len(temp_ranges)-1)]
                if len(temp_midpoints) == len(acceptance_rates):
                    correlation = np.corrcoef(temp_midpoints, acceptance_rates)[0, 1]
                    self.metrics["cooling_efficiency"] = correlation
        
        # Calculate temperature impact (effect of temperature on solution quality)
        if len(self.metrics["temperature_history"]) > 10 and len(self.raw_data["fitness_history"]) > 10:
            # Simplistic measure: correlation between temperature and fitness improvement
            fitness_improvements = []
            for i in range(1, len(self.raw_data["fitness_history"])):
                improvement = self.raw_data["fitness_history"][i-1] - self.raw_data["fitness_history"][i]
                fitness_improvements.append(improvement)
            
            # Ensure lengths match
            min_length = min(len(fitness_improvements), len(self.metrics["temperature_history"][:-1]))
            if min_length > 1:
                correlation = np.corrcoef(
                    self.metrics["temperature_history"][:min_length], 
                    fitness_improvements[:min_length]
                )[0, 1]
                self.metrics["temperature_impact"] = correlation


class GeneticAlgorithmMetricCollector(MetricCollector):
    """Metric collector for Genetic Algorithm."""
    
    def __init__(self, config_name: str, run_id: Optional[str] = None):
        """
        Initialize the Genetic Algorithm metric collector.
        
        Args:
            config_name: Configuration name
            run_id: Optional unique identifier for this run
        """
        super().__init__("GeneticAlgorithm", config_name, run_id)
        
        # Genetic Algorithm specific metrics
        self.metrics.update({
            "generations": 0,
            "population_size": 0,
            "population_diversity": [],
            "selection_pressure": None,
            "crossover_success_rate": None,
            "mutation_impact": None,
            "elitism_impact": None,
            "avg_fitness_history": [],
            "std_fitness_history": [],
            "best_fitness_per_gen": [],
            "stagnation_generations": 0,
        })
        
        # Raw data for GA-specific analysis
        self.raw_data.update({
            "crossover_operations": 0,
            "successful_crossovers": 0,
            "mutation_operations": 0,
            "successful_mutations": 0,
            "selection_counts": {},  # Track how often each individual is selected
            "parent_fitness": [],
            "offspring_fitness": [],
        })
    
    def record_generation(self, generation: int, population: List[Any], 
                         best_fitness: float, avg_fitness: float, std_fitness: float,
                         diversity: float):
        """
        Record metrics for a generation.
        
        Args:
            generation: Current generation number
            population: Current population
            best_fitness: Best fitness in the population
            avg_fitness: Average fitness in the population
            std_fitness: Standard deviation of fitness in the population
            diversity: Population diversity measure
        """
        self.metrics["generations"] = generation
        self.metrics["population_size"] = len(population)
        self.metrics["population_diversity"].append(diversity)
        self.metrics["avg_fitness_history"].append(avg_fitness)
        self.metrics["std_fitness_history"].append(std_fitness)
        self.metrics["best_fitness_per_gen"].append(best_fitness)
        
        # Update best fitness if improved
        if best_fitness < self.metrics["best_fitness"]:
            self.metrics["best_fitness"] = best_fitness
            self.metrics["iterations_to_best"] = generation
        else:
            # Check for stagnation
            self.metrics["stagnation_generations"] += 1
        
        # Record as an iteration for common metrics
        self.record_iteration(best_fitness, best_fitness < self.metrics["best_fitness"])
    
    def record_selection(self, individual_id: Any):
        """
        Record selection of an individual.
        
        Args:
            individual_id: Identifier for the selected individual
        """
        if individual_id not in self.raw_data["selection_counts"]:
            self.raw_data["selection_counts"][individual_id] = 0
        self.raw_data["selection_counts"][individual_id] += 1
    
    def record_crossover(self, parent1_fitness: float, parent2_fitness: float, 
                        offspring_fitness: float, successful: bool):
        """
        Record a crossover operation.
        
        Args:
            parent1_fitness: Fitness of first parent
            parent2_fitness: Fitness of second parent
            offspring_fitness: Fitness of offspring
            successful: Whether the crossover produced a valid offspring
        """
        self.raw_data["crossover_operations"] += 1
        self.raw_data["parent_fitness"].append((parent1_fitness, parent2_fitness))
        self.raw_data["offspring_fitness"].append(offspring_fitness)
        
        if successful:
            self.raw_data["successful_crossovers"] += 1
        
        # Record function evaluation
        self.record_function_evaluation(offspring_fitness)
    
    def record_mutation(self, pre_mutation_fitness: float, post_mutation_fitness: float, successful: bool):
        """
        Record a mutation operation.
        
        Args:
            pre_mutation_fitness: Fitness before mutation
            post_mutation_fitness: Fitness after mutation
            successful: Whether the mutation produced a valid individual
        """
        self.raw_data["mutation_operations"] += 1
        
        if successful:
            self.raw_data["successful_mutations"] += 1
        
        # Record function evaluation
        self.record_function_evaluation(post_mutation_fitness)
    
    def calculate_derived_metrics(self):
        """Calculate Genetic Algorithm specific derived metrics."""
        super().calculate_derived_metrics()
        
        # Calculate crossover success rate
        if self.raw_data["crossover_operations"] > 0:
            self.metrics["crossover_success_rate"] = (
                self.raw_data["successful_crossovers"] / self.raw_data["crossover_operations"]
            )
        
        # Calculate mutation impact (average fitness change due to mutation)
        if len(self.raw_data["parent_fitness"]) > 0 and len(self.raw_data["offspring_fitness"]) > 0:
            parent_avg = np.mean([np.mean([p1, p2]) for p1, p2 in self.raw_data["parent_fitness"]])
            offspring_avg = np.mean(self.raw_data["offspring_fitness"])
            self.metrics["mutation_impact"] = parent_avg - offspring_avg
        
        # Calculate selection pressure (variance in selection frequency)
        if self.raw_data["selection_counts"]:
            selection_counts = list(self.raw_data["selection_counts"].values())
            if len(selection_counts) > 1:
                self.metrics["selection_pressure"] = np.var(selection_counts)
        
        # Calculate elitism impact (if elitism was used)
        if len(self.metrics["best_fitness_per_gen"]) > 1:
            improvements = [
                self.metrics["best_fitness_per_gen"][i-1] - self.metrics["best_fitness_per_gen"][i]
                for i in range(1, len(self.metrics["best_fitness_per_gen"]))
            ]
            if improvements:
                self.metrics["elitism_impact"] = np.mean(improvements)


class IslandGAMetricCollector(GeneticAlgorithmMetricCollector):
    """Metric collector for Island Genetic Algorithm."""
    
    def __init__(self, config_name: str, run_id: Optional[str] = None):
        """
        Initialize the Island Genetic Algorithm metric collector.
        
        Args:
            config_name: Configuration name
            run_id: Optional unique identifier for this run
        """
        super().__init__(config_name, run_id)
        self.algorithm_name = "IslandGA"
        
        # Island GA specific metrics
        self.metrics.update({
            "num_islands": 0,
            "migration_events": 0,
            "migration_impact": None,
            "island_diversity": [],
            "inter_island_diversity": None,
            "migration_success_rate": None,
            "island_convergence_rates": [],
            "topology_efficiency": None,
        })
        
        # Raw data for Island GA-specific analysis
        self.raw_data.update({
            "island_best_fitness": {},  # Track best fitness per island
            "island_avg_fitness": {},   # Track average fitness per island
            "migration_details": [],    # Details of each migration event
            "pre_migration_fitness": [],
            "post_migration_fitness": [],
        })
    
    def set_num_islands(self, num_islands: int):
        """
        Set the number of islands.
        
        Args:
            num_islands: Number of islands
        """
        self.metrics["num_islands"] = num_islands
        
        # Initialize per-island tracking
        for i in range(num_islands):
            self.raw_data["island_best_fitness"][i] = []
            self.raw_data["island_avg_fitness"][i] = []
    
    def record_island_metrics(self, island_idx: int, best_fitness: float, avg_fitness: float, diversity: float):
        """
        Record metrics for a specific island.
        
        Args:
            island_idx: Island index
            best_fitness: Best fitness on this island
            avg_fitness: Average fitness on this island
            diversity: Diversity measure for this island
        """
        if island_idx not in self.raw_data["island_best_fitness"]:
            self.raw_data["island_best_fitness"][island_idx] = []
        if island_idx not in self.raw_data["island_avg_fitness"]:
            self.raw_data["island_avg_fitness"][island_idx] = []
        
        self.raw_data["island_best_fitness"][island_idx].append(best_fitness)
        self.raw_data["island_avg_fitness"][island_idx].append(avg_fitness)
        
        # Update overall best fitness if improved
        if best_fitness < self.metrics["best_fitness"]:
            self.metrics["best_fitness"] = best_fitness
    
    def record_migration(self, source_island: int, target_island: int, 
                        migrants_fitness: List[float], success: bool,
                        pre_migration_best: float, post_migration_best: float):
        """
        Record a migration event.
        
        Args:
            source_island: Source island index
            target_island: Target island index
            migrants_fitness: Fitness values of migrants
            success: Whether migration was successful
            pre_migration_best: Best fitness on target island before migration
            post_migration_best: Best fitness on target island after migration
        """
        self.metrics["migration_events"] += 1
        
        migration_detail = {
            "source_island": source_island,
            "target_island": target_island,
            "migrants_fitness": migrants_fitness,
            "success": success,
            "pre_migration_best": pre_migration_best,
            "post_migration_best": post_migration_best,
            "improvement": pre_migration_best - post_migration_best
        }
        
        self.raw_data["migration_details"].append(migration_detail)
        self.raw_data["pre_migration_fitness"].append(pre_migration_best)
        self.raw_data["post_migration_fitness"].append(post_migration_best)
    
    def record_inter_island_diversity(self, diversity_value: float):
        """
        Record diversity between islands.
        
        Args:
            diversity_value: Measure of diversity between islands
        """
        self.metrics["inter_island_diversity"] = diversity_value
    
    def calculate_derived_metrics(self):
        """Calculate Island GA specific derived metrics."""
        super().calculate_derived_metrics()
        
        # Calculate migration impact (average improvement due to migration)
        if self.raw_data["pre_migration_fitness"] and self.raw_data["post_migration_fitness"]:
            improvements = [
                pre - post for pre, post in zip(
                    self.raw_data["pre_migration_fitness"], 
                    self.raw_data["post_migration_fitness"]
                )
            ]
            if improvements:
                self.metrics["migration_impact"] = np.mean(improvements)
        
        # Calculate migration success rate
        if self.raw_data["migration_details"]:
            successful_migrations = sum(1 for m in self.raw_data["migration_details"] if m["success"])
            self.metrics["migration_success_rate"] = successful_migrations / len(self.raw_data["migration_details"])
        
        # Calculate island convergence rates
        for island_idx, fitness_history in self.raw_data["island_best_fitness"].items():
            if len(fitness_history) > 10:
                initial_fitness = fitness_history[0]
                final_fitness = fitness_history[-1]
                total_iterations = len(fitness_history)
                
                if initial_fitness != final_fitness and total_iterations > 1:
                    convergence_rate = (initial_fitness - final_fitness) / (total_iterations - 1)
                    self.metrics["island_convergence_rates"].append(convergence_rate)
        
        # Calculate topology efficiency (if multiple islands)
        if self.metrics["num_islands"] > 1 and self.raw_data["migration_details"]:
            # Simple measure: average improvement per migration event
            improvements = [m["improvement"] for m in self.raw_data["migration_details"] if m["success"]]
            if improvements:
                self.metrics["topology_efficiency"] = np.mean(improvements)


def create_metric_collector(algorithm_type: str, config_name: str, run_id: Optional[str] = None) -> MetricCollector:
    """
    Factory function to create the appropriate metric collector based on algorithm type.
    
    Args:
        algorithm_type: Type of algorithm ("HillClimbing", "SimulatedAnnealing", "GeneticAlgorithm", "IslandGA")
        config_name: Configuration name
        run_id: Optional unique identifier for this run
    
    Returns:
        Appropriate metric collector instance
    """
    if algorithm_type == "HillClimbing":
        return HillClimbingMetricCollector(config_name, run_id)
    elif algorithm_type == "SimulatedAnnealing":
        return SimulatedAnnealingMetricCollector(config_name, run_id)
    elif algorithm_type == "GeneticAlgorithm":
        return GeneticAlgorithmMetricCollector(config_name, run_id)
    elif algorithm_type == "IslandGA":
        return IslandGAMetricCollector(config_name, run_id)
    else:
        logger.warning(f"Unknown algorithm type: {algorithm_type}. Using base MetricCollector.")
        return MetricCollector(algorithm_type, config_name, run_id)
