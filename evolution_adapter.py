"""
Adapter module for evolution.py to provide class-based algorithm implementations.

This module wraps the function-based implementations in evolution.py to provide
class-based algorithm implementations compatible with the API.
"""

import sys
import os
from copy import deepcopy

# Add upload directory to path
sys.path.append('/home/ubuntu/upload')

# Import function-based implementations
from evolution import hill_climbing, simulated_annealing

class HillClimbing:
    """
    Class-based wrapper for hill_climbing function.
    """
    
    def __init__(self, initial_solution, mutation_operator=None, restart_operator=None, 
                 max_iterations=1000, random_restarts=0, intensive_search=False, 
                 execution_mode=None):
        """
        Initialize the Hill Climbing algorithm.
        
        Args:
            initial_solution: Initial solution
            mutation_operator: Mutation operator (not used in this implementation)
            restart_operator: Restart operator (not used in this implementation)
            max_iterations: Maximum number of iterations
            random_restarts: Number of random restarts (not used in this implementation)
            intensive_search: Whether to use intensive search (not used in this implementation)
            execution_mode: Execution mode (not used in this implementation)
        """
        self.initial_solution = initial_solution
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
    
    def __iter__(self):
        """
        Make the algorithm iterable.
        
        Returns:
            Iterator
        """
        return self
    
    def __next__(self):
        """
        Get the next solution in the optimization process.
        
        Returns:
            Next solution
        
        Raises:
            StopIteration: When the optimization is complete
        """
        if self.current_iteration == 0:
            # First iteration, run hill climbing
            self.best_solution, self.best_fitness, self.history = hill_climbing(
                initial_solution=self.initial_solution,
                max_iterations=self.max_iterations,
                verbose=False
            )
            self.current_iteration = self.max_iterations
            return self.best_solution
        else:
            # Optimization complete
            raise StopIteration

class SimulatedAnnealing:
    """
    Class-based wrapper for simulated_annealing function.
    """
    
    def __init__(self, initial_solution, mutation_operator=None, initial_temperature=100.0,
                 cooling_rate=0.95, min_temperature=0.1, iterations_per_temp=20,
                 execution_mode=None):
        """
        Initialize the Simulated Annealing algorithm.
        
        Args:
            initial_solution: Initial solution
            mutation_operator: Mutation operator (not used in this implementation)
            initial_temperature: Initial temperature
            cooling_rate: Cooling rate
            min_temperature: Minimum temperature
            iterations_per_temp: Iterations per temperature
            execution_mode: Execution mode (not used in this implementation)
        """
        self.initial_solution = initial_solution
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.iterations_per_temp = iterations_per_temp
        self.current_iteration = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
    
    def __iter__(self):
        """
        Make the algorithm iterable.
        
        Returns:
            Iterator
        """
        return self
    
    def __next__(self):
        """
        Get the next solution in the optimization process.
        
        Returns:
            Next solution
        
        Raises:
            StopIteration: When the optimization is complete
        """
        if self.current_iteration == 0:
            # First iteration, run simulated annealing
            self.best_solution, self.best_fitness, self.history = simulated_annealing(
                initial_solution=self.initial_solution,
                initial_temperature=self.initial_temperature,
                cooling_rate=self.cooling_rate,
                min_temperature=self.min_temperature,
                iterations_per_temp=self.iterations_per_temp,
                verbose=False
            )
            
            # Calculate total iterations
            temperature = self.initial_temperature
            total_iterations = 0
            while temperature > self.min_temperature:
                total_iterations += self.iterations_per_temp
                temperature *= self.cooling_rate
            
            self.current_iteration = total_iterations
            return self.best_solution
        else:
            # Optimization complete
            raise StopIteration
