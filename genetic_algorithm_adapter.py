"""
Adapter module for genetic_algorithms.py to provide class-based algorithm implementations.

This module wraps the function-based implementations in genetic_algorithms.py to provide
class-based algorithm implementations compatible with the API.
"""

import sys
import os
from copy import deepcopy

# Add upload directory to path
sys.path.append('/home/ubuntu/upload')

# Import function-based implementations
from genetic_algorithms import genetic_algorithm
from solution import LeagueSolution

class GeneticAlgorithm:
    """
    Class-based wrapper for genetic_algorithm function.
    """
    
    def __init__(self, initial_solution, population_size=50, selection_operator=None, 
                 crossover_operator=None, mutation_operator=None, crossover_rate=0.8, 
                 mutation_rate=0.1, max_generations=100, execution_mode=None):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            initial_solution: Initial solution
            population_size: Size of the population
            selection_operator: Selection operator
            crossover_operator: Crossover operator
            mutation_operator: Mutation operator
            crossover_rate: Crossover rate
            mutation_rate: Mutation rate
            max_generations: Maximum number of generations
            execution_mode: Execution mode (not used in this implementation)
        """
        self.initial_solution = initial_solution
        self.population_size = population_size
        self.selection_operator = selection_operator
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.current_generation = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
        
        # Extract problem parameters from initial solution
        if hasattr(initial_solution, 'teams'):
            self.num_teams = len(initial_solution.teams)
            self.team_size = len(initial_solution.teams[0].players) if initial_solution.teams and initial_solution.teams[0].players else 0
            self.max_budget = 1000  # Default budget, will be overridden by solution validity check
            
            # Extract position requirements
            if initial_solution.teams and initial_solution.teams[0].players:
                positions = {}
                for player in initial_solution.teams[0].players:
                    if player.position not in positions:
                        positions[player.position] = 0
                    positions[player.position] += 1
                self.position_requirements = positions
            else:
                self.position_requirements = {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}  # Default formation
        else:
            self.num_teams = 5  # Default number of teams
            self.team_size = 11  # Default team size
            self.max_budget = 1000  # Default budget
            self.position_requirements = {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}  # Default formation
    
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
        if self.current_generation == 0:
            # First generation, run genetic algorithm
            
            # Extract players from initial solution
            players_data = []
            for team in self.initial_solution.teams:
                for player in team.players:
                    players_data.append(player)
            
            # Set up problem parameters
            problem_params = {
                "num_teams": self.num_teams,
                "team_size": self.team_size,
                "max_budget": self.max_budget,
                "position_requirements": self.position_requirements
            }
            
            # Set up GA parameters
            ga_params = {
                "population_size": self.population_size,
                "max_generations": self.max_generations,
                "selection_operator": self.selection_operator,
                "crossover_operator": self.crossover_operator,
                "mutation_operator": self.mutation_operator,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "elitism_size": 1,  # Keep the best solution
                "verbose": False
            }
            
            # Run genetic algorithm
            self.best_solution, self.best_fitness, history_data = genetic_algorithm(
                players_data=players_data,
                problem_params=problem_params,
                ga_params=ga_params
            )
            
            # Extract fitness history
            self.history = [entry['best_fitness'] for entry in history_data]
            
            # Set current generation to max to indicate completion
            self.current_generation = self.max_generations
            
            return self.best_solution
        else:
            # Optimization complete
            raise StopIteration
