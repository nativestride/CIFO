"""
Selection operators adapter module for Fantasy League Optimization.

This module provides adapter classes for selection operators in the original codebase.
"""

import sys
import logging
import random

# Add project root to path
sys.path.append('/home/ubuntu/upload')

# Import selection operators
from selection_operators import selection_tournament, selection_ranking, selection_boltzmann

# Configure logger
logger = logging.getLogger(__name__)

class TournamentSelection:
    """
    Adapter class for tournament selection operator.
    """
    
    def __init__(self, tournament_size=3):
        """
        Initialize the TournamentSelection adapter.
        
        Args:
            tournament_size: Size of tournament (optional)
        """
        self.tournament_size = tournament_size
    
    def __call__(self, population):
        """
        Apply tournament selection to the population.
        
        Args:
            population: List of solutions
            
        Returns:
            Selected solution
        """
        return selection_tournament(population, k=self.tournament_size)

def selection_roulette(population):
    """
    Roulette wheel selection implementation.
    
    Args:
        population: List of solutions
        
    Returns:
        Selected solution
    """
    if not population:
        return None
    
    # Calculate fitness values (assuming lower is better)
    fitness_values = [sol.fitness() for sol in population]
    
    # Handle infinite fitness values
    max_fitness = max(f for f in fitness_values if f != float('inf'))
    if all(f == float('inf') for f in fitness_values):
        # If all solutions have infinite fitness, select randomly
        return random.choice(population)
    
    # Transform fitness for maximization (higher value = higher probability)
    # Add small constant to avoid division by zero
    transformed_fitness = [max_fitness - f if f != float('inf') else 0 for f in fitness_values]
    
    # Calculate total fitness
    total_fitness = sum(transformed_fitness)
    
    if total_fitness == 0:
        # If total fitness is zero, select randomly
        return random.choice(population)
    
    # Calculate selection probabilities
    probabilities = [f / total_fitness for f in transformed_fitness]
    
    # Select solution based on probabilities
    return random.choices(population, weights=probabilities, k=1)[0]

class RouletteWheelSelection:
    """
    Adapter class for roulette wheel selection.
    """
    
    def __init__(self):
        """
        Initialize the RouletteWheelSelection adapter.
        """
        pass
    
    def __call__(self, population):
        """
        Apply roulette wheel selection to the population.
        
        Args:
            population: List of solutions
            
        Returns:
            Selected solution
        """
        return selection_roulette(population)

class RankingSelection:
    """
    Adapter class for ranking selection.
    """
    
    def __init__(self):
        """
        Initialize the RankingSelection adapter.
        """
        pass
    
    def __call__(self, population):
        """
        Apply ranking selection to the population.
        
        Args:
            population: List of solutions
            
        Returns:
            Selected solution
        """
        return selection_ranking(population)

class BoltzmannSelection:
    """
    Adapter class for Boltzmann selection.
    """
    
    def __init__(self, temperature=1.0):
        """
        Initialize the BoltzmannSelection adapter.
        
        Args:
            temperature: Temperature parameter (optional)
        """
        self.temperature = temperature
    
    def __call__(self, population):
        """
        Apply Boltzmann selection to the population.
        
        Args:
            population: List of solutions
            
        Returns:
            Selected solution
        """
        from experiment_utils import safe_exp
        return selection_boltzmann(population, temperature=self.temperature, safe_exp_func=safe_exp)
