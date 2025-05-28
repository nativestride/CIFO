"""
Operators adapter module for Fantasy League Optimization.

This module provides adapter classes for operators that were previously in operators.py
but have been refactored into separate modules.
"""

import sys
import os

# Add upload directory to path
sys.path.append('/home/ubuntu/upload')

# Import operators from their new locations
from mutation_operators import mutate_swap, SwapMutation as OriginalSwapMutation
from crossover_operators import crossover_one_point, crossover_two_point, crossover_uniform

# Create adapter classes
class SwapMutation:
    """
    Adapter class for SwapMutation.
    """
    
    def __init__(self):
        """
        Initialize the SwapMutation operator.
        """
        pass
    
    def __call__(self, solution):
        """
        Apply the mutation operator to the solution.
        
        Args:
            solution: Solution to mutate
        
        Returns:
            Mutated solution
        """
        return mutate_swap(solution)

class RandomRestartOperator:
    """
    Adapter class for RandomRestartOperator.
    """
    
    def __init__(self):
        """
        Initialize the RandomRestartOperator.
        """
        pass
    
    def __call__(self, solution):
        """
        Apply the restart operator to the solution.
        
        Args:
            solution: Solution to restart
        
        Returns:
            New random solution
        """
        # Create a new random solution with the same structure
        from copy import deepcopy
        import random
        
        # Create a deep copy of the solution
        new_solution = deepcopy(solution)
        
        # Shuffle players between teams
        all_players = []
        for team in new_solution.teams:
            all_players.extend(team.players)
            team.players = []
        
        # Shuffle players
        random.shuffle(all_players)
        
        # Distribute players to teams
        players_per_team = len(all_players) // len(new_solution.teams)
        for i, team in enumerate(new_solution.teams):
            start_idx = i * players_per_team
            end_idx = start_idx + players_per_team if i < len(new_solution.teams) - 1 else len(all_players)
            team.players = all_players[start_idx:end_idx]
        
        return new_solution
