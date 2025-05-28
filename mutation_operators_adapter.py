"""
Mutation operators adapter module for Fantasy League Optimization.

This module provides adapter classes for mutation operators in the original codebase.
"""

import sys
import logging
from copy import deepcopy

# Add project root to path
sys.path.append('/home/ubuntu/upload')

# Import mutation operators
from mutation_operators import (
    mutate_swap,
    mutate_swap_constrained,
    mutate_team_shift,
    mutate_targeted_player_exchange,
    mutate_shuffle_within_team_constrained
)

# Configure logger
logger = logging.getLogger(__name__)

class MutationOperator:
    """
    Base class for mutation operators.
    """
    
    def __call__(self, solution):
        """
        Apply mutation to the solution.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        raise NotImplementedError("Subclasses must implement __call__")

class SwapMutation(MutationOperator):
    """
    Adapter class for swap mutation operator.
    """
    
    def __call__(self, solution):
        """
        Apply swap mutation to the solution.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        return mutate_swap(solution)

class ConstrainedSwapMutation(MutationOperator):
    """
    Adapter class for constrained swap mutation operator.
    """
    
    def __call__(self, solution):
        """
        Apply constrained swap mutation to the solution.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        return mutate_swap_constrained(solution)

class TeamShiftMutation(MutationOperator):
    """
    Adapter class for team shift mutation operator.
    """
    
    def __call__(self, solution):
        """
        Apply team shift mutation to the solution.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        return mutate_team_shift(solution)

class TargetedPlayerExchangeMutation(MutationOperator):
    """
    Adapter class for targeted player exchange mutation operator.
    """
    
    def __call__(self, solution):
        """
        Apply targeted player exchange mutation to the solution.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        return mutate_targeted_player_exchange(solution)

class ShuffleWithinTeamMutation(MutationOperator):
    """
    Adapter class for shuffle within team mutation operator.
    """
    
    def __call__(self, solution):
        """
        Apply shuffle within team mutation to the solution.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        return mutate_shuffle_within_team_constrained(solution)
