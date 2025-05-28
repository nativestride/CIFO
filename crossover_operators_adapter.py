"""
Crossover operators adapter module for Fantasy League Optimization.

This module provides adapter classes for crossover operators in the original codebase.
"""

import sys
import logging
from copy import deepcopy

# Add project root to path
sys.path.append('/home/ubuntu/upload')

# Import crossover operators
from crossover_operators import (
    crossover_one_point, 
    crossover_two_point, 
    crossover_uniform,
    crossover_one_point_prefer_valid,
    crossover_two_point_prefer_valid,
    crossover_uniform_prefer_valid
)

# Configure logger
logger = logging.getLogger(__name__)

class OnePointCrossover:
    """
    Adapter class for one-point crossover operator.
    """
    
    def __init__(self, prefer_valid=True, max_attempts=10):
        """
        Initialize the OnePointCrossover adapter.
        
        Args:
            prefer_valid: Whether to prefer valid solutions (optional)
            max_attempts: Maximum number of attempts to generate valid solution (optional)
        """
        self.prefer_valid = prefer_valid
        self.max_attempts = max_attempts
    
    def __call__(self, parent1, parent2):
        """
        Apply one-point crossover to the parents.
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
            
        Returns:
            Child solution
        """
        if self.prefer_valid:
            return crossover_one_point_prefer_valid(parent1, parent2, self.max_attempts)
        else:
            return crossover_one_point(parent1, parent2)

class TwoPointCrossover:
    """
    Adapter class for two-point crossover operator.
    """
    
    def __init__(self, prefer_valid=True, max_attempts=10):
        """
        Initialize the TwoPointCrossover adapter.
        
        Args:
            prefer_valid: Whether to prefer valid solutions (optional)
            max_attempts: Maximum number of attempts to generate valid solution (optional)
        """
        self.prefer_valid = prefer_valid
        self.max_attempts = max_attempts
    
    def __call__(self, parent1, parent2):
        """
        Apply two-point crossover to the parents.
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
            
        Returns:
            Child solution
        """
        if self.prefer_valid:
            return crossover_two_point_prefer_valid(parent1, parent2, self.max_attempts)
        else:
            return crossover_two_point(parent1, parent2)

class UniformCrossover:
    """
    Adapter class for uniform crossover operator.
    """
    
    def __init__(self, prefer_valid=True, max_attempts=10):
        """
        Initialize the UniformCrossover adapter.
        
        Args:
            prefer_valid: Whether to prefer valid solutions (optional)
            max_attempts: Maximum number of attempts to generate valid solution (optional)
        """
        self.prefer_valid = prefer_valid
        self.max_attempts = max_attempts
    
    def __call__(self, parent1, parent2):
        """
        Apply uniform crossover to the parents.
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
            
        Returns:
            Child solution
        """
        if self.prefer_valid:
            return crossover_uniform_prefer_valid(parent1, parent2, self.max_attempts)
        else:
            return crossover_uniform(parent1, parent2)
