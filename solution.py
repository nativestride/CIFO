"""
Solution classes for Fantasy League Optimization.

This module provides Solution classes for the Fantasy League Optimization problem.
"""

from player_team_classes import Player, Team

# Exception class needed by evolution.py
class InsufficientPlayersForPositionError(Exception):
    """Custom exception for errors when player positional quotas cannot be met."""
    pass

class Solution:
    """
    Solution class for Fantasy League Optimization.
    """
    
    def __init__(self, teams=None):
        """
        Initialize a solution.
        
        Args:
            teams: List of Team objects
        """
        self.teams = teams or []
    
    def fitness(self):
        """
        Calculate the fitness of the solution.
        
        Returns:
            Fitness value (lower is better)
        """
        # Calculate standard deviation of team skills
        team_skills = [team.total_skill() for team in self.teams]
        if not team_skills:
            return float('inf')
        
        # Calculate mean skill
        mean_skill = sum(team_skills) / len(team_skills)
        
        # Calculate standard deviation
        variance = sum((skill - mean_skill) ** 2 for skill in team_skills) / len(team_skills)
        std_dev = variance ** 0.5
        
        return std_dev
    
    def __repr__(self):
        return f"Solution(teams={len(self.teams)}, fitness={self.fitness():.2f})"

# Stub classes needed by evolution.py and other algorithm modules
class LeagueSolution(Solution):
    """
    Stub class for LeagueSolution.
    """
    def __init__(self, teams=None, **kwargs):
        super().__init__(teams=teams)
        # Store kwargs for compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)

class LeagueHillClimbingSolution(LeagueSolution):
    """
    Stub class for LeagueHillClimbingSolution.
    """
    def get_neighbors(self, max_neighbors_total=None):
        """Stub method for get_neighbors."""
        return []

class LeagueSASolution(LeagueSolution):
    """
    Stub class for LeagueSASolution.
    """
    def get_random_neighbor(self):
        """Stub method for get_random_neighbor."""
        import copy
        return copy.deepcopy(self)
