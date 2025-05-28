"""
Solution Factory module for Fantasy League Optimization.

This module provides a factory for creating valid solutions for the Fantasy League Optimization problem.
"""

import sys
import os
import random
from pathlib import Path

# Add upload directory to path
sys.path.append('/home/ubuntu/upload')

# Import solution module
from solution import Solution
from player_team_classes import Player, Team

class SolutionFactory:
    """
    Factory for creating valid solutions for the Fantasy League Optimization problem.
    """
    
    def __init__(self, players, num_teams, budget, gk_count, def_count, mid_count, fwd_count):
        """
        Initialize the solution factory.
        
        Args:
            players: List of Player objects
            num_teams: Number of teams
            budget: Budget per team
            gk_count: Number of goalkeepers per team
            def_count: Number of defenders per team
            mid_count: Number of midfielders per team
            fwd_count: Number of forwards per team
        """
        self.players = players
        self.num_teams = num_teams
        self.budget = budget
        self.gk_count = gk_count
        self.def_count = def_count
        self.mid_count = mid_count
        self.fwd_count = fwd_count
    
    def create_random_solution(self):
        """
        Create a random valid solution.
        
        Returns:
            Solution object
        """
        # Group players by position
        players_by_position = {
            'GK': [p for p in self.players if p.position == 'GK'],
            'DEF': [p for p in self.players if p.position == 'DEF'],
            'MID': [p for p in self.players if p.position == 'MID'],
            'FWD': [p for p in self.players if p.position == 'FWD']
        }
        
        # Shuffle players
        for position in players_by_position:
            random.shuffle(players_by_position[position])
        
        # Create teams
        teams = []
        for i in range(self.num_teams):
            team = Team(id=i+1)
            teams.append(team)
        
        # Assign goalkeepers
        self._assign_players(teams, players_by_position['GK'], self.gk_count)
        
        # Assign defenders
        self._assign_players(teams, players_by_position['DEF'], self.def_count)
        
        # Assign midfielders
        self._assign_players(teams, players_by_position['MID'], self.mid_count)
        
        # Assign forwards
        self._assign_players(teams, players_by_position['FWD'], self.fwd_count)
        
        # Create solution
        solution = Solution(teams=teams)
        
        return solution
    
    def _assign_players(self, teams, players, count_per_team):
        """
        Assign players to teams.
        
        Args:
            teams: List of Team objects
            players: List of Player objects
            count_per_team: Number of players per team
        """
        # Create a copy of players list
        available_players = players.copy()
        
        # Assign players to teams
        for team in teams:
            for _ in range(count_per_team):
                if not available_players:
                    break
                
                # Find a player that fits within budget
                player = self._find_player_within_budget(available_players, team)
                
                if player:
                    team.add_player(player)
                    available_players.remove(player)
    
    def _find_player_within_budget(self, players, team):
        """
        Find a player that fits within the team's budget.
        
        Args:
            players: List of Player objects
            team: Team object
        
        Returns:
            Player object or None
        """
        # Calculate remaining budget
        remaining_budget = self.budget - team.total_cost()
        
        # Find players within budget
        affordable_players = [p for p in players if p.cost <= remaining_budget]
        
        if affordable_players:
            return random.choice(affordable_players)
        
        return None
