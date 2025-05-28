"""
Player and Team classes for Fantasy League Optimization.

This module provides the Player and Team classes used by the optimization algorithms.
"""

class Player:
    """
    Player class for Fantasy League Optimization.
    """
    
    def __init__(self, id, name, position, skill, cost):
        """
        Initialize a player.
        
        Args:
            id: Player ID
            name: Player name
            position: Player position (GK, DEF, MID, FWD)
            skill: Player skill level
            cost: Player cost
        """
        self.id = id
        self.name = name
        self.position = position
        self.skill = skill
        self.cost = cost
    
    def __repr__(self):
        return f"Player(id={self.id}, name='{self.name}', position='{self.position}', skill={self.skill}, cost={self.cost})"


class Team:
    """
    Team class for Fantasy League Optimization.
    """
    
    def __init__(self, id, players=None):
        """
        Initialize a team.
        
        Args:
            id: Team ID
            players: List of Player objects
        """
        self.id = id
        self.players = players or []
    
    def add_player(self, player):
        """
        Add a player to the team.
        
        Args:
            player: Player object
        """
        self.players.append(player)
    
    def remove_player(self, player):
        """
        Remove a player from the team.
        
        Args:
            player: Player object
        """
        self.players.remove(player)
    
    def total_skill(self):
        """
        Calculate the total skill of the team.
        
        Returns:
            Total skill
        """
        return sum(player.skill for player in self.players)
    
    def total_cost(self):
        """
        Calculate the total cost of the team.
        
        Returns:
            Total cost
        """
        return sum(player.cost for player in self.players)
    
    def __repr__(self):
        return f"Team(id={self.id}, players={len(self.players)})"
