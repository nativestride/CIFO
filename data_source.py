"""
Simple data source module for Fantasy League Optimization.

This module provides a simple data source for the Fantasy League Optimization website
using CSV/JSON files instead of a database.
"""

import os
import json
import csv
import pandas as pd
from pathlib import Path

# Import Player and Team classes
from player_team_classes import Player, Team

class DataSource:
    """
    Simple data source for Fantasy League Optimization using CSV/JSON files.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the data source.
        
        Args:
            data_dir: Directory for data files (default: /home/ubuntu/fantasy_league_dashboard/data)
        """
        self.data_dir = data_dir or '/home/ubuntu/fantasy_league_dashboard/data'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data files
        self.players_file = os.path.join(self.data_dir, 'players.json')
        self.config_file = os.path.join(self.data_dir, 'config.json')
        
        # Create default data if files don't exist
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize data files with default values if they don't exist."""
        # Initialize players from CSV if JSON doesn't exist
        if not os.path.exists(self.players_file):
            # Try to load from CSV
            csv_path = '/home/ubuntu/upload/players.csv'
            if os.path.exists(csv_path):
                self._load_players_from_csv(csv_path)
            else:
                # Create empty players file
                self._save_json(self.players_file, [])
        
        # Initialize configuration if it doesn't exist
        if not os.path.exists(self.config_file):
            default_config = {
                'num_teams': 5,
                'budget': 750,
                'gk_count': 1,
                'def_count': 2,
                'mid_count': 2,
                'fwd_count': 2
            }
            self._save_json(self.config_file, default_config)
    
    def _load_players_from_csv(self, csv_path):
        """
        Load players from CSV file.
        
        Args:
            csv_path: Path to CSV file
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Convert to list of dictionaries
            players = []
            for _, row in df.iterrows():
                # Map CSV columns to player attributes
                player = {
                    'id': int(row.get('ID', len(players) + 1)),
                    'name': row.get('Name', f'Player {len(players) + 1}'),
                    'position': row.get('Position', 'Unknown'),
                    'skill': float(row.get('Skill', 0)),
                    'cost': float(row.get('Salary', row.get('Salary (€M)', 0)))
                }
                players.append(player)
            
            # Save players to JSON
            self._save_json(self.players_file, players)
        except Exception as e:
            print(f"Error loading players from CSV: {e}")
            # Create empty players file
            self._save_json(self.players_file, [])
    
    def _save_json(self, file_path, data):
        """
        Save data to JSON file.
        
        Args:
            file_path: Path to JSON file
            data: Data to save
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_json(self, file_path):
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            Loaded data
        """
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_players(self):
        """
        Get all players.
        
        Returns:
            List of player dictionaries
        """
        return self._load_json(self.players_file) or []
    
    def get_configuration(self):
        """
        Get the configuration.
        
        Returns:
            Configuration dictionary
        """
        return self._load_json(self.config_file) or {}
    
    def validate_configuration(self):
        """
        Validate the configuration.
        
        Returns:
            Tuple of (is_valid, message)
        """
        config = self.get_configuration()
        players = self.get_players()
        
        # Check if configuration is complete
        required_fields = ['num_teams', 'budget', 'gk_count', 'def_count', 'mid_count', 'fwd_count']
        for field in required_fields:
            if field not in config:
                return False, f"Missing required configuration field: {field}"
        
        # Check if there are enough players
        if not players:
            return False, "No players available"
        
        # Calculate required players per position
        num_teams = config['num_teams']
        required_players = {
            'GK': num_teams * config['gk_count'],
            'DEF': num_teams * config['def_count'],
            'MID': num_teams * config['mid_count'],
            'FWD': num_teams * config['fwd_count']
        }
        
        # Count available players per position
        available_players = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for player in players:
            position = player.get('position')
            if position in available_players:
                available_players[position] += 1
        
        # Check if there are enough players per position
        for position, required in required_players.items():
            if available_players[position] < required:
                return False, f"Not enough {position} players: {available_players[position]} available, {required} required"
        
        return True, "Configuration is valid"
    
    def get_player_objects(self):
        """
        Get all players as Player objects.
        
        Returns:
            List of Player objects
        """
        players_data = self.get_players()
        return [
            Player(
                id=player['id'],
                name=player['name'],
                position=player['position'],
                skill=player['skill'],
                cost=player['cost']
            )
            for player in players_data
        ]
