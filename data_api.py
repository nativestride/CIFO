"""
Data API module for the Fantasy League optimization website.

This module provides functions to convert player data from various sources
(CSV, pickle files, etc.) into a standardized JSON format for the web application.
"""

import os
import json
import pandas as pd
import pickle
import numpy as np

class DataAPI:
    """
    Class for exposing player and optimization data through JSON API.
    """
    
    def __init__(self, data_dir='/home/ubuntu/upload'):
        """
        Initialize the Data API.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.players_cache = None
        self.solutions_cache = {}
    
    def get_players_from_csv(self, csv_file='players.csv'):
        """
        Get player data from CSV file.
        
        Args:
            csv_file: CSV file name
            
        Returns:
            List of player dictionaries
        """
        # Check if players are already cached
        if self.players_cache is not None:
            return self.players_cache
        
        # Load CSV file
        csv_path = os.path.join(self.data_dir, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Read CSV file
        players_df = pd.read_csv(csv_path, sep=';')
        
        # Convert to list of dictionaries
        players = []
        for _, row in players_df.iterrows():
            player = {
                'id': int(row[0]),
                'name': row['Name'],
                'position': row['Position'],
                'skill': int(row['Skill']),
                'cost': int(row['Salary (€M)'])
            }
            players.append(player)
        
        # Cache players
        self.players_cache = players
        
        return players
    
    def get_players_from_pickle(self, pickle_file):
        """
        Get player data from a pickle file containing a solution.
        
        Args:
            pickle_file: Pickle file name
            
        Returns:
            List of player dictionaries
        """
        # Load pickle file
        pickle_path = os.path.join(self.data_dir, pickle_file)
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        
        # Load solution from pickle
        with open(pickle_path, 'rb') as f:
            solution = pickle.load(f)
        
        # Extract players from solution
        players = []
        for player in solution.get_all_players():
            player_dict = {
                'id': player.id,
                'name': player.name,
                'position': player.position,
                'skill': player.skill,
                'cost': player.cost
            }
            players.append(player_dict)
        
        return players
    
    def get_solution_from_pickle(self, pickle_file):
        """
        Get solution data from a pickle file.
        
        Args:
            pickle_file: Pickle file name
            
        Returns:
            Dictionary with solution data
        """
        # Check if solution is already cached
        if pickle_file in self.solutions_cache:
            return self.solutions_cache[pickle_file]
        
        # Load pickle file
        pickle_path = os.path.join(self.data_dir, pickle_file)
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        
        # Load solution from pickle
        with open(pickle_path, 'rb') as f:
            solution = pickle.load(f)
        
        # Extract solution data
        solution_data = {
            'fitness': solution.fitness(),
            'total_cost': solution.total_cost(),
            'total_skill': solution.total_skill(),
            'skill_std_dev': solution.skill_std_dev(),
            'teams': []
        }
        
        # Extract team data
        for team in solution.teams:
            team_data = {
                'id': team.id,
                'name': f'Team {team.id + 1}',
                'players': [],
                'total_cost': team.total_cost(),
                'total_skill': team.total_skill(),
                'average_skill': team.average_skill()
            }
            
            # Extract player data
            for player in team.players:
                player_data = {
                    'id': player.id,
                    'name': player.name,
                    'position': player.position,
                    'skill': player.skill,
                    'cost': player.cost
                }
                team_data['players'].append(player_data)
            
            solution_data['teams'].append(team_data)
        
        # Cache solution
        self.solutions_cache[pickle_file] = solution_data
        
        return solution_data
    
    def get_available_solutions(self):
        """
        Get list of available solution pickle files.
        
        Returns:
            List of solution file names
        """
        # Get all pickle files in data directory
        pickle_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        
        # Filter for solution files
        solution_files = [f for f in pickle_files if f.startswith('best_solution_')]
        
        return solution_files
    
    def get_algorithm_from_filename(self, filename):
        """
        Extract algorithm name from solution filename.
        
        Args:
            filename: Solution filename
            
        Returns:
            Algorithm name
        """
        # Extract algorithm name from filename
        # Format: best_solution_all_algos_ALGORITHM_VARIANT.pkl
        parts = filename.split('_')
        if len(parts) >= 5:
            return parts[4]
        
        return 'Unknown'
    
    def get_all_solutions(self):
        """
        Get data for all available solutions.
        
        Returns:
            Dictionary mapping algorithm names to solution data
        """
        # Get available solution files
        solution_files = self.get_available_solutions()
        
        # Load solutions
        solutions = {}
        for filename in solution_files:
            algorithm = self.get_algorithm_from_filename(filename)
            try:
                solution_data = self.get_solution_from_pickle(filename)
                solutions[filename] = {
                    'algorithm': algorithm,
                    'data': solution_data
                }
            except Exception as e:
                print(f"Error loading solution {filename}: {e}")
        
        return solutions
    
    def save_players_to_json(self, output_file='players.json'):
        """
        Save player data to JSON file.
        
        Args:
            output_file: Output JSON file name
            
        Returns:
            Path to the saved file
        """
        # Get player data
        players = self.get_players_from_csv()
        
        # Save to JSON file
        output_path = os.path.join(self.data_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(players, f, indent=2)
        
        return output_path
    
    def save_solutions_to_json(self, output_file='solutions.json'):
        """
        Save all solution data to JSON file.
        
        Args:
            output_file: Output JSON file name
            
        Returns:
            Path to the saved file
        """
        # Get solution data
        solutions = self.get_all_solutions()
        
        # Convert numpy types to native Python types
        solutions_json = self._make_serializable(solutions)
        
        # Save to JSON file
        output_path = os.path.join(self.data_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(solutions_json, f, indent=2)
        
        return output_path
    
    def _make_serializable(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        else:
            return obj


# Example usage
if __name__ == '__main__':
    # Create Data API
    data_api = DataAPI()
    
    # Save player data to JSON
    players_json_path = data_api.save_players_to_json()
    print(f"Player data saved to {players_json_path}")
    
    # Save solution data to JSON
    solutions_json_path = data_api.save_solutions_to_json()
    print(f"Solution data saved to {solutions_json_path}")
