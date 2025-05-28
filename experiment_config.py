"""
Centralized experiment configuration module for Fantasy League Optimization.

This module provides a centralized configuration system for experiments,
accessible from both the website and Jupyter notebooks.
"""

import os
import json
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append('/home/ubuntu/fantasy_league_dashboard')

# Import database module if available
try:
    from db_schema import FantasyLeagueDB
except ImportError:
    pass

class ExperimentConfig:
    """
    Centralized experiment configuration for Fantasy League Optimization.
    Provides a unified interface for configuring experiments from both
    the website and Jupyter notebooks.
    """
    
    def __init__(self, config_path=None, config_id=None, data_source='auto'):
        """
        Initialize the experiment configuration.
        
        Args:
            config_path: Path to JSON configuration file (optional)
            config_id: Configuration ID in SQLite database (optional)
            data_source: Data source type ('csv', 'sqlite', or 'auto')
        """
        self.config = {}
        self.data_source = data_source
        
        # Determine data source if 'auto'
        if self.data_source == 'auto':
            # Check if SQLite database exists
            db_path = '/home/ubuntu/fantasy_league_dashboard/fantasy_league.db'
            if os.path.exists(db_path):
                self.data_source = 'sqlite'
            else:
                self.data_source = 'csv'
        
        # Load configuration
        if config_path:
            self.load_from_json(config_path)
        elif config_id and self.data_source == 'sqlite':
            self.load_from_sqlite(config_id)
        else:
            self.set_default_config()
    
    def load_from_json(self, config_path):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading configuration from {config_path}: {e}")
            self.set_default_config()
    
    def load_from_sqlite(self, config_id):
        """
        Load configuration from SQLite database.
        
        Args:
            config_id: Configuration ID in SQLite database
        """
        try:
            # Import database module
            from db_schema import FantasyLeagueDB
            
            # Get configuration from database
            db = FantasyLeagueDB()
            config = db.get_configuration(config_id)
            
            if config:
                # Convert to expected format
                self.config = {
                    'name': config['name'],
                    'description': config['description'],
                    'problem': {
                        'num_teams': config['num_teams'],
                        'budget': config['budget'],
                        'team_composition': {
                            'GK': config['gk_count'],
                            'DEF': config['def_count'],
                            'MID': config['mid_count'],
                            'FWD': config['fwd_count']
                        }
                    },
                    'algorithms': self._get_default_algorithms()
                }
            else:
                print(f"Configuration with ID {config_id} not found")
                self.set_default_config()
        except ImportError:
            print("SQLite database module not available")
            self.set_default_config()
        except Exception as e:
            print(f"Error loading configuration from SQLite: {e}")
            self.set_default_config()
    
    def save_to_json(self, config_path):
        """
        Save configuration to a JSON file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving configuration to {config_path}: {e}")
    
    def save_to_sqlite(self):
        """
        Save configuration to SQLite database.
        
        Returns:
            Configuration ID if successful, None otherwise
        """
        try:
            # Import database module
            from db_schema import FantasyLeagueDB
            
            # Get configuration from database
            db = FantasyLeagueDB()
            
            # Convert to expected format
            team_comp = self.config.get('problem', {}).get('team_composition', {})
            
            # Add or update configuration
            config_id = db.add_configuration(
                name=self.config.get('name', 'Unnamed Configuration'),
                description=self.config.get('description', ''),
                num_teams=self.config.get('problem', {}).get('num_teams', 4),
                budget=self.config.get('problem', {}).get('budget', 1000),
                gk_count=team_comp.get('GK', 1),
                def_count=team_comp.get('DEF', 4),
                mid_count=team_comp.get('MID', 4),
                fwd_count=team_comp.get('FWD', 2),
                is_default=False
            )
            
            return config_id
        except ImportError:
            print("SQLite database module not available")
            return None
        except Exception as e:
            print(f"Error saving configuration to SQLite: {e}")
            return None
    
    def set_default_config(self):
        """
        Set default configuration.
        """
        self.config = {
            'name': 'Default Configuration',
            'description': 'Default configuration for Fantasy League Optimization',
            'problem': {
                'num_teams': 5,
                'budget': 1000,
                'team_composition': {
                    'GK': 1,
                    'DEF': 4,
                    'MID': 4,
                    'FWD': 2
                }
            },
            'algorithms': self._get_default_algorithms()
        }
    
    def _get_default_algorithms(self):
        """
        Get default algorithm configurations.
        
        Returns:
            Dictionary of algorithm configurations
        """
        return {
            'HillClimbing': {
                'max_iterations': 1000,
                'random_restarts': 0,
                'intensive_search': False
            },
            'SimulatedAnnealing': {
                'max_iterations': 1000,
                'initial_temperature': 100,
                'cooling_rate': 0.95,
                'min_temperature': 0.1
            },
            'GeneticAlgorithm': {
                'population_size': 50,
                'max_generations': 100,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1,
                'selection_method': 'tournament',
                'crossover_method': 'onePoint'
            },
            'IslandGA': {
                'num_islands': 4,
                'island_population_size': 20,
                'max_generations': 100,
                'migration_frequency': 10,
                'migration_rate': 0.2,
                'migration_topology': 'ring',
                'crossover_rate': 0.8,
                'mutation_rate': 0.1
            }
        }
    
    def get_algorithm_config(self, algorithm_name):
        """
        Get configuration for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary of algorithm configuration parameters
        """
        return self.config.get('algorithms', {}).get(algorithm_name, {})
    
    def set_algorithm_config(self, algorithm_name, params):
        """
        Set configuration for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            params: Dictionary of algorithm configuration parameters
        """
        if 'algorithms' not in self.config:
            self.config['algorithms'] = {}
        
        self.config['algorithms'][algorithm_name] = params
    
    def get_problem_config(self):
        """
        Get problem configuration.
        
        Returns:
            Dictionary of problem configuration parameters
        """
        return self.config.get('problem', {})
    
    def set_problem_config(self, problem_config):
        """
        Set problem configuration.
        
        Args:
            problem_config: Dictionary of problem configuration parameters
        """
        self.config['problem'] = problem_config
    
    def get_team_composition(self):
        """
        Get team composition.
        
        Returns:
            Dictionary of player counts by position
        """
        return self.config.get('problem', {}).get('team_composition', {})
    
    def set_team_composition(self, team_composition):
        """
        Set team composition.
        
        Args:
            team_composition: Dictionary of player counts by position
        """
        if 'problem' not in self.config:
            self.config['problem'] = {}
        
        self.config['problem']['team_composition'] = team_composition
    
    def validate(self, players_df=None):
        """
        Validate the configuration against available players.
        
        Args:
            players_df: DataFrame of players (optional)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get problem configuration
        problem = self.get_problem_config()
        num_teams = problem.get('num_teams', 0)
        team_comp = self.get_team_composition()
        
        # Check if team composition is defined
        if not team_comp:
            return False, "Team composition not defined"
        
        # Check if number of teams is defined
        if num_teams <= 0:
            return False, "Number of teams must be positive"
        
        # If no players DataFrame is provided, try to load from data source
        if players_df is None:
            if self.data_source == 'sqlite':
                try:
                    # Import database module
                    from db_schema import FantasyLeagueDB
                    
                    # Get players from database
                    db = FantasyLeagueDB()
                    players = db.get_all_players()
                    players_df = pd.DataFrame(players)
                except Exception as e:
                    return False, f"Error loading players from SQLite: {e}"
            else:
                try:
                    # Load from CSV
                    csv_path = '/home/ubuntu/upload/players.csv'
                    players_df = pd.read_csv(csv_path, sep=';')
                    
                    # Rename columns to match expected format
                    players_df = players_df.rename(columns={
                        'Name': 'name',
                        'Position': 'position',
                        'Skill': 'skill',
                        'Salary (€M)': 'cost'
                    })
                except Exception as e:
                    return False, f"Error loading players from CSV: {e}"
        
        # Count players by position
        position_counts = players_df['position'].value_counts().to_dict()
        
        # Check if we have enough players of each position
        errors = []
        
        for position, count in team_comp.items():
            required = num_teams * count
            available = position_counts.get(position, 0)
            
            if available < required:
                errors.append(f"Not enough {position} players: {available} available, {required} required")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, "Configuration is valid"
    
    def create_solution_factory(self, players_df=None):
        """
        Create a SolutionFactory instance for the current configuration.
        
        Args:
            players_df: DataFrame of players (optional)
            
        Returns:
            SolutionFactory instance
        """
        # Import solution module
        sys.path.append('/home/ubuntu/upload')
        from solution import SolutionFactory
        
        # If no players DataFrame is provided, try to load from data source
        if players_df is None:
            if self.data_source == 'sqlite':
                try:
                    # Import database module
                    from db_schema import FantasyLeagueDB
                    
                    # Get players from database
                    db = FantasyLeagueDB()
                    players = db.get_all_players()
                    players_df = pd.DataFrame(players)
                except Exception as e:
                    raise ValueError(f"Error loading players from SQLite: {e}")
            else:
                try:
                    # Load from CSV
                    csv_path = '/home/ubuntu/upload/players.csv'
                    players_df = pd.read_csv(csv_path, sep=';')
                    
                    # Rename columns to match expected format
                    players_df = players_df.rename(columns={
                        'Name': 'name',
                        'Position': 'position',
                        'Skill': 'skill',
                        'Salary (€M)': 'cost'
                    })
                except Exception as e:
                    raise ValueError(f"Error loading players from CSV: {e}")
        
        # Get problem configuration
        problem = self.get_problem_config()
        budget = problem.get('budget', 1000)
        team_comp = self.get_team_composition()
        
        # Create SolutionFactory
        return SolutionFactory(
            players_df,
            budget=budget,
            gk_count=team_comp.get('GK', 1),
            def_count=team_comp.get('DEF', 4),
            mid_count=team_comp.get('MID', 4),
            fwd_count=team_comp.get('FWD', 2)
        )


# Example usage
if __name__ == '__main__':
    # Create experiment configuration
    config = ExperimentConfig()
    
    # Print configuration
    print("Default Configuration:")
    print(json.dumps(config.config, indent=2))
    
    # Validate configuration
    is_valid, message = config.validate()
    print(f"Validation: {message}")
    
    # Save to JSON
    config.save_to_json('/home/ubuntu/fantasy_league_dashboard/default_config.json')
    
    # Save to SQLite if available
    if config.data_source == 'sqlite':
        config_id = config.save_to_sqlite()
        print(f"Saved to SQLite with ID: {config_id}")
