"""
Data adapter module for Fantasy League Optimization.

This module provides a data adapter layer that allows the optimizer to work with
either CSV or SQLite data sources transparently.
"""

import os
import pandas as pd
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append('/home/ubuntu/fantasy_league_dashboard')

# Import database module
from db_schema import FantasyLeagueDB

class DataAdapter:
    """
    Data adapter for Fantasy League Optimization.
    Provides a unified interface for accessing player and configuration data
    from either CSV or SQLite data sources.
    """
    
    def __init__(self, data_source='sqlite', csv_path=None, db_path=None):
        """
        Initialize the data adapter.
        
        Args:
            data_source: Data source type ('csv' or 'sqlite')
            csv_path: Path to the CSV file (required if data_source is 'csv')
            db_path: Path to the SQLite database file (optional if data_source is 'sqlite')
        """
        self.data_source = data_source.lower()
        
        if self.data_source == 'csv':
            if csv_path is None:
                raise ValueError("CSV path is required for CSV data source")
            self.csv_path = csv_path
        elif self.data_source == 'sqlite':
            self.db = FantasyLeagueDB(db_path) if db_path else FantasyLeagueDB()
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
    
    def get_players(self):
        """
        Get all players from the data source.
        
        Returns:
            DataFrame containing player data
        """
        if self.data_source == 'csv':
            # Read from CSV
            df = pd.read_csv(self.csv_path, sep=';')
            
            # Rename columns to match the expected format
            df = df.rename(columns={
                'Name': 'name',
                'Position': 'position',
                'Skill': 'skill',
                'Salary (€M)': 'cost'
            })
            
            # Ensure ID column is named 'id'
            if 0 in df.columns:
                df = df.rename(columns={0: 'id'})
            
            return df
        else:
            # Get from SQLite
            players = self.db.get_all_players()
            return pd.DataFrame(players)
    
    def get_configuration(self, config_id=None):
        """
        Get a configuration from the data source.
        
        Args:
            config_id: Configuration ID (optional, only used for SQLite)
            
        Returns:
            Dictionary containing configuration data
        """
        if self.data_source == 'csv':
            # For CSV, return a default configuration
            return {
                'id': 1,
                'name': 'Default Configuration',
                'description': 'Default configuration for CSV data source',
                'num_teams': 4,
                'budget': 1000,
                'gk_count': 1,
                'def_count': 4,
                'mid_count': 4,
                'fwd_count': 2
            }
        else:
            # Get from SQLite
            if config_id is None:
                # Get default configuration
                config = self.db.get_default_configuration()
                if config is None:
                    # Create default configuration if none exists
                    config_id = self.db.create_default_configuration()
                    config = self.db.get_configuration(config_id)
            else:
                # Get specific configuration
                config = self.db.get_configuration(config_id)
            
            return config
    
    def validate_configuration(self, config_id=None, num_teams=None, gk_count=None, 
                              def_count=None, mid_count=None, fwd_count=None):
        """
        Validate if a configuration is feasible with the current player pool.
        
        Args:
            config_id: Configuration ID (optional)
            num_teams: Number of teams (optional, used if config_id not provided)
            gk_count: Number of goalkeepers per team (optional, used if config_id not provided)
            def_count: Number of defenders per team (optional, used if config_id not provided)
            mid_count: Number of midfielders per team (optional, used if config_id not provided)
            fwd_count: Number of forwards per team (optional, used if config_id not provided)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.data_source == 'csv':
            # For CSV, validate manually
            players_df = self.get_players()
            
            # If config_id is provided, get configuration
            if config_id is not None:
                config = self.get_configuration(config_id)
                num_teams = config['num_teams']
                gk_count = config['gk_count']
                def_count = config['def_count']
                mid_count = config['mid_count']
                fwd_count = config['fwd_count']
            
            # Count players by position
            position_counts = players_df['position'].value_counts().to_dict()
            
            # Check if we have enough players of each position
            required_gk = num_teams * gk_count
            required_def = num_teams * def_count
            required_mid = num_teams * mid_count
            required_fwd = num_teams * fwd_count
            
            errors = []
            
            if 'GK' not in position_counts or position_counts['GK'] < required_gk:
                available = position_counts.get('GK', 0)
                errors.append(f"Not enough goalkeepers: {available} available, {required_gk} required")
            
            if 'DEF' not in position_counts or position_counts['DEF'] < required_def:
                available = position_counts.get('DEF', 0)
                errors.append(f"Not enough defenders: {available} available, {required_def} required")
            
            if 'MID' not in position_counts or position_counts['MID'] < required_mid:
                available = position_counts.get('MID', 0)
                errors.append(f"Not enough midfielders: {available} available, {required_mid} required")
            
            if 'FWD' not in position_counts or position_counts['FWD'] < required_fwd:
                available = position_counts.get('FWD', 0)
                errors.append(f"Not enough forwards: {available} available, {required_fwd} required")
            
            if errors:
                return False, "; ".join(errors)
            
            return True, "Configuration is valid"
        else:
            # Use SQLite validation
            return self.db.validate_configuration(
                config_id, num_teams, gk_count, def_count, mid_count, fwd_count
            )
    
    def save_optimization_result(self, algorithm, parameters, result_data, fitness, configuration_id=None):
        """
        Save an optimization result.
        
        Args:
            algorithm: Algorithm name
            parameters: Algorithm parameters (dict)
            result_data: Result data (dict)
            fitness: Fitness value
            configuration_id: Configuration ID (optional)
            
        Returns:
            ID of the new optimization result or None if not supported
        """
        if self.data_source == 'csv':
            # For CSV, just print the result (no persistent storage)
            print(f"Optimization result (CSV mode):")
            print(f"  Algorithm: {algorithm}")
            print(f"  Parameters: {parameters}")
            print(f"  Fitness: {fitness}")
            return None
        else:
            # Save to SQLite
            return self.db.save_optimization_result(
                algorithm, parameters, result_data, fitness, configuration_id
            )
    
    def create_solution_factory(self, config_id=None):
        """
        Create a SolutionFactory instance for the current data source and configuration.
        
        Args:
            config_id: Configuration ID (optional)
            
        Returns:
            SolutionFactory instance
        """
        # Import here to avoid circular imports
        sys.path.append('/home/ubuntu/upload')
        from solution import SolutionFactory
        
        # Get players
        players_df = self.get_players()
        
        # Get configuration
        config = self.get_configuration(config_id)
        
        # Create SolutionFactory
        return SolutionFactory(
            players_df,
            budget=config['budget'],
            gk_count=config['gk_count'],
            def_count=config['def_count'],
            mid_count=config['mid_count'],
            fwd_count=config['fwd_count']
        )


# Example usage
if __name__ == '__main__':
    # Test CSV adapter
    csv_path = '/home/ubuntu/upload/players.csv'
    csv_adapter = DataAdapter(data_source='csv', csv_path=csv_path)
    
    print("CSV Adapter Test:")
    players_df = csv_adapter.get_players()
    print(f"Player count: {len(players_df)}")
    
    config = csv_adapter.get_configuration()
    print(f"Configuration: {config['name']}")
    
    is_valid, message = csv_adapter.validate_configuration(
        num_teams=4, gk_count=1, def_count=4, mid_count=4, fwd_count=2
    )
    print(f"Validation: {message}")
    
    # Test SQLite adapter
    sqlite_adapter = DataAdapter(data_source='sqlite')
    
    print("\nSQLite Adapter Test:")
    players_df = sqlite_adapter.get_players()
    print(f"Player count: {len(players_df)}")
    
    config = sqlite_adapter.get_configuration()
    print(f"Configuration: {config['name']}")
    
    is_valid, message = sqlite_adapter.validate_configuration(
        num_teams=4, gk_count=1, def_count=4, mid_count=4, fwd_count=2
    )
    print(f"Validation: {message}")
