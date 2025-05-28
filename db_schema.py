"""
SQLite database schema and CRUD operations for Fantasy League Optimization.

This module provides the database schema and CRUD operations for players and configurations,
while maintaining compatibility with the existing CSV-based workflow.
"""

import os
import sqlite3
import pandas as pd
import json
from contextlib import contextmanager

class FantasyLeagueDB:
    """
    SQLite database manager for Fantasy League Optimization.
    Provides CRUD operations for players and configurations.
    """
    
    def __init__(self, db_path='/home/ubuntu/fantasy_league_dashboard/fantasy_league.db'):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.initialize_db()
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            SQLite connection object
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
        finally:
            conn.close()
    
    def initialize_db(self):
        """
        Initialize the database schema if it doesn't exist.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create players table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                position TEXT NOT NULL,
                skill INTEGER NOT NULL,
                cost INTEGER NOT NULL,
                active BOOLEAN DEFAULT 1
            )
            ''')
            
            # Create configurations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS configurations (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                num_teams INTEGER NOT NULL,
                budget INTEGER NOT NULL,
                gk_count INTEGER NOT NULL,
                def_count INTEGER NOT NULL,
                mid_count INTEGER NOT NULL,
                fwd_count INTEGER NOT NULL,
                is_default BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create optimization_results table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_results (
                id INTEGER PRIMARY KEY,
                configuration_id INTEGER,
                algorithm TEXT NOT NULL,
                parameters TEXT NOT NULL,
                result_data TEXT,
                fitness REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (configuration_id) REFERENCES configurations (id)
            )
            ''')
            
            conn.commit()
    
    # Player CRUD operations
    
    def get_all_players(self):
        """
        Get all active players from the database.
        
        Returns:
            List of player dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM players WHERE active = 1')
            return [dict(row) for row in cursor.fetchall()]
    
    def get_player(self, player_id):
        """
        Get a player by ID.
        
        Args:
            player_id: Player ID
            
        Returns:
            Player dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM players WHERE id = ?', (player_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def add_player(self, name, position, skill, cost):
        """
        Add a new player to the database.
        
        Args:
            name: Player name
            position: Player position (GK, DEF, MID, FWD)
            skill: Player skill level
            cost: Player cost
            
        Returns:
            ID of the new player
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO players (name, position, skill, cost) VALUES (?, ?, ?, ?)',
                (name, position, skill, cost)
            )
            conn.commit()
            return cursor.lastrowid
    
    def update_player(self, player_id, name=None, position=None, skill=None, cost=None):
        """
        Update a player in the database.
        
        Args:
            player_id: Player ID
            name: New player name (optional)
            position: New player position (optional)
            skill: New player skill level (optional)
            cost: New player cost (optional)
            
        Returns:
            True if successful, False if player not found
        """
        # Get current player data
        player = self.get_player(player_id)
        if not player:
            return False
        
        # Update with new values or keep existing ones
        name = name if name is not None else player['name']
        position = position if position is not None else player['position']
        skill = skill if skill is not None else player['skill']
        cost = cost if cost is not None else player['cost']
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE players SET name = ?, position = ?, skill = ?, cost = ? WHERE id = ?',
                (name, position, skill, cost, player_id)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_player(self, player_id):
        """
        Soft delete a player (mark as inactive).
        
        Args:
            player_id: Player ID
            
        Returns:
            True if successful, False if player not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE players SET active = 0 WHERE id = ?', (player_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def hard_delete_player(self, player_id):
        """
        Hard delete a player from the database.
        
        Args:
            player_id: Player ID
            
        Returns:
            True if successful, False if player not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM players WHERE id = ?', (player_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    # Configuration CRUD operations
    
    def get_all_configurations(self):
        """
        Get all configurations from the database.
        
        Returns:
            List of configuration dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM configurations ORDER BY is_default DESC, name')
            return [dict(row) for row in cursor.fetchall()]
    
    def get_configuration(self, config_id):
        """
        Get a configuration by ID.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            Configuration dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM configurations WHERE id = ?', (config_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_default_configuration(self):
        """
        Get the default configuration.
        
        Returns:
            Default configuration dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM configurations WHERE is_default = 1 LIMIT 1')
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def add_configuration(self, name, num_teams, budget, gk_count, def_count, mid_count, fwd_count, 
                         description=None, is_default=False):
        """
        Add a new configuration to the database.
        
        Args:
            name: Configuration name
            num_teams: Number of teams
            budget: Budget per team
            gk_count: Number of goalkeepers per team
            def_count: Number of defenders per team
            mid_count: Number of midfielders per team
            fwd_count: Number of forwards per team
            description: Configuration description (optional)
            is_default: Whether this is the default configuration (optional)
            
        Returns:
            ID of the new configuration
        """
        # If this is the default configuration, unset any existing default
        if is_default:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE configurations SET is_default = 0')
                conn.commit()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO configurations 
                   (name, description, num_teams, budget, gk_count, def_count, mid_count, fwd_count, is_default) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (name, description, num_teams, budget, gk_count, def_count, mid_count, fwd_count, is_default)
            )
            conn.commit()
            return cursor.lastrowid
    
    def update_configuration(self, config_id, name=None, num_teams=None, budget=None, 
                            gk_count=None, def_count=None, mid_count=None, fwd_count=None,
                            description=None, is_default=None):
        """
        Update a configuration in the database.
        
        Args:
            config_id: Configuration ID
            name: New configuration name (optional)
            num_teams: New number of teams (optional)
            budget: New budget per team (optional)
            gk_count: New number of goalkeepers per team (optional)
            def_count: New number of defenders per team (optional)
            mid_count: New number of midfielders per team (optional)
            fwd_count: New number of forwards per team (optional)
            description: New configuration description (optional)
            is_default: Whether this is the default configuration (optional)
            
        Returns:
            True if successful, False if configuration not found
        """
        # Get current configuration data
        config = self.get_configuration(config_id)
        if not config:
            return False
        
        # Update with new values or keep existing ones
        name = name if name is not None else config['name']
        num_teams = num_teams if num_teams is not None else config['num_teams']
        budget = budget if budget is not None else config['budget']
        gk_count = gk_count if gk_count is not None else config['gk_count']
        def_count = def_count if def_count is not None else config['def_count']
        mid_count = mid_count if mid_count is not None else config['mid_count']
        fwd_count = fwd_count if fwd_count is not None else config['fwd_count']
        description = description if description is not None else config['description']
        is_default = is_default if is_default is not None else config['is_default']
        
        # If this is the default configuration, unset any existing default
        if is_default:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE configurations SET is_default = 0')
                conn.commit()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''UPDATE configurations 
                   SET name = ?, description = ?, num_teams = ?, budget = ?, 
                       gk_count = ?, def_count = ?, mid_count = ?, fwd_count = ?, is_default = ? 
                   WHERE id = ?''',
                (name, description, num_teams, budget, gk_count, def_count, mid_count, fwd_count, is_default, config_id)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_configuration(self, config_id):
        """
        Delete a configuration from the database.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            True if successful, False if configuration not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM configurations WHERE id = ?', (config_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    # Optimization results operations
    
    def save_optimization_result(self, algorithm, parameters, result_data, fitness, configuration_id=None):
        """
        Save an optimization result to the database.
        
        Args:
            algorithm: Algorithm name
            parameters: Algorithm parameters (dict)
            result_data: Result data (dict)
            fitness: Fitness value
            configuration_id: Configuration ID (optional)
            
        Returns:
            ID of the new optimization result
        """
        # Convert parameters and result_data to JSON
        parameters_json = json.dumps(parameters)
        result_data_json = json.dumps(result_data)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO optimization_results 
                   (configuration_id, algorithm, parameters, result_data, fitness) 
                   VALUES (?, ?, ?, ?, ?)''',
                (configuration_id, algorithm, parameters_json, result_data_json, fitness)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_optimization_results(self, limit=10):
        """
        Get recent optimization results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of optimization result dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT r.*, c.name as configuration_name 
                   FROM optimization_results r
                   LEFT JOIN configurations c ON r.configuration_id = c.id
                   ORDER BY r.created_at DESC LIMIT ?''',
                (limit,)
            )
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                # Parse JSON fields
                result['parameters'] = json.loads(result['parameters'])
                result['result_data'] = json.loads(result['result_data'])
                results.append(result)
            return results
    
    # Data import/export operations
    
    def import_players_from_csv(self, csv_path):
        """
        Import players from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Number of players imported
        """
        # Read CSV file
        df = pd.read_csv(csv_path, sep=';')
        
        # Insert players into database
        count = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                cursor.execute(
                    'INSERT INTO players (id, name, position, skill, cost) VALUES (?, ?, ?, ?, ?)',
                    (int(row[0]), row['Name'], row['Position'], int(row['Skill']), int(row['Salary (€M)']))
                )
                count += 1
            conn.commit()
        
        return count
    
    def export_players_to_csv(self, csv_path):
        """
        Export players to a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Number of players exported
        """
        # Get all active players
        players = self.get_all_players()
        
        # Create DataFrame
        df = pd.DataFrame(players)
        df = df.rename(columns={
            'name': 'Name',
            'position': 'Position',
            'skill': 'Skill',
            'cost': 'Salary (€M)'
        })
        
        # Reorder columns to match original CSV
        df = df[['id', 'Name', 'Position', 'Skill', 'Salary (€M)']]
        
        # Write to CSV
        df.to_csv(csv_path, sep=';', index=False)
        
        return len(players)
    
    # Validation operations
    
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
        # If config_id is provided, get configuration from database
        if config_id is not None:
            config = self.get_configuration(config_id)
            if not config:
                return False, "Configuration not found"
            
            num_teams = config['num_teams']
            gk_count = config['gk_count']
            def_count = config['def_count']
            mid_count = config['mid_count']
            fwd_count = config['fwd_count']
        
        # Get player counts by position
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT position, COUNT(*) as count FROM players WHERE active = 1 GROUP BY position')
            position_counts = {row['position']: row['count'] for row in cursor.fetchall()}
        
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
    
    # Utility methods
    
    def create_default_configuration(self):
        """
        Create a default configuration if none exists.
        
        Returns:
            ID of the default configuration
        """
        # Check if a default configuration already exists
        default_config = self.get_default_configuration()
        if default_config:
            return default_config['id']
        
        # Create default configuration
        return self.add_configuration(
            name="Default Configuration",
            description="Default configuration with 4 teams",
            num_teams=4,
            budget=1000,
            gk_count=1,
            def_count=4,
            mid_count=4,
            fwd_count=2,
            is_default=True
        )
    
    def initialize_with_sample_data(self):
        """
        Initialize the database with sample data.
        
        Returns:
            True if successful
        """
        # Check if players table is empty
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM players')
            player_count = cursor.fetchone()['count']
        
        # If players table is empty, import from CSV
        if player_count == 0:
            csv_path = '/home/ubuntu/upload/players.csv'
            if os.path.exists(csv_path):
                self.import_players_from_csv(csv_path)
        
        # Create default configuration if none exists
        self.create_default_configuration()
        
        # Create additional sample configurations
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM configurations')
            config_count = cursor.fetchone()['count']
        
        if config_count <= 1:
            self.add_configuration(
                name="Small League",
                description="Configuration with 2 teams",
                num_teams=2,
                budget=800,
                gk_count=1,
                def_count=3,
                mid_count=3,
                fwd_count=1,
                is_default=False
            )
            
            self.add_configuration(
                name="Large League",
                description="Configuration with 8 teams",
                num_teams=8,
                budget=1200,
                gk_count=1,
                def_count=4,
                mid_count=4,
                fwd_count=2,
                is_default=False
            )
        
        return True


# Example usage
if __name__ == '__main__':
    db = FantasyLeagueDB()
    db.initialize_with_sample_data()
    print("Database initialized with sample data")
    
    # Print player count
    players = db.get_all_players()
    print(f"Player count: {len(players)}")
    
    # Print configurations
    configs = db.get_all_configurations()
    print(f"Configuration count: {len(configs)}")
    for config in configs:
        print(f"  - {config['name']}: {config['num_teams']} teams")
