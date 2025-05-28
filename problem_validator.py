"""
Validation module for Fantasy League Optimization.

This module provides validation functions for checking problem feasibility
and handling unsolvable cases in both web and notebook workflows.
"""

import os
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

class ProblemValidator:
    """
    Problem validator for Fantasy League Optimization.
    Provides validation functions for checking problem feasibility
    and handling unsolvable cases.
    """
    
    @staticmethod
    def validate_with_csv(csv_path, num_teams, gk_count, def_count, mid_count, fwd_count):
        """
        Validate problem feasibility using CSV data.
        
        Args:
            csv_path: Path to CSV file
            num_teams: Number of teams
            gk_count: Number of goalkeepers per team
            def_count: Number of defenders per team
            mid_count: Number of midfielders per team
            fwd_count: Number of forwards per team
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Load players from CSV
            df = pd.read_csv(csv_path, sep=';')
            
            # Count players by position
            position_counts = df['Position'].value_counts().to_dict()
            
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
            
            # Check if at least one player of each position is required
            if gk_count == 0 and def_count == 0 and mid_count == 0 and fwd_count == 0:
                errors.append("At least one player position must have a non-zero count")
            
            # Check if team composition is valid
            total_players = gk_count + def_count + mid_count + fwd_count
            if total_players == 0:
                errors.append("Team composition must include at least one player")
            
            if errors:
                return False, "; ".join(errors)
            
            return True, "Configuration is valid"
        except Exception as e:
            return False, f"Error validating with CSV: {str(e)}"
    
    @staticmethod
    def validate_with_sqlite(config_id=None, num_teams=None, gk_count=None, def_count=None, mid_count=None, fwd_count=None):
        """
        Validate problem feasibility using SQLite data.
        
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
        try:
            # Import database module
            from db_schema import FantasyLeagueDB
            
            # Create database instance
            db = FantasyLeagueDB()
            
            # Validate configuration
            return db.validate_configuration(
                config_id=config_id,
                num_teams=num_teams,
                gk_count=gk_count,
                def_count=def_count,
                mid_count=mid_count,
                fwd_count=fwd_count
            )
        except ImportError:
            return False, "SQLite database module not available"
        except Exception as e:
            return False, f"Error validating with SQLite: {str(e)}"
    
    @staticmethod
    def validate_configuration(data_source='auto', config_id=None, num_teams=None, 
                              gk_count=None, def_count=None, mid_count=None, fwd_count=None,
                              csv_path=None):
        """
        Validate problem feasibility using the specified data source.
        
        Args:
            data_source: Data source type ('csv', 'sqlite', or 'auto')
            config_id: Configuration ID (optional, for SQLite)
            num_teams: Number of teams (optional)
            gk_count: Number of goalkeepers per team (optional)
            def_count: Number of defenders per team (optional)
            mid_count: Number of midfielders per team (optional)
            fwd_count: Number of forwards per team (optional)
            csv_path: Path to CSV file (optional, for CSV)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Determine data source if 'auto'
        if data_source == 'auto':
            # Check if SQLite database exists
            db_path = '/home/ubuntu/fantasy_league_dashboard/fantasy_league.db'
            if os.path.exists(db_path):
                data_source = 'sqlite'
            else:
                data_source = 'csv'
        
        # Validate based on data source
        if data_source == 'csv':
            # Use default CSV path if not provided
            if csv_path is None:
                csv_path = '/home/ubuntu/upload/players.csv'
            
            # Check if CSV file exists
            if not os.path.exists(csv_path):
                return False, f"CSV file not found: {csv_path}"
            
            return ProblemValidator.validate_with_csv(
                csv_path=csv_path,
                num_teams=num_teams,
                gk_count=gk_count,
                def_count=def_count,
                mid_count=mid_count,
                fwd_count=fwd_count
            )
        else:  # sqlite
            return ProblemValidator.validate_with_sqlite(
                config_id=config_id,
                num_teams=num_teams,
                gk_count=gk_count,
                def_count=def_count,
                mid_count=mid_count,
                fwd_count=fwd_count
            )
    
    @staticmethod
    def validate_team_composition(gk_count, def_count, mid_count, fwd_count):
        """
        Validate team composition.
        
        Args:
            gk_count: Number of goalkeepers per team
            def_count: Number of defenders per team
            mid_count: Number of midfielders per team
            fwd_count: Number of forwards per team
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Check if counts are non-negative
        if gk_count < 0 or def_count < 0 or mid_count < 0 or fwd_count < 0:
            errors.append("Player counts cannot be negative")
        
        # Check if at least one player of each position is required
        if gk_count == 0 and def_count == 0 and mid_count == 0 and fwd_count == 0:
            errors.append("At least one player position must have a non-zero count")
        
        # Check if team composition is valid
        total_players = gk_count + def_count + mid_count + fwd_count
        if total_players == 0:
            errors.append("Team composition must include at least one player")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, "Team composition is valid"
    
    @staticmethod
    def validate_budget(budget):
        """
        Validate budget.
        
        Args:
            budget: Budget per team
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if budget <= 0:
            return False, "Budget must be positive"
        
        return True, "Budget is valid"
    
    @staticmethod
    def validate_num_teams(num_teams):
        """
        Validate number of teams.
        
        Args:
            num_teams: Number of teams
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if num_teams <= 0:
            return False, "Number of teams must be positive"
        
        return True, "Number of teams is valid"
    
    @staticmethod
    def check_solution_feasibility(solution):
        """
        Check if a solution is feasible.
        
        Args:
            solution: Solution object
            
        Returns:
            Tuple of (is_feasible, error_message)
        """
        try:
            # Check if solution is valid
            if solution is None:
                return False, "Solution is None"
            
            # Check if solution has teams
            if not hasattr(solution, 'teams') or not solution.teams:
                return False, "Solution has no teams"
            
            # Check if all teams have players
            for i, team in enumerate(solution.teams):
                if not hasattr(team, 'players') or not team.players:
                    return False, f"Team {i+1} has no players"
            
            # Check if solution satisfies budget constraint
            for i, team in enumerate(solution.teams):
                if team.total_cost() > solution.budget:
                    return False, f"Team {i+1} exceeds budget: {team.total_cost()} > {solution.budget}"
            
            # Check if solution satisfies position constraints
            for i, team in enumerate(solution.teams):
                positions = [player.position for player in team.players]
                position_counts = {pos: positions.count(pos) for pos in set(positions)}
                
                if position_counts.get('GK', 0) != solution.gk_count:
                    return False, f"Team {i+1} has {position_counts.get('GK', 0)} goalkeepers, expected {solution.gk_count}"
                
                if position_counts.get('DEF', 0) != solution.def_count:
                    return False, f"Team {i+1} has {position_counts.get('DEF', 0)} defenders, expected {solution.def_count}"
                
                if position_counts.get('MID', 0) != solution.mid_count:
                    return False, f"Team {i+1} has {position_counts.get('MID', 0)} midfielders, expected {solution.mid_count}"
                
                if position_counts.get('FWD', 0) != solution.fwd_count:
                    return False, f"Team {i+1} has {position_counts.get('FWD', 0)} forwards, expected {solution.fwd_count}"
            
            return True, "Solution is feasible"
        except Exception as e:
            return False, f"Error checking solution feasibility: {str(e)}"


# Example usage
if __name__ == '__main__':
    # Validate with CSV
    is_valid, message = ProblemValidator.validate_with_csv(
        csv_path='/home/ubuntu/upload/players.csv',
        num_teams=5,
        gk_count=1,
        def_count=4,
        mid_count=4,
        fwd_count=2
    )
    print(f"CSV Validation: {message}")
    
    # Validate with SQLite
    is_valid, message = ProblemValidator.validate_with_sqlite(
        num_teams=5,
        gk_count=1,
        def_count=4,
        mid_count=4,
        fwd_count=2
    )
    print(f"SQLite Validation: {message}")
    
    # Validate team composition
    is_valid, message = ProblemValidator.validate_team_composition(
        gk_count=1,
        def_count=4,
        mid_count=4,
        fwd_count=2
    )
    print(f"Team Composition Validation: {message}")
    
    # Validate budget
    is_valid, message = ProblemValidator.validate_budget(budget=1000)
    print(f"Budget Validation: {message}")
    
    # Validate number of teams
    is_valid, message = ProblemValidator.validate_num_teams(num_teams=5)
    print(f"Number of Teams Validation: {message}")
