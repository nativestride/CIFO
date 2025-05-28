"""
Optimizer adapter module for Fantasy League Optimization.

This module provides adapters for running optimization algorithms with
either Excel/CSV or SQLite data sources.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('/home/ubuntu/fantasy_league_dashboard')
sys.path.append('/home/ubuntu/upload')

# Import optimizer modules
from solution import Solution
from solution_factory import SolutionFactory
from evolution_adapter import HillClimbing, SimulatedAnnealing
from genetic_algorithm_adapter import GeneticAlgorithm, IslandGeneticAlgorithm
from operators_adapter import SwapMutation, RandomRestartOperator
from crossover_operators_adapter import OnePointCrossover, TwoPointCrossover, UniformCrossover
from selection_operators_adapter import TournamentSelection, RouletteWheelSelection, RankingSelection
from mutation_operators_adapter import SwapMutation as GaSwapMutation
from config_adapter import ExecutionMode

# Import validation module
from problem_validator import ProblemValidator

class OptimizerAdapter:
    """
    Adapter for running optimization algorithms with either Excel/CSV or SQLite data sources.
    """
    
    def __init__(self, data_source='auto', csv_path=None, config_id=None):
        """
        Initialize the optimizer adapter.
        
        Args:
            data_source: Data source type ('csv', 'sqlite', or 'auto')
            csv_path: Path to CSV file (optional, for CSV)
            config_id: Configuration ID (optional, for SQLite)
        """
        self.data_source = data_source
        self.csv_path = csv_path
        self.config_id = config_id
        self.solution_factory = None
        self.players_df = None
        self.config = None
        
        # Determine data source if 'auto'
        if self.data_source == 'auto':
            # Check if SQLite database exists
            db_path = '/home/ubuntu/fantasy_league_dashboard/fantasy_league.db'
            if os.path.exists(db_path):
                self.data_source = 'sqlite'
            else:
                self.data_source = 'csv'
        
        # Initialize data source
        self._initialize_data_source()
    
    def _initialize_data_source(self):
        """
        Initialize the data source and load configuration.
        """
        if self.data_source == 'csv':
            # Use default CSV path if not provided
            if self.csv_path is None:
                self.csv_path = '/home/ubuntu/upload/players.csv'
            
            # Check if CSV file exists
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
            # Load players from CSV
            self.players_df = pd.read_csv(self.csv_path, sep=';')
            
            # Rename columns to match expected format
            self.players_df = self.players_df.rename(columns={
                'Name': 'name',
                'Position': 'position',
                'Skill': 'skill',
                'Salary (€M)': 'cost'
            })
            
            # Set default configuration
            self.config = {
                'num_teams': 5,
                'budget': 1000,
                'gk_count': 1,
                'def_count': 4,
                'mid_count': 4,
                'fwd_count': 2
            }
        else:  # sqlite
            # Import database module
            from db_schema import FantasyLeagueDB
            
            # Create database instance
            db = FantasyLeagueDB()
            
            # Get players from database
            players = db.get_all_players()
            self.players_df = pd.DataFrame(players)
            
            # Get configuration from database
            if self.config_id is None:
                # Get default configuration
                config = db.get_default_configuration()
                if config is None:
                    # Create default configuration if none exists
                    self.config_id = db.create_default_configuration()
                    config = db.get_configuration(self.config_id)
            else:
                # Get specific configuration
                config = db.get_configuration(self.config_id)
            
            # Set configuration
            self.config = {
                'num_teams': config['num_teams'],
                'budget': config['budget'],
                'gk_count': config['gk_count'],
                'def_count': config['def_count'],
                'mid_count': config['mid_count'],
                'fwd_count': config['fwd_count']
            }
    
    def create_solution_factory(self):
        """
        Create a SolutionFactory instance for the current data source and configuration.
        
        Returns:
            SolutionFactory instance
        """
        # Create SolutionFactory
        self.solution_factory = SolutionFactory(
            self.players_df,
            budget=self.config['budget'],
            gk_count=self.config['gk_count'],
            def_count=self.config['def_count'],
            mid_count=self.config['mid_count'],
            fwd_count=self.config['fwd_count']
        )
        
        return self.solution_factory
    
    def validate_configuration(self):
        """
        Validate the current configuration against available players.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return ProblemValidator.validate_configuration(
            data_source=self.data_source,
            config_id=self.config_id,
            num_teams=self.config['num_teams'],
            gk_count=self.config['gk_count'],
            def_count=self.config['def_count'],
            mid_count=self.config['mid_count'],
            fwd_count=self.config['fwd_count'],
            csv_path=self.csv_path
        )
    
    def run_hill_climbing(self, params=None, callback=None):
        """
        Run Hill Climbing optimization.
        
        Args:
            params: Dictionary of algorithm parameters (optional)
            callback: Callback function for iteration updates (optional)
            
        Returns:
            Best solution found
        """
        # Set default parameters if not provided
        if params is None:
            params = {
                'max_iterations': 1000,
                'random_restarts': 0,
                'intensive_search': False
            }
        
        # Validate configuration
        is_valid, message = self.validate_configuration()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {message}")
        
        # Create solution factory if not already created
        if self.solution_factory is None:
            self.create_solution_factory()
        
        # Create initial solution
        initial_solution = self.solution_factory.create_random_valid_solution()
        
        # Configure operators
        operators = [SwapMutation()]
        if params.get('random_restarts', 0) > 0:
            operators.append(RandomRestartOperator(params['random_restarts']))
        
        # Create Hill Climbing instance
        hill_climbing = HillClimbing(
            initial_solution=initial_solution,
            operators=operators,
            max_iterations=params.get('max_iterations', 1000),
            intensive_local_search=params.get('intensive_search', False),
            execution_mode=ExecutionMode.SINGLE_PROCESSOR
        )
        
        # Set random restarts if specified
        if params.get('random_restarts', 0) > 0:
            hill_climbing.random_restarts = params['random_restarts']
        
        # Register callback if provided
        if callback:
            hill_climbing.register_iteration_callback(callback)
        
        # Run optimization
        best_solution = hill_climbing.evolve()
        
        # Validate solution
        is_feasible, message = ProblemValidator.check_solution_feasibility(best_solution)
        if not is_feasible:
            raise ValueError(f"Infeasible solution: {message}")
        
        return best_solution
    
    def run_simulated_annealing(self, params=None, callback=None):
        """
        Run Simulated Annealing optimization.
        
        Args:
            params: Dictionary of algorithm parameters (optional)
            callback: Callback function for iteration updates (optional)
            
        Returns:
            Best solution found
        """
        # Set default parameters if not provided
        if params is None:
            params = {
                'max_iterations': 1000,
                'initial_temperature': 100,
                'cooling_rate': 0.95,
                'min_temperature': 0.1
            }
        
        # Validate configuration
        is_valid, message = self.validate_configuration()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {message}")
        
        # Create solution factory if not already created
        if self.solution_factory is None:
            self.create_solution_factory()
        
        # Create initial solution
        initial_solution = self.solution_factory.create_random_valid_solution()
        
        # Configure operators
        operators = [SwapMutation()]
        
        # Create Simulated Annealing instance
        simulated_annealing = SimulatedAnnealing(
            initial_solution=initial_solution,
            operators=operators,
            max_iterations=params.get('max_iterations', 1000),
            initial_temperature=params.get('initial_temperature', 100),
            cooling_rate=params.get('cooling_rate', 0.95),
            min_temperature=params.get('min_temperature', 0.1),
            execution_mode=ExecutionMode.SINGLE_PROCESSOR
        )
        
        # Register callback if provided
        if callback:
            simulated_annealing.register_iteration_callback(callback)
        
        # Run optimization
        best_solution = simulated_annealing.evolve()
        
        # Validate solution
        is_feasible, message = ProblemValidator.check_solution_feasibility(best_solution)
        if not is_feasible:
            raise ValueError(f"Infeasible solution: {message}")
        
        return best_solution
    
    def run_genetic_algorithm(self, params=None, callback=None):
        """
        Run Genetic Algorithm optimization.
        
        Args:
            params: Dictionary of algorithm parameters (optional)
            callback: Callback function for generation updates (optional)
            
        Returns:
            Best solution found
        """
        # Set default parameters if not provided
        if params is None:
            params = {
                'population_size': 50,
                'max_generations': 100,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1,
                'selection_method': 'tournament',
                'crossover_method': 'onePoint'
            }
        
        # Validate configuration
        is_valid, message = self.validate_configuration()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {message}")
        
        # Create solution factory if not already created
        if self.solution_factory is None:
            self.create_solution_factory()
        
        # Create initial population
        population_size = params.get('population_size', 50)
        initial_population = [self.solution_factory.create_random_valid_solution() 
                             for _ in range(population_size)]
        
        # Configure selection operator
        selection_method = params.get('selection_method', 'tournament')
        if selection_method == "tournament":
            selection_operator = TournamentSelection(tournament_size=3)
        elif selection_method == "roulette":
            selection_operator = RouletteWheelSelection()
        else:  # ranking
            selection_operator = RankingSelection()
        
        # Configure crossover operator
        crossover_method = params.get('crossover_method', 'onePoint')
        if crossover_method == "onePoint":
            crossover_operator = OnePointCrossover()
        elif crossover_method == "twoPoint":
            crossover_operator = TwoPointCrossover()
        else:  # uniform
            crossover_operator = UniformCrossover()
        
        # Configure mutation operator
        mutation_operator = GaSwapMutation()
        
        # Create Genetic Algorithm instance
        genetic_algorithm = GeneticAlgorithm(
            initial_population=initial_population,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            crossover_probability=params.get('crossover_rate', 0.8),
            mutation_probability=params.get('mutation_rate', 0.1),
            max_generations=params.get('max_generations', 100),
            execution_mode=ExecutionMode.SINGLE_PROCESSOR
        )
        
        # Register callback if provided
        if callback:
            genetic_algorithm.register_generation_callback(callback)
        
        # Run optimization
        best_solution = genetic_algorithm.evolve()
        
        # Validate solution
        is_feasible, message = ProblemValidator.check_solution_feasibility(best_solution)
        if not is_feasible:
            raise ValueError(f"Infeasible solution: {message}")
        
        return best_solution
    
    def run_island_ga(self, params=None, callback=None):
        """
        Run Island Genetic Algorithm optimization.
        
        Args:
            params: Dictionary of algorithm parameters (optional)
            callback: Callback function for generation updates (optional)
            
        Returns:
            Best solution found
        """
        # Set default parameters if not provided
        if params is None:
            params = {
                'num_islands': 4,
                'island_population_size': 20,
                'max_generations': 100,
                'migration_frequency': 10,
                'migration_rate': 0.2,
                'migration_topology': 'ring',
                'crossover_rate': 0.8,
                'mutation_rate': 0.1
            }
        
        # Validate configuration
        is_valid, message = self.validate_configuration()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {message}")
        
        # Create solution factory if not already created
        if self.solution_factory is None:
            self.create_solution_factory()
        
        # Create initial populations for each island
        num_islands = params.get('num_islands', 4)
        island_population_size = params.get('island_population_size', 20)
        island_populations = []
        for i in range(num_islands):
            island_population = [self.solution_factory.create_random_valid_solution() 
                               for _ in range(island_population_size)]
            island_populations.append(island_population)
        
        # Configure selection operator (using tournament selection for all islands)
        selection_operator = TournamentSelection(tournament_size=3)
        
        # Configure crossover operator (using one-point crossover for all islands)
        crossover_operator = OnePointCrossover()
        
        # Configure mutation operator
        mutation_operator = GaSwapMutation()
        
        # Map topology string to enum
        migration_topology = params.get('migration_topology', 'ring')
        if migration_topology == "ring":
            topology = IslandGeneticAlgorithm.MigrationTopology.RING
        elif migration_topology == "random_pair":
            topology = IslandGeneticAlgorithm.MigrationTopology.RANDOM_PAIR
        else:  # broadcast_best
            topology = IslandGeneticAlgorithm.MigrationTopology.BROADCAST_BEST
        
        # Create Island Genetic Algorithm instance
        island_ga = IslandGeneticAlgorithm(
            island_populations=island_populations,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            crossover_probability=params.get('crossover_rate', 0.8),
            mutation_probability=params.get('mutation_rate', 0.1),
            max_generations=params.get('max_generations', 100),
            migration_frequency=params.get('migration_frequency', 10),
            migration_rate=params.get('migration_rate', 0.2),
            migration_topology=topology,
            execution_mode=ExecutionMode.SINGLE_PROCESSOR
        )
        
        # Register callback if provided
        if callback:
            island_ga.register_generation_callback(callback)
        
        # Run optimization
        best_solution = island_ga.evolve()
        
        # Validate solution
        is_feasible, message = ProblemValidator.check_solution_feasibility(best_solution)
        if not is_feasible:
            raise ValueError(f"Infeasible solution: {message}")
        
        return best_solution
    
    def run_optimization(self, algorithm, params=None, callback=None):
        """
        Run optimization with the specified algorithm.
        
        Args:
            algorithm: Algorithm name ('HillClimbing', 'SimulatedAnnealing', 'GeneticAlgorithm', or 'IslandGA')
            params: Dictionary of algorithm parameters (optional)
            callback: Callback function for updates (optional)
            
        Returns:
            Best solution found
        """
        if algorithm == 'HillClimbing':
            return self.run_hill_climbing(params, callback)
        elif algorithm == 'SimulatedAnnealing':
            return self.run_simulated_annealing(params, callback)
        elif algorithm == 'GeneticAlgorithm':
            return self.run_genetic_algorithm(params, callback)
        elif algorithm == 'IslandGA':
            return self.run_island_ga(params, callback)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def save_solution(self, solution, algorithm, params):
        """
        Save solution to the data source.
        
        Args:
            solution: Solution object
            algorithm: Algorithm name
            params: Dictionary of algorithm parameters
            
        Returns:
            ID of the saved solution or None if not supported
        """
        if self.data_source == 'sqlite':
            # Import database module
            from db_schema import FantasyLeagueDB
            
            # Create database instance
            db = FantasyLeagueDB()
            
            # Convert solution to dictionary
            solution_dict = {
                'teams': [
                    {
                        'id': team.id,
                        'players': [
                            {
                                'id': player.id,
                                'name': player.name,
                                'position': player.position,
                                'skill': player.skill,
                                'cost': player.cost
                            }
                            for player in team.players
                        ]
                    }
                    for team in solution.teams
                ]
            }
            
            # Save optimization result
            return db.save_optimization_result(
                algorithm=algorithm,
                parameters=params,
                result_data=solution_dict,
                fitness=solution.fitness(),
                configuration_id=self.config_id
            )
        else:
            # For CSV, just print the result (no persistent storage)
            print(f"Solution saved (CSV mode):")
            print(f"  Algorithm: {algorithm}")
            print(f"  Parameters: {params}")
            print(f"  Fitness: {solution.fitness()}")
            return None


# Example usage for Jupyter notebooks
def run_optimization_from_notebook(algorithm='GeneticAlgorithm', csv_path=None, params=None):
    """
    Run optimization from a Jupyter notebook.
    
    Args:
        algorithm: Algorithm name ('HillClimbing', 'SimulatedAnnealing', 'GeneticAlgorithm', or 'IslandGA')
        csv_path: Path to CSV file (optional)
        params: Dictionary of algorithm parameters (optional)
        
    Returns:
        Best solution found
    """
    # Create optimizer adapter
    adapter = OptimizerAdapter(data_source='csv', csv_path=csv_path)
    
    # Run optimization
    return adapter.run_optimization(algorithm, params)
