"""
API endpoints for the Fantasy League optimization website.

This module provides RESTful API endpoints for optimization visualization
using CSV/JSON data sources instead of a database.
"""

from flask import Flask, request, jsonify, Response, stream_with_context, Blueprint
import json
import threading
import queue
import time
import sys
import os

# Add project root to path
sys.path.append('/home/ubuntu/fantasy_league_dashboard')

# Import data source module
from data_source import DataSource

# Add upload directory to path for optimizer modules
sys.path.append('/home/ubuntu/upload')

# Import optimizer modules
from solution import Solution
from player_team_classes import Player, Team
from evolution_adapter import HillClimbing, SimulatedAnnealing
from genetic_algorithm_adapter import GeneticAlgorithm
from operators_adapter import SwapMutation, RandomRestartOperator
from crossover_operators import OnePointCrossover, TwoPointCrossover, UniformCrossover
from selection_operators import TournamentSelection, RouletteWheelSelection, RankingSelection
from mutation_operators import SwapMutation as GaSwapMutation
from config import ExecutionMode

# Create Flask app
app = Blueprint('api', __name__)

# Initialize data source
data_source = DataSource()

# Global variables for optimization state
optimization_running = False
optimization_thread = None
event_queue = queue.Queue()
current_solution = None
best_solution = None
current_fitness = float('inf')
best_fitness = float('inf')
iteration_count = 0
total_iterations = 0
player_movements = []

# Data API endpoints

@app.route('/players', methods=['GET'])
def get_players():
    """API endpoint to get all players."""
    players = data_source.get_players()
    
    return jsonify({
        'success': True,
        'players': players
    })

@app.route('/configuration', methods=['GET'])
def get_configuration():
    """API endpoint to get the configuration."""
    configuration = data_source.get_configuration()
    
    return jsonify({
        'success': True,
        'configuration': configuration
    })

@app.route('/configurations', methods=['GET'])
def get_configurations():
    """API endpoint to get all available configurations."""
    # Return predefined configurations as an array (not an object/dictionary)
    configurations = [
        {
            'id': 'HillClimbing',
            'name': 'Hill Climbing',
            'description': 'A simple local search algorithm that iteratively moves to better neighboring solutions.',
            'parameters': {
                'max_iterations': 1000,
                'max_no_improvement': 100,
                'random_restarts': 0
            }
        },
        {
            'id': 'SimulatedAnnealing',
            'name': 'Simulated Annealing',
            'description': 'A probabilistic technique that accepts worse solutions with decreasing probability over time.',
            'parameters': {
                'initial_temperature': 100.0,
                'cooling_rate': 0.95,
                'min_temperature': 0.1,
                'iterations_per_temp': 10
            }
        },
        {
            'id': 'GeneticAlgorithm',
            'name': 'Genetic Algorithm',
            'description': 'An evolutionary algorithm inspired by natural selection.',
            'parameters': {
                'population_size': 50,
                'max_generations': 100,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1,
                'selection_method': 'tournament',
                'crossover_method': 'onePoint'
            }
        }
    ]
    
    return jsonify({
        'success': True,
        'configurations': configurations
    })

@app.route('/configuration/validate', methods=['GET'])
def validate_configuration():
    """API endpoint to validate the configuration."""
    is_valid, message = data_source.validate_configuration()
    
    return jsonify({
        'success': True,
        'is_valid': is_valid,
        'message': message
    })

# Algorithm API endpoints

@app.route('/algorithms', methods=['GET'])
def get_algorithms():
    """API endpoint to get available algorithms and their parameters."""
    algorithms = {
        'HillClimbing': {
            'name': 'Hill Climbing',
            'parameters': {
                'max_iterations': {
                    'type': 'number',
                    'min': 10,
                    'max': 10000,
                    'default': 1000,
                    'step': 10,
                    'label': 'Max Iterations'
                },
                'random_restarts': {
                    'type': 'number',
                    'min': 0,
                    'max': 20,
                    'default': 0,
                    'step': 1,
                    'label': 'Random Restarts'
                },
                'intensive_search': {
                    'type': 'boolean',
                    'default': False,
                    'label': 'Intensive Search'
                }
            }
        },
        'SimulatedAnnealing': {
            'name': 'Simulated Annealing',
            'parameters': {
                'initial_temperature': {
                    'type': 'number',
                    'min': 1,
                    'max': 1000,
                    'default': 100,
                    'step': 1,
                    'label': 'Initial Temperature'
                },
                'cooling_rate': {
                    'type': 'number',
                    'min': 0.1,
                    'max': 0.99,
                    'default': 0.95,
                    'step': 0.01,
                    'label': 'Cooling Rate'
                },
                'min_temperature': {
                    'type': 'number',
                    'min': 0.1,
                    'max': 10,
                    'default': 1,
                    'step': 0.1,
                    'label': 'Min Temperature'
                },
                'iterations_per_temp': {
                    'type': 'number',
                    'min': 1,
                    'max': 100,
                    'default': 10,
                    'step': 1,
                    'label': 'Iterations per Temperature'
                }
            }
        },
        'GeneticAlgorithm': {
            'name': 'Genetic Algorithm',
            'parameters': {
                'population_size': {
                    'type': 'number',
                    'min': 10,
                    'max': 1000,
                    'default': 50,
                    'step': 10,
                    'label': 'Population Size'
                },
                'max_generations': {
                    'type': 'number',
                    'min': 10,
                    'max': 1000,
                    'default': 100,
                    'step': 10,
                    'label': 'Max Generations'
                },
                'crossover_rate': {
                    'type': 'number',
                    'min': 0.1,
                    'max': 1.0,
                    'default': 0.8,
                    'step': 0.1,
                    'label': 'Crossover Rate'
                },
                'mutation_rate': {
                    'type': 'number',
                    'min': 0.01,
                    'max': 0.5,
                    'default': 0.1,
                    'step': 0.01,
                    'label': 'Mutation Rate'
                },
                'selection_method': {
                    'type': 'select',
                    'options': ['tournament', 'roulette', 'ranking'],
                    'default': 'tournament',
                    'label': 'Selection Method'
                },
                'crossover_method': {
                    'type': 'select',
                    'options': ['onePoint', 'twoPoint', 'uniform'],
                    'default': 'onePoint',
                    'label': 'Crossover Method'
                }
            }
        }
    }
    
    return jsonify({
        'success': True,
        'algorithms': algorithms
    })

# Optimization API endpoints

@app.route('/optimize/start', methods=['POST'])
def start_optimization():
    """API endpoint to start optimization."""
    global optimization_running, optimization_thread, event_queue
    global current_solution, best_solution, current_fitness, best_fitness
    global iteration_count, total_iterations, player_movements
    
    # Check if optimization is already running
    if optimization_running:
        return jsonify({
            'success': False,
            'message': 'Optimization is already running'
        }), 400
    
    # Get request data
    data = request.json
    algorithm_name = data.get('algorithm', 'GeneticAlgorithm')
    parameters = data.get('parameters', {})
    
    # Validate configuration
    is_valid, message = data_source.validate_configuration()
    if not is_valid:
        return jsonify({
            'success': False,
            'message': f'Invalid configuration: {message}'
        }), 400
    
    # Reset optimization state
    event_queue = queue.Queue()
    current_solution = None
    best_solution = None
    current_fitness = float('inf')
    best_fitness = float('inf')
    iteration_count = 0
    player_movements = []
    
    # Get players and configuration
    players = data_source.get_player_objects()
    config = data_source.get_configuration()
    
    # Set total iterations based on algorithm
    if algorithm_name == 'HillClimbing':
        total_iterations = parameters.get('max_iterations', 1000)
    elif algorithm_name == 'SimulatedAnnealing':
        initial_temp = parameters.get('initial_temperature', 100)
        cooling_rate = parameters.get('cooling_rate', 0.95)
        min_temp = parameters.get('min_temperature', 1)
        iterations_per_temp = parameters.get('iterations_per_temp', 10)
        
        # Calculate total iterations
        temp = initial_temp
        total_iterations = 0
        while temp > min_temp:
            total_iterations += iterations_per_temp
            temp *= cooling_rate
    elif algorithm_name == 'GeneticAlgorithm':
        total_iterations = parameters.get('max_generations', 100)
    
    # Start optimization in a separate thread
    optimization_running = True
    optimization_thread = threading.Thread(
        target=run_optimization,
        args=(algorithm_name, parameters, players, config)
    )
    optimization_thread.daemon = True
    optimization_thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Started {algorithm_name} optimization'
    })

@app.route('/optimize/status', methods=['GET'])
def get_optimization_status():
    """API endpoint to get optimization status."""
    global optimization_running, current_solution, best_solution
    global current_fitness, best_fitness, iteration_count, total_iterations
    
    # Check if optimization is running
    if not optimization_running and current_solution is None:
        return jsonify({
            'success': True,
            'running': False,
            'message': 'No optimization has been run'
        })
    
    # Get current status
    status = {
        'running': optimization_running,
        'iteration': iteration_count,
        'total_iterations': total_iterations,
        'progress': iteration_count / total_iterations if total_iterations > 0 else 0,
        'current_fitness': current_fitness,
        'best_fitness': best_fitness
    }
    
    # Add solution if available
    if best_solution is not None:
        status['solution'] = solution_to_dict(best_solution)
    
    return jsonify({
        'success': True,
        'status': status
    })

@app.route('/optimize/events', methods=['GET'])
def get_optimization_events():
    """API endpoint to stream optimization events."""
    def generate():
        global event_queue, optimization_running
        
        # Send initial event
        yield f"data: {json.dumps({'type': 'connected'})}\n\n"
        
        # Stream events
        while True:
            try:
                # Get event from queue with timeout
                event = event_queue.get(timeout=1)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # Send heartbeat event
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            
            # Check if optimization is still running
            if not optimization_running and event_queue.empty():
                # Send completed event
                yield f"data: {json.dumps({'type': 'completed'})}\n\n"
                break
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )

@app.route('/optimize/stop', methods=['POST'])
def stop_optimization():
    """API endpoint to stop optimization."""
    global optimization_running
    
    # Check if optimization is running
    if not optimization_running:
        return jsonify({
            'success': False,
            'message': 'No optimization is running'
        }), 400
    
    # Stop optimization
    optimization_running = False
    
    return jsonify({
        'success': True,
        'message': 'Optimization stopped'
    })

@app.route('/optimize/reset', methods=['POST'])
def reset_optimization():
    """API endpoint to reset optimization state."""
    global optimization_running, current_solution, best_solution
    global current_fitness, best_fitness, iteration_count, total_iterations
    global player_movements
    
    # Check if optimization is running
    if optimization_running:
        return jsonify({
            'success': False,
            'message': 'Cannot reset while optimization is running'
        }), 400
    
    # Reset optimization state
    current_solution = None
    best_solution = None
    current_fitness = float('inf')
    best_fitness = float('inf')
    iteration_count = 0
    total_iterations = 0
    player_movements = []
    
    return jsonify({
        'success': True,
        'message': 'Optimization state reset'
    })

# Helper functions

def run_optimization(algorithm_name, parameters, players, config):
    """
    Run optimization in a separate thread.
    
    Args:
        algorithm_name: Name of the algorithm to run
        parameters: Algorithm parameters
        players: List of Player objects
        config: Configuration dictionary
    """
    global optimization_running, current_solution, best_solution
    global current_fitness, best_fitness, iteration_count, event_queue
    
    try:
        # Create initial solution with teams
        teams = create_initial_teams(players, config)
        initial_solution = Solution(teams=teams)
        
        # Create algorithm
        algorithm = create_algorithm(algorithm_name, parameters, initial_solution)
        
        # Run optimization
        for solution in algorithm:
            # Check if optimization should stop
            if not optimization_running:
                break
            
            # Update state
            current_solution = solution
            current_fitness = solution.fitness()
            iteration_count += 1
            
            # Update best solution
            if current_fitness < best_fitness:
                best_solution = solution
                best_fitness = current_fitness
                
                # Add player movement event
                if len(player_movements) > 0:
                    # Compare with previous best solution
                    movement = compare_solutions(player_movements[-1]['solution'], solution)
                    
                    if movement:
                        player_movements.append({
                            'iteration': iteration_count,
                            'solution': solution,
                            'movement': movement
                        })
                        
                        # Add event to queue
                        event_queue.put({
                            'type': 'player_movement',
                            'iteration': iteration_count,
                            'movement': movement
                        })
                else:
                    # First best solution
                    player_movements.append({
                        'iteration': iteration_count,
                        'solution': solution,
                        'movement': None
                    })
            
            # Add event to queue
            event_queue.put({
                'type': 'iteration',
                'iteration': iteration_count,
                'current_fitness': current_fitness,
                'best_fitness': best_fitness
            })
            
            # Sleep to avoid overwhelming the queue
            time.sleep(0.01)
        
        # Add final event
        event_queue.put({
            'type': 'completed',
            'iteration': iteration_count,
            'current_fitness': current_fitness,
            'best_fitness': best_fitness
        })
    
    except Exception as e:
        # Add error event
        event_queue.put({
            'type': 'error',
            'message': str(e)
        })
    
    finally:
        # Mark optimization as complete
        optimization_running = False

def create_initial_teams(players, config):
    """
    Create initial teams with players.
    
    Args:
        players: List of Player objects
        config: Configuration dictionary
    
    Returns:
        List of Team objects
    """
    # Get configuration values
    num_teams = config.get('num_teams', 5)
    team_size = config.get('team_size', 11)
    
    # Create teams
    teams = []
    for i in range(num_teams):
        team = Team(
            id=i,
            name=f'Team {i+1}',
            players=[]
        )
        teams.append(team)
    
    # Assign players to teams
    for i, player in enumerate(players):
        team_id = i % num_teams
        teams[team_id].players.append(player)
    
    return teams

def create_algorithm(algorithm_name, parameters, initial_solution):
    """
    Create algorithm instance.
    
    Args:
        algorithm_name: Name of the algorithm to create
        parameters: Algorithm parameters
        initial_solution: Initial solution
    
    Returns:
        Algorithm instance
    """
    if algorithm_name == 'HillClimbing':
        # Get parameters
        max_iterations = parameters.get('max_iterations', 1000)
        random_restarts = parameters.get('random_restarts', 0)
        intensive_search = parameters.get('intensive_search', False)
        
        # Create algorithm
        if random_restarts > 0:
            return HillClimbing(
                initial_solution=initial_solution,
                max_iterations=max_iterations,
                max_no_improvement=max_iterations // 10,
                random_restart_operator=RandomRestartOperator(num_restarts=random_restarts)
            )
        elif intensive_search:
            return HillClimbing(
                initial_solution=initial_solution,
                max_iterations=max_iterations * 2,
                max_no_improvement=max_iterations // 5
            )
        else:
            return HillClimbing(
                initial_solution=initial_solution,
                max_iterations=max_iterations,
                max_no_improvement=max_iterations // 10
            )
    
    elif algorithm_name == 'SimulatedAnnealing':
        # Get parameters
        initial_temperature = parameters.get('initial_temperature', 100)
        cooling_rate = parameters.get('cooling_rate', 0.95)
        min_temperature = parameters.get('min_temperature', 1)
        iterations_per_temp = parameters.get('iterations_per_temp', 10)
        
        # Create algorithm
        return SimulatedAnnealing(
            initial_solution=initial_solution,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            iterations_per_temp=iterations_per_temp
        )
    
    elif algorithm_name == 'GeneticAlgorithm':
        # Get parameters
        population_size = parameters.get('population_size', 50)
        max_generations = parameters.get('max_generations', 100)
        crossover_rate = parameters.get('crossover_rate', 0.8)
        mutation_rate = parameters.get('mutation_rate', 0.1)
        selection_method = parameters.get('selection_method', 'tournament')
        crossover_method = parameters.get('crossover_method', 'onePoint')
        
        # Create selection operator
        if selection_method == 'tournament':
            selection_operator = TournamentSelection(tournament_size=3)
        elif selection_method == 'roulette':
            selection_operator = RouletteWheelSelection()
        elif selection_method == 'ranking':
            selection_operator = RankingSelection()
        else:
            selection_operator = TournamentSelection(tournament_size=3)
        
        # Create crossover operator
        if crossover_method == 'onePoint':
            crossover_operator = OnePointCrossover()
        elif crossover_method == 'twoPoint':
            crossover_operator = TwoPointCrossover()
        elif crossover_method == 'uniform':
            crossover_operator = UniformCrossover()
        else:
            crossover_operator = OnePointCrossover()
        
        # Create mutation operator
        mutation_operator = GaSwapMutation()
        
        # Create algorithm
        return GeneticAlgorithm(
            initial_solution=initial_solution,
            population_size=population_size,
            max_generations=max_generations,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            crossover_rate=crossover_rate,
            mutation_operator=mutation_operator,
            mutation_rate=mutation_rate,
            execution_mode=ExecutionMode.SINGLE_PROCESSOR
        )
    
    else:
        raise ValueError(f'Unknown algorithm: {algorithm_name}')

def solution_to_dict(solution):
    """
    Convert solution to dictionary.
    
    Args:
        solution: Solution object
    
    Returns:
        Dictionary representation of the solution
    """
    # Get teams
    teams = []
    for team in solution.teams:
        # Get players
        players = []
        for player in team.players:
            players.append({
                'id': player.id,
                'name': player.name,
                'position': player.position,
                'skill': player.skill,
                'cost': player.cost
            })
        
        # Add team
        teams.append({
            'id': team.id,
            'name': team.name,
            'players': players
        })
    
    return {
        'teams': teams,
        'fitness': solution.fitness()
    }

def compare_solutions(old_solution, new_solution):
    """
    Compare two solutions and return player movements.
    
    Args:
        old_solution: Old solution
        new_solution: New solution
    
    Returns:
        List of player movements
    """
    # Get team assignments
    old_assignments = {}
    for team in old_solution.teams:
        for player in team.players:
            old_assignments[player.id] = team.id
    
    new_assignments = {}
    for team in new_solution.teams:
        for player in team.players:
            new_assignments[player.id] = team.id
    
    # Find movements
    movements = []
    for player_id, new_team_id in new_assignments.items():
        if player_id in old_assignments and old_assignments[player_id] != new_team_id:
            # Get player
            player = None
            for team in new_solution.teams:
                if team.id == new_team_id:
                    for p in team.players:
                        if p.id == player_id:
                            player = p
                            break
            
            if player:
                movements.append({
                    'player_id': player_id,
                    'player_name': player.name,
                    'player_position': player.position,
                    'old_team_id': old_assignments[player_id],
                    'new_team_id': new_team_id
                })
    
    return movements
