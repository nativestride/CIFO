"""
Flask API for the Fantasy League optimization website.

This module provides API endpoints for retrieving player data,
running optimization algorithms, and streaming real-time updates.
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import time
import threading
import queue
import sys

# Add the upload directory to the path to import the Fantasy League modules
sys.path.append('/home/ubuntu/upload')

# Import Fantasy League modules
from solution import Solution, SolutionFactory
from evolution import HillClimbing, SimulatedAnnealing
from genetic_algorithms import GeneticAlgorithm
from ga_island_model import IslandGeneticAlgorithm
from operators import SwapMutation, RandomRestartOperator
from crossover_operators import OnePointCrossover, TwoPointCrossover, UniformCrossover
from selection_operators import TournamentSelection, RouletteWheelSelection, RankingSelection
from mutation_operators import SwapMutation as GaSwapMutation
from config import ExecutionMode

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

# Load player data
PLAYERS_FILE = '/home/ubuntu/upload/players.csv'
players_df = None

def load_players():
    """Load player data from CSV file."""
    global players_df
    players_df = pd.read_csv(PLAYERS_FILE, sep=';')
    return players_df

@app.route('/api/players', methods=['GET'])
def get_players():
    """API endpoint to get all players."""
    global players_df
    if players_df is None:
        load_players()
    
    # Convert DataFrame to list of dictionaries
    players_list = players_df.to_dict('records')
    
    return jsonify({
        'success': True,
        'players': players_list
    })

@app.route('/api/algorithms', methods=['GET'])
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
                'max_iterations': {
                    'type': 'number',
                    'min': 10,
                    'max': 10000,
                    'default': 1000,
                    'step': 10,
                    'label': 'Max Iterations'
                },
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
                    'min': 0.5,
                    'max': 0.99,
                    'default': 0.95,
                    'step': 0.01,
                    'label': 'Cooling Rate'
                },
                'min_temperature': {
                    'type': 'number',
                    'min': 0.001,
                    'max': 10,
                    'default': 0.1,
                    'step': 0.001,
                    'label': 'Minimum Temperature'
                }
            }
        },
        'GeneticAlgorithm': {
            'name': 'Genetic Algorithm',
            'parameters': {
                'population_size': {
                    'type': 'number',
                    'min': 10,
                    'max': 200,
                    'default': 50,
                    'step': 5,
                    'label': 'Population Size'
                },
                'max_generations': {
                    'type': 'number',
                    'min': 10,
                    'max': 1000,
                    'default': 100,
                    'step': 5,
                    'label': 'Max Generations'
                },
                'crossover_rate': {
                    'type': 'number',
                    'min': 0.1,
                    'max': 1.0,
                    'default': 0.8,
                    'step': 0.05,
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
                    'options': [
                        {'value': 'tournament', 'label': 'Tournament'},
                        {'value': 'roulette', 'label': 'Roulette Wheel'},
                        {'value': 'ranking', 'label': 'Ranking'}
                    ],
                    'default': 'tournament',
                    'label': 'Selection Method'
                },
                'crossover_method': {
                    'type': 'select',
                    'options': [
                        {'value': 'onePoint', 'label': 'One Point'},
                        {'value': 'twoPoint', 'label': 'Two Point'},
                        {'value': 'uniform', 'label': 'Uniform'}
                    ],
                    'default': 'onePoint',
                    'label': 'Crossover Method'
                }
            }
        },
        'IslandGA': {
            'name': 'Island Genetic Algorithm',
            'parameters': {
                'num_islands': {
                    'type': 'number',
                    'min': 2,
                    'max': 10,
                    'default': 4,
                    'step': 1,
                    'label': 'Number of Islands'
                },
                'island_population_size': {
                    'type': 'number',
                    'min': 5,
                    'max': 50,
                    'default': 20,
                    'step': 5,
                    'label': 'Island Population Size'
                },
                'max_generations': {
                    'type': 'number',
                    'min': 10,
                    'max': 1000,
                    'default': 100,
                    'step': 5,
                    'label': 'Max Generations'
                },
                'migration_frequency': {
                    'type': 'number',
                    'min': 1,
                    'max': 50,
                    'default': 10,
                    'step': 1,
                    'label': 'Migration Frequency'
                },
                'migration_rate': {
                    'type': 'number',
                    'min': 0.05,
                    'max': 0.5,
                    'default': 0.2,
                    'step': 0.05,
                    'label': 'Migration Rate'
                },
                'migration_topology': {
                    'type': 'select',
                    'options': [
                        {'value': 'ring', 'label': 'Ring'},
                        {'value': 'random_pair', 'label': 'Random Pair'},
                        {'value': 'broadcast_best', 'label': 'Broadcast Best'}
                    ],
                    'default': 'ring',
                    'label': 'Migration Topology'
                },
                'crossover_rate': {
                    'type': 'number',
                    'min': 0.1,
                    'max': 1.0,
                    'default': 0.8,
                    'step': 0.05,
                    'label': 'Crossover Rate'
                },
                'mutation_rate': {
                    'type': 'number',
                    'min': 0.01,
                    'max': 0.5,
                    'default': 0.1,
                    'step': 0.01,
                    'label': 'Mutation Rate'
                }
            }
        }
    }
    
    return jsonify({
        'success': True,
        'algorithms': algorithms
    })

@app.route('/api/problem_constraints', methods=['GET'])
def get_problem_constraints():
    """API endpoint to get problem constraints."""
    constraints = {
        'budget': {
            'type': 'number',
            'min': 500,
            'max': 2000,
            'default': 1000,
            'step': 50,
            'label': 'Budget (€M)'
        },
        'gk_count': {
            'type': 'number',
            'min': 1,
            'max': 3,
            'default': 1,
            'step': 1,
            'label': 'Goalkeepers'
        },
        'def_count': {
            'type': 'number',
            'min': 3,
            'max': 6,
            'default': 4,
            'step': 1,
            'label': 'Defenders'
        },
        'mid_count': {
            'type': 'number',
            'min': 3,
            'max': 6,
            'default': 4,
            'step': 1,
            'label': 'Midfielders'
        },
        'fwd_count': {
            'type': 'number',
            'min': 1,
            'max': 4,
            'default': 2,
            'step': 1,
            'label': 'Forwards'
        }
    }
    
    return jsonify({
        'success': True,
        'constraints': constraints
    })

def hill_climbing_callback(solution, iteration, best_solution, best_fitness):
    """Callback function for Hill Climbing algorithm."""
    global current_solution, best_solution, current_fitness, best_fitness, iteration_count
    
    current_solution = solution
    current_fitness = solution.fitness()
    
    if best_solution is not None:
        best_fitness = best_fitness
    
    iteration_count = iteration
    
    # Add event to queue
    event_queue.put({
        'type': 'iteration',
        'iteration': iteration,
        'current_fitness': current_fitness,
        'best_fitness': best_fitness
    })

def simulated_annealing_callback(solution, iteration, temperature, best_solution, best_fitness, accepted):
    """Callback function for Simulated Annealing algorithm."""
    global current_solution, best_solution, current_fitness, best_fitness, iteration_count
    
    current_solution = solution
    current_fitness = solution.fitness()
    
    if best_solution is not None:
        best_fitness = best_fitness
    
    iteration_count = iteration
    
    # Add event to queue
    event_queue.put({
        'type': 'iteration',
        'iteration': iteration,
        'current_fitness': current_fitness,
        'best_fitness': best_fitness,
        'temperature': temperature,
        'accepted': accepted
    })

def genetic_algorithm_callback(population, generation, best_individual, best_fitness):
    """Callback function for Genetic Algorithm."""
    global current_solution, best_solution, current_fitness, best_fitness, iteration_count
    
    current_solution = best_individual
    current_fitness = best_fitness
    
    iteration_count = generation
    
    # Add event to queue
    event_queue.put({
        'type': 'iteration',
        'iteration': generation,
        'current_fitness': current_fitness,
        'best_fitness': best_fitness,
        'population_size': len(population),
        'diversity': calculate_diversity(population)
    })

def island_ga_callback(islands, generation, best_individual, best_fitness, migration_occurred):
    """Callback function for Island Genetic Algorithm."""
    global current_solution, best_solution, current_fitness, best_fitness, iteration_count
    
    current_solution = best_individual
    current_fitness = best_fitness
    
    iteration_count = generation
    
    # Add event to queue
    event_queue.put({
        'type': 'iteration',
        'iteration': generation,
        'current_fitness': current_fitness,
        'best_fitness': best_fitness,
        'islands_count': len(islands),
        'migration_occurred': migration_occurred,
        'island_diversity': [calculate_diversity(island) for island in islands]
    })

def calculate_diversity(population):
    """Calculate diversity of a population."""
    if not population:
        return 0
    
    # Simple diversity measure: average pairwise distance
    total_distance = 0
    count = 0
    
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            # Calculate distance between solutions
            distance = population[i].distance(population[j])
            total_distance += distance
            count += 1
    
    return total_distance / max(1, count)

def run_hill_climbing(solution_factory, params):
    """Run Hill Climbing optimization."""
    global optimization_running, current_solution, best_solution, current_fitness, best_fitness, iteration_count, total_iterations, player_movements
    
    # Reset state
    current_solution = None
    best_solution = None
    current_fitness = float('inf')
    best_fitness = float('inf')
    iteration_count = 0
    player_movements = []
    
    # Get parameters
    max_iterations = params.get('max_iterations', 1000)
    random_restarts = params.get('random_restarts', 0)
    intensive_search = params.get('intensive_search', False)
    
    total_iterations = max_iterations
    
    # Create initial solution
    initial_solution = solution_factory.create_random_valid_solution()
    
    # Configure operators
    operators = [SwapMutation()]
    if random_restarts > 0:
        operators.append(RandomRestartOperator(random_restarts))
    
    # Create Hill Climbing instance
    hill_climbing = HillClimbing(
        initial_solution=initial_solution,
        operators=operators,
        max_iterations=max_iterations,
        intensive_local_search=intensive_search,
        execution_mode=ExecutionMode.SINGLE_PROCESSOR
    )
    
    # Register callback
    hill_climbing.register_iteration_callback(hill_climbing_callback)
    
    # Run optimization
    try:
        best_solution = hill_climbing.evolve()
        best_fitness = best_solution.fitness()
        
        # Add final event
        event_queue.put({
            'type': 'complete',
            'best_fitness': best_fitness
        })
    except Exception as e:
        event_queue.put({
            'type': 'error',
            'message': str(e)
        })
    finally:
        optimization_running = False

def run_simulated_annealing(solution_factory, params):
    """Run Simulated Annealing optimization."""
    global optimization_running, current_solution, best_solution, current_fitness, best_fitness, iteration_count, total_iterations, player_movements
    
    # Reset state
    current_solution = None
    best_solution = None
    current_fitness = float('inf')
    best_fitness = float('inf')
    iteration_count = 0
    player_movements = []
    
    # Get parameters
    max_iterations = params.get('max_iterations', 1000)
    initial_temperature = params.get('initial_temperature', 100)
    cooling_rate = params.get('cooling_rate', 0.95)
    min_temperature = params.get('min_temperature', 0.1)
    
    total_iterations = max_iterations
    
    # Create initial solution
    initial_solution = solution_factory.create_random_valid_solution()
    
    # Configure operators
    operators = [SwapMutation()]
    
    # Create Simulated Annealing instance
    simulated_annealing = SimulatedAnnealing(
        initial_solution=initial_solution,
        operators=operators,
        max_iterations=max_iterations,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        min_temperature=min_temperature,
        execution_mode=ExecutionMode.SINGLE_PROCESSOR
    )
    
    # Register callback
    simulated_annealing.register_iteration_callback(simulated_annealing_callback)
    
    # Run optimization
    try:
        best_solution = simulated_annealing.evolve()
        best_fitness = best_solution.fitness()
        
        # Add final event
        event_queue.put({
            'type': 'complete',
            'best_fitness': best_fitness
        })
    except Exception as e:
        event_queue.put({
            'type': 'error',
            'message': str(e)
        })
    finally:
        optimization_running = False

def run_genetic_algorithm(solution_factory, params):
    """Run Genetic Algorithm optimization."""
    global optimization_running, current_solution, best_solution, current_fitness, best_fitness, iteration_count, total_iterations, player_movements
    
    # Reset state
    current_solution = None
    best_solution = None
    current_fitness = float('inf')
    best_fitness = float('inf')
    iteration_count = 0
    player_movements = []
    
    # Get parameters
    population_size = params.get('population_size', 50)
    max_generations = params.get('max_generations', 100)
    crossover_rate = params.get('crossover_rate', 0.8)
    mutation_rate = params.get('mutation_rate', 0.1)
    selection_method = params.get('selection_method', 'tournament')
    crossover_method = params.get('crossover_method', 'onePoint')
    
    total_iterations = max_generations
    
    # Create initial population
    initial_population = [solution_factory.create_random_valid_solution() 
                         for _ in range(population_size)]
    
    # Configure selection operator
    if selection_method == "tournament":
        selection_operator = TournamentSelection(tournament_size=3)
    elif selection_method == "roulette":
        selection_operator = RouletteWheelSelection()
    else:  # ranking
        selection_operator = RankingSelection()
    
    # Configure crossover operator
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
        crossover_probability=crossover_rate,
        mutation_probability=mutation_rate,
        max_generations=max_generations,
        execution_mode=ExecutionMode.SINGLE_PROCESSOR
    )
    
    # Register callback
    genetic_algorithm.register_generation_callback(genetic_algorithm_callback)
    
    # Run optimization
    try:
        best_solution = genetic_algorithm.evolve()
        best_fitness = best_solution.fitness()
        
        # Add final event
        event_queue.put({
            'type': 'complete',
            'best_fitness': best_fitness
        })
    except Exception as e:
        event_queue.put({
            'type': 'error',
            'message': str(e)
        })
    finally:
        optimization_running = False

def run_island_ga(solution_factory, params):
    """Run Island Genetic Algorithm optimization."""
    global optimization_running, current_solution, best_solution, current_fitness, best_fitness, iteration_count, total_iterations, player_movements
    
    # Reset state
    current_solution = None
    best_solution = None
    current_fitness = float('inf')
    best_fitness = float('inf')
    iteration_count = 0
    player_movements = []
    
    # Get parameters
    num_islands = params.get('num_islands', 4)
    island_population_size = params.get('island_population_size', 20)
    max_generations = params.get('max_generations', 100)
    migration_frequency = params.get('migration_frequency', 10)
    migration_rate = params.get('migration_rate', 0.2)
    migration_topology = params.get('migration_topology', 'ring')
    crossover_rate = params.get('crossover_rate', 0.8)
    mutation_rate = params.get('mutation_rate', 0.1)
    
    total_iterations = max_generations
    
    # Create initial populations for each island
    island_populations = []
    for i in range(num_islands):
        island_population = [solution_factory.create_random_valid_solution() 
                           for _ in range(island_population_size)]
        island_populations.append(island_population)
    
    # Configure selection operator (using tournament selection for all islands)
    selection_operator = TournamentSelection(tournament_size=3)
    
    # Configure crossover operator (using one-point crossover for all islands)
    crossover_operator = OnePointCrossover()
    
    # Configure mutation operator
    mutation_operator = GaSwapMutation()
    
    # Map topology string to enum
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
        crossover_probability=crossover_rate,
        mutation_probability=mutation_rate,
        max_generations=max_generations,
        migration_frequency=migration_frequency,
        migration_rate=migration_rate,
        migration_topology=topology,
        execution_mode=ExecutionMode.SINGLE_PROCESSOR
    )
    
    # Register callback
    island_ga.register_generation_callback(island_ga_callback)
    
    # Run optimization
    try:
        best_solution = island_ga.evolve()
        best_fitness = best_solution.fitness()
        
        # Add final event
        event_queue.put({
            'type': 'complete',
            'best_fitness': best_fitness
        })
    except Exception as e:
        event_queue.put({
            'type': 'error',
            'message': str(e)
        })
    finally:
        optimization_running = False

@app.route('/api/optimize', methods=['POST'])
def start_optimization():
    """API endpoint to start optimization."""
    global optimization_running, optimization_thread, event_queue, total_iterations
    
    # Check if optimization is already running
    if optimization_running:
        return jsonify({
            'success': False,
            'message': 'Optimization is already running'
        })
    
    # Get request data
    data = request.json
    algorithm = data.get('algorithm')
    algorithm_params = data.get('algorithm_params', {})
    problem_params = data.get('problem_params', {})
    
    # Validate algorithm
    if algorithm not in ['HillClimbing', 'SimulatedAnnealing', 'GeneticAlgorithm', 'IslandGA']:
        return jsonify({
            'success': False,
            'message': 'Invalid algorithm'
        })
    
    # Create solution factory
    budget = problem_params.get('budget', 1000)
    gk_count = problem_params.get('gk_count', 1)
    def_count = problem_params.get('def_count', 4)
    mid_count = problem_params.get('mid_count', 4)
    fwd_count = problem_params.get('fwd_count', 2)
    
    # Load players if not already loaded
    global players_df
    if players_df is None:
        load_players()
    
    solution_factory = SolutionFactory(
        players_df,
        budget=budget,
        gk_count=gk_count,
        def_count=def_count,
        mid_count=mid_count,
        fwd_count=fwd_count
    )
    
    # Clear event queue
    while not event_queue.empty():
        event_queue.get()
    
    # Set optimization running flag
    optimization_running = True
    
    # Start optimization thread
    if algorithm == 'HillClimbing':
        optimization_thread = threading.Thread(
            target=run_hill_climbing,
            args=(solution_factory, algorithm_params)
        )
    elif algorithm == 'SimulatedAnnealing':
        optimization_thread = threading.Thread(
            target=run_simulated_annealing,
            args=(solution_factory, algorithm_params)
        )
    elif algorithm == 'GeneticAlgorithm':
        optimization_thread = threading.Thread(
            target=run_genetic_algorithm,
            args=(solution_factory, algorithm_params)
        )
    elif algorithm == 'IslandGA':
        optimization_thread = threading.Thread(
            target=run_island_ga,
            args=(solution_factory, algorithm_params)
        )
    
    optimization_thread.daemon = True
    optimization_thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Started {algorithm} optimization'
    })

@app.route('/api/stop', methods=['POST'])
def stop_optimization():
    """API endpoint to stop optimization."""
    global optimization_running
    
    optimization_running = False
    
    return jsonify({
        'success': True,
        'message': 'Optimization stopped'
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """API endpoint to get optimization status."""
    global optimization_running, current_solution, best_solution, current_fitness, best_fitness, iteration_count, total_iterations
    
    return jsonify({
        'success': True,
        'running': optimization_running,
        'current_fitness': current_fitness,
        'best_fitness': best_fitness,
        'iteration': iteration_count,
        'total_iterations': total_iterations
    })

@app.route('/api/events', methods=['GET'])
def stream_events():
    """API endpoint to stream optimization events."""
    @stream_with_context
    def generate():
        while True:
            # Check if optimization is running
            if not optimization_running and event_queue.empty():
                # Send a keep-alive message
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                time.sleep(1)
                continue
            
            # Get event from queue (non-blocking)
            try:
                event = event_queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # Send a keep-alive message
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                time.sleep(0.1)
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Load player data
    load_players()
    
    # Run app
    app.run(host='0.0.0.0', port=5000, threaded=True)
