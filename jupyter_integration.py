"""
Jupyter notebook integration module for Fantasy League Optimization.

This module provides functions for integrating the optimizer with Jupyter notebooks,
allowing seamless switching between Excel/CSV and SQLite data sources.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append('/home/ubuntu/fantasy_league_dashboard')
sys.path.append('/home/ubuntu/upload')

# Import optimizer modules
from optimizer_adapter import OptimizerAdapter, run_optimization_from_notebook
from problem_validator import ProblemValidator

def load_players_from_csv(csv_path=None):
    """
    Load players from a CSV file.
    
    Args:
        csv_path: Path to CSV file (optional)
        
    Returns:
        DataFrame of players
    """
    # Use default CSV path if not provided
    if csv_path is None:
        csv_path = '/home/ubuntu/upload/players.csv'
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load players from CSV
    df = pd.read_csv(csv_path, sep=';')
    
    # Rename columns to match expected format if needed
    if 'Name' in df.columns and 'name' not in df.columns:
        df = df.rename(columns={
            'Name': 'name',
            'Position': 'position',
            'Skill': 'skill',
            'Salary (€M)': 'cost'
        })
    
    return df

def validate_configuration(num_teams, gk_count, def_count, mid_count, fwd_count, budget=1000, csv_path=None):
    """
    Validate a configuration against available players.
    
    Args:
        num_teams: Number of teams
        gk_count: Number of goalkeepers per team
        def_count: Number of defenders per team
        mid_count: Number of midfielders per team
        fwd_count: Number of forwards per team
        budget: Budget per team (optional)
        csv_path: Path to CSV file (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    return ProblemValidator.validate_configuration(
        data_source='csv',
        num_teams=num_teams,
        gk_count=gk_count,
        def_count=def_count,
        mid_count=mid_count,
        fwd_count=fwd_count,
        csv_path=csv_path
    )

def run_hill_climbing(num_teams=5, gk_count=1, def_count=4, mid_count=4, fwd_count=2, budget=1000, 
                     max_iterations=1000, random_restarts=0, intensive_search=False, csv_path=None):
    """
    Run Hill Climbing optimization from a Jupyter notebook.
    
    Args:
        num_teams: Number of teams (optional)
        gk_count: Number of goalkeepers per team (optional)
        def_count: Number of defenders per team (optional)
        mid_count: Number of midfielders per team (optional)
        fwd_count: Number of forwards per team (optional)
        budget: Budget per team (optional)
        max_iterations: Maximum number of iterations (optional)
        random_restarts: Number of random restarts (optional)
        intensive_search: Whether to use intensive search (optional)
        csv_path: Path to CSV file (optional)
        
    Returns:
        Best solution found
    """
    # Create optimizer adapter
    adapter = OptimizerAdapter(data_source='csv', csv_path=csv_path)
    
    # Set configuration
    adapter.config = {
        'num_teams': num_teams,
        'budget': budget,
        'gk_count': gk_count,
        'def_count': def_count,
        'mid_count': mid_count,
        'fwd_count': fwd_count
    }
    
    # Set parameters
    params = {
        'max_iterations': max_iterations,
        'random_restarts': random_restarts,
        'intensive_search': intensive_search
    }
    
    # Run optimization
    return adapter.run_hill_climbing(params)

def run_simulated_annealing(num_teams=5, gk_count=1, def_count=4, mid_count=4, fwd_count=2, budget=1000, 
                           max_iterations=1000, initial_temperature=100, cooling_rate=0.95, min_temperature=0.1, 
                           csv_path=None):
    """
    Run Simulated Annealing optimization from a Jupyter notebook.
    
    Args:
        num_teams: Number of teams (optional)
        gk_count: Number of goalkeepers per team (optional)
        def_count: Number of defenders per team (optional)
        mid_count: Number of midfielders per team (optional)
        fwd_count: Number of forwards per team (optional)
        budget: Budget per team (optional)
        max_iterations: Maximum number of iterations (optional)
        initial_temperature: Initial temperature (optional)
        cooling_rate: Cooling rate (optional)
        min_temperature: Minimum temperature (optional)
        csv_path: Path to CSV file (optional)
        
    Returns:
        Best solution found
    """
    # Create optimizer adapter
    adapter = OptimizerAdapter(data_source='csv', csv_path=csv_path)
    
    # Set configuration
    adapter.config = {
        'num_teams': num_teams,
        'budget': budget,
        'gk_count': gk_count,
        'def_count': def_count,
        'mid_count': mid_count,
        'fwd_count': fwd_count
    }
    
    # Set parameters
    params = {
        'max_iterations': max_iterations,
        'initial_temperature': initial_temperature,
        'cooling_rate': cooling_rate,
        'min_temperature': min_temperature
    }
    
    # Run optimization
    return adapter.run_simulated_annealing(params)

def run_genetic_algorithm(num_teams=5, gk_count=1, def_count=4, mid_count=4, fwd_count=2, budget=1000, 
                         population_size=50, max_generations=100, crossover_rate=0.8, mutation_rate=0.1, 
                         selection_method='tournament', crossover_method='onePoint', csv_path=None):
    """
    Run Genetic Algorithm optimization from a Jupyter notebook.
    
    Args:
        num_teams: Number of teams (optional)
        gk_count: Number of goalkeepers per team (optional)
        def_count: Number of defenders per team (optional)
        mid_count: Number of midfielders per team (optional)
        fwd_count: Number of forwards per team (optional)
        budget: Budget per team (optional)
        population_size: Population size (optional)
        max_generations: Maximum number of generations (optional)
        crossover_rate: Crossover rate (optional)
        mutation_rate: Mutation rate (optional)
        selection_method: Selection method (optional)
        crossover_method: Crossover method (optional)
        csv_path: Path to CSV file (optional)
        
    Returns:
        Best solution found
    """
    # Create optimizer adapter
    adapter = OptimizerAdapter(data_source='csv', csv_path=csv_path)
    
    # Set configuration
    adapter.config = {
        'num_teams': num_teams,
        'budget': budget,
        'gk_count': gk_count,
        'def_count': def_count,
        'mid_count': mid_count,
        'fwd_count': fwd_count
    }
    
    # Set parameters
    params = {
        'population_size': population_size,
        'max_generations': max_generations,
        'crossover_rate': crossover_rate,
        'mutation_rate': mutation_rate,
        'selection_method': selection_method,
        'crossover_method': crossover_method
    }
    
    # Run optimization
    return adapter.run_genetic_algorithm(params)

def run_island_ga(num_teams=5, gk_count=1, def_count=4, mid_count=4, fwd_count=2, budget=1000, 
                 num_islands=4, island_population_size=20, max_generations=100, migration_frequency=10, 
                 migration_rate=0.2, migration_topology='ring', crossover_rate=0.8, mutation_rate=0.1, 
                 csv_path=None):
    """
    Run Island Genetic Algorithm optimization from a Jupyter notebook.
    
    Args:
        num_teams: Number of teams (optional)
        gk_count: Number of goalkeepers per team (optional)
        def_count: Number of defenders per team (optional)
        mid_count: Number of midfielders per team (optional)
        fwd_count: Number of forwards per team (optional)
        budget: Budget per team (optional)
        num_islands: Number of islands (optional)
        island_population_size: Island population size (optional)
        max_generations: Maximum number of generations (optional)
        migration_frequency: Migration frequency (optional)
        migration_rate: Migration rate (optional)
        migration_topology: Migration topology (optional)
        crossover_rate: Crossover rate (optional)
        mutation_rate: Mutation rate (optional)
        csv_path: Path to CSV file (optional)
        
    Returns:
        Best solution found
    """
    # Create optimizer adapter
    adapter = OptimizerAdapter(data_source='csv', csv_path=csv_path)
    
    # Set configuration
    adapter.config = {
        'num_teams': num_teams,
        'budget': budget,
        'gk_count': gk_count,
        'def_count': def_count,
        'mid_count': mid_count,
        'fwd_count': fwd_count
    }
    
    # Set parameters
    params = {
        'num_islands': num_islands,
        'island_population_size': island_population_size,
        'max_generations': max_generations,
        'migration_frequency': migration_frequency,
        'migration_rate': migration_rate,
        'migration_topology': migration_topology,
        'crossover_rate': crossover_rate,
        'mutation_rate': mutation_rate
    }
    
    # Run optimization
    return adapter.run_island_ga(params)

def visualize_solution(solution):
    """
    Visualize a solution in a Jupyter notebook.
    
    Args:
        solution: Solution object
    """
    if solution is None:
        print("No solution to visualize")
        return
    
    # Print solution summary
    print(f"Solution Fitness: {solution.fitness()}")
    print(f"Total Cost: {solution.total_cost()}")
    print(f"Total Skill: {solution.total_skill()}")
    print(f"Skill Standard Deviation: {solution.skill_std_dev()}")
    
    # Create figure with subplots for each team
    num_teams = len(solution.teams)
    fig, axes = plt.subplots(1, num_teams, figsize=(5 * num_teams, 5))
    
    # Handle case with only one team
    if num_teams == 1:
        axes = [axes]
    
    # Plot each team
    for i, team in enumerate(solution.teams):
        ax = axes[i]
        
        # Create pitch
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.add_patch(plt.Rectangle((0, 0), 100, 100, fill=True, color='green', alpha=0.6))
        ax.add_patch(plt.Rectangle((0, 0), 16, 100, fill=True, color='green', alpha=0.3))
        ax.add_patch(plt.Rectangle((84, 0), 16, 100, fill=True, color='green', alpha=0.3))
        ax.add_patch(plt.Rectangle((0, 40), 16, 20, fill=False, color='white'))
        ax.add_patch(plt.Rectangle((84, 40), 16, 20, fill=False, color='white'))
        ax.add_patch(plt.Circle((8, 50), 8, fill=False, color='white'))
        ax.add_patch(plt.Circle((92, 50), 8, fill=False, color='white'))
        ax.add_patch(plt.Circle((50, 50), 8, fill=False, color='white'))
        
        # Plot players
        positions = {
            'GK': [(8, 50)],
            'DEF': [(25, 20), (25, 40), (25, 60), (25, 80)],
            'MID': [(50, 20), (50, 40), (50, 60), (50, 80)],
            'FWD': [(75, 30), (75, 50), (75, 70)]
        }
        
        # Count players by position
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for player in team.players:
            position_counts[player.position] += 1
        
        # Plot players by position
        for player in team.players:
            pos = player.position
            idx = position_counts[pos] - 1
            position_counts[pos] -= 1
            
            if idx < len(positions[pos]):
                x, y = positions[pos][idx]
                
                # Plot player
                if pos == 'GK':
                    color = 'yellow'
                elif pos == 'DEF':
                    color = 'cyan'
                elif pos == 'MID':
                    color = 'lime'
                else:  # FWD
                    color = 'magenta'
                
                ax.add_patch(plt.Circle((x, y), 5, fill=True, color=color))
                ax.text(x, y, str(player.skill), ha='center', va='center', fontsize=10, fontweight='bold')
                ax.text(x, y - 8, player.name.split(' ')[0], ha='center', va='center', fontsize=8)
        
        # Set title
        ax.set_title(f"Team {i+1}\nSkill: {team.total_skill()}, Cost: {team.total_cost()}€M")
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def compare_algorithms(num_teams=5, gk_count=1, def_count=4, mid_count=4, fwd_count=2, budget=1000, csv_path=None):
    """
    Compare different optimization algorithms on the same problem.
    
    Args:
        num_teams: Number of teams (optional)
        gk_count: Number of goalkeepers per team (optional)
        def_count: Number of defenders per team (optional)
        mid_count: Number of midfielders per team (optional)
        fwd_count: Number of forwards per team (optional)
        budget: Budget per team (optional)
        csv_path: Path to CSV file (optional)
        
    Returns:
        Dictionary of solutions by algorithm
    """
    # Validate configuration
    is_valid, message = validate_configuration(
        num_teams=num_teams,
        gk_count=gk_count,
        def_count=def_count,
        mid_count=mid_count,
        fwd_count=fwd_count,
        csv_path=csv_path
    )
    
    if not is_valid:
        raise ValueError(f"Invalid configuration: {message}")
    
    # Run algorithms
    results = {}
    
    print("Running Hill Climbing...")
    results['Hill Climbing'] = run_hill_climbing(
        num_teams=num_teams,
        gk_count=gk_count,
        def_count=def_count,
        mid_count=mid_count,
        fwd_count=fwd_count,
        budget=budget,
        max_iterations=500,
        csv_path=csv_path
    )
    
    print("Running Simulated Annealing...")
    results['Simulated Annealing'] = run_simulated_annealing(
        num_teams=num_teams,
        gk_count=gk_count,
        def_count=def_count,
        mid_count=mid_count,
        fwd_count=fwd_count,
        budget=budget,
        max_iterations=500,
        csv_path=csv_path
    )
    
    print("Running Genetic Algorithm...")
    results['Genetic Algorithm'] = run_genetic_algorithm(
        num_teams=num_teams,
        gk_count=gk_count,
        def_count=def_count,
        mid_count=mid_count,
        fwd_count=fwd_count,
        budget=budget,
        population_size=50,
        max_generations=50,
        csv_path=csv_path
    )
    
    print("Running Island Genetic Algorithm...")
    results['Island GA'] = run_island_ga(
        num_teams=num_teams,
        gk_count=gk_count,
        def_count=def_count,
        mid_count=mid_count,
        fwd_count=fwd_count,
        budget=budget,
        num_islands=4,
        island_population_size=20,
        max_generations=50,
        csv_path=csv_path
    )
    
    # Compare results
    print("\nResults Summary:")
    for algorithm, solution in results.items():
        print(f"{algorithm}: Fitness = {solution.fitness()}, Skill = {solution.total_skill()}, Cost = {solution.total_cost()}")
    
    # Plot comparison
    algorithms = list(results.keys())
    fitness_values = [results[algo].fitness() for algo in algorithms]
    skill_values = [results[algo].total_skill() for algo in algorithms]
    cost_values = [results[algo].total_cost() for algo in algorithms]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Fitness comparison
    axes[0].bar(algorithms, fitness_values, color='skyblue')
    axes[0].set_title('Fitness Comparison')
    axes[0].set_ylabel('Fitness (lower is better)')
    axes[0].set_xticklabels(algorithms, rotation=45, ha='right')
    
    # Skill comparison
    axes[1].bar(algorithms, skill_values, color='lightgreen')
    axes[1].set_title('Total Skill Comparison')
    axes[1].set_ylabel('Total Skill')
    axes[1].set_xticklabels(algorithms, rotation=45, ha='right')
    
    # Cost comparison
    axes[2].bar(algorithms, cost_values, color='salmon')
    axes[2].set_title('Total Cost Comparison')
    axes[2].set_ylabel('Total Cost (€M)')
    axes[2].set_xticklabels(algorithms, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return results


# Example usage in a Jupyter notebook:
"""
from jupyter_integration import load_players_from_csv, validate_configuration, run_hill_climbing, visualize_solution, compare_algorithms

# Load players
players_df = load_players_from_csv()
print(f"Loaded {len(players_df)} players")

# Validate configuration
is_valid, message = validate_configuration(num_teams=5, gk_count=1, def_count=4, mid_count=4, fwd_count=2)
print(f"Configuration validation: {message}")

# Run optimization
solution = run_hill_climbing(num_teams=5, gk_count=1, def_count=4, mid_count=4, fwd_count=2)

# Visualize solution
visualize_solution(solution)

# Compare algorithms
results = compare_algorithms(num_teams=5, gk_count=1, def_count=4, mid_count=4, fwd_count=2)
"""
