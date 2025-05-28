#!/usr/bin/env python
"""
Generate dummy metrics JSON files for testing the dashboard.
"""

import json
import os
import random
import numpy as np
import uuid

METRICS_DIR = "/home/ubuntu/fantasy_league_dashboard/metrics_data"
NUM_RUNS_PER_CONFIG = 3
CONFIGURATIONS = {
    "HillClimbing": ["HillClimbing_Std", "HillClimbing_Intensive", "HillClimbing_RandomRestart_Test"],
    "SimulatedAnnealing": ["SimulatedAnnealing_Std", "SimulatedAnnealing_Enhanced"],
    "GeneticAlgorithm": ["GA_Tournament_OnePoint", "GA_Ranking_Uniform", "GA_TwoPointCrossover_Test", "GA_Hybrid_Optimized"],
    "IslandGA": ["GA_Island_Model_Test"]
}

def generate_dummy_metrics(algorithm_name, config_name, run_id):
    """Generate a single dummy metrics dictionary."""
    base_fitness = random.uniform(50, 150)
    best_fitness = base_fitness - random.uniform(0, base_fitness * 0.3)
    runtime = random.uniform(5, 120)
    iterations = random.randint(100, 10000)
    evaluations = iterations * random.randint(10, 100)
    
    # Generate history data
    history_len = random.randint(50, 200)
    fitness_history = sorted([base_fitness - random.uniform(0, base_fitness * 0.3) * (i / history_len) 
                              for i in range(history_len)], reverse=True)
    fitness_history = [f + random.uniform(-2, 2) for f in fitness_history] # Add noise
    fitness_history[0] = base_fitness # Start with base
    fitness_history[-1] = best_fitness # End with best
    
    metrics = {
        "best_fitness": best_fitness,
        "runtime_seconds": runtime,
        "iterations": iterations,
        "function_evaluations": evaluations,
        "convergence_rate": random.uniform(0.01, 0.5),
        "solution_stability": random.uniform(0.1, 5.0),
        "best_fitness_history": fitness_history,
        "parameters": {"param1": random.choice(["A", "B"]), "param2": random.randint(1, 10)}
    }
    
    raw_data = {
        "fitness_history": fitness_history
    }
    
    # Add algorithm-specific metrics
    if algorithm_name == "HillClimbing":
        metrics.update({
            "neighbors_generated": evaluations * random.uniform(0.8, 1.2),
            "neighbors_evaluated": evaluations,
            "local_optima_count": random.randint(0, 5),
            "plateau_length": random.randint(0, 50) if random.random() < 0.3 else 0,
            "improvement_rate": random.uniform(0.01, 0.2),
        })
    elif algorithm_name == "SimulatedAnnealing":
        metrics.update({
            "acceptance_rate": random.uniform(0.1, 0.8),
            "worse_solutions_accepted": random.randint(0, int(iterations * 0.2)),
            "cooling_efficiency": random.uniform(0.5, 0.99),
            "temperature_impact": random.uniform(0.1, 1.0),
            "temperature_history": sorted([random.uniform(0.1, 100) for _ in range(history_len)], reverse=True)
        })
        raw_data["temperature_history"] = metrics["temperature_history"]
    elif algorithm_name in ["GeneticAlgorithm", "IslandGA"]:
        metrics.update({
            "generations": iterations, # Assuming 1 iter = 1 gen for simplicity
            "population_size": random.randint(50, 200),
            "crossover_success_rate": random.uniform(0.5, 0.95),
            "mutation_impact": random.uniform(0.01, 0.1),
            "selection_pressure": random.uniform(1.1, 2.0),
            "population_diversity": [random.uniform(0.1, 0.8) for _ in range(history_len)]
        })
        raw_data["population_diversity"] = metrics["population_diversity"]
        
    if algorithm_name == "IslandGA":
        num_islands = random.randint(2, 8)
        metrics.update({
            "num_islands": num_islands,
            "migration_events": random.randint(10, 100),
            "migration_impact": random.uniform(0.05, 0.3),
            "migration_success_rate": random.uniform(0.7, 1.0),
            "topology_efficiency": random.uniform(0.6, 0.95),
            "inter_island_diversity": [random.uniform(0.2, 0.9) for _ in range(history_len // 5)] # Less frequent
        })
        raw_data["island_best_fitness"] = { 
            f"island_{i}": sorted([random.uniform(best_fitness, base_fitness) for _ in range(history_len)], reverse=True) 
            for i in range(num_islands)
        }
        raw_data["inter_island_diversity"] = metrics["inter_island_diversity"]

    return {
        "algorithm_name": algorithm_name,
        "config_name": config_name,
        "run_id": run_id,
        "metrics": metrics,
        "raw_data": raw_data
    }

def main():
    if not os.path.exists(METRICS_DIR):
        os.makedirs(METRICS_DIR)
        print(f"Created directory: {METRICS_DIR}")
    
    total_files = 0
    for algo, configs in CONFIGURATIONS.items():
        for config in configs:
            for i in range(NUM_RUNS_PER_CONFIG):
                run_id = str(uuid.uuid4())
                dummy_data = generate_dummy_metrics(algo, config, run_id)
                
                filename = f"{config}_run_{i+1}_{run_id[:8]}.json"
                filepath = os.path.join(METRICS_DIR, filename)
                
                try:
                    with open(filepath, "w") as f:
                        json.dump(dummy_data, f, indent=4)
                    total_files += 1
                except Exception as e:
                    print(f"Error writing file {filepath}: {e}")
                    
    print(f"Generated {total_files} dummy metrics files in {METRICS_DIR}")

if __name__ == "__main__":
    main()
