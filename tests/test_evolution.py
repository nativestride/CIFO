import unittest
from copy import deepcopy
import sys
import os
import random

# Adjust sys.path to allow importing from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evolution import hill_climbing_random_restart
from solution import LeagueSolution, LeagueHillClimbingSolution, InsufficientPlayersForPositionError

class TestEvolutionAlgorithms(unittest.TestCase):

    def setUp(self):
        """Setup common parameters for tests."""
        self.test_problem_params = {
            'num_teams': 2,
            'team_size': 3, # GK, DEF, MID
            'max_budget': 500.0, # Increased budget
            'position_requirements': {"GK": 1, "DEF": 1, "MID": 1}
        }

        # Create a pool of players larger than needed to allow for random selection issues
        # Total players needed: 2 teams * 3 players/team = 6
        # Ensure enough players for each required position for initial solution generation
        self.mock_players_data = []
        positions_needed_flat = []
        for pos, count in self.test_problem_params['position_requirements'].items():
            positions_needed_flat.extend([pos] * count * self.test_problem_params['num_teams']) # GK, GK, DEF, DEF, MID, MID

        # Add some extra players to avoid InsufficientPlayersForPositionError during random init
        # if the core 6 players are by chance too expensive or poorly distributed for a valid random solution.
        # Let's ensure at least 10 players of diverse positions.
        available_positions = ["GK", "DEF", "MID", "FWD"] 
        
        player_pool_size = 15 # Larger pool
        for i in range(player_pool_size):
            pos = random.choice(available_positions) # More diverse positions for the pool
            self.mock_players_data.append({
                "Name": f"Player{i}",
                "Position": pos,
                "Skill": random.randint(60, 90), # Skills that are not too uniform
                "Salary": random.uniform(10.0, 80.0) # Salaries that allow combinations within budget
            })
        
        # Check if initial setup can form at least one valid solution (for sanity)
        # This is not part of the HCRR test itself but helps debug test setup
        try:
            temp_sol_params = {
                'num_teams': self.test_problem_params['num_teams'],
                'team_size': self.test_problem_params['team_size'],
                'max_budget': self.test_problem_params['max_budget'],
                'players': deepcopy(self.mock_players_data),
                'position_requirements': self.test_problem_params['position_requirements'],
                'repr': None
            }
            LeagueHillClimbingSolution(**temp_sol_params).is_valid()
        except (InsufficientPlayersForPositionError, ValueError) as e:
            print(f"Warning: Test setUp might have issues creating a valid initial solution for testing: {e}")
            # This might indicate player data or problem params need adjustment for reliable testing.


    def test_hill_climbing_random_restart_integration(self):
        """
        Integration test for the hill_climbing_random_restart algorithm.
        """
        initial_solution_params_for_hcrr = {
            'num_teams': self.test_problem_params['num_teams'],
            'team_size': self.test_problem_params['team_size'],
            'max_budget': self.test_problem_params['max_budget'],
            'players': deepcopy(self.mock_players_data), # Pass a deepcopy
            'position_requirements': self.test_problem_params['position_requirements'],
            # 'repr': None is implied for random initial solution generation by HCRR
        }

        hcrr_params = {
            "initial_solution_params": initial_solution_params_for_hcrr,
            "solution_class_for_hc": LeagueHillClimbingSolution,
            "num_restarts": 3,  # Keep small for a unit/integration test
            "max_iterations_per_hc": 20, # Small number of iterations
            "max_no_improvement_per_hc": 10, # Small number for no improvement
            "verbose": False, # Keep test output clean
            "hc_specific_kwargs": {"verbose": False} # Verbosity for inner HC
        }

        # Run the algorithm
        best_solution, best_fitness, history = hill_climbing_random_restart(**hcrr_params)

        # Assert Results
        self.assertIsNotNone(best_solution, "HCRR should return a best solution.")
        self.assertIsInstance(best_solution, LeagueHillClimbingSolution, 
                              "Best solution should be an instance of the specified solution class.")
        
        # Critical: The returned solution must be valid.
        self.assertTrue(best_solution.is_valid(), 
                        f"The best solution returned by HCRR must be valid. Fitness: {best_solution.fitness()}, "
                        f"Violations: {best_solution.get_all_violations_details()}")

        self.assertIsInstance(best_fitness, float, "Best fitness should be a float.")
        self.assertNotEqual(best_fitness, float('inf'), 
                            "Best fitness should not be infinity if a solution is found.")

        self.assertIsInstance(history, list, "History should be a list.")
        self.assertEqual(len(history), hcrr_params["num_restarts"],
                         "History length should match the number of restarts.")

        # Check if best_fitness is consistent with history
        # (only if history contains valid fitness values, not inf for failed restarts)
        for restart_fitness in history:
            if restart_fitness != float('inf'): # Ignore failed restarts in this check
                self.assertLessEqual(best_fitness, restart_fitness,
                                     "Overall best fitness should be <= fitness of any single restart.")

if __name__ == '__main__':
    unittest.main()
