import pytest
import random
from copy import deepcopy
import numpy as np # For fitness calculation/comparison in some cases

# Assuming solution.py and operators.py are in the parent directory or PYTHONPATH is set up
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solution import LeagueSolution
from operators import (
    mutate_swap_constrained,
    mutate_targeted_player_exchange,
    mutate_shuffle_within_team_constrained,
    crossover_one_point_prefer_valid,
    crossover_two_point,
    crossover_two_point_prefer_valid,
    selection_tournament_variable_k,
    genetic_algorithm_island_model, # Added
    generate_population, # Added for potential use in setup or direct testing
    # Individual operators for ga_params_per_island
    mutate_swap, 
    selection_tournament # Assuming this is the non-variable k version if needed
)
from unittest.mock import patch, MagicMock
from experiment_utils import safe_exp # Added

# --- Test Fixtures ---

@pytest.fixture
def operator_test_setup():
    """
    Provides player data, problem definition, and a helper to create solutions.
    Also provides references to some operator functions for convenience in tests.
    """
    num_teams = 2
    team_size = 3 # Requires GK, DEF, MID per team if using below reqs
    total_players_needed_for_one_solution = num_teams * team_size # 6 players

    # Player data should be sufficient for at least one solution.
    # For island model, total players needed by all individuals across all islands
    # can be large if each individual randomly samples players.
    # However, GA typically passes the *same* master player list to all individuals/islands.
    # The diversity comes from different team assignments (repr).
    # So, players_data needs to be at least total_players_needed_for_one_solution.
    # Let's make sure it's exactly that for simplicity in some tests, or slightly more for sampling.
    # The fixture already provides 8 players.
    players_data = [
        {"Name": "P0", "Position": "GK", "Skill": 70, "Salary": 100},
        {"Name": "P1", "Position": "DEF", "Skill": 75, "Salary": 120},
        {"Name": "P2", "Position": "MID", "Skill": 80, "Salary": 150},
        {"Name": "P3", "Position": "GK", "Skill": 72, "Salary": 110}, # Enough GKs for 2 teams
        {"Name": "P4", "Position": "DEF", "Skill": 77, "Salary": 130}, # Enough DEFs for 2 teams
        {"Name": "P5", "Position": "MID", "Skill": 82, "Salary": 160}, # Enough MIDs for 2 teams
        # Add a couple more to make the list slightly larger than strictly needed for 6 player slots,
        # allowing some flexibility if solution init samples or if problem_params change.
        {"Name": "P6", "Position": "FWD", "Skill": 85, "Salary": 180}, 
        {"Name": "P7", "Position": "FWD", "Skill": 88, "Salary": 190},
    ]

    problem_definition = {
        "num_teams": num_teams,
        "team_size": team_size,
        "max_budget": 1000.0, 
        "position_requirements": {"GK": 1, "DEF": 1, "MID": 1} # 1 GK, 1 DEF, 1 MID per team
    }
    # Check if total players needed by requirements matches team_size
    if sum(problem_definition["position_requirements"].values()) != team_size:
        raise ValueError("Test setup error: Sum of position_requirements must equal team_size.")
    
    # Check if players_data has enough players for one solution according to problem_definition
    if len(players_data) < total_players_needed_for_one_solution:
         raise ValueError(f"Test setup error: players_data has {len(players_data)} players, "
                          f"but {total_players_needed_for_one_solution} are needed for one solution.")


    def create_valid_solution(specific_players=None, specific_repr=None):
        # Use players from the fixture's players_data if not specified.
        # Ensure enough players are selected if specific_players is None and repr implies certain number.
        # For random repr, LeagueSolution uses the full list passed to it.
        current_players_list = deepcopy(specific_players) if specific_players else deepcopy(players_data)
        
        return LeagueSolution(
            repr=specific_repr, 
            players=current_players_list, # Pass the chosen list
            num_teams=problem_definition["num_teams"],
            team_size=problem_definition["team_size"],
            max_budget=problem_definition["max_budget"],
            position_requirements=problem_definition["position_requirements"]
        )
    
    return {
        "players_data": players_data, # Full list of 8 players
        "problem_definition": problem_definition, # For 2 teams of 3 = 6 players
        "create_valid_solution": create_valid_solution,
        "total_players_needed_for_one_solution": total_players_needed_for_one_solution,
        # Provide direct access to some operators for convenience in GA tests
        "selection_tournament": selection_tournament_variable_k, # Using variable_k version
        "crossover_one_point_prefer_valid": crossover_one_point_prefer_valid,
        "mutate_swap": mutate_swap, # Basic swap
        "mutate_swap_constrained": mutate_swap_constrained # Constrained swap
    }

# --- Tests for Mutation Operators ---

def test_mutate_swap_constrained(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    original_solution = create_sol() # Creates a random valid solution
    
    # Ensure the original solution is valid to start with
    if not original_solution.is_valid():
        # If random init sometimes fails budget (unlikely with high budget fixture), skip or adjust
        pytest.skip("Initial random solution for test_mutate_swap_constrained was not valid.")

    mutated_solution = mutate_swap_constrained(original_solution)

    assert isinstance(mutated_solution, LeagueSolution)
    assert mutated_solution is not original_solution # Ensure new instance

    # Check if a swap actually happened. It's possible no swappable positions exist or players in them.
    if mutated_solution.repr != original_solution.repr:
        diff_indices = [i for i, (o, m) in enumerate(zip(original_solution.repr, mutated_solution.repr)) if o != m]
        assert len(diff_indices) == 2, "Swap should involve two players"
        
        p1_idx, p2_idx = diff_indices[0], diff_indices[1]
        original_p1_pos = original_solution.players[p1_idx]["Position"]
        original_p2_pos = original_solution.players[p2_idx]["Position"]
        assert original_p1_pos == original_p2_pos, "Constrained swap should swap players of same position"
        
        # Check team assignments changed
        assert original_solution.repr[p1_idx] == mutated_solution.repr[p2_idx]
        assert original_solution.repr[p2_idx] == mutated_solution.repr[p1_idx]
    else:
        # If repr is same, it implies no valid swap was found.
        # This can happen if, e.g., all players of swappable positions are already on the same team.
        # Or if there are less than 2 players in any position.
        pass # Test passes, operator correctly returned an unchanged solution (or copy)

    # Test edge case: no swappable positions (e.g. all players of a pos on same team, or only 1 of each pos)
    # Requires a specific setup.
    pd = operator_test_setup["problem_definition"]
    players_no_swap = [ # 2 teams, 3 players each. GK, DEF, MID per team.
        {"Name": "GK1", "Position": "GK", "Skill": 70, "Salary": 100}, # Team 0
        {"Name": "DEF1", "Position": "DEF", "Skill": 70, "Salary": 100},# Team 0
        {"Name": "MID1", "Position": "MID", "Skill": 70, "Salary": 100},# Team 0
        {"Name": "GK2", "Position": "GK", "Skill": 70, "Salary": 100}, # Team 1
        {"Name": "DEF2", "Position": "DEF", "Skill": 70, "Salary": 100},# Team 1
        {"Name": "MID2", "Position": "MID", "Skill": 70, "Salary": 100},# Team 1
    ]
     # Assign GKs to team 0, DEFs to team 0, MIDs to team 0. No, this makes teams invalid.
     # We need a valid assignment where swaps are not possible for mutate_swap_constrained.
     # e.g. P0(GK,T0), P1(DEF,T0), P2(MID,T0), P3(GK,T1), P4(DEF,T1), P5(MID,T1)
     # Here, GK from T0 cannot swap with GK from T1 as they are the only GKs on their teams for the swap logic.
     # The logic for mutate_swap_constrained looks for *two* players of the same position *overall*
     # and then swaps their team assignments. So, as long as there are >=2 players of one position,
     # a swap should occur unless they are already on different teams and the repr is already optimal for the swap.
     # The only case it doesn't swap is if all players of a position are on the *same* team *after* a swap,
     # or if there's only one player of each required position type.
     # For this test, let's use players where only 1 of each position exists.
    players_singleton_pos = [
        {"Name": "GK1", "Position": "GK", "Skill": 70, "Salary": 100},
        {"Name": "DEF1", "Position": "DEF", "Skill": 70, "Salary": 100},
        {"Name": "MID1", "Position": "MID", "Skill": 70, "Salary": 100},
        # Need 3 more for 2 teams of 3, but let's make them unique positions not in reqs to force no swap
        {"Name": "FWD1", "Position": "FWD", "Skill": 70, "Salary": 100}, 
        {"Name": "SUB1", "Position": "SUB", "Skill": 70, "Salary": 100},
        {"Name": "WING1", "Position": "WING", "Skill": 70, "Salary": 100},
    ]
    # This setup is problematic as random_initial_representation would fail for these players.
    # Instead, let's test with a valid solution where all GKs are on one team, all DEFs on another, etc.
    # No, mutate_swap_constrained swaps players of the same position regardless of their team.
    # The only way it would not change repr is if no two players share a position.
    
    # Create a solution where only one player per position exists (among those considered for assignment)
    # This means the `swappable_positions` list inside the operator will be empty.
    one_of_each_pos_players = [
        {"Name": "GK_S", "Position": "GK", "Skill": 90, "Salary": 100},
        {"Name": "DEF_S", "Position": "DEF", "Skill": 90, "Salary": 100},
        {"Name": "MID_S", "Position": "MID", "Skill": 90, "Salary": 100},
        # Fill remaining slots with players of *other* positions to ensure the above are unique
        {"Name": "FWD1", "Position": "FWD", "Skill": 70, "Salary": 100}, 
        {"Name": "FWD2", "Position": "FWD", "Skill": 70, "Salary": 100},
        {"Name": "FWD3", "Position": "FWD", "Skill": 70, "Salary": 100},
    ]
    # This setup will make LeagueSolution's random_initial_representation fail if FWD isn't in reqs.
    # The fixture has GK, DEF, MID as required.
    # Let's use the create_sol helper with a specific list of players for the solution.
    sol_no_swap_possible = create_sol(specific_players=one_of_each_pos_players)
    # If sol_no_swap_possible uses random_initial_representation, it might fail if it can't meet quotas.
    # We need to ensure the solution is valid *before* mutation.
    # The simplest way: ensure players list for the solution has <2 of any position.
    players_for_no_swap_test = [
        operator_test_setup["players_data"][0], # GK
        operator_test_setup["players_data"][1], # DEF
        operator_test_setup["players_data"][2], # MID
        operator_test_setup["players_data"][6], # FWD
        {"Name": "P_Extra1", "Position": "GK", "Skill": 70, "Salary": 100}, # Second GK
        {"Name": "P_Extra2", "Position": "DEF", "Skill": 70, "Salary": 100}, # Second DEF
    ] # This list has 2 GK, 2 DEF, 1 MID, 1 FWD. For team size 3, 2 teams. Total 6 players.
      # This setup *will* allow swaps for GK and DEF.
      # To prevent swap: solution.players must have <2 of any position.
      # This implies the problem definition itself (team_size, num_teams) is very small
      # or player pool is very diverse.
      # This edge case is hard to reliably test without very specific solution construction.
      # The operator's internal check `if not swappable_positions:` covers it.
      # If we provide a solution where this is true, it should return an identical repr.


def test_mutate_shuffle_within_team_constrained_corrected(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    # Create a solution with players P0-P5.
    # P0(GK,S70), P1(DEF,S75), P2(MID,S80), P3(GK,S72), P4(DEF,S77), P5(MID,S82)
    # Repr for example: [0,0,0,1,1,1] -> T0: P0,P1,P2; T1: P3,P4,P5
    original_solution = create_sol(specific_repr=[0,0,0,1,1,1])
    
    if not original_solution.is_valid():
        pytest.skip("Initial solution for test_mutate_shuffle_within_team_constrained was not valid.")

    mutated_solution = mutate_shuffle_within_team_constrained(original_solution)

    assert isinstance(mutated_solution, LeagueSolution)
    assert mutated_solution is not original_solution

    if mutated_solution.repr != original_solution.repr:
        # Identify the two players whose teams changed
        swapped_indices = [i for i, (o_team, m_team) in enumerate(zip(original_solution.repr, mutated_solution.repr)) if o_team != m_team]
        assert len(swapped_indices) == 2, "Exactly two players should have their teams changed"

        p1_idx, p2_idx = swapped_indices[0], swapped_indices[1]
        
        # Assert they have the same position
        p1_pos = original_solution.players[p1_idx]["Position"]
        p2_pos = original_solution.players[p2_idx]["Position"]
        assert p1_pos == p2_pos, "Swapped players must be of the same position"

        # Assert they were originally on different teams
        original_p1_team = original_solution.repr[p1_idx]
        original_p2_team = original_solution.repr[p2_idx]
        assert original_p1_team != original_p2_team, "Swapped players must have been on different teams"

        # Assert their new teams are the swapped versions of their original teams
        assert mutated_solution.repr[p1_idx] == original_p2_team
        assert mutated_solution.repr[p2_idx] == original_p1_team
    else:
        # This implies no valid shuffle swap was found (e.g., all players of a position on one team,
        # or one team is empty, or no players of same position on different teams).
        pass


def test_mutate_targeted_player_exchange(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    players = operator_test_setup["players_data"][:operator_test_setup["total_players_needed"]]
    # Skills: P0(70), P1(75), P2(80), P3(72), P4(77), P5(82)
    # Make Team 0 high skill, Team 1 low skill for a clear test
    # Team 0: P2(MID,S80), P4(DEF,S77), P5(MID,S82) -> Avg ~79.6 (needs one GK)
    # Team 1: P0(GK,S70), P1(DEF,S75), P3(GK,S72) -> Avg ~72.3 (needs one MID)
    # Let's use players: P2(MID,80), P4(DEF,77), P5(MID,82) vs P0(GK,70), P1(DEF,75), P3(GK,72)
    # This setup is hard because positions must be met.
    # Fixture: {"GK": 1, "DEF": 1, "MID": 1}
    # Team0 (high skill): P5(MID,82), P4(DEF,77), P3(GK,72) -> Avg: (82+77+72)/3 = 77
    # Team1 (low skill): P2(MID,80), P1(DEF,75), P0(GK,70) -> Avg: (80+75+70)/3 = 75
    # This setup is not ideal, as P2 from low skill team is better than P5 from high skill team.
    # Let's use:
    # T0: P5(MID,S82), P4(DEF,S77), P0(GK,S70) -> Avg skill: (82+77+70)/3 = 76.33
    # T1: P2(MID,S80), P1(DEF,S75), P3(GK,S72) -> Avg skill: (80+75+72)/3 = 75.66
    # Targeted exchange should swap a MID from T0 (P5) with MID from T1 (P2) if possible.
    # Or DEF P4 with DEF P1. Or GK P0 with GK P3.
    
    # Using P0-P5 for the 6 slots.
    # P0(GK,70), P1(DEF,75), P2(MID,80), P3(GK,72), P4(DEF,77), P5(MID,82)
    # Team 0 (Low skill): P0 (GK, 70), P1 (DEF, 75), P2 (MID, 80) -> Avg skill = 75
    # Team 1 (High skill): P3 (GK, 72), P4 (DEF, 77), P5 (MID, 82) -> Avg skill = 77
    # Operator should try to swap e.g. P5(MID,T1) with P2(MID,T0)
    
    original_solution = create_sol(specific_repr=[0,0,0,1,1,1]) # P0,P1,P2 on T0; P3,P4,P5 on T1
    if not original_solution.is_valid():
        pytest.skip("Initial solution for mutate_targeted_player_exchange was not valid.")

    mutated_solution = mutate_targeted_player_exchange(original_solution)
    assert isinstance(mutated_solution, LeagueSolution)
    assert mutated_solution is not original_solution

    if mutated_solution.repr != original_solution.repr:
        swapped_indices = [i for i, (o, m) in enumerate(zip(original_solution.repr, mutated_solution.repr)) if o != m]
        assert len(swapped_indices) == 2
        
        p1_idx, p2_idx = swapped_indices[0], swapped_indices[1]
        p1_original_team = original_solution.repr[p1_idx]
        p2_original_team = original_solution.repr[p2_idx]

        # Check they were from different teams and swapped
        assert p1_original_team != p2_original_team
        assert mutated_solution.repr[p1_idx] == p2_original_team
        assert mutated_solution.repr[p2_idx] == p1_original_team
        
        # Check they have the same position
        assert original_solution.players[p1_idx]["Position"] == original_solution.players[p2_idx]["Position"]

        # Harder to assert specific teams (highest/lowest) without re-calculating avg skills here,
        # but the core swap logic is what's being tested.
    # else: pass, no suitable swap found based on operator's logic


# --- Tests for Crossover Operators ---

def test_crossover_one_point_prefer_valid(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    # P0-P5 for parent1, P0-P5 for parent2 (but solutions will shuffle them)
    parent1 = create_sol(specific_repr=[0,0,0,1,1,1]) # T0: P0,P1,P2 ; T1: P3,P4,P5
    parent2 = create_sol(specific_repr=[1,1,1,0,0,0]) # T0: P3,P4,P5 ; T1: P0,P1,P2

    if not parent1.is_valid() or not parent2.is_valid():
         pytest.skip("Parent solutions for crossover test were not valid.")

    child = crossover_one_point_prefer_valid(parent1, parent2, max_attempts=5)
    assert isinstance(child, LeagueSolution)
    assert child.is_valid(), "Child from prefer_valid crossover should be valid if parents are"

    # Check if repr is a mix. This depends on the cut point.
    # For a cut point `k`, child.repr = parent1.repr[:k] + parent2.repr[k:]
    # It's hard to assert a specific cut point was used, but we can check if it's not identical to parents
    # unless parents were identical or crossover produced one of the parents by chance.
    if parent1.repr != parent2.repr: # Only if parents are different
        # If child is valid, it might be identical to one parent if cut point is 0 or len(repr)
        # or if the "prefer_valid" part kicked in and returned a parent.
        # The operator itself tries to make a valid child. If it fails after N attempts, it returns fitter parent.
        # This test primarily ensures it returns a valid solution.
        pass


# --- Tests for Selection Operators ---

def test_selection_tournament_variable_k(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    population_size = 10
    population = []
    
    # Create population with varying fitness (using skill as a proxy for fitness variation here)
    # Lower skill sum for a team could lead to "better" std dev if other teams are similar,
    # or "worse" if it makes std dev higher. We just need some variation.
    # For simplicity, assign fixed repr and vary player quality.
    
    base_players = operator_test_setup["players_data"][:operator_test_setup["total_players_needed"]]
    
    for i in range(population_size):
        current_players = deepcopy(base_players)
        for p_idx, p_data in enumerate(current_players):
            # Slightly alter skill to create different fitness values
            current_players[p_idx]["Skill"] = p_data["Skill"] + i*2 - population_size 
        
        # Use a fixed valid repr for all solutions in this test population
        # T0: P0,P1,P2 ; T1: P3,P4,P5
        sol = create_sol(specific_players=current_players, specific_repr=[0,0,0,1,1,1])
        if sol.is_valid(): # Ensure solution is valid before adding
            population.append(sol)
        # If a solution isn't valid with this setup, the test setup for fitness variation might need adjustment.
    
    if len(population) < 2 : # Need at least a few for meaningful selection test
        pytest.skip("Could not create a sufficiently large valid population for selection test.")

    # Test with k smaller than population size
    selected_k_small = selection_tournament_variable_k(population, k=3)
    assert selected_k_small in population

    # Test with k equal to population size
    selected_k_equal = selection_tournament_variable_k(population, k=len(population))
    assert selected_k_equal in population
    
    # Test with k larger than population size
    selected_k_large = selection_tournament_variable_k(population, k=len(population) + 5)
    assert selected_k_large in population
    # The best one should be selected if k >= population_size in a deterministic tournament
    population.sort(key=lambda s: s.fitness())
    assert selected_k_large == population[0] # Smallest fitness is best

    # Test with k=1
    selected_k_one = selection_tournament_variable_k(population, k=1)
    assert selected_k_one in population # Should pick one random individual

    # Probabilistic nature is hard to assert strictly in unit test,
    # but we check it returns a member and handles k correctly.
    # Running many times and checking distribution is more of an integration/statistical test.

# --- Tests for Two-Point Crossover Operators ---

def test_crossover_two_point_valid_parents(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    # Repr length must be >= 3. Let's use total_players_needed = 6
    # P0-P5: [0,0,0,1,1,1] and [1,1,0,0,0,1]
    parent1_repr = [0,0,0,1,1,1]
    parent2_repr = [1,1,0,0,0,1]
    parent1 = create_sol(specific_repr=parent1_repr)
    parent2 = create_sol(specific_repr=parent2_repr)

    assert parent1.is_valid() and parent2.is_valid(), "Parents must be valid for this test"

    # Mock random.randint to control cut points k1 and k2
    # L = 6. k1 from [1, L-2=4], k2 from [k1+1, L-1=5]
    # Let k1 = 2, k2 = 4
    # Child = P1[:2] + P2[2:4] + P1[4:]
    #       = [0,0] + [0,0] + [1,1] = [0,0,0,0,1,1]
    expected_child_repr = [0,0,0,0,1,1]
    
    with patch('random.randint') as mock_randint:
        # First call for k1, second for k2
        mock_randint.side_effect = [2, 4] 
        child = crossover_two_point(parent1, parent2)

    assert isinstance(child, LeagueSolution)
    assert child.repr == expected_child_repr
    assert child.num_teams == parent1.num_teams
    assert child.team_size == parent1.team_size
    assert child.players is parent1.players # Should be a shared reference
    assert child.is_valid(), "Child from controlled two-point crossover should be valid if parents are and repr is constructed correctly"

def test_crossover_two_point_short_representation(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    players_data = operator_test_setup["players_data"]
    problem_def = operator_test_setup["problem_definition"]
    
    # Create a scenario with very few players (e.g., 2 players for a 2-team, 1-player-per-team setup)
    # This makes len(repr) = 2, which is too short for two_point_crossover
    short_players_data = players_data[:2]
    short_problem_def = problem_def.copy()
    short_problem_def["num_teams"] = 2
    short_problem_def["team_size"] = 1
    short_problem_def["position_requirements"] = {"GK": 1} # Assuming P0, P1 are GKs or compatible

    # Modify players to fit simple position req
    short_players_data[0]["Position"] = "GK"
    short_players_data[1]["Position"] = "GK"
    
    parent1 = LeagueSolution(repr=[0], players=[short_players_data[0]], num_teams=1, team_size=1, max_budget=100, position_requirements={"GK":1})
    parent2 = LeagueSolution(repr=[0], players=[short_players_data[1]], num_teams=1, team_size=1, max_budget=100, position_requirements={"GK":1})
    # Adjusting for a length 2 representation:
    # P0 (GK, S70), P1 (DEF, S75) -> problem_def team_size = 1, num_teams = 2
    # parent1_repr = [0], parent2_repr = [1] -> This is len 1.
    # Need len(repr) = 2 for the L < 3 check.
    # So, 2 players, 2 teams, 1 player/team. Player indices 0, 1.
    # repr could be [team_for_player_0, team_for_player_1]
    # Example: P0 is GK, P1 is GK.
    # parent1_repr = [0,1] (P0 on T0, P1 on T1)
    # parent2_repr = [1,0] (P0 on T1, P1 on T0)
    
    # Ensure players have different skills for fitness comparison
    p_fitter = deepcopy(short_players_data[0]) # P0
    p_less_fit = deepcopy(short_players_data[1]) # P1
    p_fitter["Skill"] = 100 # Higher skill = lower (better) team skill_std_dev if teams are uniform
    p_less_fit["Skill"] = 50

    # We need solutions where fitness can be reliably different.
    # Let parent1 be fitter.
    # For LeagueSolution, fitness is sum of team skill std devs.
    # If team_size=1, std dev is 0. So fitness will be 0 for both.
    # Let's use a slightly larger setup for more meaningful fitness.
    # 3 players, num_teams=1, team_size=3. repr len = 3.
    # This will pass L < 3 if k1, k2 are chosen appropriately.
    # The function's L < 3 check is strict.
    # If L=2, it should return fitter parent.
    
    players_for_len2 = [
        {"Name": "P0", "Position": "GK", "Skill": 100, "Salary": 100}, # Fitter base
        {"Name": "P1", "Position": "GK", "Skill": 50, "Salary": 100}, # Less fit base
    ]
    # Solutions assign these 2 players to 2 teams.
    parent1_sol = LeagueSolution([0,1], num_teams=2, team_size=1, max_budget=200, players=players_for_len2, position_requirements={"GK":1})
    parent2_sol = LeagueSolution([1,0], num_teams=2, team_size=1, max_budget=200, players=players_for_len2, position_requirements={"GK":1})
    
    # Manually set fitness values for clarity in test, assuming P1 is fitter
    # Mock fitness() method
    parent1_sol.fitness = MagicMock(return_value=10.0) # Fitter
    parent2_sol.fitness = MagicMock(return_value=20.0)

    child = crossover_two_point(parent1_sol, parent2_sol)
    assert child.repr == parent1_sol.repr # Should be deepcopy of parent1
    assert child.fitness() == parent1_sol.fitness()
    assert child is not parent1_sol # Ensure it's a copy

    # Test with parent2 being fitter
    parent1_sol.fitness = MagicMock(return_value=20.0)
    parent2_sol.fitness = MagicMock(return_value=10.0) # Fitter
    child = crossover_two_point(parent1_sol, parent2_sol)
    assert child.repr == parent2_sol.repr
    assert child.fitness() == parent2_sol.fitness()
    assert child is not parent2_sol

def test_crossover_two_point_identical_parents(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    parent_repr = [0,0,1,1,0,1] # Length 6
    parent1 = create_sol(specific_repr=parent_repr)
    parent2 = create_sol(specific_repr=parent_repr) # Identical parent

    assert parent1.is_valid(), "Parent must be valid for this test"
    
    # No need to mock randint, result should always be the same as parents
    child = crossover_two_point(parent1, parent2)
    assert isinstance(child, LeagueSolution)
    assert child.repr == parent_repr

def test_crossover_two_point_prefer_valid_produces_valid_child(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    parent1 = create_sol(specific_repr=[0,0,0,1,1,1])
    parent2 = create_sol(specific_repr=[1,1,0,0,0,1])

    # Mock crossover_two_point to return a specific child
    # This child will then have its is_valid method checked.
    mock_child_solution = create_sol(specific_repr=[0,0,0,0,1,1]) # Expected from a specific crossover
    mock_child_solution.is_valid = MagicMock(return_value=True)

    with patch('tests.test_operators.crossover_two_point', return_value=mock_child_solution) as mock_crossover_op:
        child = crossover_two_point_prefer_valid(parent1, parent2, max_attempts=3)
        mock_crossover_op.assert_called_once_with(parent1, parent2)
        mock_child_solution.is_valid.assert_called_once()
        assert child is mock_child_solution # Should be the one returned by the mocked op


def test_crossover_two_point_prefer_valid_produces_invalid_then_valid(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    parent1 = create_sol(specific_repr=[0,0,0,1,1,1])
    parent2 = create_sol(specific_repr=[1,1,0,0,0,1])

    invalid_child = create_sol(specific_repr=[1,1,1,1,1,1]) # Some invalid child
    invalid_child.is_valid = MagicMock(return_value=False)
    
    valid_child = create_sol(specific_repr=[0,0,1,1,0,0]) # Some valid child
    valid_child.is_valid = MagicMock(return_value=True)

    # Mock crossover_two_point to return invalid, then valid
    with patch('tests.test_operators.crossover_two_point') as mock_crossover_op:
        mock_crossover_op.side_effect = [invalid_child, valid_child, invalid_child] # Third one shouldn't be called
        
        child = crossover_two_point_prefer_valid(parent1, parent2, max_attempts=3)
        
        assert mock_crossover_op.call_count == 2
        invalid_child.is_valid.assert_called_once()
        valid_child.is_valid.assert_called_once()
        assert child is valid_child

def test_crossover_two_point_prefer_valid_always_invalid_returns_fitter_parent(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    parent1 = create_sol(specific_repr=[0,0,0,1,1,1]) # Assume fitter
    parent2 = create_sol(specific_repr=[1,1,0,0,0,1]) # Assume less fit

    parent1.fitness = MagicMock(return_value=10.0) # Fitter
    parent2.fitness = MagicMock(return_value=20.0)

    invalid_child = create_sol(specific_repr=[1,1,1,1,1,1])
    invalid_child.is_valid = MagicMock(return_value=False)

    with patch('tests.test_operators.crossover_two_point', return_value=invalid_child) as mock_crossover_op:
        returned_child = crossover_two_point_prefer_valid(parent1, parent2, max_attempts=3)
        
        assert mock_crossover_op.call_count == 3
        invalid_child.is_valid.call_count == 3 # Called for each attempt
        
        assert returned_child.repr == parent1.repr # Deepcopy of fitter parent
        assert returned_child.fitness() == parent1.fitness()
        assert returned_child is not parent1 # Ensure it's a copy

def test_crossover_two_point_prefer_valid_parents_differing_fitness(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    parent1 = create_sol(specific_repr=[0,0,0,1,1,1])
    parent2 = create_sol(specific_repr=[1,1,1,0,0,0])

    # Case 1: parent1 is fitter
    parent1.fitness = MagicMock(return_value=10.0)
    parent2.fitness = MagicMock(return_value=20.0)

    # Mock crossover_two_point to always produce an invalid child
    # The child's actual representation doesn't matter here, only its validity.
    invalid_child_mock = create_sol() # Create a valid solution instance just for mocking
    invalid_child_mock.is_valid = MagicMock(return_value=False)

    with patch('tests.test_operators.crossover_two_point', return_value=invalid_child_mock) as mock_basic_crossover:
        result_child1_fitter = crossover_two_point_prefer_valid(parent1, parent2, max_attempts=2)
        
        assert mock_basic_crossover.call_count == 2
        invalid_child_mock.is_valid.call_count == 2
        assert result_child1_fitter.repr == parent1.repr
        assert result_child1_fitter.fitness() == parent1.fitness()
        assert result_child1_fitter is not parent1 # Must be a deepcopy

    # Reset mocks for next part of test
    invalid_child_mock.is_valid.reset_mock()
    mock_basic_crossover.reset_mock(return_value=invalid_child_mock, side_effect=None) # Reset side_effect too

    # Case 2: parent2 is fitter
    parent1.fitness = MagicMock(return_value=20.0)
    parent2.fitness = MagicMock(return_value=10.0)

    with patch('tests.test_operators.crossover_two_point', return_value=invalid_child_mock) as mock_basic_crossover_2:
        result_child2_fitter = crossover_two_point_prefer_valid(parent1, parent2, max_attempts=2)

        assert mock_basic_crossover_2.call_count == 2
        invalid_child_mock.is_valid.call_count == 2 # is_valid called twice more
        assert result_child2_fitter.repr == parent2.repr
        assert result_child2_fitter.fitness() == parent2.fitness()
        assert result_child2_fitter is not parent2 # Must be a deepcopy


# --- General Operator Considerations ---

def test_operators_return_new_instances(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    sol1 = create_sol(specific_repr=[0,0,0,1,1,1])
    sol2 = create_sol(specific_repr=[1,1,1,0,0,0])

    if not sol1.is_valid() or not sol2.is_valid():
        pytest.skip("Initial solutions for new instance test were not valid.")

    # Mutation
    mut_s1 = mutate_swap_constrained(sol1)
    assert mut_s1 is not sol1
    
    mut_s2 = mutate_shuffle_within_team_constrained(sol1)
    assert mut_s2 is not sol1

    mut_s3 = mutate_targeted_player_exchange(sol1)
    assert mut_s3 is not sol1

    # Crossover
    cross_child = crossover_one_point_prefer_valid(sol1, sol2)
    assert cross_child is not sol1
    assert cross_child is not sol2

# Add more tests for other operators and edge cases as needed.
# Example: test what happens if a mutation operator is called with an empty solution.players list
# (though __init__ or random_initial_representation might prevent this state).
# Example: test crossover if parents have different numbers of teams/players (should ideally be caught by operator).

# Test that two_point_crossover is also included in the new instance check
def test_operators_return_new_instances_includes_two_point(operator_test_setup):
    create_sol = operator_test_setup["create_valid_solution"]
    sol1 = create_sol(specific_repr=[0,0,0,1,1,1]) # Len 6
    sol2 = create_sol(specific_repr=[1,1,1,0,0,0]) # Len 6

    if not sol1.is_valid() or not sol2.is_valid():
        pytest.skip("Initial solutions for new instance test were not valid.")
    
    # Mock random.randint for crossover_two_point to ensure it doesn't fallback to parent copy due to L<3
    # L=6. k1 from [1,4], k2 from [k1+1,5]. Let k1=1, k2=2
    with patch('random.randint') as mock_randint:
        mock_randint.side_effect = [1,2] # k1=1, k2=2
        cross_child_two_point = crossover_two_point(sol1, sol2)
        assert cross_child_two_point is not sol1
        assert cross_child_two_point is not sol2

    cross_child_two_point_prefer_valid = crossover_two_point_prefer_valid(sol1, sol2)
    # This might return a copy of a parent if all attempts make invalid children.
    # To ensure it's a new instance even in that case (deepcopied parent):
    if cross_child_two_point_prefer_valid.repr == sol1.repr:
        assert cross_child_two_point_prefer_valid is not sol1
    elif cross_child_two_point_prefer_valid.repr == sol2.repr:
        assert cross_child_two_point_prefer_valid is not sol2
    # Otherwise, it's a new child, so inherently a new instance.
    # A stronger check:
    assert cross_child_two_point_prefer_valid is not sol1
    assert cross_child_two_point_prefer_valid is not sol2


# --- Tests for Genetic Algorithm - Island Model ---

def test_genetic_algorithm_island_model_integration(operator_test_setup):
    players_data_fixture = operator_test_setup["players_data"]
    problem_params_fixture = operator_test_setup["problem_definition"]

    # Ensure players_data has enough for one solution (6 players for 2 teams of 3)
    # The fixture setup should guarantee this.
    # num_teams = 2, team_size = 3. Total players in a solution = 6.
    # Fixture players_data has 8 players.

    test_island_model_ga_params = {
        "num_islands": 2,
        "island_population_size": 8, # Adjusted to be small but reasonable
        "max_generations_total": 5,
        "migration_frequency": 2,
        "num_migrants": 1, # Send 1 migrant
        "migration_topology": "ring",
        "verbose": False,
        "ga_params_per_island": {
            "population_size": 8, # Match island_population_size
            "selection_operator": operator_test_setup["selection_tournament"],
            "selection_params": {"k": 2},
            "crossover_operator": operator_test_setup["crossover_one_point_prefer_valid"],
            "crossover_rate": 0.8,
            "mutation_operator": operator_test_setup["mutate_swap_constrained"], # Using constrained version
            "mutation_rate": 0.2,
            "elitism": True,
            "elitism_size": 1,
            "verbose": False,
            "safe_exp_func": safe_exp, # For Boltzmann, though tournament is used here
        }
    }

    # Run the algorithm
    # Need to use deepcopy for players_data if it's modified by GA,
    # but GA and its sub-functions (generate_population, LeagueSolution) are expected
    # to handle copying of players_data internally if they modify it for an individual.
    # The main players_data list passed to the GA function itself should be the master list.
    
    best_solution, best_fitness, history = genetic_algorithm_island_model(
        deepcopy(players_data_fixture), # Pass a copy to be safe if GA modifies it
        problem_params_fixture, 
        test_island_model_ga_params
    )

    # Assert Results
    assert best_solution is not None, "Island GA should return a best solution."
    assert isinstance(best_solution, LeagueSolution), "Best solution should be an instance of LeagueSolution."
    
    # It's crucial that the returned solution is valid.
    is_valid_check = best_solution.is_valid()
    if not is_valid_check:
        violations = best_solution.get_all_violations_details()
        pytest.fail(f"Best solution from Island GA is not valid. Violations: {violations}")
    
    assert isinstance(best_fitness, float), "Best fitness should be a float."
    assert best_fitness != float('inf'), "Best fitness should not be infinity if a solution is found."

    assert isinstance(history, list), "History should be a list."
    # History includes initial fitness + one entry per generation
    assert len(history) == test_island_model_ga_params["max_generations_total"] + 1, \
        f"History length expected {test_island_model_ga_params['max_generations_total'] + 1}, got {len(history)}"

    # The best fitness found should be the last recorded fitness in history
    assert best_fitness == history[-1], "Best fitness should match the last entry in history."

    # Check if the problem parameters in the solution match the input
    assert best_solution.num_teams == problem_params_fixture["num_teams"]
    assert best_solution.team_size == problem_params_fixture["team_size"]
    assert best_solution.max_budget == problem_params_fixture["max_budget"]
    # Position requirements are not directly stored in solution in the same dict format, but are used.

    # Check if the number of players assigned in the solution matches num_teams * team_size
    # This depends on how repr is structured. If repr is list of team_ids for each player in players_data:
    # For this problem, repr is a list where index is player_id and value is team_id.
    # The length of repr should be equal to the number of players involved in *one* solution.
    # This is determined by generate_population -> LeagueSolution.random_initial_representation,
    # which uses players up to problem_params_fixture['num_teams'] * problem_params_fixture['team_size'].
    # So, len(best_solution.repr) should correspond to this.
    # However, the current LeagueSolution structure takes the *full* players_data list
    # and assigns teams to players from that list. The number of players *assigned* to teams
    # is implicitly problem_params_fixture['num_teams'] * problem_params_fixture['team_size']
    # due to how random_initial_representation works (it picks this many players).
    # The length of best_solution.repr is len(players_data_fixture).
    # This aspect of solution representation vs. problem definition needs to be clear.
    # For now, we assume the solution structure is internally consistent.
    
    # A simple check for one of the migration topologies (e.g., ring)
    # This is harder to check deeply without more invasive mocking or specific event logging.
    # For now, the fact it runs and produces a valid output is the main integration check.
