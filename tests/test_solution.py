import pytest
import random
import numpy as np
from copy import deepcopy

# Assuming solution.py is in the parent directory or PYTHONPATH is set up
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solution import LeagueSolution, InsufficientPlayersForPositionError

# --- Test Fixtures ---

@pytest.fixture
def basic_problem_setup():
    """Provides a basic set of players and problem definition for tests."""
    num_teams = 2
    team_size = 3
    total_players_needed = num_teams * team_size # 2 * 3 = 6 players

    players_data = [
        {"Name": "Player1", "Position": "GK", "Skill": 70, "Salary": 100},
        {"Name": "Player2", "Position": "DEF", "Skill": 75, "Salary": 120},
        {"Name": "Player3", "Position": "MID", "Skill": 80, "Salary": 150},
        {"Name": "Player4", "Position": "FWD", "Skill": 85, "Salary": 180},
        {"Name": "Player5", "Position": "GK", "Skill": 72, "Salary": 110},
        {"Name": "Player6", "Position": "DEF", "Skill": 77, "Salary": 130},
        {"Name": "Player7", "Position": "MID", "Skill": 82, "Salary": 160}, # Extra MID for 'too_many'
        {"Name": "Player8", "Position": "DEF", "Skill": 78, "Salary": 140}, # Extra DEF for 'too_many'
    ] 
    # Initial players_data has 8 players, total_players_needed is 6.

    problem_definition = {
        "num_teams": num_teams,
        "team_size": team_size,
        "max_budget": 500.0, # GK(100) + DEF(120) + MID(150) = 370. FWD(180)
        "position_requirements": {"GK": 1, "DEF": 1, "MID": 1} # Sum = 3 (team_size)
    }
    # This setup requires 1 GK, 1 DEF, 1 MID per team.
    return {"players_data": players_data, "problem_definition": problem_definition}

# --- Tests for LeagueSolution.random_initial_representation ---

def test_random_initial_exact_player_count(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    # Take exactly the number of players needed
    players = basic_problem_setup["players_data"][:pd["num_teams"] * pd["team_size"]]
    
    sol = LeagueSolution(
        players=deepcopy(players),
        num_teams=pd["num_teams"],
        team_size=pd["team_size"],
        max_budget=pd["max_budget"],
        position_requirements=pd["position_requirements"]
    )
    
    assert sol.repr is not None
    assert len(sol.repr) == pd["num_teams"] * pd["team_size"]
    assert all(team_id != -1 for team_id in sol.repr), "Not all players were assigned"
    assert len(sol.players) == pd["num_teams"] * pd["team_size"]

def test_random_initial_too_few_players(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    # Provide fewer players than total needed
    players = basic_problem_setup["players_data"][:pd["num_teams"] * pd["team_size"] - 1]

    with pytest.raises(ValueError, match="Insufficient players"):
        LeagueSolution(
            players=deepcopy(players),
            num_teams=pd["num_teams"],
            team_size=pd["team_size"],
            max_budget=pd["max_budget"],
            position_requirements=pd["position_requirements"]
        )

def test_random_initial_too_many_players(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    # players_data fixture already has more players than needed (8 vs 6)
    players_fixture = basic_problem_setup["players_data"] 
    
    sol = LeagueSolution(
        players=deepcopy(players_fixture), # Pass the larger list
        num_teams=pd["num_teams"],
        team_size=pd["team_size"],
        max_budget=pd["max_budget"],
        position_requirements=pd["position_requirements"]
    )
    
    expected_player_count = pd["num_teams"] * pd["team_size"]
    assert len(sol.players) == expected_player_count, "self.players was not down-sampled"
    assert sol.repr is not None
    assert len(sol.repr) == expected_player_count
    assert all(team_id != -1 for team_id in sol.repr), "Not all players (from sampled list) were assigned"

def test_random_initial_insufficient_players_for_position(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    players_fixture = basic_problem_setup["players_data"]
    
    # Modify players_data so a position cannot be filled.
    # Requirements: {"GK": 1, "DEF": 1, "MID": 1} per team. Total 2 GKs, 2 DEFs, 2 MIDs needed.
    # Original fixture has: GK: P1, P5; DEF: P2, P6, P8; MID: P3, P7; FWD: P4
    # Let's remove all GKs from the first 6 players that would be chosen if exact.
    players_modified = [p for p in players_fixture[:6] if p["Position"] != "GK"]
    # Add some other players to make up the count to 6, but no GKs
    players_modified.extend([
        {"Name": "NoGK1", "Position": "DEF", "Skill": 70, "Salary": 100},
        {"Name": "NoGK2", "Position": "MID", "Skill": 70, "Salary": 100} 
    ]) # players_modified will have 6 players, but 0 GKs.
    
    with pytest.raises(InsufficientPlayersForPositionError, match="Insufficient players of position 'GK'"):
        LeagueSolution(
            players=deepcopy(players_modified), # Use the GK-starved list
            num_teams=pd["num_teams"],
            team_size=pd["team_size"],
            max_budget=pd["max_budget"],
            position_requirements=pd["position_requirements"]
        )

def test_random_initial_valid_assignments(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    # Use only the exact number of players needed for this test's focus
    players = basic_problem_setup["players_data"][:pd["num_teams"] * pd["team_size"]]

    sol = LeagueSolution(
        players=deepcopy(players),
        num_teams=pd["num_teams"],
        team_size=pd["team_size"],
        max_budget=pd["max_budget"],
        position_requirements=pd["position_requirements"]
    )
    
    assert sol.repr is not None
    
    teams_check = [[] for _ in range(pd["num_teams"])]
    for player_idx, team_id in enumerate(sol.repr):
        teams_check[team_id].append(sol.players[player_idx])

    for team_id, team_players in enumerate(teams_check):
        assert len(team_players) == pd["team_size"], f"Team {team_id} has incorrect size"
        
        current_team_positions = {pos: 0 for pos in pd["position_requirements"]}
        for player_in_team in team_players:
            pos = player_in_team["Position"]
            if pos in current_team_positions:
                current_team_positions[pos] += 1
        
        assert current_team_positions == pd["position_requirements"], \
            f"Team {team_id} does not meet positional quotas. Got {current_team_positions}"


# --- Tests for LeagueSolution.is_valid() ---

def test_is_valid_true(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    players = basic_problem_setup["players_data"][:pd["num_teams"] * pd["team_size"]] # exact 6 players
    
    # Manually create a known valid representation.
    # Team 0: P1 (GK, S:100), P2 (DEF, S:120), P3 (MID, S:150) -> Total Salary: 370
    # Team 1: P5 (GK, S:110), P6 (DEF, S:130), P4 (FWD, but we'll use P7 MID for this example if available)
    # Problem def: {"GK": 1, "DEF": 1, "MID": 1}. Max budget 500.
    # P4 is FWD, not in requirements. P7 is MID.
    # Players for this example:
    # P1(GK,100), P2(DEF,120), P3(MID,150), P4(FWD,180), P5(GK,110), P6(DEF,130)
    # Let's adjust players list to ensure we can make a valid assignment for test_is_valid_true
    
    test_players_for_is_valid = [
        {"Name": "P1_GK", "Position": "GK", "Skill": 70, "Salary": 100}, # Team 0
        {"Name": "P2_DEF", "Position": "DEF", "Skill": 75, "Salary": 120}, # Team 0
        {"Name": "P3_MID", "Position": "MID", "Skill": 80, "Salary": 150}, # Team 0
        {"Name": "P4_GK", "Position": "GK", "Skill": 72, "Salary": 110},  # Team 1
        {"Name": "P5_DEF", "Position": "DEF", "Skill": 77, "Salary": 130},  # Team 1
        {"Name": "P6_MID", "Position": "MID", "Skill": 82, "Salary": 100},  # Team 1
    ]
    # Team 0 Salary: 100+120+150 = 370 (Ok)
    # Team 1 Salary: 110+130+100 = 340 (Ok)

    sol = LeagueSolution(
        repr=[0, 0, 0, 1, 1, 1], # P1,P2,P3 to Team 0; P4,P5,P6 to Team 1
        players=deepcopy(test_players_for_is_valid),
        num_teams=pd["num_teams"], team_size=pd["team_size"],
        max_budget=pd["max_budget"], position_requirements=pd["position_requirements"]
    )
    assert sol.is_valid() is True

def test_is_valid_false_team_size(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    players = basic_problem_setup["players_data"][:pd["num_teams"] * pd["team_size"]]
    
    sol = LeagueSolution(
        repr=[0, 0, 1, 1, 1, 1], # Team 0 has 2, Team 1 has 4. Both violate size 3.
        players=deepcopy(players),
        num_teams=pd["num_teams"], team_size=pd["team_size"],
        max_budget=pd["max_budget"], position_requirements=pd["position_requirements"]
    )
    assert sol.is_valid() is False

def test_is_valid_false_position_quota(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    # P1(GK), P2(DEF), P3(MID), P4(FWD), P5(GK), P6(DEF)
    players_for_test = basic_problem_setup["players_data"][:6] 
    
    # Team 0: P1(GK), P2(DEF), P5(GK) -> Violates MID quota, surplus GK
    # Team 1: P3(MID), P4(FWD), P6(DEF) -> P4 is FWD, not in position_requirements.
    # If FWD is not in requirements, it will fail. If it is, team might be ok.
    # Current reqs: {"GK": 1, "DEF": 1, "MID": 1}
    # repr for P1,P2,P5 to T0; P3,P4,P6 to T1
    # T0: P1(GK), P2(DEF), P5(GK) -> GK:2, DEF:1, MID:0 -> Fails
    # T1: P3(MID), P4(FWD), P6(DEF) -> If P4 (FWD) cannot be counted, this team is also invalid positionally
    # Or, if P4 is counted as an unrecognized position, it's invalid.

    sol = LeagueSolution(
        repr=[0, 0, 1, 1, 0, 1], # P1(T0), P2(T0), P3(T1), P4(T1), P5(T0), P6(T1)
        players=deepcopy(players_for_test),
        num_teams=pd["num_teams"], team_size=pd["team_size"],
        max_budget=pd["max_budget"], position_requirements=pd["position_requirements"]
    )
    assert sol.is_valid() is False

def test_is_valid_false_budget(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    # Max budget is 500
    # Team 0: P1(GK,S:100), P2(DEF,S:120), P3(MID,S:150) -> Salary 370 (OK)
    # Team 1: P4(GK,S:300), P5(DEF,S:130), P6(MID,S:100) -> Salary 530 (Over budget)
    players_for_budget_test = [
        {"Name": "P1_GK", "Position": "GK", "Skill": 70, "Salary": 100}, # Team 0
        {"Name": "P2_DEF", "Position": "DEF", "Skill": 75, "Salary": 120}, # Team 0
        {"Name": "P3_MID", "Position": "MID", "Skill": 80, "Salary": 150}, # Team 0
        {"Name": "P4_GK_Exp", "Position": "GK", "Skill": 72, "Salary": 300},  # Team 1 - Expensive GK
        {"Name": "P5_DEF", "Position": "DEF", "Skill": 77, "Salary": 130},  # Team 1
        {"Name": "P6_MID", "Position": "MID", "Skill": 82, "Salary": 100},  # Team 1
    ]
    sol = LeagueSolution(
        repr=[0, 0, 0, 1, 1, 1],
        players=deepcopy(players_for_budget_test),
        num_teams=pd["num_teams"], team_size=pd["team_size"],
        max_budget=pd["max_budget"], position_requirements=pd["position_requirements"]
    )
    assert sol.is_valid() is False

# --- Tests for LeagueSolution.fitness() ---

def test_fitness_valid_solution(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    test_players_for_fitness = [ # Skills: T0 (70,75,80), T1 (72,77,82)
        {"Name": "P1", "Position": "GK", "Skill": 70, "Salary": 100}, 
        {"Name": "P2", "Position": "DEF", "Skill": 75, "Salary": 120}, 
        {"Name": "P3", "Position": "MID", "Skill": 80, "Salary": 150}, 
        {"Name": "P4", "Position": "GK", "Skill": 72, "Salary": 110},  
        {"Name": "P5", "Position": "DEF", "Skill": 77, "Salary": 130},  
        {"Name": "P6", "Position": "MID", "Skill": 82, "Salary": 100},  
    ]
    sol = LeagueSolution(
        repr=[0, 0, 0, 1, 1, 1], # Valid assignment
        players=deepcopy(test_players_for_fitness),
        num_teams=pd["num_teams"], team_size=pd["team_size"],
        max_budget=pd["max_budget"], position_requirements=pd["position_requirements"]
    )
    assert sol.is_valid() is True # Precondition for this fitness test
    fitness = sol.fitness()
    assert isinstance(fitness, float)
    assert fitness >= 0.0
    # Team0 avg skill: (70+75+80)/3 = 75
    # Team1 avg skill: (72+77+82)/3 = 77
    # Expected fitness: np.std([75, 77]) = np.std([-1, 1]) with mean 76 = sqrt(((75-76)^2 + (77-76)^2)/2) = sqrt((1+1)/2) = 1.0
    assert fitness == pytest.approx(1.0)


def test_fitness_invalid_solution(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    players = basic_problem_setup["players_data"][:pd["num_teams"] * pd["team_size"]]
    
    sol = LeagueSolution(
        repr=[0, 0, 1, 1, 1, 1], # Invalid (team size violation)
        players=deepcopy(players),
        num_teams=pd["num_teams"], team_size=pd["team_size"],
        max_budget=pd["max_budget"], position_requirements=pd["position_requirements"]
    )
    assert sol.is_valid() is False # Precondition
    assert sol.fitness() == float('inf')

# --- Tests for Fitness Caching ---

def test_fitness_caching_active(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    test_players = [
        {"Name": "P1", "Position": "GK", "Skill": 70, "Salary": 100}, 
        {"Name": "P2", "Position": "DEF", "Skill": 75, "Salary": 120}, 
        {"Name": "P3", "Position": "MID", "Skill": 80, "Salary": 150}, 
        {"Name": "P4", "Position": "GK", "Skill": 72, "Salary": 110},  
        {"Name": "P5", "Position": "DEF", "Skill": 77, "Salary": 130},  
        {"Name": "P6", "Position": "MID", "Skill": 82, "Salary": 100},  
    ]
    sol = LeagueSolution(
        repr=[0,0,0,1,1,1], players=deepcopy(test_players),
        num_teams=pd["num_teams"], team_size=pd["team_size"],
        max_budget=pd["max_budget"], position_requirements=pd["position_requirements"]
    )
    
    first_fitness = sol.fitness() # Calculate and cache
    assert sol._fitness_cache == pytest.approx(first_fitness) # Cache is set
    
    dummy_value = -100.0
    sol._fitness_cache = dummy_value # Manually change cache
    
    second_fitness = sol.fitness() # Should return from cache
    assert second_fitness == dummy_value

def test_fitness_caching_new_object(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    test_players = [
        {"Name": "P1", "Position": "GK", "Skill": 70, "Salary": 100}, 
        {"Name": "P2", "Position": "DEF", "Skill": 75, "Salary": 120}, 
        {"Name": "P3", "Position": "MID", "Skill": 80, "Salary": 150}, 
        {"Name": "P4", "Position": "GK", "Skill": 72, "Salary": 110},  
        {"Name": "P5", "Position": "DEF", "Skill": 77, "Salary": 130},  
        {"Name": "P6", "Position": "MID", "Skill": 82, "Salary": 100},  
    ]
    repr_val = [0,0,0,1,1,1]

    sol1 = LeagueSolution(
        repr=deepcopy(repr_val), players=deepcopy(test_players),
        num_teams=pd["num_teams"], team_size=pd["team_size"],
        max_budget=pd["max_budget"], position_requirements=pd["position_requirements"]
    )
    fitness1 = sol1.fitness() # Calculates and caches in sol1

    sol2 = LeagueSolution(
        repr=deepcopy(repr_val), players=deepcopy(test_players),
        num_teams=pd["num_teams"], team_size=pd["team_size"],
        max_budget=pd["max_budget"], position_requirements=pd["position_requirements"]
    )
    assert sol2._fitness_cache is None # New object should have empty cache
    fitness2 = sol2.fitness() # Should recalculate
    
    assert fitness1 == pytest.approx(fitness2) # Values should be same
    assert sol1._fitness_cache == pytest.approx(fitness1)
    assert sol2._fitness_cache == pytest.approx(fitness2)
    # If we had a global call counter, we'd check it incremented twice.
    # Here, we rely on _fitness_cache being None initially for sol2.

def test_fitness_caching_deepcopy(basic_problem_setup):
    pd = basic_problem_setup["problem_definition"]
    test_players = [
        {"Name": "P1", "Position": "GK", "Skill": 70, "Salary": 100}, 
        {"Name": "P2", "Position": "DEF", "Skill": 75, "Salary": 120}, 
        {"Name": "P3", "Position": "MID", "Skill": 80, "Salary": 150}, 
        {"Name": "P4", "Position": "GK", "Skill": 72, "Salary": 110},  
        {"Name": "P5", "Position": "DEF", "Skill": 77, "Salary": 130},  
        {"Name": "P6", "Position": "MID", "Skill": 82, "Salary": 100},  
    ]
    sol_orig = LeagueSolution(
        repr=[0,0,0,1,1,1], players=deepcopy(test_players),
        num_teams=pd["num_teams"], team_size=pd["team_size"],
        max_budget=pd["max_budget"], position_requirements=pd["position_requirements"]
    )
    
    original_fitness = sol_orig.fitness() # Calculate and cache
    
    sol_copy = deepcopy(sol_orig)
    
    assert sol_copy._fitness_cache is not None
    assert sol_copy._fitness_cache == pytest.approx(original_fitness)
    
    # Calling fitness on copy should use its copied cache
    # To strictly test this, we'd need to ensure no recalculation.
    # For now, we check the cache value is correct and fitness() returns it.
    dummy_val_for_orig_cache = -555.0 # Change original's cache
    sol_orig._fitness_cache = dummy_val_for_orig_cache

    assert sol_copy.fitness() == pytest.approx(original_fitness), "Deepcopy did not use its own cache or cache was wrong"
    assert sol_orig.fitness() == pytest.approx(dummy_val_for_orig_cache), "Original's cache was not changed as expected"

# More tests can be added for edge cases, specific constraints, etc.
# For example, testing behavior with empty player list (if allowed by __init__ or random_initial_representation).
# Testing behavior when position_requirements are complex or team_size is large.
# Testing with players list that doesn't have 'Skill' or 'Salary' (though fitness already logs errors).
