import random
import logging
from copy import deepcopy
import numpy as np
from solution import LeagueSolution

logger = logging.getLogger(__name__)

def mutate_swap(solution):
    """
    Swaps two players in the solution.

    Args:
        solution (LeagueSolution): The solution to mutate.

    Returns:
        LeagueSolution: The mutated solution.
    """
    new_solution_instance = deepcopy(solution)  # Uses LeagueSolution.__deepcopy__

    new_repr = new_solution_instance.repr[:]  # Shallow copy of list of ints
    if len(new_repr) < 2:  # Cannot swap if less than 2 players
        return new_solution_instance  # Return unmodified copy

    i, j = random.sample(range(len(new_repr)), 2)
    new_repr[i], new_repr[j] = (
        new_repr[j],
        new_repr[i],
    )  # Modifies the shallow copy new_repr

    # This returns a BRAND NEW object, not the modified new_solution_instance.
    # Also, no explicit _fitness_cache = None on the *returned* object.
    return solution.__class__(
        repr=new_repr,
        num_teams=solution.num_teams,
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players,
        position_requirements=solution.position_requirements,
    )


def mutate_swap_constrained(solution):
    """
    Swaps two players in the solution, ensuring that the positions of the players are the same.

    Args:
        solution (LeagueSolution): The solution to mutate.

    Returns:
        LeagueSolution: The mutated solution.
    """
    new_solution_instance = deepcopy(solution)

    if not new_solution_instance.players:  # Check on the instance
        return new_solution_instance

    position_map = {}
    # Ensure players list is accessed from the instance for consistency if it were mutable (though it's shared)
    for idx, player in enumerate(new_solution_instance.players):
        pos = player["Position"]
        if pos not in position_map:
            position_map[pos] = []
        position_map[pos].append(idx)

    swappable_positions = [
        pos for pos, players_in_pos in position_map.items() if len(players_in_pos) >= 2
    ]
    if not swappable_positions:
        return new_solution_instance  # Return copy if no swap is possible

    pos_to_swap = random.choice(swappable_positions)
    idx1, idx2 = random.sample(position_map[pos_to_swap], 2)

    new_repr = new_solution_instance.repr[:]  # Operate on the instance's representation
    new_repr[idx1], new_repr[idx2] = new_repr[idx2], new_repr[idx1]

    return solution.__class__(  # Use original solution's class type for instantiation
        repr=new_repr,
        num_teams=solution.num_teams,  # Use original solution's parameters
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players,
        position_requirements=solution.position_requirements,
    )


def mutate_team_shift(solution: LeagueSolution) -> LeagueSolution:
    """
    Shifts all players' team assignments by a random amount, modulo the number of teams.
    Ensures the shift amount is not zero if there's more than one team.

    Args:
        solution (LeagueSolution): The solution to mutate.

    Returns:
        LeagueSolution: The mutated solution.
    """
    mutated_solution = deepcopy(solution)

    if mutated_solution.num_teams <= 1:
        # No shift possible or meaningful with 0 or 1 team.
        logger.debug(
            "mutate_team_shift: Not enough teams to perform a shift. Returning original solution copy."
        )
        return mutated_solution

    # Generate a random shift amount (non-zero)
    # random.randint(a, b) includes both a and b.
    # We want a shift from 1 to num_teams - 1.
    shift = random.randint(1, mutated_solution.num_teams - 1)

    # Apply the shift to the representation of the mutated_solution
    # The representation is a list of team_ids for each player.
    original_repr = list(
        mutated_solution.repr
    )  # Work on a copy if iterating and modifying, though list comprehension below avoids this need for source list

    mutated_solution.repr = [
        (team_id + shift) % mutated_solution.num_teams for team_id in original_repr
    ]

    # Invalidate the fitness cache as the representation has changed
    mutated_solution._fitness_cache = None

    logger.debug(
        f"mutate_team_shift: Shifted teams by {shift}. Original first player team: {original_repr[0] if original_repr else 'N/A'}, New: {mutated_solution.repr[0] if mutated_solution.repr else 'N/A'}"
    )

    return mutated_solution


def mutate_targeted_player_exchange(solution):
    """
    Swaps two players in the solution, ensuring that the positions of the players are the same.

    Args:
        solution (LeagueSolution): The solution to mutate.

    Returns:
        LeagueSolution: The mutated solution.
    """
    new_solution_instance = deepcopy(solution)

    if (
        not new_solution_instance.players or new_solution_instance.num_teams < 2
    ):  # Checks on instance
        return new_solution_instance

    # Get team player indices from the instance's representation
    teams_player_indices = [[] for _ in range(new_solution_instance.num_teams)]
    for player_idx, team_id in enumerate(
        new_solution_instance.repr
    ):  # Use instance's repr
        teams_player_indices[team_id].append(player_idx)

    avg_skills = []
    for team_p_indices in teams_player_indices:
        if not team_p_indices:  # Empty team
            avg_skills.append(0)
            continue
        # Access players from the instance
        skill_sum = sum(
            new_solution_instance.players[p_idx]["Skill"] for p_idx in team_p_indices
        )
        avg_skills.append(skill_sum / len(team_p_indices))

    if not avg_skills or len(avg_skills) < 2:
        return new_solution_instance  # Unmodified copy

    highest_team_idx = np.argmax(avg_skills)
    lowest_team_idx = np.argmin(avg_skills)

    if highest_team_idx == lowest_team_idx:
        return new_solution_instance  # Unmodified copy

    high_team_players_by_pos = {}
    for p_idx in teams_player_indices[highest_team_idx]:
        # Access players from the instance
        pos = new_solution_instance.players[p_idx]["Position"]
        if pos not in high_team_players_by_pos:
            high_team_players_by_pos[pos] = []
        high_team_players_by_pos[pos].append(p_idx)

    low_team_players_by_pos = {}
    for p_idx in teams_player_indices[lowest_team_idx]:
        # Access players from the instance
        pos = new_solution_instance.players[p_idx]["Position"]
        if pos not in low_team_players_by_pos:
            low_team_players_by_pos[pos] = []
        low_team_players_by_pos[pos].append(p_idx)

    common_positions = set(high_team_players_by_pos.keys()) & set(
        low_team_players_by_pos.keys()
    )
    if not common_positions:
        return new_solution_instance  # Unmodified copy

    pos_to_swap = random.choice(list(common_positions))

    high_player_p_idx = random.choice(high_team_players_by_pos[pos_to_swap])
    low_player_p_idx = random.choice(low_team_players_by_pos[pos_to_swap])

    new_repr = new_solution_instance.repr[:]  # Operate on instance's repr
    new_repr[high_player_p_idx], new_repr[low_player_p_idx] = (
        new_repr[low_player_p_idx],
        new_repr[high_player_p_idx],
    )

    return solution.__class__(  # Use original solution's class type
        repr=new_repr,
        num_teams=solution.num_teams,  # Use original solution's parameters
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players,
        position_requirements=solution.position_requirements,
    )


def mutate_shuffle_within_team_constrained(solution):
    """
    Performs a single, position-constrained swap of players between two different teams.
    One player is chosen from a random team, and another player of the same position
    is chosen from a different team. Their team assignments are then swapped.
    If a valid swap cannot be performed, an effectively unchanged (but new) copy of the solution is returned.
    """
    mutated_solution = deepcopy(solution)

    # Basic condition checks
    if not mutated_solution.players or mutated_solution.num_teams < 2:
        return mutated_solution  # Return deepcopied instance, effectively unchanged

    # 1. Pick one chosen_team_id at random.
    chosen_team_id = random.randint(0, mutated_solution.num_teams - 1)

    # 2. Get all player_indices_in_chosen_team from mutated_solution.repr.
    player_indices_in_chosen_team = [
        idx
        for idx, team_id in enumerate(mutated_solution.repr)
        if team_id == chosen_team_id
    ]
    if not player_indices_in_chosen_team:
        return mutated_solution  # Chosen team is empty

    # 3. Randomly select one player_to_move_idx from player_indices_in_chosen_team.
    player_to_move_idx = random.choice(player_indices_in_chosen_team)

    # 4. Get the position_of_player_to_move.
    # Ensure player_to_move_idx is valid for the players list (should be if repr is consistent).
    if not (0 <= player_to_move_idx < len(mutated_solution.players)):
        return mutated_solution
    position_of_player_to_move = mutated_solution.players[player_to_move_idx][
        "Position"
    ]

    # 5. Find a list of candidate_swap_indices_from_other_teams.
    # These candidates must be on a *different* team and have the *same position*.
    candidate_swap_indices_from_other_teams = [
        other_idx
        for other_idx, other_team_id in enumerate(mutated_solution.repr)
        if other_team_id != chosen_team_id
        and (0 <= other_idx < len(mutated_solution.players))
        and mutated_solution.players[other_idx]["Position"]
        == position_of_player_to_move
    ]

    if not candidate_swap_indices_from_other_teams:
        return mutated_solution  # No valid swap partner found

    # 6. Randomly select one player_from_other_team_idx from candidates.
    player_from_other_team_idx = random.choice(candidate_swap_indices_from_other_teams)

    # 7. Perform the swap directly in mutated_solution.repr
    # Player player_to_move_idx (from chosen_team_id) gets the team of player_from_other_team_idx.
    # Player player_from_other_team_idx gets chosen_team_id.

    # Store the original team of player_from_other_team_idx before changing it
    original_team_of_swapped_player = mutated_solution.repr[player_from_other_team_idx]

    mutated_solution.repr[player_to_move_idx] = original_team_of_swapped_player
    mutated_solution.repr[player_from_other_team_idx] = chosen_team_id

    mutated_solution._fitness_cache = (
        None  # Invalidate fitness cache as repr has changed
    )

    return mutated_solution
