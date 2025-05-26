import random
import numpy as np
from copy import deepcopy
from solution import LeagueSolution, LeagueHillClimbingSolution # for GA with local search
from evolution import hill_climbing # for GA with local search
import logging 

# Solution module is imported by the main script, and then LeagueSolution are passed around
# Operators that CREATE solutions,

# Get a logger for this module
logger = logging.getLogger(__name__)

# Default safe_exp function if not provided (can be overridden by passing a different one)

def default_safe_exp(x, max_value=700): # Max value from original main script
    """
    Used to prevent numerical overflow or underflow errors when calculating np.exp(x) 
    This is important for Simmulated Annealing and Boltzmann Selection in Genetic Algorithms

    Args:
        x (float): The input value to calculate the exponential of.
        max_value (float): The maximum value to clip the input to. Defaults to 700.
    
    Returns:
        float: The exponential of the input value, clipped to the maximum value.
    """
    return np.exp(np.clip(x, -max_value, max_value))

#----------------------------------------------------------------------------------------------#
# MUTATION OPERATORS
#----------------------------------------------------------------------------------------------#

def mutate_swap(solution): 
    """
    Swaps two players in the solution.
    
    Args:
        solution (LeagueSolution): The solution to mutate.
    
    Returns:
        LeagueSolution: The mutated solution.
    """
    new_solution_instance = deepcopy(solution) # Uses LeagueSolution.__deepcopy__
    
    new_repr = new_solution_instance.repr[:] # Shallow copy of list of ints
    if len(new_repr) < 2: # Cannot swap if less than 2 players
        return new_solution_instance # Return unmodified copy

    i, j = random.sample(range(len(new_repr)), 2)
    new_repr[i], new_repr[j] = new_repr[j], new_repr[i] # Modifies the shallow copy new_repr
    
    # This returns a BRAND NEW object, not the modified new_solution_instance.
    # Also, no explicit _fitness_cache = None on the *returned* object.
    return solution.__class__( 
        repr=new_repr, 
        num_teams=solution.num_teams,
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players, 
        position_requirements=solution.position_requirements
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
    
    if not new_solution_instance.players: # Check on the instance
        return new_solution_instance

    position_map = {}
    # Ensure players list is accessed from the instance for consistency if it were mutable (though it's shared)
    for idx, player in enumerate(new_solution_instance.players):
        pos = player["Position"]
        if pos not in position_map: position_map[pos] = []
        position_map[pos].append(idx)
    
    swappable_positions = [pos for pos, players_in_pos in position_map.items() if len(players_in_pos) >= 2]
    if not swappable_positions:
        return new_solution_instance # Return copy if no swap is possible

    pos_to_swap = random.choice(swappable_positions)
    idx1, idx2 = random.sample(position_map[pos_to_swap], 2)
    
    new_repr = new_solution_instance.repr[:] # Operate on the instance's representation
    new_repr[idx1], new_repr[idx2] = new_repr[idx2], new_repr[idx1]
    
    return solution.__class__( # Use original solution's class type for instantiation
        repr=new_repr,
        num_teams=solution.num_teams, # Use original solution's parameters
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players,
        position_requirements=solution.position_requirements
    )

# In operators.py

# Ensure these are imported at the top if not already:
# import random
# from copy import deepcopy
# from solution import LeagueSolution # Assuming this is your base solution class
# import logging
# logger = logging.getLogger(__name__)


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
        logger.debug("mutate_team_shift: Not enough teams to perform a shift. Returning original solution copy.")
        return mutated_solution

    # Generate a random shift amount (non-zero)
    # random.randint(a, b) includes both a and b.
    # We want a shift from 1 to num_teams - 1.
    shift = random.randint(1, mutated_solution.num_teams - 1)

    # Apply the shift to the representation of the mutated_solution
    # The representation is a list of team_ids for each player.
    original_repr = list(mutated_solution.repr) # Work on a copy if iterating and modifying, though list comprehension below avoids this need for source list
    
    mutated_solution.repr = [(team_id + shift) % mutated_solution.num_teams for team_id in original_repr]
    
    # Invalidate the fitness cache as the representation has changed
    mutated_solution._fitness_cache = None
    
    logger.debug(f"mutate_team_shift: Shifted teams by {shift}. Original first player team: {original_repr[0] if original_repr else 'N/A'}, New: {mutated_solution.repr[0] if mutated_solution.repr else 'N/A'}")
    
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

    if not new_solution_instance.players or new_solution_instance.num_teams < 2: # Checks on instance
        return new_solution_instance

    # Get team player indices from the instance's representation
    teams_player_indices = [[] for _ in range(new_solution_instance.num_teams)]
    for player_idx, team_id in enumerate(new_solution_instance.repr): # Use instance's repr
        teams_player_indices[team_id].append(player_idx)

    avg_skills = []
    for team_p_indices in teams_player_indices:
        if not team_p_indices: # Empty team
            avg_skills.append(0)
            continue
        # Access players from the instance
        skill_sum = sum(new_solution_instance.players[p_idx]["Skill"] for p_idx in team_p_indices)
        avg_skills.append(skill_sum / len(team_p_indices))
        
    if not avg_skills or len(avg_skills) < 2 : return new_solution_instance # Unmodified copy

    highest_team_idx = np.argmax(avg_skills)
    lowest_team_idx = np.argmin(avg_skills)
    
    if highest_team_idx == lowest_team_idx: return new_solution_instance # Unmodified copy
            
    high_team_players_by_pos = {}
    for p_idx in teams_player_indices[highest_team_idx]:
        # Access players from the instance
        pos = new_solution_instance.players[p_idx]["Position"]
        if pos not in high_team_players_by_pos: high_team_players_by_pos[pos] = []
        high_team_players_by_pos[pos].append(p_idx)
    
    low_team_players_by_pos = {}
    for p_idx in teams_player_indices[lowest_team_idx]:
        # Access players from the instance
        pos = new_solution_instance.players[p_idx]["Position"]
        if pos not in low_team_players_by_pos: low_team_players_by_pos[pos] = []
        low_team_players_by_pos[pos].append(p_idx)
            
    common_positions = set(high_team_players_by_pos.keys()) & set(low_team_players_by_pos.keys())
    if not common_positions: 
        return new_solution_instance # Unmodified copy
            
    pos_to_swap = random.choice(list(common_positions))
    
    high_player_p_idx = random.choice(high_team_players_by_pos[pos_to_swap])
    low_player_p_idx = random.choice(low_team_players_by_pos[pos_to_swap])
    
    new_repr = new_solution_instance.repr[:] # Operate on instance's repr
    new_repr[high_player_p_idx], new_repr[low_player_p_idx] = new_repr[low_player_p_idx], new_repr[high_player_p_idx]
    
    return solution.__class__( # Use original solution's class type
        repr=new_repr,
        num_teams=solution.num_teams, # Use original solution's parameters
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players,
        position_requirements=solution.position_requirements
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
        return mutated_solution # Return deepcopied instance, effectively unchanged

    # 1. Pick one chosen_team_id at random.
    chosen_team_id = random.randint(0, mutated_solution.num_teams - 1)

    # 2. Get all player_indices_in_chosen_team from mutated_solution.repr.
    player_indices_in_chosen_team = [
        idx for idx, team_id in enumerate(mutated_solution.repr) if team_id == chosen_team_id
    ]
    if not player_indices_in_chosen_team:
        return mutated_solution # Chosen team is empty

    # 3. Randomly select one player_to_move_idx from player_indices_in_chosen_team.
    player_to_move_idx = random.choice(player_indices_in_chosen_team)

    # 4. Get the position_of_player_to_move.
    # Ensure player_to_move_idx is valid for the players list (should be if repr is consistent).
    if not (0 <= player_to_move_idx < len(mutated_solution.players)):
        return mutated_solution 
    position_of_player_to_move = mutated_solution.players[player_to_move_idx]["Position"]

    # 5. Find a list of candidate_swap_indices_from_other_teams.
    # These candidates must be on a *different* team and have the *same position*.
    candidate_swap_indices_from_other_teams = [
        other_idx for other_idx, other_team_id in enumerate(mutated_solution.repr)
        if other_team_id != chosen_team_id and \
           (0 <= other_idx < len(mutated_solution.players)) and \
           mutated_solution.players[other_idx]["Position"] == position_of_player_to_move
    ]

    if not candidate_swap_indices_from_other_teams:
        return mutated_solution # No valid swap partner found

    # 6. Randomly select one player_from_other_team_idx from candidates.
    player_from_other_team_idx = random.choice(candidate_swap_indices_from_other_teams)
    
    # 7. Perform the swap directly in mutated_solution.repr
    # Player player_to_move_idx (from chosen_team_id) gets the team of player_from_other_team_idx.
    # Player player_from_other_team_idx gets chosen_team_id.
    
    # Store the original team of player_from_other_team_idx before changing it
    original_team_of_swapped_player = mutated_solution.repr[player_from_other_team_idx]

    mutated_solution.repr[player_to_move_idx] = original_team_of_swapped_player
    mutated_solution.repr[player_from_other_team_idx] = chosen_team_id
    
    mutated_solution._fitness_cache = None # Invalidate fitness cache as repr has changed

    return mutated_solution
    
# CROSSOVER OPERATORS --------

def crossover_one_point(parent1, parent2):
    # Ensure parents have representations and they are lists
    if not hasattr(parent1, 'repr') or not hasattr(parent2, 'repr') or \
       not isinstance(parent1.repr, list) or not isinstance(parent2.repr, list) or \
       len(parent1.repr) != len(parent2.repr) or len(parent1.repr) < 2: # Min length for valid cut
        # Fallback: return a copy of the fitter parent if crossover isn't possible
        return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)

    # Cut point must be between 1 and len-1 to ensure segments from both parents
    # random.randint(a,b) includes b. So, for len=2, cut can be 1.
    # For len=L, cut can be from 1 to L-1.
    # Original was (1, len-2), which for len=2 is (1,0) -> error. For len=3, (1,1) -> cut=1.
    # A cut point of 1 means child gets parent1.repr[0] and rest from parent2.
    # A cut point of L-1 means child gets parent1.repr[0...L-2] and parent2.repr[L-1].
    min_cut = 1
    max_cut = len(parent1.repr) -1 
    if min_cut > max_cut : # e.g. if len(parent1.repr) is 1 (though earlier check prevents)
         return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)
    cut = random.randint(min_cut, max_cut) 

    child_repr = parent1.repr[:cut] + parent2.repr[cut:]
    
    return parent1.__class__( # Use parent1's class type
        repr=child_repr,
        num_teams=parent1.num_teams, team_size=parent1.team_size,
        max_budget=parent1.max_budget, players=parent1.players,
        position_requirements=parent1.position_requirements
    )

def crossover_one_point_prefer_valid(parent1, parent2, max_attempts=10):
    for _ in range(max_attempts):
        child = crossover_one_point(parent1, parent2) # This now handles its own creation
        if child.is_valid():
            return child
    
    # If no valid child found, return a deepcopy of the better parent
    return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)


def crossover_two_point(parent1: LeagueSolution, parent2: LeagueSolution) -> LeagueSolution:
    """
    Performs a two-point crossover between two parent solutions.

    The child's representation is created by taking the first segment from parent1 (up to cut point k1),
    the middle segment from parent2 (between cut points k1 and k2), and the final segment from parent1 (from k2 onwards).
    Child = P1[0:k1] + P2[k1:k2] + P1[k2:L]

    Args:
        parent1: The first parent solution (LeagueSolution).
        parent2: The second parent solution (LeagueSolution).

    Returns:
        A new LeagueSolution instance representing the child. If crossover is not possible
        (e.g., due to incompatible parent representations or representation length < 3),
        a deepcopy of the fitter parent is returned.
    """
    if not hasattr(parent1, 'repr') or not hasattr(parent2, 'repr') or \
       not isinstance(parent1.repr, list) or not isinstance(parent2.repr, list) or \
       len(parent1.repr) != len(parent2.repr):
        logger.debug("Two-point crossover fallback: Incompatible parent representations.")
        return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)

    L = len(parent1.repr)
    # Required conditions for k1 and k2: 1 <= k1 < k2 <= L-1.
    # This implies L must be at least 3 (e.g., k1=1, k2=2 for L=3).
    if L < 3:
        logger.debug(f"Two-point crossover fallback: Representation length {L} is less than 3.")
        return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)

    # Select two distinct crossover points k1, k2.
    # k1 can range from 1 to L-2 (inclusive, to ensure k2 can be k1+1).
    k1 = random.randint(1, L - 2)
    # k2 can range from k1+1 to L-1 (inclusive).
    k2 = random.randint(k1 + 1, L - 1)

    child_repr = parent1.repr[:k1] + parent2.repr[k1:k2] + parent1.repr[k2:]
    
    return parent1.__class__(
        repr=child_repr,
        num_teams=parent1.num_teams, team_size=parent1.team_size,
        max_budget=parent1.max_budget, players=parent1.players,
        position_requirements=parent1.position_requirements
    )

def crossover_two_point_prefer_valid(parent1: LeagueSolution, parent2: LeagueSolution, max_attempts: int = 10) -> LeagueSolution:
    """
    Attempts to create a valid child solution using two-point crossover, trying multiple times.

    This operator repeatedly calls `crossover_two_point`. If a valid child
    (checked using `child.is_valid()`) is produced within `max_attempts`,
    that child is returned. If no valid child is found after `max_attempts`,
    a deepcopy of the fitter parent (based on `parent.fitness()`) is returned as a fallback.

    Args:
        parent1: The first parent solution.
        parent2: The second parent solution.
        max_attempts: The maximum number of attempts to generate a valid child.

    Returns:
        A new LeagueSolution instance, either a valid child from crossover or
        a deepcopy of the fitter parent.
    """
    for attempt in range(max_attempts):
        child = crossover_two_point(parent1, parent2)
        if child.is_valid():
            logger.debug(f"Two-point prefer_valid: Valid child found on attempt {attempt + 1}.")
            return child
    logger.debug(f"Two-point prefer_valid: No valid child after {max_attempts} attempts. Returning fitter parent.")
    return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)


def crossover_uniform(parent1, parent2):
    if not hasattr(parent1, 'repr') or not hasattr(parent2, 'repr') or \
       len(parent1.repr) != len(parent2.repr):
        return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)

    child_repr = [
        parent1.repr[i] if random.random() < 0.5 else parent2.repr[i]
        for i in range(len(parent1.repr))
    ]
    
    return parent1.__class__(
        repr=child_repr,
        num_teams=parent1.num_teams, team_size=parent1.team_size,
        max_budget=parent1.max_budget, players=parent1.players,
        position_requirements=parent1.position_requirements
    )

def crossover_uniform_prefer_valid(parent1, parent2, max_attempts=10):
    for _ in range(max_attempts):
        child = crossover_uniform(parent1, parent2)
        if child.is_valid():
            return child
    return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)

# SELECTION OPERATORS --------

def selection_tournament(population, k=3): # k can be passed via selection_params
    # Ensure k is not larger than population size
    actual_k = min(k, len(population))
    if actual_k <= 0: # Handle empty population or invalid k
        return random.choice(population) if population else None 

    selected_candidates = random.sample(population, actual_k)
    # Sort by fitness (assuming lower is better)
    selected_candidates.sort(key=lambda sol: sol.fitness()) 
    return selected_candidates[0]

# selection_tournament_variable_k is essentially the same as above with the k clamping.
# We can merge them or keep selection_tournament simple if k is always pre-validated.
# The main script used selection_tournament_variable_k, so let's keep its robust k handling.

def selection_tournament_variable_k(population: list, k: int = 3, k_percentage: float = None) -> 'LeagueSolution':
    """
    Selects an individual using tournament selection.
    The tournament size can be a fixed integer 'k' or a 'k_percentage' of the population size.
    If k_percentage is provided and valid (0 < k_percentage <= 1), it overrides the fixed 'k'.

    Args:
        population (list): The current population of solution instances.
        k (int): The fixed tournament size (number of participants). Used if k_percentage is None.
        k_percentage (float, optional): The percentage of the population to use as tournament size.
                                         Overrides 'k' if provided and valid. Defaults to None.

    Returns:
        LeagueSolution: The winning individual from the tournament, or None if the population is empty.
    """
    if not population:
        logger.warning("selection_tournament_variable_k: Population is empty. Cannot select.")
        return None

    actual_k = 0
    if k_percentage is not None and 0 < k_percentage <= 1:
        # Calculate k based on percentage, ensuring it's at least 1
        actual_k = max(1, int(len(population) * k_percentage))
        logger.debug(f"Tournament selection: Using k_percentage {k_percentage*100:.1f}%, actual_k={actual_k} from pop_size={len(population)}")
    else:
        actual_k = k
        logger.debug(f"Tournament selection: Using fixed k={k}, actual_k={actual_k} (k_percentage not used or invalid)")
    
    # Ensure k is not larger than population size
    actual_k = min(actual_k, len(population))
    
    # If actual_k became 0 (e.g., from a very small k_percentage on a small pop, before max(1,...)
    # or if initial k was 0), and population is not empty, set to 1.
    if actual_k == 0 and len(population) > 0:
        actual_k = 1
        logger.debug(f"Tournament selection: actual_k was 0, adjusted to 1 for non-empty population.")
    elif actual_k == 0: # Population was empty, already handled, but good to be safe.
        logger.warning("selection_tournament_variable_k: actual_k is 0 and population is empty. Cannot select.")
        return None

    tournament_participants = random.sample(population, actual_k)
    
    # Return the fittest individual from the tournament (assuming minimization problem)
    # Ensure all participants have a fitness value calculated.
    try:
        winner = min(tournament_participants, key=lambda x: x.fitness())
    except Exception as e:
        logger.error(f"Error during fitness comparison in tournament selection: {e}", exc_info=True)
        # Fallback: return a random participant if fitness comparison fails
        winner = random.choice(tournament_participants) if tournament_participants else None

    return winner


def selection_ranking(population):
    if not population: return None
    # Sort by fitness (lower is better, so best individuals come first)
    sorted_pop = sorted(population, key=lambda s: s.fitness())
    
    # Ranks: best is rank N, second best is N-1, ..., worst is 1 (for probability assignment)
    # This means the individual at sorted_pop[0] (best fitness) gets rank N.
    # sorted_pop[N-1] (worst fitness) gets rank 1.
    num_individuals = len(sorted_pop)
    # Assign ranks such that higher rank value means better for probability calculation
    # (e.g. best gets rank 'num_individuals', worst gets rank '1')
    ranks = list(range(1, num_individuals + 1)) # ranks [1, 2, ..., N]
    
    # Probabilities: sum of ranks is N*(N+1)/2
    # Best individual (sorted_pop[0]) should get prob proportional to N.
    # Worst individual (sorted_pop[N-1]) should get prob proportional to 1.
    # So, weights should be ranks in descending order of value for the sorted_pop.
    # sorted_pop[0] uses weight ranks[N-1] (which is N)
    # sorted_pop[1] uses weight ranks[N-2] (which is N-1)
    # ...
    # sorted_pop[N-1] uses weight ranks[0] (which is 1)
    
    # Original: probs = [r / total for r in ranks[::-1]]
    # ranks = [1, 2, 3, 4, 5] for len=5. Ranks sum = 15.
    # ranks[::-1] = [5, 4, 3, 2, 1]. Probs = [5/15, 4/15, ..., 1/15].
    # This assigns highest probability to sorted_pop[0] (best fitness), which is correct.
    if num_individuals == 0: return None
    
    # Calculate sum of ranks (1 to N) = N*(N+1)/2
    sum_of_ranks = num_individuals * (num_individuals + 1) / 2.0
    if sum_of_ranks == 0 : # only if num_individuals is 0, handled above
        return random.choice(sorted_pop) if sorted_pop else None

    # Probabilities for sorted_pop: best (index 0) gets highest prob.
    # Probability for sorted_pop[i] is (N-i) / sum_of_ranks
    probabilities = [(num_individuals - i) / sum_of_ranks for i in range(num_individuals)]
    
    return random.choices(sorted_pop, weights=probabilities, k=1)[0]


def selection_boltzmann(population, temperature=1.0, safe_exp_func=default_safe_exp):
    if not population: return None
    
    fitness_values = [sol.fitness() for sol in population]
    
    # Handle inf fitness: give them zero chance of being selected directly
    # For minimization: lower fitness is better. We need to transform fitness so higher value = higher prob.
    # Option 1: Max_fitness - fitness (if max_fitness is known/estimable)
    # Option 2: 1 / (fitness + epsilon) (as original)
    
    # Using 1 / (fitness + epsilon)
    # If fitness is inf, 1/inf -> 0. If fitness is 0, 1/epsilon.
    inverted_fitness = []
    for f_val in fitness_values:
        if f_val == float('inf'):
            inverted_fitness.append(0.0) # Zero contribution to selection if infinitely bad
        else:
            # Add a small constant to prevent division by zero if fitness can be 0
            # and to scale very small fitness values reasonably if needed.
            inverted_fitness.append(1.0 / (f_val + 1e-9)) 

    # Calculate Boltzmann probabilities using the provided safe_exp_func
    # Higher inverted_fitness (from lower original fitness) should lead to higher probability
    try:
        # Scale inverted_fitness before exp to prevent overflow even with safe_exp if values are huge
        # This step is optional but can help if inverted_fitness values become very large.
        # max_inv_fitness = max(inverted_fitness) if inverted_fitness else 0
        # scaled_inv_fitness = [f / (max_inv_fitness + 1e-9) for f in inverted_fitness] if max_inv_fitness > 0 else inverted_fitness
        # For now, directly use inverted_fitness with safe_exp.
        
        boltzmann_exp_values = [safe_exp_func(f / temperature) for f in inverted_fitness]
    except OverflowError: # Should be caught by safe_exp_func, but as a fallback
        # Handle extreme case: if safe_exp still overflows (e.g. bad temperature or extreme fitness values)
        # Fallback to uniform random choice or ranking
        # For now, let it propagate if safe_exp fails, or implement robust fallback.
        # A simple fallback: if sum is 0, use uniform.
        # This part might need more robust handling for extreme numerical inputs.
        # If all fitness are inf, all inverted_fitness are 0, all boltzmann_exp_values might be equal (e.g., 1 from exp(0))
        # If temperature is near 0, exponent can be huge.
        # safe_exp_func should prevent overflow in exp, but division by temperature can still be an issue.
        pass # Assuming safe_exp_func handles it.

    total_boltzmann_value = sum(boltzmann_exp_values)
    
    if total_boltzmann_value == 0: # All fitness were inf, or exp values became 0
        # Fallback to uniform random selection if all probabilities are zero
        return random.choice(population) if population else None
        
    probabilities = [b_val / total_boltzmann_value for b_val in boltzmann_exp_values]
    
    return random.choices(population, weights=probabilities, k=1)[0]


# GENETIC ALGORITHM --------

def generate_population(
    players_data: list,
    population_size: int,
    num_teams: int,
    team_size: int,
    max_budget: float,
    position_requirements: dict,
    logger_instance: logging.Logger,
    max_initial_solution_attempts: int = 50  
) -> list:
    """
    Generates an initial population of valid LeagueSolution instances.
    Tries to meet positional quotas and team sizes. Budget is checked by is_valid().
    """
    population = []
    # Use the passed parameter directly for clarity within this function
    # max_attempts_per_solution_in_loop = max_initial_solution_attempts # Or just use max_initial_solution_attempts

    # Overall safeguard for total attempts across all solutions
    total_attempts_allowed = population_size * max_initial_solution_attempts * 2 

    current_total_attempts = 0
    solutions_generated = 0

    logger_instance.info(f"Attempting to generate initial population of size {population_size} (max attempts per solution: {max_initial_solution_attempts})...")

    while solutions_generated < population_size and current_total_attempts < total_attempts_allowed:
        current_total_attempts += 1
        attempts_for_this_solution = 0 # Renamed for clarity within the inner loop

        current_solution = None
        while attempts_for_this_solution < max_initial_solution_attempts: # Use the parameter
            attempts_for_this_solution += 1
            try:
                solution = LeagueSolution(
                    num_teams=num_teams,
                    team_size=team_size,
                    max_budget=max_budget,
                    players=deepcopy(players_data),
                    position_requirements=position_requirements
                )
                if solution.is_valid():
                    current_solution = solution
                    break 
            except InsufficientPlayersForPositionError as e_pos:
                if attempts_for_this_solution % 10 == 0:
                    logger_instance.warning(
                        f"generate_population: Attempt {attempts_for_this_solution}/{max_initial_solution_attempts} for sol {solutions_generated + 1} - {e_pos}"
                    )
            except ValueError as e_val:
                if attempts_for_this_solution % 10 == 0:
                    logger_instance.warning(
                        f"generate_population: Attempt {attempts_for_this_solution}/{max_initial_solution_attempts} for sol {solutions_generated + 1} - Value error: {e_val}"
                    )
            except Exception as e_unexpected:
                logger_instance.error(
                    f"generate_population: Unexpected error on attempt {attempts_for_this_solution} for sol {solutions_generated + 1}: {e_unexpected}",
                    exc_info=True
                )
        
        if current_solution:
            population.append(current_solution)
            solutions_generated += 1
            if solutions_generated % (population_size // 10 or 1) == 0:
                logger_instance.info(f"  Generated {solutions_generated}/{population_size} valid initial solutions...")
        else:
            logger_instance.warning(
                f"generate_population: Failed to generate a valid solution for individual {solutions_generated + 1} "
                f"after {max_initial_solution_attempts} attempts. Continuing..."
            )

    if solutions_generated < population_size:
        logger_instance.warning(
            f"generate_population: Only generated {solutions_generated}/{population_size} valid solutions "
            f"after {current_total_attempts} total attempts. Population will be smaller than requested."
        )
    
    if not population and population_size > 0:
        logger_instance.error("generate_population: Failed to generate ANY valid solutions for the initial population.")
        # raise RuntimeError("Failed to generate any valid solutions for the initial GA population.") # Optionally raise

    logger_instance.info(f"generate_population: Finished. Generated {len(population)} solutions.")
    return population


def genetic_algorithm(
    players_data: list,
    problem_params: dict, # Dictionary: num_teams, team_size, max_budget, position_requirements
    ga_params: dict      # Dictionary: population_size, max_generations, operators, rates, etc.
) -> tuple:
    """
    Genetic Algorithm implementation.

    problem_params: {'num_teams', 'team_size', 'max_budget', 'position_requirements'}
    ga_params: {
        'population_size', 'max_generations',
        'selection_operator' (callable), 'selection_params' (dict),
        'crossover_operator' (callable), 'crossover_rate' (float),
        'mutation_operator' (callable), 'mutation_rate' (float, prob_apply_mutation),
        'elitism_size' (int),
        'max_initial_solution_attempts' (int, optional for generate_population),
        'local_search_func' (optional callable),
        'local_search_params' (optional dict for local_search_func,
                               e.g., {'frequency': 5, 'num_to_improve': 1,
                                      'max_iterations': 50, 'max_no_improvement': 20}),
        'safe_exp_func' (optional callable, for Boltzmann selection),
        'verbose' (bool)
    }
    Returns:
        Tuple: (best_solution_overall, best_fitness_overall, history_of_best_fitness)
    """
    # Use the module-level logger
    global logger # If logger is indeed a global in operators.py
                  # Or, if it's just defined at module scope, direct use is fine.
                  # If not, it should be passed or properly imported.
                  # For now, assuming 'logger' is accessible as the module's logger.

    num_teams = problem_params['num_teams']
    team_size = problem_params['team_size']
    max_budget = problem_params['max_budget']
    position_requirements = problem_params['position_requirements']

    population_size = ga_params['population_size']
    max_generations = ga_params['max_generations']
    
    selection_operator = ga_params['selection_operator']
    selection_kwargs = ga_params.get('selection_params', {}).copy()
    if selection_operator.__name__ == 'selection_boltzmann' and 'safe_exp_func' not in selection_kwargs:
        # Ensure default_safe_exp is defined or imported
        selection_kwargs['safe_exp_func'] = ga_params.get('safe_exp_func', default_safe_exp)

    crossover_operator = ga_params['crossover_operator']
    crossover_rate = ga_params['crossover_rate']
    
    mutation_operator = ga_params['mutation_operator']
    prob_apply_mutation = ga_params.get('mutation_rate', 0.1)

    elitism_size = ga_params.get('elitism_size', 0)
    if 'elitism' in ga_params and isinstance(ga_params['elitism'], bool):
        if ga_params['elitism'] and elitism_size == 0:
            elitism_size = 1 # Default to 1 if elitism is true but size is 0
        elif not ga_params['elitism']:
            elitism_size = 0 # Ensure size is 0 if elitism is false
    
    local_search_func = ga_params.get('local_search_func', None)
    local_search_config = ga_params.get('local_search_params', {})
    
    verbose = ga_params.get('verbose', False)

    # Get max_initial_solution_attempts from ga_params for generate_population
    max_init_attempts_for_population = ga_params.get('max_initial_solution_attempts', 50) # Default

    # Initialize population
    population = generate_population(
        players_data,
        population_size,
        num_teams,
        team_size,
        max_budget,
        position_requirements,
        logger_instance=logger, # Pass the module-level logger
        max_initial_solution_attempts=max_init_attempts_for_population # Pass the new parameter
    )
    if not population:
        logger.error("Failed to initialize GA population. Cannot proceed.")
        raise RuntimeError("Failed to initialize GA population.")

    try:
        best_solution_overall = min(population, key=lambda s: s.fitness())
        best_fitness_overall = best_solution_overall.fitness()
    except ValueError: 
        logger.error("Initial population is empty after generation. Cannot determine initial best.")
        raise RuntimeError("Initial population empty, cannot determine initial best.")
        
    history = [best_fitness_overall]
    
    if verbose:
        logger.info(f"Initial best fitness: {best_fitness_overall:.4f}")
    
    for generation in range(max_generations):
        population.sort(key=lambda x: x.fitness())
        
        new_population = []
        if elitism_size > 0:
            actual_elitism_size = min(elitism_size, len(population))
            new_population.extend(deepcopy(population[:actual_elitism_size]))
            
        while len(new_population) < population_size:
            if not population:
                logger.warning(f"GA Gen {generation + 1}: Parent population depleted. Loop will break.")
                break 

            parent1 = selection_operator(population, **selection_kwargs)
            parent2 = selection_operator(population, **selection_kwargs)
            
            if parent1 is None or parent2 is None:
                logger.warning(f"GA Gen {generation + 1}: Parent selection failed (pop size: {len(population)}). Using random fallback.")
                if population:
                    parent1 = deepcopy(random.choice(population)) if parent1 is None and population else parent1
                    parent2 = deepcopy(random.choice(population)) if parent2 is None and population else parent2
                else:
                    logger.error(f"GA Gen {generation + 1}: Population empty, cannot select parents. Breaking breeding.")
                    break 
                if parent1 is None or parent2 is None:
                    logger.error(f"GA Gen {generation + 1}: Fallback parent selection also failed. Breaking breeding.")
                    break

            child1 = None
            if random.random() < crossover_rate:
                child1 = crossover_operator(parent1, parent2)
            else:
                child1 = deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)
            
            if random.random() < prob_apply_mutation:
                if child1 is not None:
                    child1 = mutation_operator(child1)
                else:
                    logger.warning(f"GA Gen {generation + 1}: child1 is None before mutation. Using parent1 as fallback.")
                    child1 = deepcopy(parent1) 

            if child1 is not None and child1.is_valid():
                new_population.append(child1)
            else:
                logger.debug(f"GA Gen {generation + 1}: Generated child was invalid or None. Not added.")

        if not new_population:
            logger.error(f"GA Gen {generation + 1}: new_population is empty. Elitism failed or no valid offspring. Stopping.")
            break 

        population = new_population[:population_size] 

        idx_to_fill_from_current = 0
        while len(population) < population_size and new_population: 
            population.append(deepcopy(new_population[idx_to_fill_from_current % len(new_population)]))
            idx_to_fill_from_current +=1
            if idx_to_fill_from_current > len(new_population) * 2: 
                 logger.warning(f"GA Gen {generation + 1}: Could not fill population to size despite fallbacks.")
                 break
        
        if not population:
            logger.critical(f"GA Gen {generation + 1}: Population became critically empty. Stopping algorithm.")
            break

        # --- CORRECTED LOCAL SEARCH SECTION ---
        if local_search_func and (generation + 1) % local_search_config.get("frequency", 5) == 0:
            num_to_improve = local_search_config.get("num_to_improve", max(1, int(population_size * 0.1)))
            
            population.sort(key=lambda x: x.fitness())
            
            logger.info(f"GA Gen {generation + 1}: Applying local search to {min(num_to_improve, len(population))} individuals.")

            for i in range(min(num_to_improve, len(population))):
                solution_to_improve = population[i]
                original_fitness = solution_to_improve.fitness()

                ls_call_params = local_search_config.copy()
                ls_call_params.pop("frequency", None)
                ls_call_params.pop("num_to_improve", None)
                ls_call_params.setdefault('verbose', verbose)

                logger.debug(f"  Applying LS to individual {i} (fitness: {original_fitness:.4f}) with params: {ls_call_params}")
                
                improved_solution_object_from_ls = local_search_func(
                    solution_to_improve,
                    **ls_call_params
                )

                if improved_solution_object_from_ls:
                    improved_fit = improved_solution_object_from_ls.fitness()

                    if improved_fit < original_fitness:
                        population[i] = LeagueSolution(
                            repr=list(improved_solution_object_from_ls.repr),
                            num_teams=problem_params['num_teams'],
                            team_size=problem_params['team_size'],
                            max_budget=problem_params['max_budget'],
                            players=players_data, # Use master players_data for consistency
                            position_requirements=problem_params['position_requirements']
                        )
                        logger.info(f"    LS improved individual {i}. Old fitness: {original_fitness:.4f}, New fitness: {population[i].fitness():.4f}")
                    else:
                        logger.debug(f"    LS did not improve individual {i}. Original: {original_fitness:.4f}, LS attempt: {improved_fit:.4f}")
                else:
                    logger.warning(f"    Local search function returned None for individual {i} (original fitness: {original_fitness:.4f}).")
        # --- END OF CORRECTED LOCAL SEARCH SECTION ---
        
        try:
            current_gen_best_sol = min(population, key=lambda s: s.fitness())
            current_gen_best_fit = current_gen_best_sol.fitness()
        except ValueError:
            logger.error(f"GA Gen {generation + 1}: Population empty. Using overall best for history.")
            current_gen_best_fit = best_fitness_overall
        
        if current_gen_best_fit < best_fitness_overall:
            best_solution_overall = deepcopy(current_gen_best_sol)
            best_fitness_overall = current_gen_best_fit
            if verbose:
                logger.info(f"GA Gen {generation + 1}: New best fitness: {best_fitness_overall:.4f}")
        elif verbose and (generation + 1) % 5 == 0:
            logger.info(f"GA Gen {generation + 1}: Current gen best: {current_gen_best_fit:.4f} (Overall best: {best_fitness_overall:.4f})")
        
        history.append(best_fitness_overall)
    
    if verbose:
        logger.info(f"Genetic Algorithm finished. Final best fitness: {best_fitness_overall:.4f}")
    return best_solution_overall, best_fitness_overall, history

# --- Helper function for Island Model GA ---
import random
from copy import deepcopy
import logging

# Assuming LeagueSolution and InsufficientPlayersForPositionError are imported
# from solution import LeagueSolution, InsufficientPlayersForPositionError
# Assuming selection_boltzmann and default_safe_exp are available if used
# from .operators import selection_boltzmann, default_safe_exp # Or however they are imported

# Get a logger for this module (if not already defined at the top of operators.py)
# logger = logging.getLogger(__name__) # This should be defined at the module level

def _evolve_one_generation(
    current_population: list,
    problem_params: dict, # Contains num_teams, team_size, max_budget, position_requirements
    single_gen_ga_params: dict, # Standard GA params for one generation
    players_data: list,
    # Add logger instance if not globally available or if passing explicitly
    logger_instance: logging.Logger 
) -> list:
    """
    Evolves a population for a single generation using specified GA operators.

    This helper function is primarily used by the Island Model GA. It applies
    selection, crossover, and mutation to the `current_population` based on
    parameters defined in `single_gen_ga_params`. It also handles elitism.

    Args:
        current_population: The list of LeagueSolution instances to evolve.
        problem_params: Dictionary containing problem definition like 'num_teams',
                        'team_size', 'max_budget', 'position_requirements'. Used for
                        potential re-generation of solutions if population shrinks too much.
        single_gen_ga_params: Dictionary with GA parameters for this generation, including:
                              'population_size', 'selection_operator', 'selection_params',
                              'crossover_operator', 'crossover_rate', 'mutation_operator',
                              'mutation_rate', 'elitism_size'.
                              It can also include 'safe_exp_func' if Boltzmann selection is used.
        players_data: The master list of player data, used if new solutions need to be
                      generated from scratch as a fallback.
        logger_instance: An instance of a logger for logging messages.


    Returns:
        A new list of LeagueSolution instances representing the next generation.
        The size of the returned list aims to match `single_gen_ga_params['population_size']`.
    """
    # Ensure necessary classes are available in this scope if not globally imported
    # This is just for clarity; they should be imported at the module level.
    from solution import LeagueSolution, InsufficientPlayersForPositionError
    
    population_size = single_gen_ga_params['population_size']
    
    selection_operator = single_gen_ga_params['selection_operator']
    selection_kwargs = single_gen_ga_params.get('selection_params', {}).copy()
    # Assuming selection_boltzmann is a function object if used
    # from .operators import selection_boltzmann, default_safe_exp # Example import
    if selection_operator.__name__ == 'selection_boltzmann' and 'safe_exp_func' not in selection_kwargs:
        # Ensure default_safe_exp is defined or imported if used as a default
        # For example: from .operators import default_safe_exp
        selection_kwargs['safe_exp_func'] = single_gen_ga_params.get('safe_exp_func', default_safe_exp)

    crossover_operator = single_gen_ga_params['crossover_operator']
    crossover_rate = single_gen_ga_params['crossover_rate']
    
    mutation_operator = single_gen_ga_params['mutation_operator']
    prob_apply_mutation = single_gen_ga_params.get('mutation_rate', 0.1)

    # Elitism is usually controlled by elitism_size. If elitism_size > 0, elitism is active.
    elitism_size = single_gen_ga_params.get('elitism_size', 0)

    if not current_population:
        logger_instance.warning("_evolve_one_generation: Received empty current_population. Cannot evolve.")
        # Attempt to fill with new random solutions if problem_params and players_data are available
        # This is a recovery attempt for an empty island.
        new_pop_attempt = []
        for _ in range(population_size):
            try:
                solution = LeagueSolution(
                    num_teams=problem_params['num_teams'],
                    team_size=problem_params['team_size'],
                    max_budget=problem_params['max_budget'],
                    players=deepcopy(players_data),
                    position_requirements=problem_params['position_requirements']
                )
                if solution.is_valid():
                    new_pop_attempt.append(solution)
            except (ValueError, InsufficientPlayersForPositionError) as e_gen:
                logger_instance.error(f"_evolve_one_generation: Error generating recovery solution: {e_gen}")
        if new_pop_attempt:
             logger_instance.info(f"_evolve_one_generation: Recovered with {len(new_pop_attempt)} new solutions for empty population.")
             return new_pop_attempt
        else:
             logger_instance.error("_evolve_one_generation: Failed to recover empty population with new solutions.")
             return []


    # Sort current population by fitness (ascending, lower is better)
    # This is crucial for elitism and some rank-based selections.
    current_population.sort(key=lambda x: x.fitness())
    
    new_population = []
    if elitism_size > 0:
        # Ensure we don't take more elites than available in the population
        actual_elitism_size = min(elitism_size, len(current_population))
        new_population.extend(deepcopy(current_population[:actual_elitism_size]))
        
    # Main loop to fill the new population
    while len(new_population) < population_size:
        if not current_population: # Should not happen if initial population was not empty
            logger_instance.warning("_evolve_one_generation: current_population became empty during evolution loop. Breaking.")
            break

        # Parent selection
        parent1 = selection_operator(current_population, **selection_kwargs)
        parent2 = selection_operator(current_population, **selection_kwargs)
        
        # Fallback if parent selection fails (e.g., population too small for tournament k)
        if parent1 is None:
            parent1 = deepcopy(random.choice(current_population)) # Choose a random survivor
            logger_instance.debug("_evolve_one_generation: parent1 selection failed, using random survivor.")
        if parent2 is None:
            parent2 = deepcopy(random.choice(current_population)) # Choose another random survivor
            logger_instance.debug("_evolve_one_generation: parent2 selection failed, using random survivor.")

        # Crossover
        child1 = None # Initialize child1
        if random.random() < crossover_rate:
            child1 = crossover_operator(parent1, parent2)
        else:
            # If no crossover, one parent (e.g., the fitter one) moves to the next stage
            # This assumes fitness() is cheap or cached.
            child1 = deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)
            
        # Mutation
        if random.random() < prob_apply_mutation:
            # Ensure child1 is not None before mutation (it shouldn't be with the above logic)
            if child1 is not None:
                child1 = mutation_operator(child1)
            else: # Should ideally not happen
                logger_instance.warning("_evolve_one_generation: child1 was None before mutation attempt. Re-selecting a parent.")
                child1 = deepcopy(parent1) # Fallback to a parent

        # Add to new population if valid
        # Crossover operators with '_prefer_valid' should already return valid or a parent.
        # If using a crossover that can produce invalid children, this check is important.
        if child1 is not None and child1.is_valid():
            new_population.append(child1)
        else:
            # Fallback: if child is invalid or None, add a copy of a good individual to maintain size.
            # This prevents population shrinkage due to invalid offspring.
            logger_instance.debug("_evolve_one_generation: Child was invalid or None. Adding a good individual as fallback.")
            if new_population and elitism_size > 0: # If elites are already in new_population
                new_population.append(deepcopy(new_population[0])) # Add the best elite again
            elif current_population: # Otherwise, add a random individual from current (parent) population
                new_population.append(deepcopy(random.choice(current_population)))
            # If new_population is still empty and current_population is also empty, the loop will break.

        # Safety break if something goes extremely wrong and population_size is huge
        if len(new_population) > population_size * 2: # Arbitrary safety limit
            logger_instance.warning("_evolve_one_generation: New population excessively large. Truncating.")
            break
            
    # Ensure population size is strictly maintained
    final_new_population = new_population[:population_size]
    
    # If underfilled after the loop (e.g., current_population became empty and fallbacks failed)
    # try to fill with more elites or random survivors from the original current_population if it had content.
    # This is a last-ditch effort to maintain size.
    idx_to_fill_from_current = 0
    while len(final_new_population) < population_size and current_population:
        # Cycle through current_population to add diverse individuals if possible
        final_new_population.append(deepcopy(current_population[idx_to_fill_from_current % len(current_population)]))
        idx_to_fill_from_current +=1
        if idx_to_fill_from_current > len(current_population) * 2: # Avoid infinite loop if all are invalid
             logger_instance.warning("_evolve_one_generation: Could not fill population to size despite fallbacks.")
             break


    if len(final_new_population) < population_size:
        logger_instance.warning(f"_evolve_one_generation: Final population size {len(final_new_population)} is less than target {population_size}.")
        # At this point, if it's still underfilled, it might indicate persistent issues
        # with generating valid solutions or very small initial populations.
        # For the island model, returning a smaller population might be acceptable,
        # or it could try to generate brand new random solutions as a last resort.
        # The initial check for empty current_population already tries this.

    return final_new_population

# --- Genetic Algorithm - Island Model ---

import random
import numpy as np
from copy import deepcopy
import logging

# Assuming these are imported correctly at the top of your operators.py file:
from solution import LeagueSolution, InsufficientPlayersForPositionError
# from .operators import _evolve_one_generation, generate_population # If in same file
# from .operators import selection_boltzmann, default_safe_exp # If used by _evolve_one_generation

# This logger should be defined at the module level in operators.py
# logger = logging.getLogger(__name__)
# For this function, we'll assume 'logger' is the module-level logger.

def genetic_algorithm_island_model(
    players_data: list,
    problem_params: dict,
    island_model_ga_params: dict
) -> tuple[LeagueSolution | None, float, list]:
    """
    Implements a Genetic Algorithm with an Island Model.

    Args:
        players_data: List of player data.
        problem_params: Dict with 'num_teams', 'team_size', 'max_budget', 'position_requirements'.
        island_model_ga_params: Dict with island model parameters including:
            'num_islands', 'island_population_size', 'max_generations_total',
            'migration_frequency', 'num_migrants', 'migration_topology',
            'ga_params_per_island' (which can include 'max_initial_solution_attempts'),
            'verbose'.

    Returns:
        Tuple: (best_solution_overall, best_fitness_overall, history_of_global_best_fitness)
    """
    global logger # Assuming logger is a module-level global in operators.py

    logger.info(f"Starting Genetic Algorithm - Island Model with params: {island_model_ga_params}")
    num_islands = island_model_ga_params['num_islands']
    island_population_size = island_model_ga_params['island_population_size']
    max_generations_total = island_model_ga_params['max_generations_total']
    migration_frequency = island_model_ga_params['migration_frequency']
    num_migrants = island_model_ga_params['num_migrants']
    migration_topology = island_model_ga_params.get('migration_topology', 'ring')
    ga_params_per_island = island_model_ga_params['ga_params_per_island'].copy() # Use a copy
    verbose = island_model_ga_params.get('verbose', False)

    # Ensure population_size in ga_params_per_island matches island_population_size
    # This is for the _evolve_one_generation helper
    if ga_params_per_island.get('population_size') != island_population_size:
        logger.warning(
            f"Adjusting ga_params_per_island['population_size'] to {island_population_size} "
            f"to match island_population_size for _evolve_one_generation."
        )
        ga_params_per_island['population_size'] = island_population_size

    # Fetch max_initial_solution_attempts for generate_population,
    # from ga_params_per_island or a top-level island_model_ga_param if preferred.
    # For consistency, let's assume it can be in ga_params_per_island.
    max_init_attempts_for_island_pop = ga_params_per_island.get('max_initial_solution_attempts', 50)


    # Initialize islands
    islands = []
    logger.info(f"Initializing {num_islands} islands, each with population size {island_population_size}.")
    for i in range(num_islands):
        if verbose:
            logger.info(f"Initializing island {i+1}/{num_islands}...")
        try:
            island_pop = generate_population( # Ensure this function is defined/imported
                players_data,
                island_population_size,
                problem_params['num_teams'],
                problem_params['team_size'],
                problem_params['max_budget'],
                problem_params['position_requirements'],
                logger_instance=logger,  # Pass the logger
                max_initial_solution_attempts=max_init_attempts_for_island_pop # Pass attempts
            )
            if not island_pop:
                # generate_population should log extensively if it fails to produce solutions
                logger.error(f"Island {i+1} failed to generate any initial valid solutions. This is critical.")
                # Depending on desired robustness, either raise error or append empty and let logic handle
                # For now, let's be strict as per your original code's intent
                raise RuntimeError(f"Island {i+1} failed to generate any initial valid solutions.")
            islands.append(island_pop)
            logger.debug(f"Island {i+1} initialized with {len(island_pop)} individuals.")
        except RuntimeError as e:
            logger.critical(f"Fatal Error: Could not initialize island {i+1}. Error: {e}")
            raise # Re-raise as it's a critical setup failure.
    
    if not islands or any(not island_pop for island_pop in islands):
        logger.critical("One or more islands could not be initialized with a population. Stopping Island GA.")
        # Return None or raise error if no valid initial state
        return None, float('inf'), []


    # Initial global best
    global_best_solution: LeagueSolution | None = None
    global_best_fitness = float('inf')

    for island_idx, island_pop in enumerate(islands):
        if not island_pop: # Should have been caught by init checks
            logger.warning(f"Island {island_idx+1} is empty after initialization. Skipping for initial best.")
            continue
        try:
            current_island_best = min(island_pop, key=lambda s: s.fitness())
            current_island_best_fitness = current_island_best.fitness()
            if current_island_best_fitness < global_best_fitness:
                global_best_fitness = current_island_best_fitness
                global_best_solution = deepcopy(current_island_best)
        except ValueError: # Should not happen if island_pop is not empty
            logger.error(f"Error finding best in initialized island {island_idx+1}. It might be empty despite checks.")
        except Exception as e_fit:
            logger.error(f"Error calculating fitness for individuals in island {island_idx+1} during initial scan: {e_fit}", exc_info=True)


    history_of_global_best_fitness = [global_best_fitness]
    if global_best_solution:
        logger.info(f"Initial global best fitness: {global_best_fitness:.4f}")
    else:
        logger.warning(f"Initial global best fitness: {global_best_fitness:.4f} (No valid initial solution found across all islands to set as best_solution_overall)")

    # Main evolutionary loop
    logger.info(f"Starting evolution for {max_generations_total} generations.")
    for current_generation in range(max_generations_total):
        log_level_gen = logging.DEBUG if not verbose else logging.INFO # Use DEBUG if not verbose for gen logs
        if (current_generation + 1) % 10 == 0 or current_generation == max_generations_total - 1 or verbose:
             logger.log(log_level_gen, f"Island GA - Gen {current_generation + 1}/{max_generations_total}. Global Best: {global_best_fitness:.4f}")
        
        # Evolution Phase (within each island)
        logger.debug(f"Gen {current_generation + 1}: Evolution phase for {num_islands} islands.")
        for i_island in range(num_islands):
            if not islands[i_island]:
                logger.warning(f"Island {i_island+1} is empty at start of gen {current_generation+1}. Skipping evolution.")
                continue
            
            logger.debug(f"Gen {current_generation + 1}: Evolving island {i_island+1}/{num_islands}.")
            # Ensure _evolve_one_generation is defined/imported
            islands[i_island] = _evolve_one_generation(
                current_population=islands[i_island],
                problem_params=problem_params, # Pass the main problem_params
                single_gen_ga_params=ga_params_per_island, # These are GA params for one gen
                players_data=players_data, # Pass main players_data
                logger_instance=logger # Pass the module-level logger
            )
            if not islands[i_island]:
                logger.warning(f"Island {i_island+1} became empty after evolution in gen {current_generation+1}.")

        # Global Best Update after evolution phase
        logger.debug(f"Gen {current_generation + 1}: Updating global best solution.")
        for island_idx, island_pop_after_evol in enumerate(islands):
            if not island_pop_after_evol:
                continue
            try:
                current_island_best_sol = min(island_pop_after_evol, key=lambda s: s.fitness())
                ind_fitness = current_island_best_sol.fitness()
                if ind_fitness < global_best_fitness:
                    global_best_fitness = ind_fitness
                    global_best_solution = deepcopy(current_island_best_sol)
                    if verbose: # Only log new global best if verbose to reduce noise
                        logger.info(f"Gen {current_generation + 1}: New global best from island {island_idx+1}! Fitness: {global_best_fitness:.4f}")
            except ValueError: # Island population might be empty
                 logger.warning(f"Island {island_idx+1} empty after evolution. Cannot determine island best.")
            except Exception as e_fit_update:
                 logger.error(f"Error updating global best from island {island_idx+1}: {e_fit_update}", exc_info=True)

        history_of_global_best_fitness.append(global_best_fitness)

        # Migration Phase
        if (current_generation + 1) % migration_frequency == 0 and num_migrants > 0 and num_islands > 1:
            logger.info(f"--- Migration event at generation {current_generation + 1} using {migration_topology} topology ---")
            
            # (Migration logic as provided in your snippet, with minor logging adjustments)
            # Ensure this logic correctly handles empty islands before attempting to sort or select migrants.
            if migration_topology == 'ring':
                migrants_to_send = [[] for _ in range(num_islands)]
                for i in range(num_islands):
                    if not islands[i]:
                        logger.warning(f"Migration (Ring): Island {i+1} is empty, cannot select migrants.")
                        continue
                    islands[i].sort(key=lambda s: s.fitness())
                    actual_num_migrants = min(num_migrants, len(islands[i]))
                    if actual_num_migrants > 0:
                        migrants_to_send[i] = deepcopy(islands[i][:actual_num_migrants])
                    logger.debug(f"Island {i+1} selected {len(migrants_to_send[i])} migrants.")

                for i_target in range(num_islands):
                    source_island_idx = (i_target - 1 + num_islands) % num_islands
                    current_migrants = migrants_to_send[source_island_idx]
                    if not current_migrants: continue

                    if not islands[i_target]:
                        islands[i_target] = current_migrants[:island_population_size] # Ensure not oversized
                        logger.info(f"Island {i_target+1} (empty) populated by {len(islands[i_target])} migrants from island {source_island_idx+1}.")
                    else:
                        islands[i_target].sort(key=lambda s: s.fitness(), reverse=True) # Worst first
                        num_to_replace = min(len(current_migrants), len(islands[i_target]))
                        
                        temp_destination_pop = islands[i_target][num_to_replace:]
                        temp_destination_pop.extend(current_migrants) # Add all selected migrants
                        islands[i_target] = sorted(temp_destination_pop, key=lambda s: s.fitness())[:island_population_size] # Keep best, ensure size
                        logger.info(f"Island {i_target+1} received {len(current_migrants)} migrants from island {source_island_idx+1}. Pop size: {len(islands[i_target])}")
            
            elif migration_topology == 'random_pair': # Your snippet had 'random_pair_exchange'
                if num_islands >= 2:
                    idx1, idx2 = random.sample(range(num_islands), 2)
                    logger.debug(f"Migration (Random Pair): Selected Island {idx1+1} and Island {idx2+1}.")
                    
                    # Using your _migrate_between_two helper, ensure it's defined or inline the logic
                    # For now, assuming _migrate_between_two is defined as in your snippet
                    _migrate_between_two(idx1, idx2, islands, num_migrants, island_population_size, logger, log_prefix="  ")
                    _migrate_between_two(idx2, idx1, islands, num_migrants, island_population_size, logger, log_prefix="  ")

            elif migration_topology == 'broadcast_best_to_all':
                # Find current overall best migrant from any island
                current_overall_best_migrant = None
                current_overall_best_migrant_fitness = float('inf')
                source_island_of_best = -1

                for idx_scan, island_pop_scan in enumerate(islands):
                    if not island_pop_scan: continue
                    try:
                        island_best = min(island_pop_scan, key=lambda s:s.fitness())
                        island_best_fit = island_best.fitness()
                        if island_best_fit < current_overall_best_migrant_fitness:
                            current_overall_best_migrant_fitness = island_best_fit
                            current_overall_best_migrant = deepcopy(island_best)
                            source_island_of_best = idx_scan
                    except ValueError: # Empty island
                        continue
                
                if current_overall_best_migrant:
                    logger.debug(f"Migration (Broadcast): Best (fitness {current_overall_best_migrant_fitness:.4f} from island {source_island_of_best+1}) to all.")
                    for i_target_island in range(num_islands):
                        if not islands[i_target_island]:
                            islands[i_target_island] = [deepcopy(current_overall_best_migrant)]
                            logger.info(f"  Island {i_target_island+1} (empty) received broadcast best.")
                            continue

                        # Avoid adding if identical representation already exists
                        already_present = any(sol.repr == current_overall_best_migrant.repr for sol in islands[i_target_island])
                        if not already_present:
                            islands[i_target_island].sort(key=lambda s: s.fitness(), reverse=True) # Worst first
                            if len(islands[i_target_island]) < island_population_size:
                                islands[i_target_island].append(deepcopy(current_overall_best_migrant))
                            elif islands[i_target_island]: # Replace worst if full
                                islands[i_target_island][0] = deepcopy(current_overall_best_migrant)
                            
                            islands[i_target_island].sort(key=lambda s: s.fitness()) # Re-sort
                            # Truncate if somehow overfilled by appending to a full list (should not happen with replace logic)
                            islands[i_target_island] = islands[i_target_island][:island_population_size] 
                            logger.info(f"  Island {i_target_island+1} received broadcast best. Pop size: {len(islands[i_target_island])}")
                        elif verbose:
                            logger.debug(f"  Island {i_target_island+1} already contains broadcast best. Not adding clone.")
                else:
                    logger.warning("Migration (Broadcast): No best migrant found to broadcast.")
            else:
                logger.warning(f"Unknown migration topology: {migration_topology}. Skipping migration.")
        elif num_islands <= 1 and (current_generation + 1) % migration_frequency == 0 :
            logger.debug(f"Migration skipped: Not enough islands ({num_islands}) for migration.")

    logger.info(f"Genetic Algorithm - Island Model finished. Final Global Best Fitness: {global_best_fitness:.4f}")
    return global_best_solution, global_best_fitness, history_of_global_best_fitness

# Helper for random pair migration (if you keep it separate)
# Ensure it's defined before being called by genetic_algorithm_island_model
# or move its logic inline.
def _migrate_between_two(source_idx, target_idx, islands_list, num_m, pop_size, logger_instance, log_prefix=""):
    if not islands_list[source_idx] or len(islands_list[source_idx]) == 0:
        logger_instance.debug(f"{log_prefix}Source island {source_idx+1} is empty. Cannot migrate.")
        return

    actual_num_migrants = min(num_m, len(islands_list[source_idx]))
    if actual_num_migrants == 0:
        logger_instance.debug(f"{log_prefix}Not enough individuals in source island {source_idx+1} to migrate ({len(islands_list[source_idx])} < {num_m} requested, or num_m is 0).")
        return

    islands_list[source_idx].sort(key=lambda s: s.fitness())
    migrants = deepcopy(islands_list[source_idx][:actual_num_migrants])
    
    if not islands_list[target_idx]: # Target empty
        islands_list[target_idx] = migrants[:pop_size] # Ensure not oversized
        logger_instance.info(f"{log_prefix}Migrated {len(islands_list[target_idx])} from Island {source_idx+1} to empty Island {target_idx+1}.")
    else:
        islands_list[target_idx].sort(key=lambda s: s.fitness(), reverse=True) # Worst first
        num_to_replace_in_target = min(len(migrants), len(islands_list[target_idx]))
        
        temp_pop = islands_list[target_idx][num_to_replace_in_target:] # Keep the better ones
        temp_pop.extend(migrants) # Add all selected migrants
        islands_list[target_idx] = sorted(temp_pop, key=lambda s: s.fitness())[:pop_size] # Keep best, ensure size
        logger_instance.info(f"{log_prefix}Migrated {len(migrants)} from Island {source_idx+1} to Island {target_idx+1}. Target pop size: {len(islands_list[target_idx])}")



def crossover_two_point(parent1: LeagueSolution, parent2: LeagueSolution) -> LeagueSolution:
    """
    Performs two-point crossover between two parent solutions.
    Selects two distinct cut points and creates a child by combining segments
    from both parents.
    """
    if not isinstance(parent1, LeagueSolution) or not isinstance(parent2, LeagueSolution): #
        logger.error("Two-point crossover: Invalid parent types. Cannot perform crossover.") #
        # Fallback: return a copy of parent1 if possible, or raise an error
        return deepcopy(parent1) if parent1 else None # Consider raising TypeError

    if len(parent1.repr) != len(parent2.repr): #
        logger.warning("Two-point crossover: Parent representation lengths differ. Returning deepcopy of parent1.") #
        return deepcopy(parent1) #

    n = len(parent1.repr) #
    if n < 3: # Need at least 3 genes to select two distinct internal cut points that result in a mix.
              # If n=2, cut1=0, cut2=1 makes child = P2[0:1] + P1[1:2] which is P2[0]+P1[1].
              # Let's allow n=2, where cut points will be 0 and 1.
        if n < 2: # Cannot perform any meaningful crossover with less than 2 items.
            logger.debug("Two-point crossover: Representation too short (length < 2). Returning deepcopy of parent1.") #
            return deepcopy(parent1) #
        # If n=2, the only distinct points are 0 and 1, leading to child_repr = p2.repr[0] + p1.repr[1]
        # which is a valid crossover for two elements.
        # If n=1, cut1=0, cut2=0 is not possible with cut1 < cut2.
        # So, a length of at least 2 is required for the chosen cut point logic.

    # Select two distinct cut points.
    # cut1 can range from 0 to n-2.
    # cut2 can range from cut1+1 to n-1.
    # This ensures 0 <= cut1 < cut2 < n.
    # Example: n=5. cut1 in [0,1,2,3]. cut2 in [cut1+1, 4].
    # if cut1=0, cut2 in [1,2,3,4].
    # if cut1=3, cut2 in [4].
    
    # To ensure meaningful segment from parent2 (i.e., cut2 > cut1):
    cut1 = random.randint(0, n - 2)  # First cut point index #
    cut2 = random.randint(cut1 + 1, n - 1) # Second cut point index, ensuring cut2 > cut1 #
    
    # Child takes: parent1 up to cut1, parent2 from cut1 to cut2, parent1 from cut2 onwards.
    child_repr = parent1.repr[:cut1] + parent2.repr[cut1:cut2] + parent1.repr[cut2:] #

    # Create the child solution
    child = parent1.__class__( #
        repr=child_repr, #
        num_teams=parent1.num_teams, #
        team_size=parent1.team_size, #
        max_budget=parent1.max_budget, #
        players=parent1.players, # Shared reference, as per LeagueSolution's design #
        position_requirements=parent1.position_requirements # Shared reference #
    )
    child._fitness_cache = None # Ensure fitness is recalculated for the new child #
    return child #

def crossover_two_point_prefer_valid(parent1: LeagueSolution, parent2: LeagueSolution, max_attempts: int = 10) -> LeagueSolution: #
    """
    Attempts two-point crossover up to `max_attempts` times to produce a valid child.
    If unsuccessful, returns a deepcopy of the fitter parent.
    """
    if not isinstance(parent1, LeagueSolution) or not isinstance(parent2, LeagueSolution): #
        logger.error("Prefer valid two-point crossover: Invalid parent types. Returning a copy of parent1 if possible.") #
        return deepcopy(parent1) if parent1 else None # Or handle error more strictly #

    for attempt in range(max_attempts): #
        child = crossover_two_point(parent1, parent2) #
        # It's possible crossover_two_point returned a copy of a parent if repr was too short
        # In that case, is_valid() will be based on that parent.
        if child.is_valid(): #
            logger.debug(f"Prefer valid two-point crossover: Valid child found on attempt {attempt + 1}.") #
            return child #
        else: #
            logger.debug(f"Prefer valid two-point crossover: Attempt {attempt + 1}, child invalid (Fitness: {child.fitness()}).") #
            
    # If max_attempts reached and no valid child found, return the fitter parent
    logger.warning(f"Prefer valid two-point crossover: Max attempts ({max_attempts}) reached to produce a valid child. Returning fitter parent.") #
    fitness1 = parent1.fitness() #
    fitness2 = parent2.fitness() #
    
    # Assuming lower fitness is better
    if fitness1 <= fitness2: #
        return deepcopy(parent1) #
    else: #
        return deepcopy(parent2) #

def hc_wrapper_for_ga(current_ga_solution, max_iterations=20, max_no_improvement=10, verbose=False, **kwargs_for_hc):
    """
    Wrapper to use hill_climbing as a local search method within GA.
    It converts a GA's LeagueSolution to a LeagueHillClimbingSolution, applies HC,
    and returns the (potentially improved) LeagueHillClimbingSolution object.
    """
    if not isinstance(current_ga_solution, LeagueSolution):
        logger.warning("hc_wrapper_for_ga: Received non-LeagueSolution object. Cannot apply HC.")
        return current_ga_solution 

    # Convert GA's LeagueSolution to LeagueHillClimbingSolution for HC
    hc_solution_instance = LeagueHillClimbingSolution(
        repr=list(current_ga_solution.repr), # Use a copy of the repr
        num_teams=current_ga_solution.num_teams,
        team_size=current_ga_solution.team_size,
        max_budget=current_ga_solution.max_budget,
        players=current_ga_solution.players, # players list is shared by reference
        position_requirements=current_ga_solution.position_requirements # shared by reference
    )

    # It's good practice to ensure the solution is valid before applying HC,
    # though GA usually maintains valid solutions.
    if not hc_solution_instance.is_valid():
        logger.warning(f"hc_wrapper_for_ga: Converted solution for HC is invalid (Fitness: {hc_solution_instance.fitness()}). Skipping HC application.")
        return current_ga_solution # Return original GA solution if conversion leads to invalid

    logger.debug(f"hc_wrapper_for_ga: Applying HC (max_iter={max_iterations}, max_no_imp={max_no_improvement}) to solution with initial fitness {hc_solution_instance.fitness()}")
    
    # Apply hill_climbing
    improved_hc_solution, _, _ = hill_climbing(
        initial_solution=hc_solution_instance,
        max_iterations=max_iterations,
        max_no_improvement=max_no_improvement,
        verbose=verbose,
        **kwargs_for_hc # Pass any other specific kwargs for HC or its get_neighbors
    )
    
    logger.debug(f"hc_wrapper_for_ga: HC application finished. Fitness before: {current_ga_solution.fitness()} (approx), Fitness after: {improved_hc_solution.fitness()}")
    # The genetic_algorithm function will take this improved_hc_solution's repr 
    # and create a new LeagueSolution object for its population.
    return improved_hc_solution