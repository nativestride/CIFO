# solution.py

from abc import ABC, abstractmethod
import random
import numpy as np
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class InsufficientPlayersForPositionError(Exception):
    """Custom exception for errors when player positional quotas cannot be met."""
    pass


class Solution(ABC):
    """
    Abstract base class for all solutions.
    Defines the interface that all solution classes must implement.
    """
    def __init__(self, repr=None):
        if repr is None:
            # This will call the random_initial_representation of the subclass
            repr = self.random_initial_representation()
        self.repr = repr

    def __repr__(self):
        return str(self.repr)

    @abstractmethod
    def fitness(self):
        pass

    @abstractmethod
    def random_initial_representation(self):
        pass


class LeagueSolution(Solution):
    """
    Solution class for the Sports League optimization problem.
    A solution is represented as a list of team assignments for each player.
    `repr[player_index] = team_id`.
    """
    DEFAULT_POS_REQ_TEAM_SIZE_7 = {"GK": 1, "DEF": 2, "MID": 2, "FWD": 2}

    def __init__(self, repr=None, num_teams=5, team_size=7, max_budget=750.0, players=None,
                 position_requirements=None):
        
        self.num_teams = num_teams
        self.team_size = team_size
        self.max_budget = float(max_budget) # Ensure float
        
        if players is None:
            logger.warning("LeagueSolution initialized without player data. Some operations might fail.")
            self.players = []
        else:
            self.players = players # Assumed to be a reference to a static list of player dicts

        if position_requirements is None:
            if self.team_size == 7: # Default for the common case
                self.position_requirements = deepcopy(self.DEFAULT_POS_REQ_TEAM_SIZE_7)
                logger.debug(f"Using default position requirements for team_size=7: {self.position_requirements}")
            else:
                raise ValueError(
                    f"position_requirements must be provided if team_size ({self.team_size}) is not 7, "
                    "as no default is defined for this specific team size."
                )
        else:
            self.position_requirements = position_requirements

        # Critical validation: Ensure position requirements sum up to the defined team size.
        if sum(self.position_requirements.values()) != self.team_size:
            raise ValueError(
                f"Configuration Error: Sum of players in position_requirements "
                f"({sum(self.position_requirements.values())}) does not match team_size ({self.team_size})."
            )
        
        # Player data and other params need to be set *before* super().__init__
        # because super().__init__ might call random_initial_representation.
        # The call to super().__init__ will trigger random_initial_representation if repr is None.
        # random_initial_representation might modify self.players (if sampling).
        super().__init__(repr=repr) 
        self._fitness_cache = None # Initialize fitness cache

    def __deepcopy__(self, memo):
        """
        Custom deepcopy to share the 'players' list and 'position_requirements'
        by reference (as they are typically static problem data), and deepcopy
        other mutable elements like 'repr' and any cache.
        """
        cls = self.__class__
        result = cls.__new__(cls) # Create a new instance without calling __init__
        memo[id(self)] = result

        # Deepcopy attributes that need to be independent for each solution instance
        result.repr = deepcopy(self.repr, memo)
        result._fitness_cache = deepcopy(self._fitness_cache, memo) # If fitness caching is implemented

        # Assign immutable attributes directly (or copy, doesn't significantly matter for them)
        result.num_teams = self.num_teams
        result.team_size = self.team_size
        result.max_budget = self.max_budget

        # Share 'players' list and 'position_requirements' dict by reference
        # This assumes these are part of the static problem definition and not modified per solution.
        result.players = self.players
        result.position_requirements = self.position_requirements

        return result

    def random_initial_representation(self) -> list:
        """
        Generates a random initial player-to-team assignment (representation)
        that guarantees team size and positional quotas are met for each team.
        Budget constraint is NOT guaranteed by this initialization method.

        The representation 'repr' is a list where index corresponds to player_id
        (i.e., index in `self.players` list after potential sampling) and the value is the team_id
        (0 to `num_teams`-1) they are assigned to.
        If `self.players` initially has more players than `num_teams * team_size`,
        `self.players` will be resampled to the required size.
        """
        num_total_players_to_assign = self.num_teams * self.team_size

        if not self.players:
            # This case should ideally be prevented by checks in __init__ or before calling.
            # If players list is empty, we cannot proceed.
            raise ValueError("Cannot generate random representation: 'self.players' is empty.")

        if len(self.players) < num_total_players_to_assign:
            raise ValueError(
                f"Insufficient players: Player data has {len(self.players)} players, "
                f"but {num_total_players_to_assign} are required "
                f"({self.num_teams} teams * {self.team_size} players/team)."
            )
        elif len(self.players) > num_total_players_to_assign:
            logger.info(
                f"Player data has {len(self.players)} players, which is more than the "
                f"{num_total_players_to_assign} required. Sampling {num_total_players_to_assign} players."
            )
            # IMPORTANT: self.players is modified here to be the sampled subset.
            # This means the representation indices will align with this new self.players list.
            self.players = random.sample(self.players, num_total_players_to_assign)
        # If len(self.players) == num_total_players_to_assign, use self.players as is.

        active_players = self.players # self.players is now correctly sized.
        
        # Initialize representation: value -1 indicates player at this index (in active_players) is not yet assigned
        player_to_team_assignment = [-1] * num_total_players_to_assign # Should match len(active_players)

        # 1. Group available player indices by their position (indices relative to active_players)
        available_player_indices_by_pos = {pos: [] for pos in self.position_requirements.keys()}
        unrecognized_positions_found = False
        for i, player_data in enumerate(active_players): # Iterate through the (potentially sampled) active_players
            pos = player_data.get("Position")
            if pos in available_player_indices_by_pos:
                available_player_indices_by_pos[pos].append(i) # Store index from active_players
            else:
                # This player has a position not listed in self.position_requirements
                # They cannot be assigned by this constrained method.
                # This implies an issue with player data or position_requirements definition.
                logger.warning(f"Player {i} (Name: {player_data.get('Name', 'N/A')}) has unrecognized position '{pos}'. "
                               f"This player cannot be assigned by constrained initialization.")
                unrecognized_positions_found = True
        
        if unrecognized_positions_found:
             logger.warning("Some players have positions not in defined requirements and won't be assigned by this method if not handled.")


        # 2. Shuffle players within each position group to ensure random selection
        for pos_group in available_player_indices_by_pos:
            random.shuffle(available_player_indices_by_pos[pos_group])

        # 3. Assign players to teams, filling positional quotas
        assigned_player_indices = set() # To keep track of players already assigned

        for team_id_current in range(self.num_teams):
            players_assigned_to_this_team = 0
            for position_needed, count_needed in self.position_requirements.items():
                for _ in range(count_needed): # For each slot of this position on this team
                    if available_player_indices_by_pos.get(position_needed): # If list exists and is not empty
                        try:
                            # Pop a player index from the available list for this position
                            player_idx_to_assign = available_player_indices_by_pos[position_needed].pop()
                            
                            # Check if already assigned (shouldn't happen if popping correctly from distinct lists)
                            if player_idx_to_assign in assigned_player_indices:
                                logger.error(f"Logic error: Player index {player_idx_to_assign} was about to be assigned twice.")
                                # This indicates a problem, potentially try to find another player
                                continue # Or handle more robustly

                            player_to_team_assignment[player_idx_to_assign] = team_id_current
                            assigned_player_indices.add(player_idx_to_assign)
                            players_assigned_to_this_team += 1
                        except IndexError:
                            # Ran out of players for this specific position before filling all team quotas
                            raise InsufficientPlayersForPositionError(
                                f"Insufficient players of position '{position_needed}' to fill all teams. "
                                f"Team {team_id_current} could not be completed for this position. "
                                "Check total player counts per position in your input player data vs. requirements."
                            )
                    else: # No players (left) for this position or list for position doesn't exist
                        raise InsufficientPlayersForPositionError(
                            f"No players available or list exhausted for position '{position_needed}' "
                            f"when trying to assign to team {team_id_current}."
                        )
        
        # Final check: ensure all assigned players were unique and total count is correct
        # num_total_players_to_assign is len(active_players) at this point.
        if len(assigned_player_indices) != num_total_players_to_assign:
            # This condition should ideally be caught by the -1 check below or indicates a logic flaw.
            logger.warning(
                f"Number of uniquely assigned players ({len(assigned_player_indices)}) "
                f"does not match total needed ({num_total_players_to_assign}). "
                "This may indicate a flaw in assignment logic or an unhandled edge case."
            )

        # Verify that all entries in player_to_team_assignment got updated from -1
        if -1 in player_to_team_assignment:
            unassigned_indices = [i for i, team_id in enumerate(player_to_team_assignment) if team_id == -1]
            # This could happen if unrecognized_positions_found was true and those players were needed,
            # or if a position quota couldn't be met for reasons not caught above (e.g. miscount).
            raise InsufficientPlayersForPositionError(
                f"Failed to assign all players. Unassigned player indices (relative to active_players): {unassigned_indices}. "
                "This may be due to players with unrecognized positions or other logic flaws."
            )

        return player_to_team_assignment

    def is_valid(self) -> bool:
        """
        Checks if the current solution representation respects all defined constraints:
        1. Correct number of players per team (team_size).
        2. Correct distribution of player positions per team (position_requirements).
        3. Total salary of each team is within max_budget.
        Assumes self.repr is a list of team_id assignments for each player index.
        """
        if not self.players:
            logger.debug("is_valid: False (no player data).")
            return False
        
        # Ensure representation length matches the number of players it's supposed to assign
        # Typically, this would be num_teams * team_size if all players from a specific pool are assigned.
        # Or, if self.players can be larger, len(self.repr) should match the number of players *being considered*.
        # Let's assume for validity, the repr length should cover exactly the players needed.
        expected_repr_length = self.num_teams * self.team_size
        if len(self.repr) != expected_repr_length:
            logger.debug(f"is_valid: False (repr length {len(self.repr)} != expected {expected_repr_length}).")
            return False

        # Check if self.players has enough entries for the representation indices
        if len(self.players) < expected_repr_length:
            logger.debug(f"is_valid: False (not enough players in self.players ({len(self.players)}) for repr length {expected_repr_length}).")
            return False

        teams_player_indices = [[] for _ in range(self.num_teams)]
        try:
            for player_idx, team_id in enumerate(self.repr): # player_idx is 0 to len(repr)-1
                if not (0 <= team_id < self.num_teams): # Check if team_id is valid
                    logger.debug(f"is_valid: False (invalid team_id {team_id} for player_idx {player_idx}).")
                    return False
                teams_player_indices[team_id].append(player_idx)
        except TypeError: # If self.repr is not iterable or contains non-integers
            logger.debug("is_valid: False (TypeError iterating over repr).")
            return False

        # Check each team for constraints
        for team_id_check, current_team_player_indices_list in enumerate(teams_player_indices):
            # 1. Check team size
            if len(current_team_player_indices_list) != self.team_size:
                logger.debug(f"is_valid: False (Team {team_id_check} size {len(current_team_player_indices_list)} != expected {self.team_size}).")
                return False

            # 2. Check position distribution and budget
            current_team_actual_positions = {pos: 0 for pos in self.position_requirements.keys()}
            current_team_total_salary = 0.0
            
            for player_idx_in_team in current_team_player_indices_list:
                # Player index must be valid for self.players list
                if not (0 <= player_idx_in_team < len(self.players)):
                     logger.debug(f"is_valid: False (Player index {player_idx_in_team} in team {team_id_check} is out of bounds for self.players).")
                     return False 
                
                player_data_dict = self.players[player_idx_in_team]
                player_actual_pos = player_data_dict.get("Position")

                if player_actual_pos not in current_team_actual_positions:
                    # This means a player has a position not defined in self.position_requirements.
                    # Such a solution should be invalid if strict adherence to defined positions is required.
                    logger.debug(f"is_valid: False (Team {team_id_check} has player {player_idx_in_team} with unrecognized position '{player_actual_pos}').")
                    return False
                current_team_actual_positions[player_actual_pos] += 1
                
                # Get salary, defaulting to 0 if missing or not convertible to float
                salary = 0.0
                try:
                    salary = float(player_data_dict.get("Salary", player_data_dict.get("Salary (€M)", 0.0)))
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse salary for player {player_idx_in_team}. Using 0.0.")
                current_team_total_salary += salary

            # Check positional quotas
            if current_team_actual_positions != self.position_requirements:
                logger.debug(f"is_valid: False (Team {team_id_check} positions {current_team_actual_positions} != expected {self.position_requirements}).")
                return False
            
            # Check budget
            if current_team_total_salary > self.max_budget:
                logger.debug(f"is_valid: False (Team {team_id_check} salary {current_team_total_salary} > budget {self.max_budget}).")
                return False
        
        logger.debug("is_valid: True.")
        return True

    def fitness(self) -> float:
        """
        Calculates the fitness of the solution. Lower fitness is better (minimization).
        Returns float('inf') for invalid solutions.
        Uses a cache to avoid re-computation if fitness has already been calculated.
        """
        # Use cached fitness if available
        if self._fitness_cache is not None:
            return self._fitness_cache

        if not self.is_valid(): # Crucial first check
            self._fitness_cache = float("inf") # Cache result for invalid solution
            return self._fitness_cache

        # Initialize list of lists to store skills for each team
        team_skills_values_list = [[] for _ in range(self.num_teams)]
        
        # Populate team skills (assuming repr is valid and players list is accessible)
        for player_idx, team_id in enumerate(self.repr):
            # Ensure player_idx is within bounds of self.players (should be if is_valid passed)
            if 0 <= player_idx < len(self.players):
                # Ensure player has "Skill" attribute, handle if missing
                try:
                    team_skills_values_list[team_id].append(self.players[player_idx]["Skill"])
                except KeyError:
                    logger.error(f"Fitness calculation error: Player index {player_idx} (Name: {self.players[player_idx].get('Name', 'N/A')}) is missing 'Skill' attribute.")
                    self._fitness_cache = float("inf") # Mark as invalid due to data error
                    return self._fitness_cache
            else:
                # This case should ideally be caught by is_valid if repr length matches expected
                logger.error(f"Fitness calculation error: Player index {player_idx} out of bounds. Solution likely invalid.")
                self._fitness_cache = float("inf") # Mark as invalid due to error
                return self._fitness_cache

        avg_team_skills = []
        for skills_in_one_team in team_skills_values_list:
            if not skills_in_one_team: 
                # This implies an empty team, which should be caught by is_valid's team_size check
                # unless team_size can be 0.
                logger.warning("Fitness calculation: An empty team's skill list encountered, though solution was valid. This is unexpected.")
                avg_team_skills.append(0.0) 
            else:
                avg_team_skills.append(np.mean(skills_in_one_team))
        
        # Calculate standard deviation of the average team skills
        calculated_fitness = np.std(avg_team_skills)
        self._fitness_cache = calculated_fitness # Cache the calculated fitness
        return calculated_fitness

    def get_teams(self) -> list[list[dict]]:
        """
        Groups players by their assigned teams based on the current representation.
        Returns a list of teams, where each team is a list of player dictionaries.
        """
        teams_by_player_data = [[] for _ in range(self.num_teams)]
        if not self.players: # If no player data, return list of empty teams
            logger.warning("get_teams called but self.players is empty or None.")
            return teams_by_player_data

        for player_idx, team_id in enumerate(self.repr):
            # Basic bounds checking for safety, though is_valid should ensure consistency
            if 0 <= player_idx < len(self.players) and 0 <= team_id < self.num_teams:
                 teams_by_player_data[team_id].append(self.players[player_idx])
            # else:
                 # logger.warning(f"get_teams: Invalid player_idx ({player_idx}) or team_id ({team_id}) in representation.")
        return teams_by_player_data
    
    def get_team_stats(self) -> list[dict]:
        """
        Calculates and returns detailed statistics for each team in the current solution.
        Statistics include average skill, total salary, and position counts.
        """
        teams_of_player_dicts = self.get_teams() # Get list of teams, where each team is list of player dicts
        list_of_team_stats = []
        
        for i, single_team_player_list in enumerate(teams_of_player_dicts):
            current_team_stats_dict = {
                "team_id": i,
                "num_players": len(single_team_player_list),
                "avg_skill": 0.0,
                "total_salary": 0.0,
                "positions": {pos: 0 for pos in self.position_requirements.keys()}, # Initialize with all required positions
                "players_names": [p.get("Name", f"Player_{idx}") for idx, p in enumerate(single_team_player_list)] # Example: list of names
                # "players_full_data": single_team_player_list # Optionally include full player data per team
            }

            if not single_team_player_list: # If team is empty
                list_of_team_stats.append(current_team_stats_dict)
                continue # Move to the next team

            # Calculate stats for non-empty team
            current_team_stats_dict["avg_skill"] = np.mean([p["Skill"] for p in single_team_player_list])
            
            current_team_total_salary = 0.0
            for p_dict in single_team_player_list:
                try:
                    current_team_total_salary += float(p_dict.get("Salary", p_dict.get("Salary (€M)", 0.0)))
                except (ValueError, TypeError):
                    logger.warning(f"Non-numeric salary for player {p_dict.get('Name', 'Unknown')} in team {i}. Treating as 0.")
            current_team_stats_dict["total_salary"] = current_team_total_salary
            
            for p_dict in single_team_player_list:
                player_actual_pos = p_dict.get("Position")
                if player_actual_pos in current_team_stats_dict["positions"]: # Only count recognized positions
                    current_team_stats_dict["positions"][player_actual_pos] += 1
            
            list_of_team_stats.append(current_team_stats_dict)
            
        return list_of_team_stats


class LeagueHillClimbingSolution(LeagueSolution):
    """
    Extends LeagueSolution for Hill Climbing, providing neighbor generation.
    Includes multiple strategies for generating neighbors to improve validity and exploration.
    """
    def get_neighbors(self, max_neighbors_total=None): # Optional: limit total neighbors
        """
        Generate valid neighboring solutions using multiple strategies:
        1. Swap players of the same position between different teams.
        2. Attempt to fix positional deficits/surpluses by targeted (any-to-any player) swaps between two teams.

        Args:
            max_neighbors_total (int, optional): If set, limits the total number of
                                               valid neighbors returned.

        Returns:
            list: A list of valid neighbor LeagueHillClimbingSolution objects.
        """
        all_potential_neighbors = []
        if not self.players:
            logger.warning("get_neighbors called on solution with no player data.")
            return all_potential_neighbors

        # --- Strategy 1: Swap players of the same position between different teams ---
        same_pos_swap_neighbors = self._generate_same_position_swaps()
        all_potential_neighbors.extend(same_pos_swap_neighbors)
        logger.debug(f"Generated {len(same_pos_swap_neighbors)} potential neighbors via same-position swaps.")

        # --- Strategy 2: Attempt to fix positional counts by general two-player swaps between two teams ---
        # This tries swapping any player from team A with any player from team B.
        general_fix_swap_neighbors = self._generate_general_inter_team_swaps()
        all_potential_neighbors.extend(general_fix_swap_neighbors)
        logger.debug(f"Generated {len(general_fix_swap_neighbors)} potential neighbors via general inter-team swaps.")
        
        # Filter for unique and valid neighbors
        valid_and_unique_neighbors = []
        seen_repr_tuples = set() # Use a set of tuples (hashable) of repr for uniqueness

        for neighbor_candidate in all_potential_neighbors:
            if neighbor_candidate.is_valid(): # Check validity first
                repr_tuple = tuple(neighbor_candidate.repr)
                if repr_tuple not in seen_repr_tuples:
                    valid_and_unique_neighbors.append(neighbor_candidate)
                    seen_repr_tuples.add(repr_tuple)
        
        logger.debug(f"Total unique and valid neighbors generated: {len(valid_and_unique_neighbors)}")
        
        # If a limit is set, sample randomly from the valid unique neighbors
        if max_neighbors_total is not None and len(valid_and_unique_neighbors) > max_neighbors_total:
            logger.debug(f"Sampling {max_neighbors_total} from {len(valid_and_unique_neighbors)} valid unique neighbors.")
            return random.sample(valid_and_unique_neighbors, max_neighbors_total)
        
        random.shuffle(valid_and_unique_neighbors) # Shuffle before returning
        return valid_and_unique_neighbors

    def _generate_same_position_swaps(self) -> list:
        """Helper: Generates neighbors by swapping same-position players between different teams."""
        potential_neighbors = []
        player_indices_by_pos = {} # Map: position_str -> list_of_player_indices_with_that_pos
        for idx, player_data in enumerate(self.players): # Assumes self.players has all players being assigned
            pos = player_data["Position"]
            if pos not in player_indices_by_pos: player_indices_by_pos[pos] = []
            player_indices_by_pos[pos].append(idx)
        
        for _, indices_in_pos_category in player_indices_by_pos.items():
            num_players_in_cat = len(indices_in_pos_category)
            if num_players_in_cat < 2: continue # Need at least two to swap

            for i in range(num_players_in_cat):
                for j in range(i + 1, num_players_in_cat):
                    player1_original_idx = indices_in_pos_category[i]
                    player2_original_idx = indices_in_pos_category[j]
                    
                    # Ensure players are currently assigned to different teams
                    if self.repr[player1_original_idx] != self.repr[player2_original_idx]:
                        new_repr_list = list(self.repr) # Create a mutable copy
                        # Swap team assignments for these two players
                        new_repr_list[player1_original_idx], new_repr_list[player2_original_idx] = \
                            new_repr_list[player2_original_idx], new_repr_list[player1_original_idx]
                        
                        # Create a new solution object of the same type as self
                        neighbor_sol = self.__class__( # Instantiates LeagueHillClimbingSolution
                            repr=new_repr_list,
                            num_teams=self.num_teams, team_size=self.team_size,
                            max_budget=self.max_budget, players=self.players,
                            position_requirements=self.position_requirements
                        )
                        potential_neighbors.append(neighbor_sol)
                        # Validity will be checked by the main get_neighbors() method
        return potential_neighbors

    def _generate_general_inter_team_swaps(self) -> list:
        """
        Helper: Generates neighbors by swapping ANY player from team A with ANY player from team B.
        This is a more disruptive move, potentially fixing major positional imbalances if the
        resulting configuration is valid.
        """
        potential_neighbors = []
        # Iterate over all unique pairs of distinct teams
        for team_a_id in range(self.num_teams):
            for team_b_id in range(team_a_id + 1, self.num_teams):
                # Get player indices currently in team A and team B
                team_a_player_indices_in_repr = [idx for idx, t_id in enumerate(self.repr) if t_id == team_a_id]
                team_b_player_indices_in_repr = [idx for idx, t_id in enumerate(self.repr) if t_id == team_b_id]

                if not team_a_player_indices_in_repr or not team_b_player_indices_in_repr:
                    continue # One of the teams is empty, no swap possible

                # Try swapping every player from team A with every player from team B
                for p1_main_idx in team_a_player_indices_in_repr:
                    for p2_main_idx in team_b_player_indices_in_repr:
                        new_repr_list = list(self.repr)
                        # Swap the team assignments: p1 now goes to team B, p2 now goes to team A
                        new_repr_list[p1_main_idx], new_repr_list[p2_main_idx] = \
                            new_repr_list[p2_main_idx], new_repr_list[p1_main_idx]
                        
                        neighbor_sol = self.__class__(
                            repr=new_repr_list, num_teams=self.num_teams, team_size=self.team_size,
                            max_budget=self.max_budget, players=self.players,
                            position_requirements=self.position_requirements
                        )
                        potential_neighbors.append(neighbor_sol)
                        # Validity will be checked by the main get_neighbors() method
        return potential_neighbors


class LeagueSASolution(LeagueSolution):
    """
    Extends LeagueSolution for Simulated Annealing, providing random neighbor generation.
    """
    def get_random_neighbor(self):
        """
        Generate a random valid neighboring solution.
        Tries a same-position swap first, then a more general swap if that fails
        or with a small probability.
        """
        if not self.players or len(self.repr) < 2: # Cannot generate neighbors
            logger.debug("SA get_random_neighbor: No players or not enough players for a swap.")
            return self.__class__(repr=list(self.repr), num_teams=self.num_teams, team_size=self.team_size,
                                  max_budget=self.max_budget, players=self.players,
                                  position_requirements=self.position_requirements)

        max_attempts_to_find_valid = 50 # Try a few times to find a valid move
        for _ in range(max_attempts_to_find_valid):
            new_repr_list = list(self.repr) # Start with a copy for modification
            
            # Choose a move type: 80% chance for same-position, 20% for general swap,
            # or if same-position isn't feasible.
            try_same_position_swap = random.random() < 0.8

            if try_same_position_swap:
                player_indices_by_pos = {}
                for idx, player_data in enumerate(self.players):
                    pos = player_data["Position"]
                    if pos not in player_indices_by_pos: player_indices_by_pos[pos] = []
                    player_indices_by_pos[pos].append(idx)
                
                swappable_positions = [p for p, ilist in player_indices_by_pos.items() if len(ilist) >= 2]
                
                if swappable_positions:
                    chosen_pos = random.choice(swappable_positions)
                    p1_idx, p2_idx = random.sample(player_indices_by_pos[chosen_pos], 2)
                    
                    if new_repr_list[p1_idx] != new_repr_list[p2_idx]: # Ensure different teams for a meaningful swap
                        new_repr_list[p1_idx], new_repr_list[p2_idx] = new_repr_list[p2_idx], new_repr_list[p1_idx]
                    else:
                        try_same_position_swap = False # Same team, not useful, fall through to general swap
                else:
                    try_same_position_swap = False # No positions eligible for this type of swap
            
            if not try_same_position_swap: # Fallback to general two-player swap
                idx1, idx2 = random.sample(range(len(new_repr_list)), 2)
                # This general swap could be between any two players, regardless of team or position
                new_repr_list[idx1], new_repr_list[idx2] = new_repr_list[idx2], new_repr_list[idx1]

            # Create and validate the neighbor from the modified representation
            neighbor_candidate = self.__class__(
                repr=new_repr_list, num_teams=self.num_teams, team_size=self.team_size,
                max_budget=self.max_budget, players=self.players,
                position_requirements=self.position_requirements
            )
            if neighbor_candidate.is_valid():
                logger.debug("SA get_random_neighbor: Found valid random neighbor.")
                return neighbor_candidate
        
        # If max_attempts reached and no valid neighbor found, return a new copy of self
        logger.debug("SA get_random_neighbor: Max attempts reached, returning copy of current solution.")
        return self.__class__(repr=list(self.repr), num_teams=self.num_teams, team_size=self.team_size,
                              max_budget=self.max_budget, players=self.players,
                              position_requirements=self.position_requirements)