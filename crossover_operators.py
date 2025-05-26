import random
import logging
from copy import deepcopy
from solution import LeagueSolution

logger = logging.getLogger(__name__)

def crossover_one_point(parent1, parent2):
    # Ensure parents have representations and they are lists
    if (
        not hasattr(parent1, "repr")
        or not hasattr(parent2, "repr")
        or not isinstance(parent1.repr, list)
        or not isinstance(parent2.repr, list)
        or len(parent1.repr) != len(parent2.repr)
        or len(parent1.repr) < 2
    ):  # Min length for valid cut
        # Fallback: return a copy of the fitter parent if crossover isn't possible
        return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)

    # Cut point must be between 1 and len-1 to ensure segments from both parents
    # random.randint(a,b) includes b. So, for len=2, cut can be 1.
    # For len=L, cut can be from 1 to L-1.
    # Original was (1, len-2), which for len=2 is (1,0) -> error. For len=3, (1,1) -> cut=1.
    # A cut point of 1 means child gets parent1.repr[0] and rest from parent2.
    # A cut point of L-1 means child gets parent1.repr[0...L-2] and parent2.repr[L-1].
    min_cut = 1
    max_cut = len(parent1.repr) - 1
    if (
        min_cut > max_cut
    ):  # e.g. if len(parent1.repr) is 1 (though earlier check prevents)
        return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)
    cut = random.randint(min_cut, max_cut)

    child_repr = parent1.repr[:cut] + parent2.repr[cut:]

    return parent1.__class__(  # Use parent1's class type
        repr=child_repr,
        num_teams=parent1.num_teams,
        team_size=parent1.team_size,
        max_budget=parent1.max_budget,
        players=parent1.players,
        position_requirements=parent1.position_requirements,
    )


def crossover_one_point_prefer_valid(parent1, parent2, max_attempts=10):
    for _ in range(max_attempts):
        child = crossover_one_point(
            parent1, parent2
        )  # This now handles its own creation
        if child.is_valid():
            return child

    # If no valid child found, return a deepcopy of the better parent
    return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)


def crossover_two_point(
    parent1: LeagueSolution, parent2: LeagueSolution
) -> LeagueSolution:
    """
    Performs two-point crossover between two parent solutions.
    Selects two distinct cut points and creates a child by combining segments
    from both parents.
    """
    if not isinstance(parent1, LeagueSolution) or not isinstance(
        parent2, LeagueSolution
    ):  #
        logger.error(
            "Two-point crossover: Invalid parent types. Cannot perform crossover."
        )  #
        # Fallback: return a copy of parent1 if possible, or raise an error
        return deepcopy(parent1) if parent1 else None  # Consider raising TypeError

    if len(parent1.repr) != len(parent2.repr):  #
        logger.warning(
            "Two-point crossover: Parent representation lengths differ. Returning deepcopy of parent1."
        )  #
        return deepcopy(parent1)  #

    n = len(parent1.repr)  #
    if (
        n < 3
    ):  # Need at least 3 genes to select two distinct internal cut points that result in a mix.
        # If n=2, cut1=0, cut2=1 makes child = P2[0:1] + P1[1:2] which is P2[0]+P1[1].
        # Let's allow n=2, where cut points will be 0 and 1.
        if n < 2:  # Cannot perform any meaningful crossover with less than 2 items.
            logger.debug(
                "Two-point crossover: Representation too short (length < 2). Returning deepcopy of parent1."
            )  #
            return deepcopy(parent1)  #
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
    cut2 = random.randint(
        cut1 + 1, n - 1
    )  # Second cut point index, ensuring cut2 > cut1 #

    # Child takes: parent1 up to cut1, parent2 from cut1 to cut2, parent1 from cut2 onwards.
    child_repr = parent1.repr[:cut1] + parent2.repr[cut1:cut2] + parent1.repr[cut2:]  #

    # Create the child solution
    child = parent1.__class__(  #
        repr=child_repr,  #
        num_teams=parent1.num_teams,  #
        team_size=parent1.team_size,  #
        max_budget=parent1.max_budget,  #
        players=parent1.players,  # Shared reference, as per LeagueSolution's design #
        position_requirements=parent1.position_requirements,  # Shared reference #
    )
    child._fitness_cache = None  # Ensure fitness is recalculated for the new child #
    return child  #


def crossover_two_point_prefer_valid(
    parent1: LeagueSolution, parent2: LeagueSolution, max_attempts: int = 10
) -> LeagueSolution:  #
    """
    Attempts two-point crossover up to `max_attempts` times to produce a valid child.
    If unsuccessful, returns a deepcopy of the fitter parent.
    """
    if not isinstance(parent1, LeagueSolution) or not isinstance(
        parent2, LeagueSolution
    ):  #
        logger.error(
            "Prefer valid two-point crossover: Invalid parent types. Returning a copy of parent1 if possible."
        )  #
        return deepcopy(parent1) if parent1 else None  # Or handle error more strictly #

    for attempt in range(max_attempts):  #
        child = crossover_two_point(parent1, parent2)  #
        # It's possible crossover_two_point returned a copy of a parent if repr was too short
        # In that case, is_valid() will be based on that parent.
        if child.is_valid():  #
            logger.debug(
                f"Prefer valid two-point crossover: Valid child found on attempt {attempt + 1}."
            )  #
            return child  #
        else:  #
            logger.debug(
                f"Prefer valid two-point crossover: Attempt {attempt + 1}, child invalid (Fitness: {child.fitness()})."
            )  #

    # If max_attempts reached and no valid child found, return the fitter parent
    logger.warning(
        f"Prefer valid two-point crossover: Max attempts ({max_attempts}) reached to produce a valid child. Returning fitter parent."
    )  #
    fitness1 = parent1.fitness()  #
    fitness2 = parent2.fitness()  #

    # Assuming lower fitness is better
    if fitness1 <= fitness2:  #
        return deepcopy(parent1)  #
    else:  #
        return deepcopy(parent2)  #


def crossover_uniform(parent1, parent2):
    if (
        not hasattr(parent1, "repr")
        or not hasattr(parent2, "repr")
        or len(parent1.repr) != len(parent2.repr)
    ):
        return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)

    child_repr = [
        parent1.repr[i] if random.random() < 0.5 else parent2.repr[i]
        for i in range(len(parent1.repr))
    ]

    return parent1.__class__(
        repr=child_repr,
        num_teams=parent1.num_teams,
        team_size=parent1.team_size,
        max_budget=parent1.max_budget,
        players=parent1.players,
        position_requirements=parent1.position_requirements,
    )


def crossover_uniform_prefer_valid(parent1, parent2, max_attempts=10):
    for _ in range(max_attempts):
        child = crossover_uniform(parent1, parent2)
        if child.is_valid():
            return child
    return deepcopy(parent1 if parent1.fitness() <= parent2.fitness() else parent2)
