import random
import logging
from solution import LeagueSolution
from experiment_utils import safe_exp

logger = logging.getLogger(__name__)

def selection_tournament(population, k=3):  # k can be passed via selection_params
    # Ensure k is not larger than population size
    actual_k = min(k, len(population))
    if actual_k <= 0:  # Handle empty population or invalid k
        return random.choice(population) if population else None

    selected_candidates = random.sample(population, actual_k)
    # Sort by fitness (assuming lower is better)
    selected_candidates.sort(key=lambda sol: sol.fitness())
    return selected_candidates[0]


def selection_tournament_variable_k(
    population: list, k: int = 3, k_percentage: float = None
) -> "LeagueSolution":
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
        logger.warning(
            "selection_tournament_variable_k: Population is empty. Cannot select."
        )
        return None

    actual_k = 0
    if k_percentage is not None and 0 < k_percentage <= 1:
        # Calculate k based on percentage, ensuring it's at least 1
        actual_k = max(1, int(len(population) * k_percentage))
        logger.debug(
            f"Tournament selection: Using k_percentage {k_percentage*100:.1f}%, actual_k={actual_k} from pop_size={len(population)}"
        )
    else:
        actual_k = k
        logger.debug(
            f"Tournament selection: Using fixed k={k}, actual_k={actual_k} (k_percentage not used or invalid)"
        )

    # Ensure k is not larger than population size
    actual_k = min(actual_k, len(population))

    # If actual_k became 0 (e.g., from a very small k_percentage on a small pop, before max(1,...)
    # or if initial k was 0), and population is not empty, set to 1.
    if actual_k == 0 and len(population) > 0:
        actual_k = 1
        logger.debug(
            f"Tournament selection: actual_k was 0, adjusted to 1 for non-empty population."
        )
    elif actual_k == 0:  # Population was empty, already handled, but good to be safe.
        logger.warning(
            "selection_tournament_variable_k: actual_k is 0 and population is empty. Cannot select."
        )
        return None

    tournament_participants = random.sample(population, actual_k)

    # Return the fittest individual from the tournament (assuming minimization problem)
    # Ensure all participants have a fitness value calculated.
    try:
        winner = min(tournament_participants, key=lambda x: x.fitness())
    except Exception as e:
        logger.error(
            f"Error during fitness comparison in tournament selection: {e}",
            exc_info=True,
        )
        # Fallback: return a random participant if fitness comparison fails
        winner = (
            random.choice(tournament_participants) if tournament_participants else None
        )

    return winner


def selection_ranking(population):
    if not population:
        return None
    # Sort by fitness (lower is better, so best individuals come first)
    sorted_pop = sorted(population, key=lambda s: s.fitness())

    # Ranks: best is rank N, second best is N-1, ..., worst is 1 (for probability assignment)
    # This means the individual at sorted_pop[0] (best fitness) gets rank N.
    # sorted_pop[N-1] (worst fitness) gets rank 1.
    num_individuals = len(sorted_pop)
    # Assign ranks such that higher rank value means better for probability calculation
    # (e.g. best gets rank 'num_individuals', worst gets rank '1')
    # ranks = list(range(1, num_individuals + 1))  # ranks [1, 2, ..., N] # This line is unused

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
    if num_individuals == 0:
        return None

    # Calculate sum of ranks (1 to N) = N*(N+1)/2
    sum_of_ranks = num_individuals * (num_individuals + 1) / 2.0
    if sum_of_ranks == 0:  # only if num_individuals is 0, handled above
        return random.choice(sorted_pop) if sorted_pop else None

    # Probabilities for sorted_pop: best (index 0) gets highest prob.
    # Probability for sorted_pop[i] is (N-i) / sum_of_ranks
    probabilities = [
        (num_individuals - i) / sum_of_ranks for i in range(num_individuals)
    ]

    return random.choices(sorted_pop, weights=probabilities, k=1)[0]


def selection_boltzmann(population, temperature=1.0, safe_exp_func=safe_exp):
    if not population:
        return None

    fitness_values = [sol.fitness() for sol in population]

    # Handle inf fitness: give them zero chance of being selected directly
    # For minimization: lower fitness is better. We need to transform fitness so higher value = higher prob.
    # Option 1: Max_fitness - fitness (if max_fitness is known/estimable)
    # Option 2: 1 / (fitness + epsilon) (as original)

    # Using 1 / (fitness + epsilon)
    # If fitness is inf, 1/inf -> 0. If fitness is 0, 1/epsilon.
    inverted_fitness = []
    for f_val in fitness_values:
        if f_val == float("inf"):
            inverted_fitness.append(
                0.0
            )  # Zero contribution to selection if infinitely bad
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

        boltzmann_exp_values = [
            safe_exp_func(f / temperature) for f in inverted_fitness
        ]
    except OverflowError:  # Should be caught by safe_exp_func, but as a fallback
        # Handle extreme case: if safe_exp still overflows (e.g. bad temperature or extreme fitness values)
        # Fallback to uniform random choice or ranking
        # For now, let it propagate if safe_exp fails, or implement robust fallback.
        # A simple fallback: if sum is 0, use uniform.
        # This part might need more robust handling for extreme numerical inputs.
        # If all fitness are inf, all inverted_fitness are 0, all boltzmann_exp_values might be equal (e.g., 1 from exp(0))
        # If temperature is near 0, exponent can be huge.
        # safe_exp_func should prevent overflow in exp, but division by temperature can still be an issue.
        pass  # Assuming safe_exp_func handles it.

    total_boltzmann_value = sum(boltzmann_exp_values)

    if total_boltzmann_value == 0:  # All fitness were inf, or exp values became 0
        # Fallback to uniform random selection if all probabilities are zero
        return random.choice(population) if population else None

    probabilities = [b_val / total_boltzmann_value for b_val in boltzmann_exp_values]

    return random.choices(population, weights=probabilities, k=1)[0]
