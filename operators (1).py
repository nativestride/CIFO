# This file has been refactored.
# Operators and algorithms have been moved to dedicated files:
# - mutation_operators.py
# - crossover_operators.py
# - selection_operators.py
# - genetic_algorithms.py
# - ga_island_model.py
# - ga_utilities.py

# Please update imports in other files to point to these new locations.
# For example: from mutation_operators import mutate_swap

# (Retain any essential base configurations or shared utilities if absolutely necessary,
# but the goal is to have operators directly imported from their new modules.)

# It's also a good practice to define __all__ if this module were to re-export symbols,
# but for this refactoring, we are moving towards direct imports from the new modules.
# __all__ = [] # Example if it were re-exporting, currently not the case.

import logging

logger = logging.getLogger(__name__)

logger.info(
    "The 'operators.py' module has been refactored. "
    "Its contents have been moved to more specific modules. "
    "Please update your imports accordingly."
)

# If there are any truly common base classes or utilities that don't fit elsewhere
# and are used by multiple new operator modules, they could remain or be moved
# to a new shared utility module (e.g., 'operator_utils.py').
# For now, assuming all specific operators have been moved out.
