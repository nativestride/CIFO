# experiment_utils.py

import logging
import sys
import numpy as np
# Import LeagueSolution for type hinting and default for FitnessCounter.
from solution import LeagueSolution

logger = logging.getLogger(__name__)

# --- Basic Utility Functions ---

def safe_exp(x, max_exp_argument=700):
    """
    Safely compute np.exp(x), clipping the argument x to prevent overflow.
    """
    clipped_x = np.clip(x, -max_exp_argument, max_exp_argument)
    return np.exp(clipped_x)


def is_notebook() -> bool:
    """
    Determines if the code is running in a Jupyter Notebook environment.

    Returns:
        bool: True if running in a Jupyter Notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            # Other type (?)
            # Check if 'google.colab' is in sys.modules for Colab environments
            if 'google.colab' in sys.modules:
                return True # Google Colab notebook
            return False
    except NameError:
        return False      # Probably standard Python interpreter
    except AttributeError: # get_ipython may not have __class__ or __name__ in all contexts
        return False


# --- Fitness Evaluation Counter ---

class FitnessCounter:
    """
    Counts fitness evaluations by temporarily wrapping .fitness() method.
    """

    def __init__(self):
        self._original_fitness_method = None
        self._wrapped_class_ref = None
        self.count = 0
        logger.debug("FitnessCounter instance created.")

    def start_counting(self, solution_class_to_wrap=LeagueSolution):
        logger.debug(
            f"Attempting to start fitness counting for: {solution_class_to_wrap.__name__}"
        )
        if (
            self._wrapped_class_ref != solution_class_to_wrap
            or self._original_fitness_method is None
        ):
            if self._wrapped_class_ref and self._original_fitness_method:
                logger.debug(
                    f"Restoring original fitness to {self._wrapped_class_ref.__name__}"
                )
                self._wrapped_class_ref.fitness = self._original_fitness_method

            self._original_fitness_method = solution_class_to_wrap.fitness
            self._wrapped_class_ref = solution_class_to_wrap
            logger.debug(
                f"Original fitness method of {solution_class_to_wrap.__name__} stored and wrapped."
            )

            counter_instance = self

            def counting_wrapper_for_fitness(solution_object_instance, *args, **kwargs):
                counter_instance.count += 1
                return counter_instance._original_fitness_method(
                    solution_object_instance, *args, **kwargs
                )

            solution_class_to_wrap.fitness = counting_wrapper_for_fitness

        self.count = 0
        logger.debug("Fitness count reset.")

    def stop_counting(self) -> int:
        logger.debug("Attempting to stop fitness counting.")
        if self._original_fitness_method and self._wrapped_class_ref:
            self._wrapped_class_ref.fitness = self._original_fitness_method
            logger.debug(
                f"Original fitness method restored to {self._wrapped_class_ref.__name__}."
            )
            self._original_fitness_method = None
            self._wrapped_class_ref = None

        current_eval_count = self.count
        self.count = 0
        logger.debug(
            f"Fitness counting stopped. Evaluations: {current_eval_count}. Count reset."
        )
        return current_eval_count
