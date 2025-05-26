import logging
from solution import LeagueSolution # For type hinting and instantiation
import traceback # For detailed error logging in recreate_solution_from_data

logger = logging.getLogger(__name__)

def extract_solution_data_for_storage(
    solution_object_to_store: LeagueSolution,
) -> dict or None:
    if solution_object_to_store is None:
        logger.debug("extract_solution_data_for_storage: None input.")
        return None

    _fitness_val = float("inf")
    if hasattr(solution_object_to_store, "fitness"):
        try:
            _fitness_val = solution_object_to_store.fitness()
        except Exception as e:
            logger.warning(f"Fitness retrieval error during extraction: {e}")

    data = {
        "repr": (
            list(solution_object_to_store.repr)
            if hasattr(solution_object_to_store, "repr")
            else None
        ),
        "fitness_value": _fitness_val,
        "num_teams": getattr(solution_object_to_store, "num_teams", None),
        "team_size": getattr(solution_object_to_store, "team_size", None),
        "solution_class_name": solution_object_to_store.__class__.__name__,
    }
    logger.debug(
        f"Extracted data for {data['solution_class_name']}: Fitness {data['fitness_value']:.4f}"
    )
    return data


def recreate_solution_from_data(
    stored_solution_data: dict,
    master_players_list_for_recreation: list,
    problem_definition_for_recreation: dict,
    solution_classes_module=None,
) -> "LeagueSolution or None":
    """
    Recreates a full solution object from its stored data representation.
    """
    if stored_solution_data is None:
        logger.warning("Attempted to recreate solution from None data.")
        return None

    sol_class_name_str = stored_solution_data.get(
        "solution_class_name", "LeagueSolution" # Default to LeagueSolution if not specified
    )
    logger.debug(f"Attempting to recreate solution of type: {sol_class_name_str}")

    ResolvedSolutionClass = None

    if solution_classes_module:
        ResolvedSolutionClass = getattr(
            solution_classes_module, sol_class_name_str, None
        )
        if ResolvedSolutionClass:
            logger.debug(f"Found class '{sol_class_name_str}' in provided module.")

    if ResolvedSolutionClass is None:
        # Fallback to trying to resolve class from globals if not found in module
        # This is less robust and generally not recommended if solution_classes_module is the expected source
        logger.debug(
            f"Class '{sol_class_name_str}' not found in provided module (or module was None). "
            "This may indicate an issue if the class is not globally available or in 'solution.py'."
        )
        # Attempting to get from globals as a last resort, assuming it might be imported into the calling scope
        # or is a very common class like the base LeagueSolution.
        # For classes like LeagueHillClimbingSolution, LeagueSASolution, they must be in the solution_classes_module.
        if sol_class_name_str == "LeagueSolution": # Special handling for base class if needed
             ResolvedSolutionClass = LeagueSolution
        # Add more specific fallbacks if other common classes are expected and might not be in the module.
        # else:
        #     ResolvedSolutionClass = globals().get(sol_class_name_str)

        if ResolvedSolutionClass:
            logger.debug(
                f"Found class '{sol_class_name_str}' in accessible scope (e.g., direct import or global)."
            )
        else:
            logger.error(
                f"Solution class '{sol_class_name_str}' could not be resolved. "
                "Ensure it's defined and correctly passed via 'solution_classes_module'. Cannot recreate."
            )
            return None
    try:
        # Instantiate the solution object using the resolved class
        recreated_sol_object = ResolvedSolutionClass(
            players=master_players_list_for_recreation,
            num_teams=stored_solution_data.get(
                "num_teams", problem_definition_for_recreation["num_teams"]
            ),
            team_size=stored_solution_data.get(
                "team_size", problem_definition_for_recreation["team_size"]
            ),
            max_budget=problem_definition_for_recreation["max_budget"],
            position_requirements=problem_definition_for_recreation[
                "position_requirements"
            ],
            # 'repr' will be set explicitly after instantiation if available
        )

        if stored_solution_data.get("repr") is not None:
            recreated_sol_object.repr = stored_solution_data["repr"]
        else:
            logger.warning(
                f"Recreating solution for '{sol_class_name_str}' without a specific representation. "
                "It will have a random initial representation from its class. Fitness may not align."
            )
        # Restore fitness if it was stored, to avoid re-computation if not needed,
        # though fitness() method should be idempotent.
        # Note: _fitness_cache is an internal detail, direct setting might be fragile.
        # It's generally better to let fitness() be called if needed.
        # If fitness_value was stored, it's for record-keeping; the object should recalc its own fitness.

        logger.debug(
            f"Successfully recreated solution object of type {sol_class_name_str}."
        )
        return recreated_sol_object
    except Exception as e_recreate:
        logger.error(
            f"Error during instantiation of solution object of type '{sol_class_name_str}': {e_recreate}"
        )
        logger.error(traceback.format_exc())
        return None
