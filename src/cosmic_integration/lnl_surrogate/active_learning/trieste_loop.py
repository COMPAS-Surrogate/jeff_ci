from __future__ import annotations

import logging
from typing import Callable, Optional

from tqdm import tqdm


def run_active_learning(
    learner,
    total_steps: int,
    steps_per_round: int,
    *,
    convergence_warnings: bool = True,
    patience: int = 3,
    kernel_diagnostics: bool = True,
    adaptive_acquisition: bool = True,
    round_callback: Optional[Callable[[int, object], None]] = None,
    callback_fail_fast: bool = True,
):
    """
    Execute the Trieste optimization loop for an :class:`~ActiveLearner` instance.

    This keeps the "loop orchestration" separate from model building, persistence,
    and plotting, while preserving the public `ActiveLearner.run(...)` API.
    """

    rounds_without_improvement = 0
    best_so_far = float("inf")

    assert total_steps > 0 and steps_per_round > 0
    num_rounds = total_steps // steps_per_round

    pbar = tqdm(total=total_steps, unit="step")
    step_counter = 0

    for round_idx in range(num_rounds):
        if adaptive_acquisition:
            explore_steps, exploit_steps = learner._adaptive_acquisition_split(round_idx, steps_per_round)
        else:
            explore_steps = int(round((2.0 / 3.0) * steps_per_round))
            exploit_steps = steps_per_round - explore_steps

        logger = logging.getLogger(__name__)
        logger.info("Round %d: %d explore + %d exploit steps", round_idx, explore_steps, exploit_steps)

        for _ in range(explore_steps):
            pbar.set_description(
                f"Exploring (best: {learner.current_best:.4f} | lnl: {learner.current_log_likelihood:.4f})"
            )
            learner._one_bo_step_with_rule(step_counter, learner.exploration_rule, "PredVar")
            step_counter += 1
            pbar.update(1)

        for _ in range(exploit_steps):
            pbar.set_description(
                f"Exploiting (best: {learner.current_best:.4f} | lnl: {learner.current_log_likelihood:.4f})"
            )
            learner._one_bo_step_with_rule(step_counter, learner.exploitation_rule, "EI")
            step_counter += 1
            pbar.update(1)

        pbar.set_description("Diagnostics & Checkpointing")

        current_uncertainty = learner._compute_model_uncertainty()
        learner.model_uncertainty_history.append(current_uncertainty)

        if kernel_diagnostics and round_idx % 2 == 0:
            learner._print_kernel_diagnostics(round_idx)

        if convergence_warnings and learner._check_convergence():
            rounds_without_improvement += 1
            logger.warning("No significant improvement for %d rounds", rounds_without_improvement)
            logger.warning("Current best: %.6f", learner.current_best)
            logger.warning("Average uncertainty: %.4f", current_uncertainty)
            logger.warning("Consider: different acquisition strategy, kernel, or more exploration")

            if rounds_without_improvement >= patience:
                logger.warning("CONVERGENCE WARNING: %d+ rounds without improvement!", patience)
                logger.warning("You may want to consider stopping manually or adjusting strategy")
        else:
            rounds_without_improvement = 0

        if learner.current_best < best_so_far:
            best_so_far = learner.current_best
            rounds_without_improvement = 0
            logger.info("New best found: %.6f", learner.current_best)

        learner._plot_diagnostics(round_idx=round_idx)
        learner.save_model(round_idx=round_idx)
        if round_callback is not None:
            try:
                round_callback(int(round_idx), learner)
            except Exception as exc:
                logger.exception("Round callback failed at round %d: %s", round_idx, exc)
                if callback_fail_fast:
                    raise

    pbar.close()

    if adaptive_acquisition and learner.acquisition_history:
        learner._print_acquisition_summary()

    return learner.current_dataset, learner.current_model
