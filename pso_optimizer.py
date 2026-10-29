"""Particle Swarm Optimization implementation for generator load allocation."""

from __future__ import annotations

import numpy as np


def repair_allocation(allocation: np.ndarray, capacities: np.ndarray, demand: float) -> np.ndarray:
    """Return a feasible allocation whose sum equals demand.

    PSO moves particles freely. This repair step clips invalid generator loads
    and redistributes any remaining demand into generators that still have
    spare capacity.
    """
    allocation = np.clip(np.asarray(allocation, dtype=float), 0.0, capacities)

    # Add or remove power until generation equals the requested demand.
    difference = demand - allocation.sum()
    for _ in range(len(capacities) + 2):
        if abs(difference) < 1e-8:
            break
        if difference > 0:
            spare = capacities - allocation
            available = spare.sum()
            if available < 1e-8:
                break
            allocation += spare / available * difference
        else:
            active = allocation.sum()
            if active < 1e-8:
                break
            allocation += allocation / active * difference

        allocation = np.clip(allocation, 0.0, capacities)
        difference = demand - allocation.sum()

    return allocation


def fitness(allocation: np.ndarray, capacities: np.ndarray, costs: np.ndarray, demand: float) -> float:
    """Calculate a lower-is-better score for one allocation.

    Generation cost is minimized. Penalties make shortage or capacity overflow
    much worse than an ordinary cost difference.
    """
    generation_cost = float(np.dot(allocation, costs))
    shortage_or_surplus = abs(float(allocation.sum()) - demand)
    overload = np.maximum(allocation - capacities, 0.0).sum()
    return generation_cost + 100_000 * shortage_or_surplus + 100_000 * overload


def optimize_loads(generators: list[dict], demand: float, settings: dict) -> dict:
    """Use PSO to find the lowest-cost safe generator load allocation."""
    capacities = np.array([item["capacity"] for item in generators], dtype=float)
    costs = np.array([item["cost_per_mw"] for item in generators], dtype=float)

    if demand <= 0:
        raise ValueError("Total demand must be greater than zero.")
    if demand > capacities.sum():
        raise ValueError(
            f"Demand ({demand:.1f} MW) exceeds grid capacity ({capacities.sum():.1f} MW)."
        )

    rng = np.random.default_rng(settings.get("random_seed"))
    swarm_size = settings["swarm_size"]
    dimensions = len(generators)

    positions = np.empty((swarm_size, dimensions))
    for index in range(swarm_size):
        # Start at different feasible load splits.
        positions[index] = repair_allocation(rng.random(dimensions) * capacities, capacities, demand)

    velocities = rng.uniform(-10.0, 10.0, size=(swarm_size, dimensions))
    personal_best_positions = positions.copy()
    personal_best_scores = np.array(
        [fitness(position, capacities, costs, demand) for position in positions]
    )
    best_index = int(np.argmin(personal_best_scores))
    global_best_position = personal_best_positions[best_index].copy()
    global_best_score = float(personal_best_scores[best_index])
    history = [global_best_score]

    for _ in range(settings["iterations"]):
        random_personal = rng.random((swarm_size, dimensions))
        random_global = rng.random((swarm_size, dimensions))
        velocities = (
            settings["inertia_weight"] * velocities
            + settings["cognitive_weight"] * random_personal * (personal_best_positions - positions)
            + settings["social_weight"] * random_global * (global_best_position - positions)
        )
        positions += velocities

        for index in range(swarm_size):
            positions[index] = repair_allocation(positions[index], capacities, demand)
            score = fitness(positions[index], capacities, costs, demand)
            if score < personal_best_scores[index]:
                personal_best_scores[index] = score
                personal_best_positions[index] = positions[index].copy()

                if score < global_best_score:
                    global_best_score = float(score)
                    global_best_position = positions[index].copy()
        history.append(global_best_score)

    return {
        "allocation": global_best_position,
        "best_fitness": global_best_score,
        "history": history,
        "total_cost": float(np.dot(global_best_position, costs)),
        "capacities": capacities,
    }
