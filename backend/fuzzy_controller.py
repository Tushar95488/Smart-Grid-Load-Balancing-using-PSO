"""Fuzzy decision layer for the hybrid fuzzy--PSO optimizer.

The fuzzy controller does not allocate megawatts.  It translates the current
grid conditions into soft priorities that the PSO uses while searching for a
safe allocation.
"""

from __future__ import annotations

from typing import Iterable


LABEL_VALUES = {
    "very_low": 0.10,
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
    "very_high": 0.95,
}
RENEWABLE_TYPES = {"renewable", "solar", "wind", "hydro"}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _membership(value: float) -> dict[str, float]:
    """Return smooth triangular memberships for a normalized input."""
    value = _clamp(value)
    return {
        "low": max(0.0, 1.0 - 2.0 * value),
        "medium": max(0.0, 1.0 - 2.0 * abs(value - 0.5)),
        "high": max(0.0, 2.0 * value - 1.0),
    }


def _and(*degrees: float) -> float:
    return min(degrees) if degrees else 0.0


def _infer(rules: Iterable[tuple[float, str]], default: float = 0.5) -> float:
    """Defuzzify Mamdani-style rules using a weighted average."""
    weighted_sum = 0.0
    activation_sum = 0.0
    for activation, label in rules:
        weighted_sum += activation * LABEL_VALUES[label]
        activation_sum += activation
    return _clamp(weighted_sum / activation_sum) if activation_sum else default


def evaluate_grid_conditions(
    generators: list[dict],
    demand: float,
    total_capacity: float,
    renewable_availability: float = 1.0,
) -> dict:
    """Evaluate grid conditions and return fuzzy priorities.

    ``renewable_availability`` is expressed as a fraction from 0 to 1.  Grid
    stress and reserve margin are calculated from the requested demand and the
    currently available generation capacity.
    """
    if total_capacity <= 0:
        raise ValueError("Total generator capacity must be greater than zero.")

    renewable_availability = _clamp(renewable_availability)
    demand_ratio = _clamp(demand / total_capacity)
    reserve_margin = _clamp((total_capacity - demand) / total_capacity)
    grid_stress = demand_ratio
    demand_membership = _membership(demand_ratio)
    stress_membership = _membership(grid_stress)
    reserve_membership = _membership(reserve_margin)
    renewable_membership = _membership(renewable_availability)

    costs = [
        float(item.get("cost_per_mwh", item.get("cost_per_mw", 0.0)))
        for item in generators
    ]
    min_cost = min(costs) if costs else 0.0
    max_cost = max(costs) if costs else 1.0
    cost_span = max(max_cost - min_cost, 1e-9)

    conventional_support = _infer(
        [
            (_and(stress_membership["high"], renewable_membership["low"]), "high"),
            (_and(stress_membership["medium"], reserve_membership["low"]), "medium"),
            (renewable_membership["high"], "low"),
        ]
    )
    battery_priority = _infer(
        [
            (reserve_membership["low"], "high"),
            (reserve_membership["medium"], "medium"),
            (reserve_membership["high"], "low"),
        ]
    )
    load_shedding_priority = _infer(
        [
            (_and(demand_membership["high"], reserve_membership["low"]), "high"),
            (stress_membership["high"], "medium"),
            (reserve_membership["high"], "low"),
        ]
    )

    generator_priorities: list[float] = []
    for generator, cost in zip(generators, costs):
        cost_norm = (cost - min_cost) / cost_span
        # Cost itself is low near zero, so the low membership is strongest for
        # the least expensive generator.
        cost_membership = _membership(cost_norm)
        generator_type = str(generator.get("type", "")).strip().lower()
        is_renewable = generator_type in RENEWABLE_TYPES or bool(
            generator.get("renewable", False)
        )

        if is_renewable:
            priority = _infer(
                [
                    (_and(renewable_membership["high"], cost_membership["low"]), "very_high"),
                    (renewable_membership["high"], "high"),
                    (renewable_membership["medium"], "medium"),
                    (renewable_membership["low"], "low"),
                ]
            )
        else:
            priority = _infer(
                [
                    (_and(stress_membership["high"], renewable_membership["low"]), "high"),
                    (cost_membership["low"], "high"),
                    (stress_membership["high"], "medium"),
                    (reserve_membership["high"], "low"),
                ]
            )
        generator_priorities.append(priority)

    return {
        "inputs": {
            "grid_stress": grid_stress,
            "renewable_availability": renewable_availability,
            "reserve_margin": reserve_margin,
            "demand_ratio": demand_ratio,
        },
        "memberships": {
            "grid_stress": stress_membership,
            "renewable_availability": renewable_membership,
            "reserve_margin": reserve_membership,
            "demand": demand_membership,
        },
        "generator_priorities": generator_priorities,
        "battery_discharge_priority": battery_priority,
        "load_shedding_priority": load_shedding_priority,
        "conventional_support": conventional_support,
    }
