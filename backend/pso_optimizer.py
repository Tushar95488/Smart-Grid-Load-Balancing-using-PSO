"""PSO allocation layer for the hybrid fuzzy--PSO smart-grid optimizer."""

from __future__ import annotations

import math

import numpy as np

from backend.fuzzy_controller import evaluate_grid_conditions
from backend.zone_allocator import allocate_power_to_zones, normalize_demand_zones


RENEWABLE_TYPES = {"renewable", "solar", "wind", "hydro"}


def validate_settings(settings: dict) -> None:
    """Validate PSO controls before creating the swarm."""
    if not isinstance(settings, dict):
        raise ValueError("PSO settings must be provided as a dictionary.")
    for key in ("swarm_size", "iterations"):
        if key not in settings:
            raise ValueError(f"PSO setting '{key}' is required.")
        value = float(settings[key])
        if not np.isfinite(value) or not value.is_integer() or value <= 0:
            raise ValueError(f"{key.replace('_', ' ').title()} must be an integer greater than zero.")
    for key in ("inertia_weight", "cognitive_weight", "social_weight"):
        if key not in settings:
            raise ValueError(f"PSO setting '{key}' is required.")
        value = float(settings[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key.replace('_', ' ').title()} must be non-negative.")


def validate_generators(generators: list[dict]) -> None:
    """Validate generator identity and non-negative physical/economic inputs."""
    if not generators:
        raise ValueError("At least one generator is required.")
    seen_names: set[str] = set()
    for index, generator in enumerate(generators):
        raw_name = str(generator.get("name", "")).strip()
        if not raw_name:
            raise ValueError(f"Generator {index + 1} must have a name.")
        normalized_name = raw_name.casefold()
        if normalized_name in seen_names:
            raise ValueError(f"Generator names must be unique: '{raw_name}' is duplicated.")
        seen_names.add(normalized_name)

        numeric_fields = {
            "capacity": generator.get("capacity", 0.0),
            "cost": generator_cost(generator),
            "available_capacity": generator.get("available_capacity", generator.get("capacity", 0.0)),
            "minimum_output": generator.get("minimum_output", 0.0),
            "ramp_rate": generator.get("ramp_rate", math.inf),
            "previous_output": generator.get("previous_output", 0.0),
        }
        for field, raw_value in numeric_fields.items():
            value = float(raw_value)
            if not np.isfinite(value) and not (field == "ramp_rate" and math.isinf(value)):
                raise ValueError(f"{raw_name} {field} must be a finite number.")
            if value < 0:
                raise ValueError(f"{raw_name} {field} cannot be negative.")


def generator_cost(generator: dict) -> float:
    """Read the current schema while keeping old project data compatible."""
    return float(generator.get("cost_per_mwh", generator.get("cost_per_mw", 0.0)))


def is_renewable(generator: dict) -> bool:
    generator_type = str(generator.get("type", "")).strip().lower()
    return generator_type in RENEWABLE_TYPES or bool(generator.get("renewable", False))


def repair_allocation(allocation: np.ndarray, capacities: np.ndarray, demand: float) -> np.ndarray:
    """Return a feasible allocation whose sum equals demand.

    This legacy helper remains available for callers using the original PSO
    API. The hybrid optimizer uses :func:`repair_hybrid_allocation` below.
    """
    allocation = np.clip(np.asarray(allocation, dtype=float), 0.0, capacities)
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


def repair_hybrid_allocation(
    allocation: np.ndarray,
    capacities: np.ndarray,
    demand: float,
    battery_capacity: float = 0.0,
    shed_limit: float = 0.0,
    minimum_outputs: np.ndarray | None = None,
) -> np.ndarray:
    """Repair generator, battery, and shedding variables to balance demand.

    The returned vector is ``[generator loads..., battery discharge,
    non-critical load shed]`` and satisfies ``generation + battery + shed ==
    demand`` when the scenario is feasible. ``capacities`` are the current
    dispatch upper bounds, while ``minimum_outputs`` are lower bounds.
    """
    upper_bounds = np.asarray(capacities, dtype=float)
    lower_bounds = (
        np.zeros_like(upper_bounds)
        if minimum_outputs is None
        else np.minimum(np.asarray(minimum_outputs, dtype=float), upper_bounds)
    )
    generator_count = len(upper_bounds)
    values = np.asarray(allocation, dtype=float).copy()
    if values.size != generator_count + 2:
        raise ValueError("Hybrid allocation has an invalid number of dimensions.")

    values[:generator_count] = np.clip(values[:generator_count], lower_bounds, upper_bounds)
    values[generator_count] = np.clip(values[generator_count], 0.0, battery_capacity)
    values[generator_count + 1] = np.clip(values[generator_count + 1], 0.0, shed_limit)

    def increase_supply(shortage: float) -> float:
        if shortage <= 1e-9:
            return 0.0
        spare = upper_bounds - values[:generator_count]
        spare_total = float(spare.sum())
        if spare_total > 1e-9:
            addition = np.minimum(spare, spare / spare_total * shortage)
            values[:generator_count] += addition
            shortage -= float(addition.sum())
        battery_spare = battery_capacity - values[generator_count]
        if shortage > 1e-9 and battery_spare > 1e-9:
            addition = min(shortage, battery_spare)
            values[generator_count] += addition
            shortage -= addition
        return max(0.0, shortage)

    supplied = float(values[:generator_count].sum() + values[generator_count])
    balanced_supply = supplied + values[generator_count + 1]
    if balanced_supply < demand:
        shortage = increase_supply(demand - balanced_supply)
        if shortage > 1e-9:
            values[generator_count + 1] = min(
                shed_limit, values[generator_count + 1] + shortage
            )
    else:
        excess = balanced_supply - demand

        # Remove unnecessary shedding before reducing generator output.
        shed_reduction = min(excess, values[generator_count + 1])
        values[generator_count + 1] -= shed_reduction
        excess -= shed_reduction

        reducible = values[:generator_count] - lower_bounds
        reducible_total = float(reducible.sum())
        if excess > 1e-9 and reducible_total > 1e-9:
            reduction = min(excess, reducible_total)
            values[:generator_count] -= reducible / reducible_total * reduction
            excess -= reduction

        if excess > 1e-9:
            values[generator_count] = max(0.0, values[generator_count] - excess)

    # Correct small floating-point residuals while respecting the same bounds.
    residual = demand - (
        values[:generator_count].sum()
        + values[generator_count]
        + values[generator_count + 1]
    )
    if residual > 1e-7:
        remaining = increase_supply(residual)
        if remaining > 1e-7:
            values[generator_count + 1] = min(
                shed_limit, values[generator_count + 1] + remaining
            )
    elif residual < -1e-7:
        excess = -residual
        shed_reduction = min(excess, values[generator_count + 1])
        values[generator_count + 1] -= shed_reduction
        excess -= shed_reduction
        reducible = values[:generator_count] - lower_bounds
        reducible_total = float(reducible.sum())
        if excess > 1e-9 and reducible_total > 1e-9:
            reduction = min(excess, reducible_total)
            values[:generator_count] -= reducible / reducible_total * reduction
            excess -= reduction
        if excess > 1e-9:
            values[generator_count] = max(0.0, values[generator_count] - excess)

    return np.clip(
        values,
        np.concatenate((lower_bounds, [0.0, 0.0])),
        np.concatenate((upper_bounds, [battery_capacity, shed_limit])),
    )


def fitness(
    allocation: np.ndarray,
    capacities: np.ndarray,
    costs: np.ndarray,
    demand: float,
    generator_priorities: np.ndarray | None = None,
    load_shedding_priority: float = 0.5,
    battery_cost: float = 0.0,
    emission_factors: np.ndarray | None = None,
    emission_weight: float = 0.0,
    minimum_outputs: np.ndarray | None = None,
    demand_zones: list[dict] | None = None,
) -> float:
    """Calculate a lower-is-better cost plus constraint penalties."""
    generator_count = len(capacities)
    generator_loads = np.asarray(allocation[:generator_count], dtype=float)
    battery_discharge = float(allocation[generator_count])
    shed = float(allocation[generator_count + 1])
    generation_cost = float(np.dot(generator_loads, costs))
    actual_cost = generation_cost + battery_discharge * battery_cost

    if generator_priorities is not None:
        adjusted_costs = costs * (1.0 - 0.35 * np.asarray(generator_priorities))
        dispatch_cost = float(np.dot(generator_loads, adjusted_costs))
    else:
        dispatch_cost = generation_cost

    if emission_factors is None:
        emissions = 0.0
    else:
        emissions = float(np.dot(generator_loads, emission_factors))

    balance_error = abs(float(generator_loads.sum() + battery_discharge + shed) - demand)
    overload = np.maximum(generator_loads - capacities, 0.0).sum()
    under_minimum = (
        0.0
        if minimum_outputs is None
        else np.maximum(np.asarray(minimum_outputs) - generator_loads, 0.0).sum()
    )
    shed_penalty = shed * (100_000.0 + 10_000.0 * load_shedding_priority)
    zone_penalty = allocate_power_to_zones(
        demand_zones, generator_loads.sum() + battery_discharge, demand
    )["weighted_shed_penalty"]
    return (
        dispatch_cost
        + battery_discharge * battery_cost
        + emission_weight * emissions
        + shed_penalty
        + zone_penalty
        + 100_000.0 * (balance_error + overload + under_minimum)
        + 0.01 * actual_cost
    )


def _bounds_for_generator(
    generator: dict,
    renewable_availability: float,
) -> tuple[float, float, float, float]:
    """Return physical, available, lower, and ramp-limited upper bounds."""
    physical_capacity = max(0.0, float(generator.get("capacity", 0.0)))
    declared_available = float(generator.get("available_capacity", physical_capacity))
    if declared_available < 0:
        raise ValueError(f"Available capacity for {generator.get('name', 'generator')} cannot be negative.")
    declared_available = min(physical_capacity, declared_available)
    if is_renewable(generator):
        declared_available *= renewable_availability

    minimum_output = max(0.0, float(generator.get("minimum_output", 0.0)))
    previous_output = max(0.0, float(generator.get("previous_output", minimum_output)))
    ramp_rate = float(generator.get("ramp_rate", math.inf))
    if ramp_rate < 0:
        raise ValueError(f"Ramp rate for {generator.get('name', 'generator')} cannot be negative.")

    if math.isfinite(ramp_rate):
        ramp_upper = previous_output + ramp_rate
        ramp_lower = max(minimum_output, previous_output - ramp_rate)
    else:
        ramp_upper = math.inf
        ramp_lower = minimum_output

    upper_bound = min(declared_available, ramp_upper)
    # A sudden outage can force a downward move beyond the normal ramp limit;
    # keep the lower operating bound feasible while still restricting ramp-up.
    lower_bound = minimum_output if ramp_lower > upper_bound else ramp_lower
    return physical_capacity, declared_available, lower_bound, upper_bound


def get_generator_dispatch_bounds(
    generators: list[dict], renewable_availability: float = 1.0
) -> dict[str, np.ndarray]:
    """Return shared generator bounds used by every comparison algorithm."""
    validate_generators(generators)
    renewable_availability = max(0.0, min(1.0, float(renewable_availability)))
    physical_capacities = []
    available_capacities = []
    lower_bounds = []
    dispatch_upper_bounds = []
    for generator in generators:
        physical, available, lower, upper = _bounds_for_generator(
            generator, renewable_availability
        )
        if upper + 1e-9 < lower:
            raise ValueError(
                f"{generator.get('name', 'Generator')} cannot meet its minimum output "
                "with its current availability and ramp limit."
            )
        physical_capacities.append(physical)
        available_capacities.append(available)
        lower_bounds.append(lower)
        dispatch_upper_bounds.append(upper)
    return {
        "physical_capacities": np.array(physical_capacities, dtype=float),
        "available_capacities": np.array(available_capacities, dtype=float),
        "minimum_outputs": np.array(lower_bounds, dtype=float),
        "dispatch_upper_bounds": np.array(dispatch_upper_bounds, dtype=float),
    }


def optimize_loads(
    generators: list[dict],
    demand: float,
    settings: dict,
    *,
    renewable_availability: float = 1.0,
    battery: dict | None = None,
    demand_zones: list[dict] | None = None,
    use_fuzzy: bool = True,
) -> dict:
    """Use PSO to find a safe allocation, optionally guided by fuzzy priorities."""
    validate_settings(settings)
    validate_generators(generators)
    demand = float(demand)
    if not np.isfinite(demand) or demand <= 0:
        raise ValueError("Total demand must be greater than zero.")

    bounds = get_generator_dispatch_bounds(generators, renewable_availability)
    physical_capacities = bounds["physical_capacities"]
    available_capacities = bounds["available_capacities"]
    lower_bounds = bounds["minimum_outputs"]
    dispatch_upper_bounds = bounds["dispatch_upper_bounds"]
    costs = np.array([generator_cost(item) for item in generators], dtype=float)
    emission_factors = np.array(
        [max(0.0, float(item.get("emission_factor", 0.0))) for item in generators],
        dtype=float,
    )

    battery = battery or {}
    battery_capacity = float(battery.get("capacity", 0.0))
    battery_soc = float(battery.get("soc", 0.0))
    if not np.isfinite(battery_capacity) or battery_capacity < 0:
        raise ValueError("Battery capacity must be non-negative.")
    if not np.isfinite(battery_soc) or battery_soc < 0 or battery_soc > 100:
        raise ValueError("Battery state of charge must be between 0 and 100.")
    battery_available = battery_capacity * battery_soc / 100.0
    battery_cost = float(battery.get("cost_per_mwh", battery.get("cost_per_mw", 0.0)))
    if not np.isfinite(battery_cost) or battery_cost < 0:
        raise ValueError("Battery cost must be non-negative.")
    emission_weight = float(settings.get("emission_cost_per_kg", 0.0))
    if not np.isfinite(emission_weight) or emission_weight < 0:
        raise ValueError("Emission cost weight must be non-negative.")

    zone_records = normalize_demand_zones(demand_zones, demand)
    zone_total = sum(float(zone["demand"]) for zone in zone_records)
    if abs(zone_total - demand) > 1e-7:
        raise ValueError(
            f"Demand zones total {zone_total:.1f} MW does not match requested demand "
            f"{demand:.1f} MW."
        )
    critical_demand = sum(
        float(zone["demand"])
        for zone in zone_records
        if zone["priority"] == "critical"
    )
    shed_limit = max(0.0, demand - critical_demand) if demand_zones else 0.0

    total_minimum = float(lower_bounds.sum())
    total_dispatch_capacity = float(dispatch_upper_bounds.sum())
    if demand + 1e-8 < total_minimum:
        raise ValueError(
            f"Demand ({demand:.1f} MW) is below the generators' minimum output "
            f"({total_minimum:.1f} MW)."
        )
    if critical_demand > total_dispatch_capacity + battery_available + 1e-8:
        raise ValueError(
            f"Critical demand ({critical_demand:.1f} MW) exceeds available supply "
            f"({total_dispatch_capacity + battery_available:.1f} MW)."
        )
    if demand > total_dispatch_capacity + battery_available + shed_limit + 1e-8:
        raise ValueError(
            f"Demand ({demand:.1f} MW) cannot be met even after shedding all eligible "
            "non-critical load."
        )

    fuzzy = None
    if use_fuzzy:
        fuzzy = evaluate_grid_conditions(
            generators,
            demand,
            max(total_dispatch_capacity, 1.0),
            renewable_availability,
        )
        generator_priorities = np.array(fuzzy["generator_priorities"], dtype=float)
        fuzzy_load_shedding_priority = fuzzy["load_shedding_priority"]
    else:
        generator_priorities = np.zeros(len(generators), dtype=float)
        fuzzy_load_shedding_priority = 0.0

    rng = np.random.default_rng(settings.get("random_seed"))
    swarm_size = int(settings["swarm_size"])
    dimensions = len(generators) + 2
    upper_bounds = np.concatenate((dispatch_upper_bounds, [battery_available, shed_limit]))
    lower_vector = np.concatenate((lower_bounds, [0.0, 0.0]))
    velocity_limit = 0.2 * upper_bounds

    positions = np.empty((swarm_size, dimensions))
    for index in range(swarm_size):
        seed_position = lower_vector + rng.random(dimensions) * (upper_bounds - lower_vector)
        positions[index] = repair_hybrid_allocation(
            seed_position,
            dispatch_upper_bounds,
            demand,
            battery_available,
            shed_limit,
            lower_bounds,
        )

    velocities = rng.uniform(
        -velocity_limit,
        velocity_limit,
        size=(swarm_size, dimensions),
    )

    def score(position: np.ndarray) -> float:
        return fitness(
            position,
            dispatch_upper_bounds,
            costs,
            demand,
            generator_priorities,
            fuzzy_load_shedding_priority,
            battery_cost,
            emission_factors,
            emission_weight,
            lower_bounds,
            zone_records,
        )

    personal_best_positions = positions.copy()
    personal_best_scores = np.array([score(position) for position in positions])
    best_index = int(np.argmin(personal_best_scores))
    global_best_position = personal_best_positions[best_index].copy()
    global_best_score = float(personal_best_scores[best_index])
    history = [global_best_score]

    for _ in range(int(settings["iterations"])):
        random_personal = rng.random((swarm_size, dimensions))
        random_global = rng.random((swarm_size, dimensions))
        velocities = (
            settings["inertia_weight"] * velocities
            + settings["cognitive_weight"]
            * random_personal
            * (personal_best_positions - positions)
            + settings["social_weight"]
            * random_global
            * (global_best_position - positions)
        )
        velocities = np.clip(velocities, -velocity_limit, velocity_limit)
        positions += velocities

        for index in range(swarm_size):
            positions[index] = repair_hybrid_allocation(
                positions[index],
                dispatch_upper_bounds,
                demand,
                battery_available,
                shed_limit,
                lower_bounds,
            )
            current_score = score(positions[index])
            if current_score < personal_best_scores[index]:
                personal_best_scores[index] = current_score
                personal_best_positions[index] = positions[index].copy()
                if current_score < global_best_score:
                    global_best_score = float(current_score)
                    global_best_position = positions[index].copy()
        history.append(global_best_score)

    allocation = global_best_position[: len(generators)]
    battery_discharge = float(global_best_position[len(generators)])
    supplied_power = float(allocation.sum() + battery_discharge)
    zone_allocation = allocate_power_to_zones(zone_records, supplied_power, demand)
    load_shed = float(zone_allocation["total_shed"])
    total_cost = float(np.dot(allocation, costs) + battery_discharge * battery_cost)
    total_emissions = float(np.dot(allocation, emission_factors))
    renewable_generation = float(
        sum(
            load
            for generator, load in zip(generators, allocation)
            if is_renewable(generator)
        )
    )
    total_supply = float(allocation.sum() + battery_discharge)
    renewable_percentage = (
        100.0 * renewable_generation / total_supply if total_supply > 1e-9 else 0.0
    )
    reserve_capacity = float(total_dispatch_capacity + battery_available - demand)
    reserve_margin = 100.0 * reserve_capacity / demand
    generator_utilization = np.divide(
        allocation,
        dispatch_upper_bounds,
        out=np.zeros_like(allocation),
        where=dispatch_upper_bounds > 1e-9,
    )

    return {
        "allocation": allocation,
        "battery_discharge": battery_discharge,
        "load_shed": load_shed,
        "best_fitness": global_best_score,
        "history": history,
        "total_cost": total_cost,
        "total_emissions": total_emissions,
        "renewable_generation": renewable_generation,
        "renewable_percentage": renewable_percentage,
        "reserve_capacity": reserve_capacity,
        "reserve_margin": reserve_margin,
        "generator_utilization": generator_utilization,
        "emission_cost": total_emissions * emission_weight,
        "capacities": dispatch_upper_bounds,
        "dispatch_upper_bounds": dispatch_upper_bounds,
        "physical_capacities": physical_capacities,
        "available_capacities": available_capacities,
        "minimum_outputs": lower_bounds,
        "original_capacities": physical_capacities,
        "renewable_availability": renewable_availability,
        "battery_available": battery_available,
        "critical_demand": critical_demand,
        "shed_limit": shed_limit,
        "served_demand": float(zone_allocation["total_supplied"]),
        "zone_allocations": zone_allocation["zones"],
        "zone_summary": zone_allocation,
        "fuzzy": fuzzy,
        "algorithm": "Fuzzy–PSO" if use_fuzzy else "PSO",
        "velocity_limit": velocity_limit,
    }
