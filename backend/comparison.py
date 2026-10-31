"""Benchmark dispatch methods on the same smart-grid scenario."""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linprog

from backend.pso_optimizer import (
    generator_cost,
    get_generator_dispatch_bounds,
    is_renewable,
    optimize_loads,
    validate_settings,
)
from backend.zone_allocator import allocate_power_to_zones, normalize_demand_zones


def _comparison_context(
    generators: list[dict],
    demand: float,
    settings: dict,
    renewable_availability: float,
    battery: dict | None,
    demand_zones: list[dict] | None,
) -> dict:
    validate_settings(settings)
    if not generators:
        raise ValueError("At least one generator is required for comparison.")
    bounds = get_generator_dispatch_bounds(generators, renewable_availability)
    battery = battery or {}
    battery_capacity = max(0.0, float(battery.get("capacity", 0.0)))
    battery_soc = max(0.0, min(100.0, float(battery.get("soc", 0.0))))
    battery_available = battery_capacity * battery_soc / 100.0
    battery_cost = max(
        0.0, float(battery.get("cost_per_mwh", battery.get("cost_per_mw", 0.0)))
    )
    zones = normalize_demand_zones(demand_zones, demand)
    zone_total = sum(float(zone["demand"]) for zone in zones)
    if abs(zone_total - demand) > 1e-7:
        raise ValueError("Demand-zone total does not match total demand.")
    critical_demand = sum(
        float(zone["demand"]) for zone in zones if zone["priority"] == "critical"
    )
    shed_limit = max(0.0, demand - critical_demand) if demand_zones else 0.0
    lower = bounds["minimum_outputs"]
    upper = bounds["dispatch_upper_bounds"]
    if demand + 1e-8 < float(lower.sum()):
        raise ValueError("Demand is below the generators' minimum output.")
    if critical_demand > float(upper.sum()) + battery_available + 1e-8:
        raise ValueError("Critical demand cannot be supplied by the available resources.")
    if demand > float(upper.sum()) + battery_available + shed_limit + 1e-8:
        raise ValueError("Demand cannot be met even after eligible load shedding.")
    return {
        "bounds": bounds,
        "battery_available": battery_available,
        "battery_cost": battery_cost,
        "zones": zones,
        "shed_limit": shed_limit,
        "costs": np.array([generator_cost(generator) for generator in generators]),
        "emissions": np.array([
            max(0.0, float(generator.get("emission_factor", 0.0)))
            for generator in generators
        ]),
        "emission_weight": max(0.0, float(settings.get("emission_cost_per_kg", 0.0))),
    }


def _result_row(
    method: str,
    loads: np.ndarray,
    battery_discharge: float,
    runtime_seconds: float,
    demand: float,
    context: dict,
) -> dict:
    zone_summary = allocate_power_to_zones(
        context["zones"], float(loads.sum() + battery_discharge), demand
    )
    load_shed = float(zone_summary["total_shed"])
    total_cost = float(
        np.dot(loads, context["costs"])
        + battery_discharge * context["battery_cost"]
    )
    total_emissions = float(np.dot(loads, context["emissions"]))
    return {
        "Method": method,
        "Cost": total_cost,
        "Emissions": total_emissions,
        "Runtime": runtime_seconds,
        "Shortage": load_shed,
        "allocation": loads,
        "battery_discharge": float(battery_discharge),
        "load_shed": load_shed,
        "zone_allocations": zone_summary["zones"],
    }


def greedy_economic_dispatch(
    generators: list[dict],
    demand: float,
    settings: dict,
    *,
    renewable_availability: float = 1.0,
    battery: dict | None = None,
    demand_zones: list[dict] | None = None,
) -> dict:
    """Dispatch the cheapest available resources first."""
    started = time.perf_counter()
    context = _comparison_context(
        generators, demand, settings, renewable_availability, battery, demand_zones
    )
    lower = context["bounds"]["minimum_outputs"]
    upper = context["bounds"]["dispatch_upper_bounds"]
    loads = lower.copy()
    remaining = max(0.0, demand - float(loads.sum()))

    resources = [
        (context["costs"][index] + context["emission_weight"] * context["emissions"][index], "generator", index)
        for index in range(len(generators))
    ]
    resources.append((context["battery_cost"], "battery", len(generators)))
    resources.sort(key=lambda item: item[0])
    battery_discharge = 0.0
    for _, resource_type, index in resources:
        if remaining <= 1e-9:
            break
        if resource_type == "battery":
            addition = min(remaining, context["battery_available"])
            battery_discharge += addition
        else:
            addition = min(remaining, upper[index] - loads[index])
            loads[index] += addition
        remaining -= addition

    return _result_row(
        "Greedy",
        loads,
        battery_discharge,
        time.perf_counter() - started,
        demand,
        context,
    )


def linear_programming_dispatch(
    generators: list[dict],
    demand: float,
    settings: dict,
    *,
    renewable_availability: float = 1.0,
    battery: dict | None = None,
    demand_zones: list[dict] | None = None,
) -> dict:
    """Solve the bounded dispatch problem with SciPy's HiGHS LP solver."""
    started = time.perf_counter()
    context = _comparison_context(
        generators, demand, settings, renewable_availability, battery, demand_zones
    )
    n = len(generators)
    carbon_adjusted_costs = context["costs"] + context["emission_weight"] * context["emissions"]
    objective = np.concatenate((carbon_adjusted_costs, [context["battery_cost"], 1_000_000.0]))
    bounds = list(zip(
        context["bounds"]["minimum_outputs"],
        context["bounds"]["dispatch_upper_bounds"],
    ))
    bounds.extend([(0.0, context["battery_available"]), (0.0, context["shed_limit"])])
    solution = linprog(
        objective,
        A_eq=np.ones((1, n + 2)),
        b_eq=np.array([demand]),
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise ValueError(f"Linear programming dispatch failed: {solution.message}")
    return _result_row(
        "Linear Programming",
        solution.x[:n],
        float(solution.x[n]),
        time.perf_counter() - started,
        demand,
        context,
    )


def compare_algorithms(
    generators: list[dict],
    demand: float,
    settings: dict,
    *,
    renewable_availability: float = 1.0,
    battery: dict | None = None,
    demand_zones: list[dict] | None = None,
    fuzzy_result: dict | None = None,
    fuzzy_runtime_seconds: float | None = None,
) -> dict:
    """Run Greedy, LP, plain PSO, and Fuzzy–PSO on identical inputs."""
    context = _comparison_context(
        generators, demand, settings, renewable_availability, battery, demand_zones
    )
    greedy = greedy_economic_dispatch(
        generators, demand, settings,
        renewable_availability=renewable_availability,
        battery=battery,
        demand_zones=demand_zones,
    )
    lp = linear_programming_dispatch(
        generators, demand, settings,
        renewable_availability=renewable_availability,
        battery=battery,
        demand_zones=demand_zones,
    )

    pso_started = time.perf_counter()
    pso_result = optimize_loads(
        generators,
        demand,
        settings,
        renewable_availability=renewable_availability,
        battery=battery,
        demand_zones=demand_zones,
        use_fuzzy=False,
    )
    pso = _result_row(
        "PSO",
        pso_result["allocation"],
        pso_result["battery_discharge"],
        time.perf_counter() - pso_started,
        demand,
        context,
    )

    if fuzzy_result is None:
        fuzzy_started = time.perf_counter()
        fuzzy_result = optimize_loads(
            generators,
            demand,
            settings,
            renewable_availability=renewable_availability,
            battery=battery,
            demand_zones=demand_zones,
            use_fuzzy=True,
        )
        fuzzy_runtime_seconds = time.perf_counter() - fuzzy_started
    fuzzy = _result_row(
        "Fuzzy–PSO",
        fuzzy_result["allocation"],
        fuzzy_result["battery_discharge"],
        float(fuzzy_runtime_seconds or fuzzy_result.get("runtime_seconds", 0.0)),
        demand,
        context,
    )

    results = [greedy, lp, pso, fuzzy]
    rows = [
        {
            "Method": item["Method"],
            "Cost": item["Cost"],
            "Emissions": item["Emissions"],
            "Runtime": item["Runtime"],
            "Shortage": item["Shortage"],
        }
        for item in results
    ]
    return {"rows": rows, "results": results}
