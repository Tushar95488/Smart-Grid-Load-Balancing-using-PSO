"""Regression tests for the hybrid smart-grid optimizer."""

import numpy as np
import pytest

from backend.pso_optimizer import optimize_loads, repair_allocation


def _settings(seed: int = 42) -> dict:
    return {
        "swarm_size": 20,
        "iterations": 40,
        "inertia_weight": 0.70,
        "cognitive_weight": 1.45,
        "social_weight": 1.45,
        "random_seed": seed,
        "emission_cost_per_kg": 0.15,
    }


def _generators() -> list[dict]:
    return [
        {
            "name": "Thermal",
            "type": "conventional",
            "capacity": 80.0,
            "available_capacity": 80.0,
            "cost_per_mwh": 6.0,
            "emission_factor": 0.8,
            "minimum_output": 0.0,
            "ramp_rate": 100.0,
            "previous_output": 0.0,
        },
        {
            "name": "Solar",
            "type": "renewable",
            "capacity": 40.0,
            "available_capacity": 35.0,
            "cost_per_mwh": 2.0,
            "emission_factor": 0.05,
            "minimum_output": 0.0,
            "ramp_rate": 100.0,
            "previous_output": 0.0,
        },
    ]


def _zones() -> list[dict]:
    return [
        {"name": "Residential", "demand": 30.0, "priority": "medium", "shed_order": 30},
        {"name": "Industrial", "demand": 20.0, "priority": "medium", "shed_order": 20},
        {"name": "Commercial", "demand": 10.0, "priority": "low", "shed_order": 10},
        {"name": "Hospital & Emergency", "demand": 10.0, "priority": "critical", "shed_order": 100},
    ]


def test_repair_allocation_keeps_values_within_capacity():
    capacities = np.array([20.0, 30.0])
    repaired = repair_allocation(np.array([-5.0, 100.0]), capacities, 25.0)

    assert np.all(repaired >= 0.0)
    assert np.all(repaired <= capacities + 1e-9)


def test_repair_allocation_matches_demand():
    capacities = np.array([20.0, 30.0])
    repaired = repair_allocation(np.zeros(2), capacities, 25.0)

    assert np.isclose(repaired.sum(), 25.0)


def test_infeasible_demand_raises_error():
    generator = [{
        "name": "Small Generator",
        "type": "conventional",
        "capacity": 10.0,
        "available_capacity": 10.0,
        "cost_per_mwh": 5.0,
        "emission_factor": 0.8,
        "minimum_output": 0.0,
        "ramp_rate": 100.0,
        "previous_output": 0.0,
    }]

    with pytest.raises(ValueError, match="cannot be met|available supply"):
        optimize_loads(generator, 20.0, _settings())


def test_same_seed_produces_same_result():
    first = optimize_loads(_generators(), 70.0, _settings(123), demand_zones=_zones())
    second = optimize_loads(_generators(), 70.0, _settings(123), demand_zones=_zones())

    np.testing.assert_allclose(first["allocation"], second["allocation"])
    assert first["battery_discharge"] == second["battery_discharge"]
    assert first["load_shed"] == second["load_shed"]
    np.testing.assert_allclose(first["history"], second["history"])


def test_optimizer_never_overloads_generator():
    result = optimize_loads(_generators(), 70.0, _settings(), demand_zones=_zones())

    assert np.all(result["allocation"] <= result["dispatch_upper_bounds"] + 1e-8)
    assert np.all(result["allocation"] <= result["physical_capacities"] + 1e-8)
    assert np.all(result["allocation"] >= result["minimum_outputs"] - 1e-8)


def test_critical_load_is_protected():
    generator = [{
        "name": "Limited Generator",
        "type": "conventional",
        "capacity": 50.0,
        "available_capacity": 50.0,
        "cost_per_mwh": 5.0,
        "emission_factor": 0.8,
        "minimum_output": 0.0,
        "ramp_rate": 100.0,
        "previous_output": 0.0,
    }]
    zones = [
        {"name": "Hospital & Emergency", "demand": 20.0, "priority": "critical", "shed_order": 100},
        {"name": "Residential", "demand": 25.0, "priority": "medium", "shed_order": 30},
        {"name": "Industrial", "demand": 25.0, "priority": "medium", "shed_order": 20},
        {"name": "Commercial", "demand": 20.0, "priority": "low", "shed_order": 10},
    ]
    result = optimize_loads(generator, 90.0, _settings(), demand_zones=zones)
    zone_by_name = {zone["name"]: zone for zone in result["zone_allocations"]}

    hospital = zone_by_name["Hospital & Emergency"]
    assert hospital["supplied_demand"] == hospital["requested_demand"]
    assert hospital["load_shed"] == 0.0
    assert zone_by_name["Commercial"]["load_shed"] > 0.0
