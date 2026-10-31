"""Sample smart-grid data used by the simulation.

All values are fictional. Power values are measured in MW, cost is Rs./MWh,
and emission_factor is kg CO2/MWh. Change these values to test weather,
outages, generator limits, and carbon-cost scenarios.
"""

GENERATORS = [
    {
        "name": "Thermal Plant",
        "type": "conventional",
        "capacity": 100.0,
        "available_capacity": 100.0,
        "cost_per_mwh": 6.5,
        "emission_factor": 0.82,
        "minimum_output": 10.0,
        "ramp_rate": 25.0,
        "previous_output": 25.0,
    },
    {
        "name": "Solar Farm",
        "type": "renewable",
        "capacity": 55.0,
        "available_capacity": 45.0,
        "cost_per_mwh": 2.5,
        "emission_factor": 0.05,
        "minimum_output": 0.0,
        "ramp_rate": 20.0,
        "previous_output": 20.0,
    },
    {
        "name": "Wind Farm",
        "type": "renewable",
        "capacity": 45.0,
        "available_capacity": 40.0,
        "cost_per_mwh": 3.0,
        "emission_factor": 0.02,
        "minimum_output": 0.0,
        "ramp_rate": 15.0,
        "previous_output": 18.0,
    },
    {
        "name": "Hydro Plant",
        "type": "renewable",
        "capacity": 90.0,
        "available_capacity": 90.0,
        "cost_per_mwh": 4.0,
        "emission_factor": 0.03,
        "minimum_output": 5.0,
        "ramp_rate": 30.0,
        "previous_output": 40.0,
    },
]

# Demand reported by each consumer zone. The optimizer supplies their total.
DEMAND_ZONES = [
    {"name": "Residential", "demand": 75.0, "priority": "medium", "shed_order": 30},
    {"name": "Industrial", "demand": 60.0, "priority": "medium", "shed_order": 20},
    {"name": "Commercial", "demand": 35.0, "priority": "low", "shed_order": 10},
    {"name": "Hospital & Emergency", "demand": 20.0, "priority": "critical", "shed_order": 100},
]

# Battery values are intentionally small and fictional for the simulation.
BATTERY = {"capacity": 40.0, "soc": 70.0, "cost_per_mwh": 1.0}

# Particle Swarm Optimization settings.
PSO_SETTINGS = {
    "swarm_size": 40,
    "iterations": 150,
    "inertia_weight": 0.70,
    "cognitive_weight": 1.45,
    "social_weight": 1.45,
    "random_seed": 42,
    "emission_cost_per_kg": 0.15,
}
