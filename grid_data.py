"""Sample smart-grid data used by the simulation.

All values are fictional and measured in megawatts (MW), except cost_per_mw.
Change these lists to test other grid scenarios.
"""

GENERATORS = [
    {"name": "Thermal Plant", "capacity": 100.0, "cost_per_mw": 6.5},
    {"name": "Solar Farm", "capacity": 55.0, "cost_per_mw": 2.5},
    {"name": "Wind Farm", "capacity": 45.0, "cost_per_mw": 3.0},
    {"name": "Hydro Plant", "capacity": 90.0, "cost_per_mw": 4.0},
]

# Demand reported by each consumer zone. The optimizer supplies their total.
DEMAND_ZONES = [
    {"name": "Residential", "demand": 75.0},
    {"name": "Industrial", "demand": 60.0},
    {"name": "Commercial", "demand": 35.0},
    {"name": "Hospital & Emergency", "demand": 20.0},
]

# Particle Swarm Optimization settings.
PSO_SETTINGS = {
    "swarm_size": 40,
    "iterations": 150,
    "inertia_weight": 0.70,
    "cognitive_weight": 1.45,
    "social_weight": 1.45,
    "random_seed": 42,
}
