# Hybrid Fuzzy–PSO Smart Grid Load Balancing

A Python smart-grid simulation that combines fuzzy logic with Particle Swarm
Optimization (PSO).

- **Fuzzy logic** evaluates grid stress, renewable availability, reserve margin,
  demand, and generator cost. It produces generator, battery, load-shedding,
  and conventional-generator priorities.
- **PSO** searches for the best allocation while respecting physical capacity,
  current availability, minimum output, ramp limits, battery state of charge,
  critical-load protection, and demand balance.

This is an educational simulation with fictional values. It is not intended to
control a real electrical grid.

## Run the interactive web application

```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

The interface supports generator outages and weather variation through
`available_capacity`, renewable uncertainty through a global forecast
multiplier, minimum generation, ramp-rate limits, previous output, emissions,
carbon-cost weighting, battery dispatch, and priority-aware zone allocation.

Demand is allocated to individual zones rather than only summed:

| Zone | Priority | Shedding behavior |
| --- | --- | --- |
| Hospital & Emergency | Critical | Protected first |
| Residential | Medium | Protected before industrial |
| Industrial | Medium | Shed second |
| Commercial | Low | Shed first |

When available supply is insufficient, the zone allocator serves critical loads
first, then residential demand, then industrial demand, and finally commercial
demand. The result reports requested demand, supplied demand, and shed demand
for every zone.

## Algorithm comparison

Every optimization run benchmarks four methods on the same scenario:

| Method | Purpose |
| --- | --- |
| Greedy | Dispatches the cheapest available resources first |
| Linear Programming | Finds the exact optimum for the bounded linear model |
| PSO | Uses particle-swarm search without fuzzy priorities |
| Fuzzy–PSO | Uses fuzzy priorities to guide the PSO search |

The comparison table reports operating cost, emissions, runtime, and shortage.
For a linear cost model, Greedy and Linear Programming may be faster or cheaper,
while Fuzzy–PSO is useful when soft grid priorities, renewable conditions, and
non-cost preferences influence dispatch.

## Streamlit dashboard

After running an optimization, the dashboard displays renewable energy share,
total emissions, reserve margin, average generator utilization, per-generator
utilization, fuzzy membership graphs, power supplied to every demand zone, and
algorithm cost comparison. It also provides scenario testing for:

- Peak demand: increases every zone by 25%.
- Cloudy day: reduces solar availability to 35% of its configured value.
- Low wind: reduces wind availability to 35% of its configured value.
- Generator failure: sets a selected generator's available capacity to zero.
- Demand increase: increases every zone by 15%.

Scenario changes are applied to a copy of the entered data, so the original
generator and demand inputs remain easy to reuse.

## Run in terminal mode

```bash
python -m backend.main
```

## Run tests

```bash
python -m pytest -q
```

The test suite covers repair feasibility, demand matching, infeasible-demand
errors, seeded reproducibility, generator overload protection, and critical
hospital-load protection.

## Generator data model

Each generator uses the following properties:

```python
{
    "name": "Solar Farm",
    "type": "renewable",
    "capacity": 55.0,             # physical maximum, MW
    "available_capacity": 45.0,   # weather/outage-limited capacity, MW
    "cost_per_mwh": 2.5,          # Rs./MWh
    "emission_factor": 0.05,      # kg CO2/MWh
    "minimum_output": 0.0,        # must-run lower bound, MW
    "ramp_rate": 20.0,             # MW per optimization interval
    "previous_output": 20.0,      # output at the previous interval, MW
}
```

The PSO dispatch upper bound is calculated as:

```text
min(capacity, available_capacity, previous_output + ramp_rate)
```

For renewable generators, the result is also multiplied by the renewable
forecast multiplier. Minimum output is enforced as a lower bound. A sudden
availability loss can force a downward move below the normal ramp limit, while
ramp-up remains restricted.

## Project flow

1. Read generator properties, battery settings, and demand-zone inputs.
2. Calculate available and ramp-limited generator bounds.
3. Apply fuzzy membership functions and rules.
4. Convert fuzzy outputs into soft dispatch priorities.
5. Create a feasible PSO swarm containing generator loads, battery discharge,
   and total non-critical load shedding.
6. Optimize operating cost, fuzzy dispatch preference, and configurable carbon
   cost.
7. Apply the controlled zone policy to the PSO-supplied power.
8. Repair every particle so generator, ramp, battery, critical-load, zone, and
   demand constraints remain satisfied.
9. Display allocation, emissions, fuzzy decisions, zone supply, and save charts
   under `results/`.

## PSO validation and velocity scaling

The backend validates swarm size, iteration count, PSO weights, generator names,
capacities, costs, available capacity, battery values, and demand feasibility
before starting. Generator names are case-insensitively unique. Demand may
exceed raw generation only when the configured non-critical zones can legally
absorb the shortage; critical demand must always be feasible.

Velocity limits scale with the current dispatch bounds:

```python
velocity_limit = 0.2 * dispatch_upper_bounds
velocities = rng.uniform(-velocity_limit, velocity_limit, size=(swarm_size, dimensions))
```

This keeps particle movement proportional when generator capacities or battery
limits change.

## Fuzzy rules implemented

```text
IF grid stress is High AND renewable availability is Low
THEN conventional generator support is High

IF renewable availability is High AND generator cost is Low
THEN renewable generator priority is Very High

IF reserve margin is Low
THEN battery discharge priority is High

IF demand is High AND reserve margin is Low
THEN non-critical load-shedding priority is High
```

## Project structure

```text
backend/
  comparison.py         # Greedy, LP, PSO, and Fuzzy–PSO benchmark runners
  fuzzy_controller.py  # Fuzzy memberships, rules, and defuzzification
  main.py               # Terminal entry point
  pso_optimizer.py      # Bounds, feasibility repair, fitness, and PSO search
  zone_allocator.py     # Priority-aware zone supply and load shedding
frontend/
  app.py                # Streamlit user interface and property editor
shared/
  grid_data.py          # Realistic sample generators and demand zones
  visualization.py      # Dashboard charts and comparison visualizations
tests/
  test_optimizer.py     # Core optimizer regression tests
```
