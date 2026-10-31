"""Interactive Streamlit frontend for the hybrid fuzzy--PSO optimizer."""

from pathlib import Path
import time

import pandas as pd
import streamlit as st

from backend.comparison import compare_algorithms
from backend.pso_optimizer import is_renewable, optimize_loads
from shared.grid_data import BATTERY, GENERATORS
from shared.visualization import save_charts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(
    page_title="Hybrid Fuzzy–PSO Smart Grid",
    page_icon="⚡",
    layout="wide",
)

DEFAULT_GENERATORS = [dict(generator) for generator in GENERATORS]
DEFAULT_ZONES = [
    ("Residential", 75.0, "medium", 30),
    ("Industrial", 60.0, "medium", 20),
    ("Commercial", 35.0, "low", 10),
    ("Hospital & Emergency", 20.0, "critical", 100),
]
SCENARIOS = [
    "Base case",
    "Peak demand",
    "Cloudy day",
    "Low wind",
    "Generator failure",
    "Demand increase",
]


def generator_defaults(index: int) -> dict:
    """Return a copy of a realistic default generator row."""
    if index < len(DEFAULT_GENERATORS):
        return dict(DEFAULT_GENERATORS[index])
    return {
        "name": f"Generator {index + 1}",
        "type": "conventional",
        "capacity": 50.0,
        "available_capacity": 50.0,
        "cost_per_mwh": 5.0,
        "emission_factor": 0.50,
        "minimum_output": 0.0,
        "ramp_rate": 25.0,
        "previous_output": 0.0,
    }


def apply_scenario(
    generators: list[dict],
    demand_zones: list[dict],
    scenario: str,
    failure_generator: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Create a scenario-specific copy without changing the input controls."""
    scenario_generators = [dict(generator) for generator in generators]
    scenario_zones = [dict(zone) for zone in demand_zones]

    if scenario == "Peak demand":
        for zone in scenario_zones:
            zone["demand"] *= 1.25
    elif scenario == "Demand increase":
        for zone in scenario_zones:
            zone["demand"] *= 1.15
    elif scenario == "Cloudy day":
        solar = [
            generator for generator in scenario_generators
            if "solar" in generator["name"].lower()
        ]
        targets = solar or [generator for generator in scenario_generators if is_renewable(generator)]
        for generator in targets:
            generator["available_capacity"] *= 0.35
    elif scenario == "Low wind":
        wind = [
            generator for generator in scenario_generators
            if "wind" in generator["name"].lower()
        ]
        targets = wind or [generator for generator in scenario_generators if is_renewable(generator)]
        for generator in targets:
            generator["available_capacity"] *= 0.35
    elif scenario == "Generator failure" and failure_generator:
        for generator in scenario_generators:
            if generator["name"] == failure_generator:
                generator["available_capacity"] = 0.0
                generator["previous_output"] = 0.0
                generator["minimum_output"] = 0.0
                break
    return scenario_generators, scenario_zones


st.title("⚡ Hybrid Fuzzy–PSO Smart Grid")
st.caption(
    "Fuzzy logic sets operating priorities; Particle Swarm Optimization finds "
    "the safest low-cost allocation under generator and grid constraints."
)

with st.sidebar:
    st.header("Hybrid Settings")
    scenario = st.selectbox("Scenario testing", SCENARIOS)
    st.caption("Choose a scenario, then run the dashboard below.")
    renewable_availability = st.slider(
        "Renewable forecast multiplier (%)", min_value=0, max_value=100, value=100, step=5,
        help="Scales the entered available_capacity for renewable generators to model uncertainty.",
    )
    emission_cost_per_kg = st.number_input(
        "Carbon cost weight (Rs./kg CO2)", min_value=0.0, value=0.15, step=0.05,
        help="Adds an emissions penalty to the PSO objective while still reporting operating cost separately.",
    )
    st.subheader("Battery")
    battery_capacity = st.number_input(
        "Battery capacity (MW)", min_value=0.0, value=float(BATTERY["capacity"]), step=5.0
    )
    battery_soc = st.slider("Battery state of charge (%)", 0, 100, int(BATTERY["soc"]), 5)
    battery_cost = st.number_input(
        "Battery discharge cost (Rs./MWh)", min_value=0.0,
        value=float(BATTERY.get("cost_per_mwh", 0.0)), step=0.5,
    )
    st.subheader("PSO Settings")
    swarm_size = st.slider("Swarm size", min_value=10, max_value=100, value=40, step=5)
    iterations = st.slider("Iterations", min_value=20, max_value=500, value=150, step=10)
    inertia_weight = st.slider("Inertia weight (w)", 0.1, 1.0, 0.70, 0.05)
    cognitive_weight = st.slider("Cognitive weight (c1)", 0.1, 3.0, 1.45, 0.05)
    social_weight = st.slider("Social weight (c2)", 0.1, 3.0, 1.45, 0.05)
    random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

st.subheader("1. Enter Realistic Generator Properties")
st.info(
    "Capacity is the physical maximum. Available capacity models weather or an "
    "outage. Minimum output and ramp rate are enforced by PSO."
)
generator_count = st.number_input("Number of generators", min_value=2, max_value=10, value=4, step=1)

generators: list[dict] = []
for index in range(generator_count):
    defaults = generator_defaults(index)
    st.markdown(f"**Generator {index + 1}**")
    basic = st.columns([2.2, 1.3, 1.0, 1.2])
    name = basic[0].text_input(
        "Name", defaults["name"], key=f"generator_name_{index}"
    )
    type_options = ["conventional", "renewable"]
    default_type = defaults.get("type", "renewable" if defaults.get("renewable") else "conventional")
    generator_type = basic[1].selectbox(
        "Type", type_options, index=type_options.index(default_type) if default_type in type_options else 0,
        key=f"generator_type_{index}"
    )
    capacity = basic[2].number_input(
        "Capacity (MW)", min_value=0.0, value=float(defaults["capacity"]), step=5.0,
        key=f"generator_capacity_{index}"
    )
    available_capacity = basic[3].number_input(
        "Available now (MW)", min_value=0.0, value=float(defaults.get("available_capacity", defaults["capacity"])), step=5.0,
        key=f"generator_available_{index}"
    )

    technical = st.columns([1.1, 1.1, 1.1, 1.1, 1.1])
    cost_per_mwh = technical[0].number_input(
        "Cost (Rs./MWh)", min_value=0.0, value=float(defaults.get("cost_per_mwh", defaults.get("cost_per_mw", 0.0))), step=0.5,
        key=f"generator_cost_{index}"
    )
    emission_factor = technical[1].number_input(
        "Emissions (kg/MWh)", min_value=0.0, value=float(defaults.get("emission_factor", 0.0)), step=0.01,
        key=f"generator_emissions_{index}"
    )
    minimum_output = technical[2].number_input(
        "Minimum output (MW)", min_value=0.0, value=float(defaults.get("minimum_output", 0.0)), step=5.0,
        key=f"generator_minimum_{index}"
    )
    ramp_rate = technical[3].number_input(
        "Ramp rate (MW/interval)", min_value=0.0, value=float(defaults.get("ramp_rate", 100.0)), step=5.0,
        key=f"generator_ramp_{index}"
    )
    previous_output = technical[4].number_input(
        "Previous output (MW)", min_value=0.0, value=float(defaults.get("previous_output", 0.0)), step=5.0,
        key=f"generator_previous_{index}"
    )
    generators.append({
        "name": name.strip() or f"Generator {index + 1}",
        "type": generator_type,
        "capacity": capacity,
        "available_capacity": available_capacity,
        "cost_per_mwh": cost_per_mwh,
        "emission_factor": emission_factor,
        "minimum_output": minimum_output,
        "ramp_rate": ramp_rate,
        "previous_output": previous_output,
    })

st.subheader("2. Enter Electricity Demand")
zone_columns = st.columns(4)
demand_zones: list[dict] = []
for index, (zone_name, default_demand, priority, shed_order) in enumerate(DEFAULT_ZONES):
    with zone_columns[index]:
        st.markdown(f"**{zone_name}**")
        demand = st.number_input(
            "Demand (MW)", min_value=0.0, value=default_demand, step=5.0,
            key=f"demand_zone_{index}", label_visibility="collapsed",
        )
        if priority == "critical":
            st.caption("Critical — protected first")
        elif zone_name == "Commercial":
            st.caption("Low — shed first")
        elif zone_name == "Industrial":
            st.caption("Medium — shed second")
        else:
            st.caption("Medium — protected before industrial")
        demand_zones.append({
            "name": zone_name,
            "demand": demand,
            "priority": priority,
            "shed_order": shed_order,
        })

failure_generator = None
if scenario == "Generator failure":
    failure_generator = st.sidebar.selectbox(
        "Generator to fail",
        [generator["name"] for generator in generators],
    )

generators, demand_zones = apply_scenario(
    generators, demand_zones, scenario, failure_generator
)
if scenario != "Base case":
    st.warning(f"Scenario active: {scenario}")

total_demand = sum(zone["demand"] for zone in demand_zones)
physical_capacity = sum(generator["capacity"] for generator in generators)
available_capacity = sum(
    min(generator["capacity"], generator["available_capacity"])
    * (renewable_availability / 100.0 if is_renewable(generator) else 1.0)
    for generator in generators
)
metric_one, metric_two, metric_three = st.columns(3)
metric_one.metric("Total demand", f"{total_demand:.1f} MW")
metric_two.metric("Physical capacity", f"{physical_capacity:.1f} MW")
metric_three.metric("Current available capacity", f"{available_capacity:.1f} MW")

st.divider()
if st.button("Run Hybrid Fuzzy–PSO Optimization", type="primary", use_container_width=True):
    settings = {
        "swarm_size": int(swarm_size),
        "iterations": int(iterations),
        "inertia_weight": float(inertia_weight),
        "cognitive_weight": float(cognitive_weight),
        "social_weight": float(social_weight),
        "random_seed": int(random_seed),
        "emission_cost_per_kg": float(emission_cost_per_kg),
    }
    battery = {
        "capacity": float(battery_capacity),
        "soc": float(battery_soc),
        "cost_per_mwh": float(battery_cost),
    }

    try:
        with st.spinner("Fuzzy controller is setting priorities and PSO is optimizing..."):
            fuzzy_started = time.perf_counter()
            result = optimize_loads(
                generators,
                total_demand,
                settings,
                renewable_availability=renewable_availability / 100.0,
                battery=battery,
                demand_zones=demand_zones,
            )
            result["runtime_seconds"] = time.perf_counter() - fuzzy_started
            comparison = compare_algorithms(
                generators,
                total_demand,
                settings,
                renewable_availability=renewable_availability / 100.0,
                battery=battery,
                demand_zones=demand_zones,
                fuzzy_result=result,
                fuzzy_runtime_seconds=result["runtime_seconds"],
            )
            save_charts(
                generators,
                result,
                output_dir=RESULTS_DIR,
                comparison_rows=comparison["rows"],
            )

        st.success("Hybrid optimization completed successfully.")
        rows = []
        for index, (generator, load, priority) in enumerate(
            zip(generators, result["allocation"], result["fuzzy"]["generator_priorities"])
        ):
            dispatch_limit = result["dispatch_upper_bounds"][index]
            emissions = float(load) * float(generator["emission_factor"])
            usage = (load / dispatch_limit * 100) if dispatch_limit else 0.0
            rows.append({
                "Generator": generator["name"],
                "Type": generator["type"].title(),
                "Load (MW)": round(float(load), 2),
                "Available (MW)": round(float(result["available_capacities"][index]), 2),
                "Dispatch limit (MW)": round(float(dispatch_limit), 2),
                "Minimum (MW)": round(float(result["minimum_outputs"][index]), 2),
                "Usage (%)": round(usage, 1),
                "Fuzzy priority": round(float(priority), 3),
                "Emissions (kg)": round(emissions, 2),
                "Status": "SAFE" if load <= dispatch_limit + 1e-6 else "OVERLOAD",
            })

        st.subheader("Optimized Result")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.subheader("Demand-Zone Allocation")
        st.dataframe(
            pd.DataFrame(result["zone_allocations"]),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Algorithm Comparison")
        st.caption(
            "All methods use the same generator bounds, battery, zone priorities, "
            "emission factors, and shortage rules. Plain PSO disables fuzzy priorities."
        )
        comparison_rows = [
            {
                "Method": row["Method"],
                "Cost (Rs./h)": round(row["Cost"], 2),
                "Emissions (kg)": round(row["Emissions"], 2),
                "Runtime (ms)": round(row["Runtime"] * 1000, 2),
                "Shortage (MW)": round(row["Shortage"], 2),
            }
            for row in comparison["rows"]
        ]
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        st.subheader("Dashboard Overview")
        dashboard_metrics = st.columns(5)
        dashboard_metrics[0].metric("Renewable energy", f"{result['renewable_percentage']:.1f}%")
        dashboard_metrics[1].metric("Total emissions", f"{result['total_emissions']:.2f} kg")
        dashboard_metrics[2].metric("Reserve margin", f"{result['reserve_margin']:.1f}%")
        dashboard_metrics[3].metric(
            "Average utilization",
            f"{float(result['generator_utilization'].mean()) * 100:.1f}%",
        )
        dashboard_metrics[4].metric("Shortage", f"{result['load_shed']:.2f} MW")
        st.subheader("Operational Metrics")
        result_metrics = st.columns(5)
        result_metrics[0].metric("Power served", f"{result['served_demand']:.2f} MW")
        result_metrics[1].metric("Battery discharge", f"{result['battery_discharge']:.2f} MW")
        result_metrics[2].metric("Load shed", f"{result['load_shed']:.2f} MW")
        result_metrics[3].metric("Operating cost", f"Rs. {result['total_cost']:.2f}/h")
        result_metrics[4].metric("Emissions", f"{result['total_emissions']:.2f} kg")

        fuzzy = result["fuzzy"]
        st.subheader("Fuzzy Decision Layer")
        fuzzy_one, fuzzy_two, fuzzy_three = st.columns(3)
        fuzzy_one.metric("Battery priority", f"{fuzzy['battery_discharge_priority']:.2f}")
        fuzzy_two.metric("Load-shedding priority", f"{fuzzy['load_shedding_priority']:.2f}")
        fuzzy_three.metric("Conventional support", f"{fuzzy['conventional_support']:.2f}")
        fuzzy_inputs = {
            "Grid stress": fuzzy["inputs"]["grid_stress"],
            "Renewable availability": fuzzy["inputs"]["renewable_availability"],
            "Reserve margin": fuzzy["inputs"]["reserve_margin"],
        }
        st.dataframe(
            pd.DataFrame([{"Input": key, "Normalized value": round(value, 3)} for key, value in fuzzy_inputs.items()]),
            use_container_width=True,
            hide_index=True,
        )

        chart_one, chart_two, chart_three, chart_four, chart_five = st.columns(5)
        with chart_one:
            st.image(RESULTS_DIR / "capacity_vs_load.png", caption="Physical, available, and dispatched capacity")
        with chart_two:
            st.image(RESULTS_DIR / "pso_convergence.png", caption="PSO convergence")
        with chart_three:
            st.image(RESULTS_DIR / "fuzzy_priorities.png", caption="Fuzzy priorities used by PSO")
        with chart_four:
            st.image(RESULTS_DIR / "emissions_by_generator.png", caption="Estimated emissions")
        with chart_five:
            st.image(RESULTS_DIR / "zone_supply.png", caption="Zone supply and shedding")
        st.image(RESULTS_DIR / "cost_comparison.png", caption="Algorithm cost comparison")
        st.image(RESULTS_DIR / "fuzzy_memberships.png", caption="Fuzzy membership graphs")
        st.image(RESULTS_DIR / "generator_utilization.png", caption="Generator utilization")
    except ValueError as error:
        st.error(str(error))

with st.expander("How this hybrid project works"):
    st.write(
        "The fuzzy controller evaluates grid stress, renewable availability, "
        "reserve margin, demand, and generator cost. It produces generator, "
        "battery, load-shedding, and conventional-support priorities. PSO then "
        "searches for the lowest-cost allocation while respecting each "
        "generator's physical capacity, current available capacity, minimum "
        "output, ramp limit, emissions, battery limit, critical-load protection, "
        "and demand balance."
    )
