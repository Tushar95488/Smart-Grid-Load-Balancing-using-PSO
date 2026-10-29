"""Interactive Streamlit frontend for Smart Grid Load Balancing using PSO."""

from pathlib import Path

import pandas as pd
import streamlit as st

from pso_optimizer import optimize_loads
from visualization import save_charts


st.set_page_config(
    page_title="Smart Grid PSO Optimizer",
    page_icon="⚡",
    layout="wide",
)


DEFAULT_GENERATORS = [
    ("Thermal Plant", 100.0, 6.5),
    ("Solar Farm", 55.0, 2.5),
    ("Wind Farm", 45.0, 3.0),
    ("Hydro Plant", 90.0, 4.0),
]
DEFAULT_ZONES = [
    ("Residential", 75.0),
    ("Industrial", 60.0),
    ("Commercial", 35.0),
    ("Hospital & Emergency", 20.0),
]


def generator_defaults(index: int) -> tuple[str, float, float]:
    """Return a sensible default for a generator input row."""
    if index < len(DEFAULT_GENERATORS):
        return DEFAULT_GENERATORS[index]
    return f"Generator {index + 1}", 50.0, 5.0


st.title("⚡ Smart Grid Load Balancing")
st.caption("Particle Swarm Optimization (PSO) simulation for safe, low-cost power allocation.")

with st.sidebar:
    st.header("PSO Settings")
    swarm_size = st.slider("Swarm size", min_value=10, max_value=100, value=40, step=5)
    iterations = st.slider("Iterations", min_value=20, max_value=500, value=150, step=10)
    inertia_weight = st.slider("Inertia weight (w)", 0.1, 1.0, 0.70, 0.05)
    cognitive_weight = st.slider("Cognitive weight (c1)", 0.1, 3.0, 1.45, 0.05)
    social_weight = st.slider("Social weight (c2)", 0.1, 3.0, 1.45, 0.05)
    random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

st.subheader("1. Enter Generator Details")
generator_count = st.number_input("Number of generators", min_value=2, max_value=10, value=4, step=1)

generators: list[dict] = []
header = st.columns([2.2, 1.1, 1.1])
header[0].markdown("**Generator name**")
header[1].markdown("**Capacity (MW)**")
header[2].markdown("**Cost per MW (Rs.)**")

for index in range(generator_count):
    default_name, default_capacity, default_cost = generator_defaults(index)
    columns = st.columns([2.2, 1.1, 1.1])
    name = columns[0].text_input("Generator name", default_name, key=f"generator_name_{index}", label_visibility="collapsed")
    capacity = columns[1].number_input("Capacity", min_value=0.0, value=default_capacity, step=5.0, key=f"generator_capacity_{index}", label_visibility="collapsed")
    cost = columns[2].number_input("Cost", min_value=0.0, value=default_cost, step=0.5, key=f"generator_cost_{index}", label_visibility="collapsed")
    generators.append({"name": name.strip() or f"Generator {index + 1}", "capacity": capacity, "cost_per_mw": cost})

st.subheader("2. Enter Electricity Demand")
zone_columns = st.columns(4)
demand_zones: list[dict] = []
for index, (zone_name, default_demand) in enumerate(DEFAULT_ZONES):
    with zone_columns[index]:
        st.markdown(f"**{zone_name}**")
        demand = st.number_input(
            "Demand (MW)", min_value=0.0, value=default_demand, step=5.0,
            key=f"demand_zone_{index}", label_visibility="collapsed",
        )
        demand_zones.append({"name": zone_name, "demand": demand})

total_demand = sum(zone["demand"] for zone in demand_zones)
total_capacity = sum(generator["capacity"] for generator in generators)
metric_one, metric_two, metric_three = st.columns(3)
metric_one.metric("Total demand", f"{total_demand:.1f} MW")
metric_two.metric("Total grid capacity", f"{total_capacity:.1f} MW")
metric_three.metric("Reserve capacity", f"{total_capacity - total_demand:.1f} MW")

st.divider()
if st.button("Optimize Load Allocation", type="primary", use_container_width=True):
    settings = {
        "swarm_size": int(swarm_size),
        "iterations": int(iterations),
        "inertia_weight": float(inertia_weight),
        "cognitive_weight": float(cognitive_weight),
        "social_weight": float(social_weight),
        "random_seed": int(random_seed),
    }

    try:
        with st.spinner("PSO is finding the best safe allocation..."):
            result = optimize_loads(generators, total_demand, settings)
            save_charts(generators, result)

        st.success("Optimization completed successfully.")
        rows = []
        for generator, load in zip(generators, result["allocation"]):
            capacity = generator["capacity"]
            usage = (load / capacity * 100) if capacity else 0.0
            rows.append({
                "Generator": generator["name"],
                "Assigned Load (MW)": round(float(load), 2),
                "Capacity (MW)": capacity,
                "Usage (%)": round(usage, 1),
                "Status": "SAFE" if load <= capacity + 1e-6 else "OVERLOAD",
            })

        st.subheader("Optimized Result")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        result_one, result_two, result_three = st.columns(3)
        result_one.metric("Power supplied", f"{sum(result['allocation']):.2f} MW")
        result_two.metric("Operating cost", f"Rs. {result['total_cost']:.2f}/hour")
        result_three.metric("Best fitness", f"{result['best_fitness']:.2f}")

        chart_one, chart_two = st.columns(2)
        with chart_one:
            st.image(Path("results") / "capacity_vs_load.png", caption="Capacity vs optimized load")
        with chart_two:
            st.image(Path("results") / "pso_convergence.png", caption="PSO convergence over iterations")
    except ValueError as error:
        st.error(str(error))

with st.expander("How this project works"):
    st.write(
        "Every particle represents one possible generator-load allocation. "
        "PSO improves these allocations over several iterations. The backend "
        "minimizes operating cost while ensuring total supply equals demand "
        "and no generator crosses its capacity."
    )
