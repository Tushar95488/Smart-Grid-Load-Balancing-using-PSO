"""Charts created after the PSO simulation finishes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _membership_curves(values: np.ndarray) -> dict[str, np.ndarray]:
    """Match the fuzzy controller's triangular membership functions."""
    return {
        "Low": np.maximum(0.0, 1.0 - 2.0 * values),
        "Medium": np.maximum(0.0, 1.0 - 2.0 * np.abs(values - 0.5)),
        "High": np.maximum(0.0, 2.0 * values - 1.0),
    }


def save_charts(
    generators: list[dict],
    result: dict,
    output_dir: str = "results",
    comparison_rows: list[dict] | None = None,
) -> None:
    """Save dashboard charts for dispatch, fuzzy logic, zones, and comparisons."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    names = [generator["name"] for generator in generators]
    allocation = result["allocation"]
    physical_capacities = result.get("physical_capacities", result["capacities"])
    available_capacities = result.get("available_capacities", result["capacities"])
    indices = np.arange(len(names))

    plt.figure(figsize=(10, 5))
    width = 0.25
    plt.bar(indices - width, physical_capacities, width, label="Physical capacity", color="#cbd5e1")
    plt.bar(indices, available_capacities, width, label="Available capacity", color="#9ecae1")
    plt.bar(indices + width, allocation, width, label="PSO assigned load", color="#238b45")
    plt.xticks(indices, names, rotation=15, ha="right")
    plt.ylabel("Power (MW)")
    plt.title("Generator Capacity vs Optimized Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination / "capacity_vs_load.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(result["history"], color="#6a3d9a", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness / total cost")
    plt.title("PSO Convergence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(destination / "pso_convergence.png", dpi=160)
    plt.close()

    priorities = result["fuzzy"]["generator_priorities"]
    plt.figure(figsize=(10, 5))
    plt.bar(names, priorities, color="#0e7490")
    plt.ylim(0, 1)
    plt.ylabel("Priority (0 to 1)")
    plt.title("Fuzzy Generator Priorities Used by PSO")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(destination / "fuzzy_priorities.png", dpi=160)
    plt.close()

    if result.get("fuzzy"):
        fuzzy_inputs = result["fuzzy"]["inputs"]
        input_labels = {
            "grid_stress": "Grid stress",
            "renewable_availability": "Renewable availability",
            "reserve_margin": "Reserve margin",
            "demand_ratio": "Demand ratio",
        }
        curve_x = np.linspace(0.0, 1.0, 200)
        curves = _membership_curves(curve_x)
        figure, axes = plt.subplots(2, 2, figsize=(11, 7))
        for axis, (key, title) in zip(axes.flat, input_labels.items()):
            for label, curve in curves.items():
                axis.plot(curve_x, curve, label=label)
            current_value = float(fuzzy_inputs[key])
            axis.axvline(current_value, color="#b91c1c", linestyle="--", linewidth=1.5)
            axis.set_title(f"{title} ({current_value:.2f})")
            axis.set_ylim(0, 1.05)
            axis.set_xlim(0, 1)
            axis.grid(alpha=0.25)
        axes[0, 0].legend(loc="upper left", fontsize=8)
        figure.suptitle("Fuzzy Membership Functions and Current Scenario")
        figure.tight_layout()
        figure.savefig(destination / "fuzzy_memberships.png", dpi=160)
        plt.close(figure)

    utilization = np.asarray(
        result.get("generator_utilization", np.zeros(len(names))), dtype=float
    )
    plt.figure(figsize=(10, 5))
    plt.bar(names, utilization * 100.0, color="#7c3aed")
    plt.ylabel("Utilization (%)")
    plt.ylim(0, 110)
    plt.title("Generator Utilization")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(destination / "generator_utilization.png", dpi=160)
    plt.close()

    emission_factors = [float(generator.get("emission_factor", 0.0)) for generator in generators]
    emissions = np.asarray(allocation, dtype=float) * np.asarray(emission_factors, dtype=float)
    plt.figure(figsize=(10, 5))
    plt.bar(names, emissions, color="#b91c1c")
    plt.ylabel("Estimated emissions (kg CO2 per interval)")
    plt.title("Estimated Generator Emissions")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(destination / "emissions_by_generator.png", dpi=160)
    plt.close()

    zone_rows = result.get("zone_allocations", [])
    if zone_rows:
        zone_names = [row["name"] for row in zone_rows]
        requested = [row["requested_demand"] for row in zone_rows]
        supplied = [row["supplied_demand"] for row in zone_rows]
        zone_indices = np.arange(len(zone_names))
        plt.figure(figsize=(10, 5))
        plt.bar(zone_indices - width / 2, requested, width, label="Requested demand", color="#cbd5e1")
        plt.bar(zone_indices + width / 2, supplied, width, label="Supplied demand", color="#2563eb")
        plt.xticks(zone_indices, zone_names, rotation=15, ha="right")
        plt.ylabel("Power (MW)")
        plt.title("Demand-Zone Supply and Controlled Shedding")
        plt.legend()
        plt.tight_layout()
        plt.savefig(destination / "zone_supply.png", dpi=160)
        plt.close()

    if comparison_rows:
        methods = [row["Method"] for row in comparison_rows]
        costs = [float(row["Cost"]) for row in comparison_rows]
        plt.figure(figsize=(9, 5))
        bars = plt.bar(methods, costs, color=["#16a34a", "#2563eb", "#f59e0b", "#7c3aed"])
        plt.ylabel("Operating cost (Rs./hour)")
        plt.title("Algorithm Cost Comparison")
        plt.xticks(rotation=15, ha="right")
        for bar, cost in zip(bars, costs):
            plt.text(bar.get_x() + bar.get_width() / 2, cost, f"{cost:.1f}", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(destination / "cost_comparison.png", dpi=160)
        plt.close()
