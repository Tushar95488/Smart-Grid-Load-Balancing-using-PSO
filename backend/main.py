"""Run the hybrid Fuzzy–PSO smart-grid project in terminal mode."""

from pathlib import Path
import time

from backend.comparison import compare_algorithms
from backend.pso_optimizer import optimize_loads
from shared.grid_data import BATTERY, DEMAND_ZONES, GENERATORS, PSO_SETTINGS
from shared.visualization import save_charts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    total_demand = sum(zone["demand"] for zone in DEMAND_ZONES)
    total_capacity = sum(generator["capacity"] for generator in GENERATORS)

    print("=" * 62)
    print(" HYBRID FUZZY–PSO SMART GRID LOAD BALANCING")
    print("=" * 62)
    print("\nDemand by zone:")
    for zone in DEMAND_ZONES:
        print(f"  {zone['name']:<24} {zone['demand']:>6.1f} MW")
    print(f"\nTotal demand: {total_demand:.1f} MW")
    print(f"Total grid capacity: {total_capacity:.1f} MW")

    fuzzy_started = time.perf_counter()
    result = optimize_loads(
        GENERATORS,
        total_demand,
        PSO_SETTINGS,
        renewable_availability=1.00,
        battery=BATTERY,
        demand_zones=DEMAND_ZONES,
    )
    fuzzy_runtime = time.perf_counter() - fuzzy_started
    comparison = compare_algorithms(
        GENERATORS,
        total_demand,
        PSO_SETTINGS,
        renewable_availability=1.0,
        battery=BATTERY,
        demand_zones=DEMAND_ZONES,
        fuzzy_result=result,
        fuzzy_runtime_seconds=fuzzy_runtime,
    )

    print("\nOptimized generator allocation:")
    print("-" * 62)
    print(f"{'Generator':<18}{'Load':>10}{'Limit':>12}{'Usage':>10}{'Status':>12}")
    print("-" * 62)
    for generator, assigned_load in zip(GENERATORS, result["allocation"]):
        capacity = result["dispatch_upper_bounds"][GENERATORS.index(generator)]
        usage = (assigned_load / capacity) * 100
        status = "SAFE" if assigned_load <= capacity + 1e-6 else "OVERLOAD"
        print(
            f"{generator['name']:<18}{assigned_load:>8.2f} MW"
            f"{capacity:>9.1f} MW{usage:>8.1f}%{status:>12}"
        )

    print("-" * 62)
    print(f"Generator power      : {sum(result['allocation']):.2f} MW")
    print(f"Battery discharge    : {result['battery_discharge']:.2f} MW")
    print(f"Load shed            : {result['load_shed']:.2f} MW")
    print(f"Total operating cost : Rs. {result['total_cost']:.2f} per hour")
    print(f"Estimated emissions  : {result['total_emissions']:.2f} kg CO2")
    print(f"Carbon cost weight   : Rs. {PSO_SETTINGS['emission_cost_per_kg']:.2f}/kg")
    print(f"Best fitness value   : {result['best_fitness']:.2f}")
    print("\nFuzzy priorities:")
    for generator, priority in zip(GENERATORS, result["fuzzy"]["generator_priorities"]):
        print(f"  {generator['name']:<24} {priority:.2f}")
    print(f"  Battery discharge priority: {result['fuzzy']['battery_discharge_priority']:.2f}")
    print(f"  Load-shedding priority:     {result['fuzzy']['load_shedding_priority']:.2f}")
    print(f"  Conventional support:       {result['fuzzy']['conventional_support']:.2f}")

    print("\nDemand-zone allocation:")
    print(f"{'Zone':<24}{'Priority':>12}{'Requested':>12}{'Supplied':>12}{'Shed':>10}")
    print("-" * 70)
    for zone in result["zone_allocations"]:
        print(
            f"{zone['name']:<24}{zone['priority']:>12}"
            f"{zone['requested_demand']:>9.1f} MW{zone['supplied_demand']:>9.1f} MW"
            f"{zone['load_shed']:>7.1f} MW"
        )

    print("\nAlgorithm comparison:")
    print(f"{'Method':<22}{'Cost':>12}{'Emissions':>14}{'Runtime (ms)':>14}{'Shortage':>12}")
    print("-" * 74)
    for row in comparison["rows"]:
        print(
            f"{row['Method']:<22}Rs. {row['Cost']:>7.2f}"
            f"{row['Emissions']:>11.2f} kg{row['Runtime'] * 1000:>11.2f}"
            f"{row['Shortage']:>9.2f} MW"
        )

    save_charts(
        GENERATORS,
        result,
        output_dir=PROJECT_ROOT / "results",
        comparison_rows=comparison["rows"],
    )
    print("\nCharts saved in the results folder:")
    print("  - capacity_vs_load.png")
    print("  - pso_convergence.png")
    print("  - fuzzy_priorities.png")
    print("  - emissions_by_generator.png")
    print("  - zone_supply.png")
    print("  - fuzzy_memberships.png")
    print("  - generator_utilization.png")
    print("  - cost_comparison.png")


if __name__ == "__main__":
    main()
