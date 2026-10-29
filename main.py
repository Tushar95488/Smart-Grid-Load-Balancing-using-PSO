"""Run the Smart Grid Load Balancing Using PSO project."""

from grid_data import DEMAND_ZONES, GENERATORS, PSO_SETTINGS
from pso_optimizer import optimize_loads
from visualization import save_charts


def main() -> None:
    total_demand = sum(zone["demand"] for zone in DEMAND_ZONES)
    total_capacity = sum(generator["capacity"] for generator in GENERATORS)

    print("=" * 62)
    print(" SMART GRID LOAD BALANCING USING PARTICLE SWARM OPTIMIZATION")
    print("=" * 62)
    print("\nDemand by zone:")
    for zone in DEMAND_ZONES:
        print(f"  {zone['name']:<24} {zone['demand']:>6.1f} MW")
    print(f"\nTotal demand: {total_demand:.1f} MW")
    print(f"Total grid capacity: {total_capacity:.1f} MW")

    result = optimize_loads(GENERATORS, total_demand, PSO_SETTINGS)

    print("\nOptimized generator allocation:")
    print("-" * 62)
    print(f"{'Generator':<18}{'Load':>10}{'Capacity':>12}{'Usage':>10}{'Status':>12}")
    print("-" * 62)
    for generator, assigned_load in zip(GENERATORS, result["allocation"]):
        capacity = generator["capacity"]
        usage = (assigned_load / capacity) * 100
        status = "SAFE" if assigned_load <= capacity + 1e-6 else "OVERLOAD"
        print(
            f"{generator['name']:<18}{assigned_load:>8.2f} MW"
            f"{capacity:>9.1f} MW{usage:>8.1f}%{status:>12}"
        )

    print("-" * 62)
    print(f"Total power supplied : {sum(result['allocation']):.2f} MW")
    print(f"Total operating cost : Rs. {result['total_cost']:.2f} per hour")
    print(f"Best fitness value   : {result['best_fitness']:.2f}")

    save_charts(GENERATORS, result)
    print("\nCharts saved in the results folder:")
    print("  - capacity_vs_load.png")
    print("  - pso_convergence.png")


if __name__ == "__main__":
    main()
