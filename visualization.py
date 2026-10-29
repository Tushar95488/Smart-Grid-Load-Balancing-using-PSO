"""Charts created after the PSO simulation finishes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_charts(generators: list[dict], result: dict, output_dir: str = "results") -> None:
    """Save capacity/load and PSO convergence charts as PNG files."""
    destination = Path(output_dir)
    destination.mkdir(exist_ok=True)

    names = [generator["name"] for generator in generators]
    allocation = result["allocation"]
    capacities = result["capacities"]
    indices = np.arange(len(names))

    plt.figure(figsize=(10, 5))
    width = 0.36
    plt.bar(indices - width / 2, capacities, width, label="Maximum capacity", color="#9ecae1")
    plt.bar(indices + width / 2, allocation, width, label="PSO assigned load", color="#238b45")
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
