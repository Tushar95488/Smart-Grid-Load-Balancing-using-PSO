"""Priority-aware demand-zone allocation and controlled load shedding."""

from __future__ import annotations


PRIORITY_PROTECTION_RANK = {
    "low": 1,
    "medium": 2,
    "critical": 3,
}
SHEDDING_PENALTY = {
    "low": 10_000.0,
    "medium": 50_000.0,
    "critical": 100_000.0,
}


def infer_zone_priority(zone: dict) -> str:
    """Infer the requested priority for older zone records."""
    if bool(zone.get("critical", False)):
        return "critical"
    explicit = str(zone.get("priority", "")).strip().lower()
    if explicit in PRIORITY_PROTECTION_RANK:
        return explicit
    name = str(zone.get("name", "")).strip().lower()
    if "hospital" in name or "emergency" in name:
        return "critical"
    if "commercial" in name:
        return "low"
    return "medium"


def normalize_demand_zones(zones: list[dict] | None, total_demand: float) -> list[dict]:
    """Return validated zone records with explicit protection metadata."""
    if not zones:
        return [{
            "name": "Total demand",
            "demand": float(total_demand),
            "priority": "critical",
            "shed_order": 100,
        }]

    normalized = []
    for index, zone in enumerate(zones):
        demand = float(zone.get("demand", 0.0))
        if demand < 0:
            raise ValueError(f"Demand for {zone.get('name', f'zone {index + 1}')} cannot be negative.")
        priority = infer_zone_priority(zone)
        name = str(zone.get("name", f"Zone {index + 1}")).strip() or f"Zone {index + 1}"

        # Residential and industrial are both medium priority, but residential
        # is protected first so industrial is shed second as requested.
        default_shed_order = {
            "hospital & emergency": 100,
            "residential": 30,
            "industrial": 20,
            "commercial": 10,
        }.get(name.lower(), 15 - index)
        normalized.append({
            "name": name,
            "demand": demand,
            "priority": priority,
            "shed_order": float(zone.get("shed_order", default_shed_order)),
        })
    return normalized


def allocate_power_to_zones(
    zones: list[dict] | None,
    available_supply: float,
    total_demand: float | None = None,
) -> dict:
    """Allocate available supply and shed loads in controlled priority order.

    Protection order is critical first, then medium-priority zones, then low
    priority zones. For equal medium priority, residential is protected before
    industrial; commercial is therefore shed first, followed by industrial.
    """
    if total_demand is None:
        total_demand = sum(float(zone.get("demand", 0.0)) for zone in zones or [])
    normalized = normalize_demand_zones(zones, float(total_demand))
    remaining_supply = max(0.0, float(available_supply))
    indexed_zones = list(enumerate(normalized))
    indexed_zones.sort(
        key=lambda item: (
            PRIORITY_PROTECTION_RANK[item[1]["priority"]],
            item[1]["shed_order"],
        ),
        reverse=True,
    )

    allocated_by_index: dict[int, dict] = {}
    weighted_shed_penalty = 0.0
    for index, zone in indexed_zones:
        requested = float(zone["demand"])
        supplied = min(requested, remaining_supply)
        shed = requested - supplied
        remaining_supply -= supplied
        weighted_shed_penalty += shed * SHEDDING_PENALTY[zone["priority"]]
        allocated_by_index[index] = {
            "name": zone["name"],
            "priority": zone["priority"],
            "requested_demand": requested,
            "supplied_demand": supplied,
            "load_shed": shed,
            "status": "SERVED" if shed < 1e-7 else "SHED",
        }

    ordered_results = [allocated_by_index[index] for index in range(len(normalized))]
    critical_shortfall = sum(
        row["load_shed"] for row in ordered_results if row["priority"] == "critical"
    )
    return {
        "zones": ordered_results,
        "total_requested": float(sum(row["requested_demand"] for row in ordered_results)),
        "total_supplied": float(sum(row["supplied_demand"] for row in ordered_results)),
        "total_shed": float(sum(row["load_shed"] for row in ordered_results)),
        "weighted_shed_penalty": float(weighted_shed_penalty),
        "critical_shortfall": float(critical_shortfall),
        "unused_supply": max(0.0, remaining_supply),
    }
