"""Finite configurable spectral instrument settings and resource semantics."""

SPECTRAL_INSTRUMENTS = ("nmr", "ir", "ms")
SPECTRAL_FIELDS = ("scan_count", "resolution_factor", "dilution_factor")
SPECTRAL_BOUNDS = {
    "scan_count": (1.0, 128.0, "count", 8.0),
    "resolution_factor": (0.5, 8.0, "dimensionless", 1.0),
    "dilution_factor": (1.0, 1000.0, "dimensionless", 1.0),
}
SPECTRAL_COST = {"nmr": 0.10, "ir": 0.04, "ms": 0.08}
SPECTRAL_SAMPLE_L = {"nmr": 0.0003, "ir": 0.00005, "ms": 0.00001}
SCAN_TIME_S = {"nmr": 2.0, "ir": 0.5, "ms": 0.1}


def acquisition_seconds(instrument: str, config: dict) -> float:
    return 5.0 + SCAN_TIME_S[instrument] * config["scan_count"]


def acquisition_cost(instrument: str, config: dict) -> float:
    return SPECTRAL_COST[instrument] + 0.0002 * acquisition_seconds(instrument, config)


def default_spectral_settings() -> dict[str, float]:
    return {key: bounds[3] for key, bounds in SPECTRAL_BOUNDS.items()}
