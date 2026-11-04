"""Explicit local adapters. No discovery service or automatic backend fallback."""

from .base import BenchmarkAdapter
from .swe import SWEBenchmark

ADAPTERS = {"swe": SWEBenchmark}


def get_benchmark(config) -> BenchmarkAdapter:
    name = config.get("benchmark", "swe")
    if name not in ADAPTERS:
        raise ValueError(f"Unknown benchmark adapter: {name}")
    return ADAPTERS[name]()


def configured_tasks(config, role, limit):
    return get_benchmark(config).load_tasks(config, role, limit)


def require_swe(config, operation):
    if get_benchmark(config).name != "swe":
        raise NotImplementedError(f"{operation} is SWE-specific; only task loading/execution/verification are adapted")
