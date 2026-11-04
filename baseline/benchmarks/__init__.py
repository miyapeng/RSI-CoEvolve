"""Explicit local adapters. No discovery service or automatic backend fallback."""

from .arc_agi_3 import ARCAGI3
from .base import BenchmarkAdapter
from .osworld import OSWorldVerified
from .swe import SWEBenchmark
from .vision2web import Vision2Web

ADAPTERS = {"swe": SWEBenchmark, "vision2web": Vision2Web, "osworld_verified": OSWorldVerified, "arc_agi_3": ARCAGI3}


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
