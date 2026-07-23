"""Local pytest config for the benchmark suite (not collected by default CI)."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow-tier benchmarks (MILP k-medoids, large column counts).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: slow-tier benchmark, only runs with --slow"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="slow tier: pass --slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
