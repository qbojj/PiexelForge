import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="Run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="Skipping slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
