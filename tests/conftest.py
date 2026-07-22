import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "blender: tests that launch real Blender background runs (slow); "
        "select with -m blender, skip with -m 'not blender'",
    )
