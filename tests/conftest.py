import os
import pkg_resources
from pytest import fixture

RESOURCES_DIR = "resources"


def pytest_addoption(parser):
    parser.addoption("--image_uri", action="store", help="image uri to test")


@fixture(scope="module")
def resource_path() -> str:
    print(os.getcwd())
    return pkg_resources.resource_filename(RESOURCES_DIR, "")


@fixture(scope="module")
def sample_file(resource_path):
    return os.path.join(resource_path, "chunker-test.txt")


@fixture(scope="module")
def rel_tol() -> float:
    """Relative tolerance for pytest approximation."""
    return 0.01
