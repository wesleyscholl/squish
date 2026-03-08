"""
tests/conftest.py
Shared pytest configuration for all Squish tests.
"""
import warnings


def pytest_addoption(parser):
    parser.addoption(
        "--model", default=None,
        help="Model hint passed to squish --model  (e.g. '14b', '7b', full path)"
    )


def pytest_configure(config):
    # SwigPy warnings from compiled MLX/Metal bindings — not actionable
    warnings.filterwarnings("ignore", message="builtin type SwigPy.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="builtin type swigvar.*", category=DeprecationWarning)
    # Starlette TestClient timeout kwarg deprecation — upstream issue
    warnings.filterwarnings("ignore", message="You should not use the 'timeout'.*", category=DeprecationWarning)
