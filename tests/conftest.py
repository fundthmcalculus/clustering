"""Global pytest configuration for the tribble-clustering test suite.

Adds a single "CI-fast" switch that trims the suite's wall-clock runtime when
it runs on a shared CI runner (notably GitHub Actions). Several tests here are
scaling / plotting benchmarks whose timings are meaningless on a noisy shared
runner (the ``benchmark`` marker already exists for the worst of them), and a
few correctness tests run on large inputs purely to exercise batching. Neither
adds signal in CI proportional to the seconds it costs.

The switch is enabled when ANY of the following is true:

  * the ``--ci-fast`` command-line option is passed,
  * the ``TRIBBLE_CI_FAST`` environment variable is truthy, or
  * the run is on GitHub Actions (``GITHUB_ACTIONS=true``) or a generic CI
    provider (``CI=true``) -- so it activates automatically in the pipeline
    with no workflow change required.

Two things happen in CI-fast mode:

  * tests marked ``@pytest.mark.ci_slow`` are deselected (scope reduction) --
    these are the pure wall-clock/plot benchmarks that assert no correctness;
    and
  * the ``ci_scale`` fixture returns reduced dataset sizes (size reduction),
    so the remaining heavy correctness tests run on smaller inputs.

Everything runs full-size locally, so developers keep the complete signal.
"""

import os

import pytest

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    """True when environment variable ``name`` is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def ci_fast_enabled() -> bool:
    """Return True when the suite should run in reduced CI-fast mode.

    Reads the environment only, so it is safe to call at import / parametrize
    time (before pytest's ``config`` object exists). ``pytest_configure``
    mirrors the ``--ci-fast`` command-line option into ``TRIBBLE_CI_FAST`` so
    this helper stays authoritative regardless of how the mode was requested.
    """
    return (
        _env_flag("TRIBBLE_CI_FAST") or _env_flag("GITHUB_ACTIONS") or _env_flag("CI")
    )


def pytest_addoption(parser):
    parser.addoption(
        "--ci-fast",
        action="store_true",
        default=False,
        help=(
            "Run the suite in reduced-scope/size mode: skip @pytest.mark.ci_slow "
            "wall-clock benchmarks and shrink dataset sizes via the ci_scale "
            "fixture. Auto-enabled on GitHub Actions (GITHUB_ACTIONS) and "
            "generic CI (CI)."
        ),
    )


def pytest_configure(config):
    # Mirror the CLI option into the environment so the module-level helper
    # (used at parametrize / import time) agrees with an explicit --ci-fast run.
    if config.getoption("--ci-fast"):
        os.environ["TRIBBLE_CI_FAST"] = "1"


def pytest_collection_modifyitems(config, items):
    if not ci_fast_enabled():
        return
    skip_ci_slow = pytest.mark.skip(
        reason="CI-fast mode: wall-clock/plot benchmark deselected "
        "(run without --ci-fast / outside CI for the full suite)"
    )
    for item in items:
        if "ci_slow" in item.keywords:
            item.add_marker(skip_ci_slow)


@pytest.fixture(scope="session")
def ci_fast() -> bool:
    """True when the suite is running in reduced CI-fast mode."""
    return ci_fast_enabled()


@pytest.fixture(scope="session")
def ci_scale(ci_fast):
    """Return a size picker: ``ci_scale(full, fast)`` -> ``fast`` in CI-fast mode.

    Use it to shrink dataset sizes without branching in each test body::

        def test_big(self, ci_scale):
            n = ci_scale(100_000, 5_000)   # 100_000 locally, 5_000 in CI
            ...
    """

    def pick(full, fast):
        return fast if ci_fast else full

    return pick
