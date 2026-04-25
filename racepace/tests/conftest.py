"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DB = FIXTURE_DIR / "fixture_session.db"


@pytest.fixture(scope="session")
def fixture_db_path() -> Path:
    """Path to the committed tiny session DB. Regenerate with `python -m racepace.tests.fixtures.build_fixture`."""
    if not FIXTURE_DB.exists():
        from racepace.tests.fixtures.build_fixture import build
        build(FIXTURE_DB)
    return FIXTURE_DB
