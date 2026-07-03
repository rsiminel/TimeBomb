"""Bootstrap for the web-layer suite: make ``web/`` and the ``timebomb/`` source root
importable (mirroring the repo-root conftest), and provide the Flask test client."""
import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[2]
for _src in ("web", "timebomb"):
    _path = str(_repo_root / _src)
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture()
def client():
    import app as web_app
    web_app.app.config["TESTING"] = True
    with web_app.app.test_client() as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_active_game():
    """The hosted game (specs/002-host-local-game) is a module-level singleton in
    game_service.py; without this, one test's active game would 409 the next."""
    import game_service
    game_service.reset_for_tests()
    yield
    game_service.reset_for_tests()
