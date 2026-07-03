"""Latency guard (SC-002 as amended, task T021): a worst-case round-start request
answers within budget — 2 s through 7 players, 3 s at 8. Thresholds carry a small
margin over the measured values (research R7) so slower machines don't flake, while
still catching a regression to a deeper default cap or an accidental second solver
pass."""
import time

import pytest


def round_start_record(n):
    names = ["Alice", "Bob", "Clara", "Darryl", "Eve", "Fred", "Gina", "Hugo"][:n]
    return {
        "setup": {"players": names, "bomb": True, "numBadOverride": None},
        "events": [{"type": "declarations", "values": [1] * n}],
    }


@pytest.mark.slow
@pytest.mark.parametrize("n,budget", [(5, 2.0), (7, 2.0), (8, 3.0)])
def test_round_start_within_budget(client, n, budget):
    start = time.perf_counter()
    resp = client.post("/api/panel", json=round_start_record(n))
    elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    assert resp.get_json()["belief"]["panel"] is not None
    assert elapsed < budget, f"N={n} took {elapsed:.2f}s (budget {budget}s)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-n0"]))
