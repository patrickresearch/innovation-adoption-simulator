import numpy as np
import pytest

from adoptsim.models.bass import bass_cumulative

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies


@hypothesis.given(
    p=st.floats(1e-4, 0.2),
    q=st.floats(0.0, 1.5),
    f0=st.floats(0.0, 0.9),
    t=st.lists(st.floats(0, 200), min_size=2, max_size=20),
)
def test_bounded_and_monotone(p: float, q: float, f0: float, t: list[float]) -> None:
    ts = np.sort(np.asarray(t))
    y = bass_cumulative(ts, p, q, 1.0, 0.0, f0)
    assert np.all(y >= f0 - 1e-12)
    assert np.all(y <= 1.0 + 1e-12)
    assert np.all(np.diff(y) >= -1e-12)
