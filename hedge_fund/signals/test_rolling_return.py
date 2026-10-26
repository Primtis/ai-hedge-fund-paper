"""Tests for RollingReturnModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hedge_fund.data.models import Price
from hedge_fund.signals import ALPHA_MODEL_REGISTRY, RollingReturnModel, get_investment_approach


def _series(n: int = 900, persistence: float = 0.6, seed: int = 7):
    """Business-day bars whose daily return carries `persistence` of the last one."""
    rng = np.random.default_rng(seed)
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = persistence * rets[i - 1] + rng.normal(0, 0.01)
    close = 100 * np.cumprod(1 + rets)
    days = pd.bdate_range("2021-01-04", periods=n).strftime("%Y-%m-%d")
    return [Price(open=c, high=c, low=c, close=float(c), volume=1_000_000 + 1000 * i, time=d)
            for i, (c, d) in enumerate(zip(close, days))]


class Bars:
    """Hands back every bar it has, ignoring end_date — the model must do its own cutting."""

    def __init__(self, bars):
        self._bars = bars

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        return self._bars


def test_registered_long_short():
    assert ALPHA_MODEL_REGISTRY["rolling_return"] is RollingReturnModel
    assert get_investment_approach("rolling_return") == "long_short"


def test_abstains_without_history():
    bars = _series(60)
    sig = RollingReturnModel().predict("X", bars[-1].time, Bars(bars))
    assert sig.value == 0.0
    assert sig.metadata["abstained"] is True


def test_no_lookahead():
    bars = _series()
    cut = 700
    full = RollingReturnModel().predict("X", bars[cut].time, Bars(bars))
    truncated = RollingReturnModel().predict("X", bars[cut].time, Bars(bars[: cut + 1]))
    assert full.value == truncated.value
    assert abs(full.value) <= 1.0


def test_forecast_tracks_next_returns_on_persistent_series():
    bars = _series()
    model, client = RollingReturnModel(horizon=1), Bars(bars)
    closes = np.array([b.close for b in bars])
    signals, realised = [], []
    for i in range(600, 880):
        signals.append(model.predict("X", bars[i].time, client).value)
        realised.append(closes[i + 1] / closes[i] - 1)
    assert np.corrcoef(signals, realised)[0, 1] > 0.15


def test_retrains_only_every_n_bars(monkeypatch):
    bars = _series()
    model, client = RollingReturnModel(retrain_every=5), Bars(bars)
    fits = []
    real = model._fit
    monkeypatch.setattr(model, "_fit", lambda *a: fits.append(1) or real(*a))
    for i in range(600, 610):
        model.predict("X", bars[i].time, client)
    assert len(fits) == 2


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"retrain_every": 0}, {"scale": 0}, {"ridge": -1}])
def test_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        RollingReturnModel(**kwargs)
