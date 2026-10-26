"""Rolling-return alpha model — a short-horizon forecast that retrains itself.

Fits a ridge regression of the next `horizon` bars' return on price and volume
features (recent returns, volatility, RSI, volume spike), using only bars
whose forward return is already known at *date*. The fit is refreshed every
`retrain_every` new bars, so the model keeps adapting to the market it is
in. Conviction is the forecast squashed into [-1, +1].

shortcut: a linear model on daily bars, so a baseline to beat before anything
heavier (GBDT, intraday bars) earns its cost — upgrade `_fit`/`_forecast` then.
"""

from __future__ import annotations

from datetime import date as _date, timedelta

import numpy as np
import pandas as pd

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

_MIN_TRAIN_ROWS = 120   # below this a fit is noise; abstain
_FEATURE_WARMUP = 21    # bars the slowest feature (20-bar window, plus a diff) needs


class RollingReturnModel(QuantModel):
    """Long when the model forecasts a positive `horizon`-bar return, short when negative."""

    investment_approach = "long_short"

    def __init__(
        self,
        *,
        horizon: int = 5,          # bars ahead the model forecasts
        train_bars: int = 500,     # trailing bars the fit may look at
        retrain_every: int = 5,    # new bars before the fit is refreshed
        ridge: float = 10.0,       # L2 strength on standardized features
        scale: float = 0.02,       # forecast return that maps to ~76% conviction (tanh(1))
    ) -> None:
        if min(horizon, train_bars, retrain_every) < 1 or ridge < 0 or scale <= 0:
            raise ValueError("horizon, train_bars, retrain_every must be >= 1; ridge >= 0; scale > 0")
        self._horizon = horizon
        self._train_bars = train_bars
        self._retrain_every = retrain_every
        self._ridge = ridge
        self._scale = scale
        # ticker -> (last bar date at fit time, fit); a backtest calls predict once a day
        self._fits: dict[str, tuple[str, dict]] = {}

    @property
    def name(self) -> str:
        return "rolling_return"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        day = date[:10]
        bars = self._bars(ticker, day, data_client)
        if bars is None or len(bars) < _MIN_TRAIN_ROWS + self._horizon + _FEATURE_WARMUP:
            return self._abstain(ticker, date, "not enough history")

        feats = _features(bars)
        fit = self._current_fit(ticker, bars, feats)
        if fit is None:
            return self._abstain(ticker, date, "not enough clean training rows")

        row = feats.iloc[-1]
        if row.isna().any():
            return self._abstain(ticker, date, "latest features incomplete")
        forecast = float(((row.to_numpy() - fit["mean"]) / fit["std"]) @ fit["coef"] + fit["intercept"])
        value = self._normalize_to_signal(float(np.tanh(forecast / self._scale)))
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=value,
            reasoning=f"forecast {forecast:+.2%} over {self._horizon} bars from a fit on {fit['rows']} bars",
            components={"forecast_return": forecast},
            metadata={"train_rows": fit["rows"], "horizon": self._horizon},
        )

    # ------------------------------------------------------------------

    def _abstain(self, ticker: str, date: str, why: str) -> Signal:
        return Signal(model_name=self.name, ticker=ticker, date=date, value=0.0,
                      reasoning=why, metadata={"abstained": True})

    def _bars(self, ticker: str, day: str, data: DataClient) -> pd.DataFrame | None:
        """Trailing daily bars up to and including *day*; nothing later is kept."""
        start = (_date.fromisoformat(day) - timedelta(days=int(self._train_bars * 1.6) + 60)).isoformat()
        prices = [p for p in data.get_prices(ticker, start, day) if p.time[:10] <= day]
        if not prices:
            return None
        frame = pd.DataFrame(
            {"close": [p.close for p in prices], "volume": [p.volume for p in prices]},
            index=[p.time[:10] for p in prices],
        )
        frame = frame[~frame.index.duplicated(keep="last")].sort_index().tail(self._train_bars)
        return frame if (frame["close"] > 0).all() else None

    def _current_fit(self, ticker: str, bars: pd.DataFrame, feats: pd.DataFrame) -> dict | None:
        cached = self._fits.get(ticker)
        # bars is capped at train_bars, so count bars newer than the last fit, not its length
        if cached is not None and cached[0] <= bars.index[-1] and (bars.index > cached[0]).sum() < self._retrain_every:
            return cached[1]
        fit = self._fit(bars, feats)
        if fit is not None:
            self._fits[ticker] = (bars.index[-1], fit)
        return fit

    def _fit(self, bars: pd.DataFrame, feats: pd.DataFrame) -> dict | None:
        # The label at bar t is the return to bar t+horizon, so the last
        # `horizon` bars have no label yet — they are never trained on.
        label = bars["close"].shift(-self._horizon) / bars["close"] - 1.0
        data = feats.assign(_y=label).dropna()
        if len(data) < _MIN_TRAIN_ROWS:
            return None
        x = data.drop(columns="_y").to_numpy()
        y = data["_y"].to_numpy()
        mean, std = x.mean(axis=0), x.std(axis=0)
        std[std == 0] = 1.0
        z = (x - mean) / std
        intercept = float(y.mean())
        gram = z.T @ z + self._ridge * np.eye(z.shape[1])
        coef = np.linalg.solve(gram, z.T @ (y - intercept))
        return {"mean": mean, "std": std, "coef": coef, "intercept": intercept, "rows": len(data)}


def _features(bars: pd.DataFrame) -> pd.DataFrame:
    close, volume = bars["close"], bars["volume"].astype(float)
    ret = close.pct_change()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
    vol_avg = volume.rolling(20).mean().replace(0, np.nan)
    return pd.DataFrame({
        "ret_1": ret,
        "ret_5": close.pct_change(5),
        "ret_10": close.pct_change(10),
        "ret_20": close.pct_change(20),
        "vol_20": ret.rolling(20).std(),
        "rsi_14": rsi,
        "vol_spike": volume / vol_avg - 1.0,
    })
