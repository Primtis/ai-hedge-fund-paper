"""Pick the data provider: Financial Datasets (default) or Yahoo Finance.

    HEDGE_FUND_DATA_PROVIDER=yfinance aihf backtest ...

Every call site that used to say ``FDClient()`` asks here instead, so adding a
provider is one branch in this file.
"""

from __future__ import annotations

import os

from hedge_fund.data.client import FDClient
from hedge_fund.data.protocol import DataClient

PROVIDER_ENV = "HEDGE_FUND_DATA_PROVIDER"
_FD_KEY = "FINANCIAL_DATASETS_API_KEY"


def data_provider() -> str:
    name = os.environ.get(PROVIDER_ENV, "financial-datasets").strip().lower()
    if name in ("yfinance", "yahoo"):
        return "yfinance"
    if name in ("financial-datasets", "fd", ""):
        return "financial-datasets"
    raise ValueError(f"{PROVIDER_ENV}={name!r}: expected 'financial-datasets' or 'yfinance'")


def data_key_env() -> str | None:
    """Environment variable holding the chosen provider's API key; None if it needs none."""
    return _FD_KEY if data_provider() == "financial-datasets" else None


def make_data_client() -> DataClient:
    """A fresh raw client (a context manager) for the chosen provider."""
    if data_provider() == "yfinance":
        from hedge_fund.data.yfinance_client import YFinanceClient
        return YFinanceClient()
    return FDClient()
