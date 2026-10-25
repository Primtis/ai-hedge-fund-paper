"""YFinanceClient against a fake Yahoo ticker — no network.

These check our mapping and point-in-time rules, not Yahoo's real output; a
live smoke test needs yahoo.com reachable.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hedge_fund.data import CachedDataClient, FDClientError
from hedge_fund.data.factory import data_key_env, data_provider, make_data_client
from hedge_fund.data.yfinance_client import YFinanceClient, _prior_quarter_end
from hedge_fund.signals.pead import PEADModel


def _bars(days: list[str], start: float = 100.0) -> pd.DataFrame:
    index = pd.DatetimeIndex(days, tz="America/New_York")
    closes = [start + i for i in range(len(days))]
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes,
                         "Volume": [1000.0] * len(days)}, index=index)


class FakeTicker:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.calls: list[tuple] = []
        self.history_error: Exception | None = None
        self.fast_info = {"lastPrice": 110.0, "previousClose": 100.0}
        self.info = {"longName": "Test Corp", "sector": "Tech", "industry": "Software", "exchange": "NMS"}
        self.insider_transactions = pd.DataFrame()
        quarter_ends = ["2025-09-30", "2025-06-30", "2025-03-31", "2024-12-31", "2024-09-30"]
        self.quarterly_income_stmt = pd.DataFrame(
            {pd.Timestamp(e): [v[i] for v in ([400, 380, 360, 340, 320], [200, 190, 180, 170, 160],
                                               [80, 76, 72, 68, 64], [40, 38, 36, 34, 32],
                                               [90, 86, 82, 78, 74], [0.4, 0.38, 0.36, 0.34, 0.32],
                                               [5, 5, 5, 5, 5])]
             for i, e in enumerate(quarter_ends)},
            index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income",
                   "EBITDA", "Diluted EPS", "Interest Expense"])
        self.quarterly_balance_sheet = pd.DataFrame(
            {pd.Timestamp(e): [1000, 2000, 500, 300, 400, 100, 50] for e in quarter_ends},
            index=["Stockholders Equity", "Total Assets", "Current Assets", "Current Liabilities",
                   "Total Debt", "Cash And Cash Equivalents", "Ordinary Shares Number"])
        self.quarterly_cashflow = pd.DataFrame(
            {pd.Timestamp(e): [60, 45] for e in quarter_ends},
            index=["Operating Cash Flow", "Free Cash Flow"])
        year_ends = ["2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31"]
        self.income_stmt = pd.DataFrame(
            {pd.Timestamp(e): [1200 - 100 * i, 30 - i, 4 - 0.1 * i] for i, e in enumerate(year_ends)},
            index=["Total Revenue", "Net Income", "Diluted EPS"])
        self.balance_sheet = pd.DataFrame(
            {pd.Timestamp(e): [900, 50] for e in year_ends}, index=["Stockholders Equity", "Ordinary Shares Number"])
        self.cashflow = pd.DataFrame({pd.Timestamp(e): [40] for e in year_ends}, index=["Free Cash Flow"])

    def history(self, **kwargs):
        self.calls.append(("history", kwargs))
        if self.history_error:
            raise self.history_error
        return _bars(["2025-11-03", "2025-11-04", "2025-11-05"])

    def get_shares_full(self, start=None, end=None):
        return pd.Series([48.0, 50.0], index=pd.DatetimeIndex(["2025-01-02", "2025-10-01"]))

    def get_earnings_dates(self, limit=12):
        # after-close release, before-open release, and a future date with no result yet
        index = pd.DatetimeIndex(
            ["2026-02-05 16:05", "2025-10-30 16:10", "2025-07-31 07:00", "2025-05-01 16:00"], tz="America/New_York")
        return pd.DataFrame({"EPS Estimate": [1.0, 1.0, 1.0, 1.0],
                             "Reported EPS": [float("nan"), 1.2, 0.9, 1.0]}, index=index)

    def get_news(self, count=10, tab="news"):
        return [
            {"content": {"title": "Old", "pubDate": "2025-06-01T12:00:00Z", "provider": {"displayName": "Wire"},
                         "canonicalUrl": {"url": "http://x/old"}}},
            {"content": {"title": "New", "pubDate": "2025-11-01T12:00:00Z", "provider": {"displayName": "Wire"},
                         "canonicalUrl": {"url": "http://x/new"}}},
        ]


@pytest.fixture
def yf_client():
    fakes: dict[str, FakeTicker] = {}

    def factory(symbol):
        fakes[symbol] = fakes.get(symbol) or FakeTicker(symbol)
        return fakes[symbol]

    client = YFinanceClient(ticker_factory=factory)
    client.fakes = fakes
    return client


def test_prices_map_to_bars_and_end_is_inclusive(yf_client):
    bars = yf_client.get_prices("TEST", "2025-11-03", "2025-11-05")
    assert [b.time for b in bars] == ["2025-11-03", "2025-11-04", "2025-11-05"]
    assert bars[0].close == 100.0 and bars[0].volume == 1000
    call = yf_client.fakes["TEST"].calls[0][1]
    assert call["end"] == "2025-11-06" and call["auto_adjust"] is False


def test_intraday_is_refused_not_silently_daily(yf_client):
    with pytest.raises(FDClientError):
        yf_client.get_prices("TEST", "2025-11-03", "2025-11-03", interval="minute")


def test_outage_raises_but_unknown_symbol_is_empty(yf_client):
    yf_client.get_prices("TEST", "2025-11-03", "2025-11-05")
    yf_client.fakes["TEST"].history_error = RuntimeError("connection reset")
    with pytest.raises(FDClientError):
        yf_client.get_prices("TEST", "2025-11-03", "2025-11-05")

    class YFPricesMissingError(Exception):
        pass

    yf_client.fakes["TEST"].history_error = YFPricesMissingError("possibly delisted")
    assert yf_client.get_prices("TEST", "2025-11-03", "2025-11-05") == []


def test_snapshot(yf_client):
    snap = yf_client.get_snapshot("TEST")
    assert snap.price == 110.0 and snap.day_change == 10.0 and snap.day_change_percent == pytest.approx(10.0)


def test_earnings_history_is_point_in_time(yf_client):
    records = yf_client.get_earnings_history("TEST")
    by_period = {r.filing_date: r for r in records}
    # the future row (no reported EPS) is dropped
    assert len(records) == 3
    # released 16:10 on 10-30: the market learns it on 10-31
    assert by_period["2025-10-31"].quarterly.eps_surprise == "BEAT"
    assert by_period["2025-10-31"].report_period == "2025-09-30"
    # released 07:00 on 07-31: known that day
    assert by_period["2025-07-31"].quarterly.eps_surprise == "MISS"
    assert by_period["2025-05-02"].quarterly.eps_surprise == "MEET"
    assert all(r.source_type == "8-K" for r in records)
    assert [r.filing_date for r in records] == sorted((r.filing_date for r in records), reverse=True)


def test_pead_runs_on_yahoo_earnings(yf_client):
    model = PEADModel()
    assert model.predict("TEST", "2025-10-30", yf_client).value == 0.0   # not public yet
    assert model.predict("TEST", "2025-10-31", yf_client).value == 1.0   # BEAT, day it was learned
    assert model.predict("TEST", "2025-11-20", yf_client).value == 0.0   # window passed


def test_prior_quarter_end():
    assert _prior_quarter_end(date(2025, 10, 30)) == date(2025, 9, 30)
    assert _prior_quarter_end(date(2026, 1, 15)) == date(2025, 12, 31)
    assert _prior_quarter_end(date(2025, 3, 31)) == date(2024, 12, 31)


def test_metrics_only_include_periods_public_by_end_date(yf_client):
    # Q3 ends 2025-09-30 -> assumed public 60 days later (2025-11-29)
    early = yf_client.get_financial_metrics("TEST", "2025-11-10")
    late = yf_client.get_financial_metrics("TEST", "2025-12-15")
    assert early[0].report_period == "2025-06-30" and early[0].period == "ttm"
    assert late[0].report_period == "2025-09-30" and late[0].period == "ttm"
    assert all(m.filing_date <= "2025-11-10" for m in early)


def test_ttm_row_sums_four_quarters(yf_client):
    row = yf_client.get_financial_metrics("TEST", "2025-12-15", limit=1)[0]
    assert row.net_margin == pytest.approx((40 + 38 + 36 + 34) / (400 + 380 + 360 + 340))
    assert row.return_on_equity == pytest.approx((40 + 38 + 36 + 34) / 1000)
    assert row.debt_to_equity == pytest.approx(0.4)
    assert row.book_value_per_share == pytest.approx(1000 / 50)
    assert row.market_cap == pytest.approx(50 * 102.0)   # shares x last close on/before the filing date


def test_annual_rows_fill_history_older_than_quarterly_window(yf_client):
    rows = yf_client.get_financial_metrics("TEST", "2025-12-15", limit=10)
    assert len(rows) >= 4   # what an analyst needs; Yahoo's quarterly window alone gives 2
    assert [r.report_period for r in rows] == sorted((r.report_period for r in rows), reverse=True)
    assert {r.period for r in rows} == {"ttm", "annual"}


def test_non_ttm_period_is_refused(yf_client):
    with pytest.raises(FDClientError):
        yf_client.get_financial_metrics("TEST", "2025-12-15", period="annual")


def test_news_respects_end_date(yf_client):
    assert [n.title for n in yf_client.get_news("TEST", "2025-12-01")] == ["New", "Old"]
    assert [n.title for n in yf_client.get_news("TEST", "2025-07-01")] == ["Old"]
    assert [n.title for n in yf_client.get_news("TEST", "2025-12-01", start_date="2025-10-01")] == ["New"]


def test_insider_trades_are_dated_by_filing_not_trade(yf_client):
    yf_client.get_prices("TEST", "2025-11-03", "2025-11-05")
    yf_client.fakes["TEST"].insider_transactions = pd.DataFrame({
        "Insider": ["A Person"], "Position": ["CEO"], "Shares": [-1000.0], "Value": [150000.0],
        "Text": ["Sale at price 150.00 per share."], "Start Date": [pd.Timestamp("2025-11-03")],
    })
    assert yf_client.get_insider_trades("TEST", "2025-11-03") == []        # Form 4 not filed yet
    trades = yf_client.get_insider_trades("TEST", "2025-11-05")
    assert trades[0].filing_date == "2025-11-05" and trades[0].transaction_type == "Sale"


def test_company_facts(yf_client):
    assert yf_client.get_company_facts("TEST").sector == "Tech"
    yf_client.fakes["TEST"].info = {}
    assert yf_client.get_company_facts("TEST") is None


def test_market_cap_uses_shares_known_at_the_date(yf_client):
    # 50.0 shares were reported 2025-10-01; the 2025-11-05 close is 102
    assert yf_client.get_market_cap("TEST", "2025-11-05") == pytest.approx(50.0 * 102.0)


def test_cache_keeps_providers_apart(tmp_path, yf_client):
    cached = CachedDataClient(yf_client, cache_dir=tmp_path)
    cached.get_prices("TEST", "2025-11-03", "2025-11-05")
    assert (tmp_path / "yfinance").is_dir()


def test_factory_selects_provider(monkeypatch):
    monkeypatch.delenv("HEDGE_FUND_DATA_PROVIDER", raising=False)
    assert data_provider() == "financial-datasets" and data_key_env() == "FINANCIAL_DATASETS_API_KEY"
    monkeypatch.setenv("HEDGE_FUND_DATA_PROVIDER", "yfinance")
    assert data_provider() == "yfinance" and data_key_env() is None
    assert isinstance(make_data_client(), YFinanceClient)
    monkeypatch.setenv("HEDGE_FUND_DATA_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        data_provider()
