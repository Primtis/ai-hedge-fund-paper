"""Yahoo Finance DataClient — free, no API key.

An alternative to FDClient for running the pipeline without a Financial
Datasets key. Select it with ``HEDGE_FUND_DATA_PROVIDER=yfinance`` (see
hedge_fund/data/factory.py). Requires ``pip install yfinance``.

Yahoo is a scraped, unofficial source, so it is weaker than FD in ways the
pipeline cares about. Know these before trusting a backtest on it:

* Daily bars only (no minute funds), split-adjusted but not dividend-adjusted.
* Fundamentals are the last ~5 quarters and ~4 fiscal years Yahoo exposes, so a
  backtest reaching further back sees no fundamentals and LLM analysts that
  need MIN_PERIODS of history will abstain. The quant ``pead`` model only needs
  earnings dates and works over the ~3 years Yahoo returns.
* Yahoo carries no SEC filing dates. Fundamentals are treated as public
  FILING_LAG_DAYS after the period ends — late rather than early, so the
  backtest errs towards less lookahead, not more.
* News is only the recent feed; older ``end_date`` values return nothing.
* Failures raise FDClientError (network, rate limit). An unknown ticker or a
  range with no bars is "no data" and returns empty. Yahoo does occasionally
  answer a transient failure with an empty frame; if a backtest looks
  suspiciously thin, rerun with the cache refreshed.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone

from hedge_fund.data.client import FDClientError
from hedge_fund.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsData,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
    PriceSnapshot,
)

logger = logging.getLogger(__name__)

FILING_LAG_DAYS = 60       # assumed gap between a period ending and its numbers being public
FORM4_LAG_DAYS = 2         # insiders must file within two business days
MEET_TOLERANCE = 0.005     # |reported - estimate| under half a cent is a MEET
_QUARTER_ENDS = ((3, 31), (6, 30), (9, 30), (12, 31))


class YFinanceClient:
    """Yahoo Finance data provider. Same surface as FDClient."""

    cache_namespace = "yfinance"  # CachedDataClient keeps our answers apart from FD's

    def __init__(self, ticker_factory=None) -> None:
        if ticker_factory is None:
            try:
                import yfinance
            except ImportError as exc:
                raise FDClientError("yfinance is not installed: pip install yfinance") from exc
            ticker_factory = yfinance.Ticker
        self._factory = ticker_factory
        self._tickers: dict[str, object] = {}

    def __enter__(self) -> YFinanceClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self._tickers.clear()

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "day",
        interval_multiplier: int = 1,
    ) -> list[Price]:
        if interval != "day" or interval_multiplier != 1:
            raise FDClientError(
                f"yfinance client serves daily bars only (asked {interval_multiplier} x {interval})",
                path="get_prices",
            )
        frame = self._history(ticker, start_date, end_date)
        bars = []
        for stamp, row in frame.iterrows():
            if any(_nan(row[c]) for c in ("Open", "High", "Low", "Close")):
                continue
            bars.append(Price(
                open=float(row["Open"]), high=float(row["High"]),
                low=float(row["Low"]), close=float(row["Close"]),
                volume=0 if _nan(row["Volume"]) else int(row["Volume"]),
                time=stamp.strftime("%Y-%m-%d"),
            ))
        return bars

    def get_snapshot(self, ticker: str) -> PriceSnapshot | None:
        def read():
            fast = self._ticker(ticker).fast_info
            try:
                return fast["lastPrice"], fast["previousClose"]
            except (KeyError, TypeError):
                return None, None  # Yahoo has no quote for this symbol
        last, previous = self._guard("get_snapshot", read) or (None, None)
        if _nan(last):
            return None
        change = None if _nan(previous) or not previous else float(last) - float(previous)
        return PriceSnapshot(
            ticker=ticker, price=float(last),
            time=datetime.now(timezone.utc).isoformat(),
            day_change=change,
            day_change_percent=None if change is None else 100.0 * change / float(previous),
        )

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        """Metrics public by *end_date*, newest first.

        Trailing-twelve-month rows come from Yahoo's quarterly statements;
        fiscal-year rows from its annual statements fill in history older than
        the quarterly window so there are enough periods for an analyst.
        """
        if period != "ttm":
            raise FDClientError(f"yfinance client supports period='ttm' only (asked {period!r})",
                                path="get_financial_metrics")
        as_of = _parse(end_date)
        quarters = self._statement_periods(ticker, annual=False)
        years = self._statement_periods(ticker, annual=True)

        rows = []  # (period_end, flows, balance, label)
        for i in range(len(quarters) - 3):
            window = quarters[i:i + 4]
            if _public(window[0]["end"]) <= as_of:
                rows.append((window[0]["end"], _sum_flows(window), window[0], "ttm"))
        oldest_ttm = rows[-1][0] if rows else None
        for year in years:
            if oldest_ttm is not None and year["end"] > oldest_ttm - timedelta(days=300):
                continue
            if _public(year["end"]) <= as_of:
                rows.append((year["end"], year, year, "annual"))
        rows.sort(key=lambda r: r[0], reverse=True)
        rows = rows[:limit]
        if not rows:
            return []

        closes = self._closes(ticker, min(_public(r[0]) for r in rows) - timedelta(days=10), as_of)
        out = []
        for index, (end, flows, balance, label) in enumerate(rows):
            previous = rows[index + 1] if index + 1 < len(rows) and rows[index + 1][3] == label else None
            filed = _public(end)
            price = _close_on_or_before(closes, filed)
            out.append(_metrics_row(ticker, end, filed, label, flows, balance,
                                    previous[1] if previous else None, price))
        return out

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        as_of = _parse(end_date)
        closes = self._closes(ticker, as_of - timedelta(days=10), as_of)
        price = _close_on_or_before(closes, as_of)
        shares = self._shares_on(ticker, as_of)
        if price is None or shares is None:
            return None
        return price * shares

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        info = self._info(ticker)
        name = info.get("longName") or info.get("shortName")
        if not name:
            return None
        return CompanyFacts(
            ticker=ticker, name=name, sector=info.get("sector"),
            industry=info.get("industry"), exchange=info.get("exchange"),
            category=info.get("quoteType"),
            location=", ".join(p for p in (info.get("city"), info.get("state"), info.get("country")) if p) or None,
        )

    # ------------------------------------------------------------------
    # Earnings
    # ------------------------------------------------------------------

    def get_earnings_history(self, ticker: str, limit: int = 12) -> list[EarningsRecord]:
        """One record per reported quarter, newest first.

        Dated by when the market learned the number: a release at or after the
        4pm close (or with no time given) counts as the next day, so a
        close-to-close decision never uses news that broke after the close.
        Yahoo has no fiscal calendar, so ``report_period`` is the calendar
        quarter end before the release; source_type is "8-K" because earnings
        are announced under 8-K Item 2.02.
        """
        frame = self._guard("get_earnings_dates",
                            lambda: self._ticker(ticker).get_earnings_dates(limit=max(limit, 4) + 4))
        if frame is None or len(frame) == 0:
            return []
        records = []
        for stamp, row in frame.iterrows():
            reported, estimate = row.get("Reported EPS"), row.get("EPS Estimate")
            if _nan(reported) or _nan(estimate):
                continue  # a future date, or a quarter with no consensus to compare against
            released = stamp.to_pydatetime()
            known = released.date() + (timedelta(days=1) if released.hour >= 16 or released.hour == 0 else timedelta(0))
            records.append(EarningsRecord(
                ticker=ticker,
                report_period=_prior_quarter_end(released.date()).isoformat(),
                source_type="8-K",
                filing_date=known.isoformat(),
                filing_datetime=released.isoformat(),
                quarterly=EarningsData(
                    earnings_per_share=float(reported),
                    estimated_earnings_per_share=float(estimate),
                    eps_surprise=_surprise(float(reported), float(estimate)),
                ),
            ))
        records.sort(key=lambda r: r.filing_date, reverse=True)
        return records[:limit]

    def get_earnings(self, ticker: str) -> Earnings | None:
        latest = self.get_earnings_history(ticker, limit=1)
        if not latest:
            return None
        record = latest[0]
        return Earnings(ticker=ticker, report_period=record.report_period, quarterly=record.quarterly)

    # ------------------------------------------------------------------
    # News and insiders
    # ------------------------------------------------------------------

    def get_news(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[CompanyNews]:
        items = self._guard("get_news", lambda: self._ticker(ticker).get_news(count=min(limit, 100))) or []
        out = []
        for item in items:
            content = item.get("content") or item
            title = content.get("title")
            stamp = content.get("pubDate") or content.get("displayTime")
            if not title or not stamp:
                continue
            day = str(stamp)[:10]
            if day > end_date or (start_date and day < start_date):
                continue
            out.append(CompanyNews(
                ticker=ticker, title=title, date=str(stamp),
                source=(content.get("provider") or {}).get("displayName") or "Yahoo Finance",
                url=(content.get("canonicalUrl") or content.get("clickThroughUrl") or {}).get("url"),
            ))
        out.sort(key=lambda n: n.date or "", reverse=True)
        return out[:limit]

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        frame = self._guard("get_insider_trades", lambda: self._ticker(ticker).insider_transactions)
        if frame is None or len(frame) == 0:
            return []
        out = []
        for _, row in frame.iterrows():
            traded = row.get("Start Date")
            if traded is None or _nan(traded):
                continue
            traded_on = traded.date() if hasattr(traded, "date") else _parse(str(traded)[:10])
            filed = traded_on + timedelta(days=FORM4_LAG_DAYS)
            if filed.isoformat() > end_date or (start_date and filed.isoformat() < start_date):
                continue
            text = str(row.get("Text") or "")
            kind = "Sale" if "sale" in text.lower() else "Purchase" if "purchase" in text.lower() else (row.get("Transaction") or None)
            shares = None if _nan(row.get("Shares")) else float(row["Shares"])
            value = None if _nan(row.get("Value")) else float(row["Value"])
            out.append(InsiderTrade(
                ticker=ticker, name=str(row.get("Insider") or "unknown"),
                title=row.get("Position") or None, filing_date=filed.isoformat(),
                transaction_date=traded_on.isoformat(), transaction_type=kind,
                transaction_shares=shares, transaction_value=value,
            ))
        out.sort(key=lambda t: t.filing_date, reverse=True)
        return out[:limit]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ticker(self, symbol: str):
        if symbol not in self._tickers:
            self._tickers[symbol] = self._factory(symbol)
        return self._tickers[symbol]

    def _guard(self, path: str, fn):
        """Run a Yahoo call; infrastructure failures become FDClientError."""
        try:
            return fn()
        except FDClientError:
            raise
        except Exception as exc:
            if _is_no_data(exc):
                return None
            raise FDClientError(f"yfinance {path} failed: {exc}", path=path) from exc

    def _info(self, symbol: str) -> dict:
        return self._guard("info", lambda: self._ticker(symbol).info) or {}

    def _history(self, symbol: str, start: str, end: str):
        import pandas as pd

        stop = (_parse(end) + timedelta(days=1)).isoformat()  # Yahoo's end is exclusive
        frame = self._guard("history", lambda: self._ticker(symbol).history(
            start=start, end=stop, interval="1d", auto_adjust=False, raise_errors=True))
        if frame is None or len(frame) == 0:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        frame = frame.copy()
        frame.index = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
        return frame

    def _closes(self, symbol: str, start: date, end: date) -> list[tuple[date, float]]:
        frame = self._history(symbol, start.isoformat(), end.isoformat())
        return [(stamp.date(), float(row["Close"])) for stamp, row in frame.iterrows() if not _nan(row["Close"])]

    def _shares_on(self, symbol: str, as_of: date) -> float | None:
        frame = self._guard("get_shares_full", lambda: self._ticker(symbol).get_shares_full(
            start=(as_of - timedelta(days=400)).isoformat(), end=(as_of + timedelta(days=1)).isoformat()))
        if frame is not None and len(frame):
            frame = frame.dropna()
            stamps = [s.date() if hasattr(s, "date") else s for s in frame.index]
            usable = [(d, v) for d, v in zip(stamps, frame.values) if d <= as_of]
            if usable:
                return float(max(usable, key=lambda p: p[0])[1])
        # info is today's share count: right for a live run, lookahead for a backtest.
        if (date.today() - as_of).days <= 7:
            shares = self._info(symbol).get("sharesOutstanding")
            return None if _nan(shares) else float(shares)
        return None

    def _statement_periods(self, symbol: str, *, annual: bool) -> list[dict]:
        """Yahoo statements as one dict per period, newest first."""
        ticker = self._ticker(symbol)
        prefix = "" if annual else "quarterly_"
        income, balance, cashflow = (
            self._guard(name, lambda name=name: getattr(ticker, prefix + name))
            for name in ("income_stmt", "balance_sheet", "cashflow")
        )
        if income is None or len(income) == 0 or income.shape[1] == 0:
            return []
        periods = []
        for column in income.columns:
            end = column.date() if hasattr(column, "date") else _parse(str(column)[:10])
            periods.append({
                "end": end,
                "revenue": _cell(income, "Total Revenue", column),
                "gross_profit": _cell(income, "Gross Profit", column),
                "operating_income": _cell(income, "Operating Income", column),
                "net_income": _cell(income, "Net Income", column),
                "ebitda": _cell(income, "EBITDA", column),
                "eps": _cell(income, "Diluted EPS", column),
                "interest_expense": _cell(income, "Interest Expense", column),
                "equity": _cell(balance, "Stockholders Equity", column),
                "assets": _cell(balance, "Total Assets", column),
                "current_assets": _cell(balance, "Current Assets", column),
                "current_liabilities": _cell(balance, "Current Liabilities", column),
                "debt": _cell(balance, "Total Debt", column),
                "cash": _cell(balance, "Cash And Cash Equivalents", column),
                "shares": _cell(balance, "Ordinary Shares Number", column) or _cell(balance, "Share Issued", column),
                "operating_cash_flow": _cell(cashflow, "Operating Cash Flow", column),
                "free_cash_flow": _cell(cashflow, "Free Cash Flow", column),
            })
        periods = [p for p in periods if p["revenue"] is not None and p["net_income"] is not None]
        periods.sort(key=lambda p: p["end"], reverse=True)
        return periods


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

_FLOW_KEYS = ("revenue", "gross_profit", "operating_income", "net_income", "ebitda", "eps",
              "interest_expense", "operating_cash_flow", "free_cash_flow")


def _sum_flows(window: list[dict]) -> dict:
    """Four quarters summed into a trailing-twelve-month flow; stocks are not summed."""
    total = {}
    for key in _FLOW_KEYS:
        values = [p[key] for p in window]
        total[key] = None if any(v is None for v in values) else sum(values)
    return total


def _metrics_row(ticker, end, filed, label, flows, balance, prior, price) -> FinancialMetrics:
    shares = balance.get("shares")
    market_cap = price * shares if price is not None and shares else None
    revenue, net_income = flows.get("revenue"), flows.get("net_income")
    equity, debt, cash = balance.get("equity"), balance.get("debt"), balance.get("cash")
    ebitda, fcf = flows.get("ebitda"), flows.get("free_cash_flow")
    enterprise = market_cap + (debt or 0.0) - (cash or 0.0) if market_cap is not None else None
    return FinancialMetrics(
        ticker=ticker, report_period=end.isoformat(), period=label, filing_date=filed.isoformat(),
        market_cap=market_cap, enterprise_value=enterprise,
        price_to_earnings_ratio=_ratio(market_cap, net_income, positive=True),
        price_to_book_ratio=_ratio(market_cap, equity, positive=True),
        price_to_sales_ratio=_ratio(market_cap, revenue, positive=True),
        enterprise_value_to_ebitda_ratio=_ratio(enterprise, ebitda, positive=True),
        enterprise_value_to_revenue_ratio=_ratio(enterprise, revenue, positive=True),
        free_cash_flow_yield=_ratio(fcf, market_cap),
        gross_margin=_ratio(flows.get("gross_profit"), revenue),
        operating_margin=_ratio(flows.get("operating_income"), revenue),
        net_margin=_ratio(net_income, revenue),
        return_on_equity=_ratio(net_income, equity),
        return_on_assets=_ratio(net_income, balance.get("assets")),
        current_ratio=_ratio(balance.get("current_assets"), balance.get("current_liabilities")),
        debt_to_equity=_ratio(debt, equity),
        debt_to_assets=_ratio(debt, balance.get("assets")),
        interest_coverage=_ratio(flows.get("operating_income"), abs(flows["interest_expense"]) if flows.get("interest_expense") else None),
        revenue_growth=_growth(revenue, prior.get("revenue") if prior else None),
        earnings_growth=_growth(net_income, prior.get("net_income") if prior else None),
        earnings_per_share=flows.get("eps"),
        book_value_per_share=_ratio(equity, shares),
        free_cash_flow_per_share=_ratio(fcf, shares),
        earnings_per_share_growth=_growth(flows.get("eps"), prior.get("eps") if prior else None),
        operating_income_growth=_growth(flows.get("operating_income"), prior.get("operating_income") if prior else None),
        free_cash_flow_growth=_growth(fcf, prior.get("free_cash_flow") if prior else None),
    )


def _public(period_end: date) -> date:
    return period_end + timedelta(days=FILING_LAG_DAYS)


def _prior_quarter_end(day: date) -> date:
    candidates = [date(day.year + shift, m, d) for shift in (-1, 0) for m, d in _QUARTER_ENDS]
    return max(c for c in candidates if c < day)


def _surprise(reported: float, estimate: float) -> str:
    if abs(reported - estimate) < MEET_TOLERANCE:
        return "MEET"
    return "BEAT" if reported > estimate else "MISS"


def _close_on_or_before(closes: list[tuple[date, float]], day: date) -> float | None:
    index = bisect_right([d for d, _ in closes], day)
    return closes[index - 1][1] if index else None


def _ratio(numerator, denominator, *, positive: bool = False) -> float | None:
    if numerator is None or denominator is None or not denominator:
        return None
    if positive and (numerator <= 0 or denominator <= 0):
        return None
    return numerator / denominator


def _growth(current, previous) -> float | None:
    if current is None or previous is None or previous <= 0:
        return None
    return current / previous - 1.0


def _cell(frame, row: str, column) -> float | None:
    if frame is None or row not in frame.index:
        return None
    value = frame.at[row, column]
    return None if _nan(value) else float(value)


def _nan(value) -> bool:
    try:
        return value is None or value != value
    except (TypeError, ValueError):
        return False


def _parse(text: str) -> date:
    return datetime.strptime(text[:10], "%Y-%m-%d").date()


def _is_no_data(exc: Exception) -> bool:
    """Yahoo's "this symbol/range has nothing" errors, as opposed to outages."""
    return type(exc).__name__ in {
        "YFPricesMissingError", "YFTickerMissingError", "YFTzMissingError", "YFEarningsDateMissing",
    }
