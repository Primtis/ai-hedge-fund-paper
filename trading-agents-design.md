# Self-Correcting Short-Term Trading Agent: Design + Roadmap

Markets: crypto and US stocks, in parallel. Mode: paper trading until gates in Phase 5 pass.
Basis: README/docs review of freqtrade (FreqAI), NautilusTrader, TradingAgents, FinRL/FinRL-X, Qlib. No source-code audit yet.

## 1. Key decision: one shared brain, two execution engines

Nobody ships a single engine that is good at both crypto and US-equity short-term trading with built-in ML.
So split by concern:

| Concern | Crypto | US stocks |
|---|---|---|
| Execution + risk | freqtrade | NautilusTrader (Interactive Brokers adapter) |
| Signal model | FreqAI (built in) | Qlib-style rolling LightGBM, run as a separate service |
| Context filter | TradingAgents (shared) | TradingAgents (shared) |
| Sizing/selection (later) | FinRL-X ensemble | FinRL-X ensemble |

Why not Nautilus for both: it supports crypto venues, but the FreqAI retrain loop would have to be rebuilt. Start with freqtrade for crypto because the self-correction is already there; add Nautilus for stocks.
Why not freqtrade for stocks: it is crypto-only.

Shared pieces (one codebase, used by both): feature definitions, label definitions, the TradingAgents context service, the performance ledger, the risk-policy file.

## 2. Layers

```
            +---------------------------------------------+
  slow      | L4 Context: TradingAgents (min-hours/daily)  |  bias/veto per symbol
            +----------------------+----------------------+
                                   | bias in [-1,0,+1], veto flag
            +----------------------v----------------------+
  fast      | L2 Signal: rolling-retrained GBDT           |  expected return + confidence
            +----------------------+----------------------+
                                   |
            +----------------------v----------------------+
  later     | L3 Sizing/selection: RL ensemble (FinRL-X)  |  position size, which signal
            +----------------------+----------------------+
                                   |
            +----------------------v----------------------+
  always    | L1 Execution + hard risk gates              |  orders; kill switch
            +---------------------------------------------+
```

Rule: upper layers can only reduce or shape risk. Only L1's gates can halt trading. LLM output never places an order directly.

## 3. Three self-correction loops

1. Retrain (L2): rolling refit on a fixed window (FreqAI: background thread, backtests emulate retraining. Qlib: rolling retraining baseline, DDG-DA for drift). Cadence: crypto every few hours, stocks daily after close. Old models kept on disk for rollback.
2. Reweight (L2/L3): track realised PnL, hit rate and slippage per signal/agent over a trailing window; shift allocation toward what is working; set weight to zero below a floor (FinRL ensemble idea: pick agents by recent Sharpe). Add a drift test (feature distribution vs training window) that forces an early retrain or a size cut.
3. Reflect (L4): TradingAgents' memory log settles past decisions against realised return (raw and alpha vs benchmark) and writes lessons used in later runs. Run `settle_all_pending()` on a schedule.

Guard against self-correction going wrong: retrained models must beat the incumbent on a held-out walk-forward window, with costs, before promotion. Never retrain on the window used to judge it.

## 4. Risk policy (L1, non-negotiable, config file in version control)

- Max risk per trade, max gross exposure, max positions, per-symbol cap.
- Daily loss limit and max drawdown: breach flattens positions and halts until a human resets.
- Stale-data and stale-model checks: no trading on data or a model older than N minutes/hours.
- Fees, spread and slippage modelled in every backtest; reject any strategy whose edge disappears at 2x assumed costs.
- API keys: trade-only permission, no withdrawal rights, IP-restricted, kept in env/secret store, never in the repo.
- Stocks: respect the pattern-day-trader rule and settlement constraints for the account type.

## 5. Roadmap

Phase 0 - Foundations (1 week): repo layout, risk-policy file, ledger schema, data download, pinned dependency versions. Decide stock broker and data vendor.
Phase 1 - Crypto baseline (1-2 weeks): freqtrade + FreqAI dry-run on a small fixed pair list (FreqAI cannot use dynamic volume pairlists). Walk-forward backtest with fees. Goal: a reproducible baseline and a benchmark to beat.
Phase 2 - Context layer (1 week): TradingAgents as a scheduled service, writing a bias/veto per symbol to a file or small API that the strategy reads. Backtest with and without it; keep it only if it improves risk-adjusted return. Check cost per run.
Phase 3 - Stocks path (2-3 weeks): Nautilus + IBKR paper account; Qlib-style LightGBM signal service with rolling retrain; same ledger and risk policy. Note Nautilus v2 is release-candidate; pin a version and decide v1 vs v2 deliberately.
Phase 4 - Reweight + drift (2 weeks): loop 2 across both paths; shadow mode first (compute weights, do not apply).
Phase 5 - Paper-trading gate (4+ weeks): both paths on paper. Promote to small live size only if: positive net PnL after costs, drawdown within policy, paper results consistent with backtest, no unexplained gaps. Otherwise iterate.
Phase 6 - RL sizing (optional): FinRL-X agents in shadow mode versus fixed sizing; adopt only if clearly better out of sample.

## 6. Connectors found in the MCP registry (none connected; search only)

| Connector | Use here | Notes |
|---|---|---|
| Interactive Brokers (IBKR) | stock broker/account data | Tools include balances, positions, orders, price history. Execution should stay in Nautilus, not the chat connector. |
| Webull | stock broker/account data | 60+ tools incl. trading; same caution. |
| Clear Street | stocks/options research, balances | |
| Crypto.com | crypto market data (tickers, candles, trades) | Market-data endpoint. |
| Alpha Vantage | stock data, indicators, news | Free tier is rate-limited. |
| FMP, Financial Datasets | fundamentals, filings | For the TradingAgents fundamentals/news side. |
| LunarCrush | social sentiment for crypto and stocks | Feeds the sentiment analyst. |

Recommendation: connect read-only market-data connectors (Alpha Vantage, Crypto.com) for research. Do not give any chat connector order-placing access; keep order flow inside the engines under the risk policy.

## 7. Known risks and open questions

- No edge is proven. LLM-agent repos backtest more easily than they trade; treat Phase 5 as the real test.
- TradingAgents is non-deterministic and costs money per run; benchmark its cost per symbol per day.
- Star counts of some repos are unusually high for their age and were not audited.
- Qlib: unreleased breaking change on artifact loading; its official dataset is disabled, so stock data must come from our own vendor.
- Nautilus ROADMAP puts built-in ML tooling out of scope, so the model service is ours to build and maintain.
- Open: account sizes, broker choice, data budget, LLM budget, jurisdiction/tax and regulatory constraints.
