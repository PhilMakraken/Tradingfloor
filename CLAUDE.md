# propfloor — autonomous prop-firm research lab & virtual trading floor

Goal: discover strategies with a high, *robust* probability of passing prop-firm
challenges (Stage 1 -> Stage 2 -> Funded) and surviving the funded phase.
Paper trading only.

## Commands
- `pip install -e .[dev]`  (Python >= 3.11)
- `pytest -q`                                  — test suite
- `python -m propfloor data ingest`            — load data/raw/*.csv (+ downloads) into data/parquet/
- `python -m propfloor data download --source dukascopy --symbols EURUSD,GBPUSD --start 2018-01-01`
- `python -m propfloor data download --source github --start 2012-01-01`  — 12 FX/gold M15 histories 2012-2022 (GitHub mirror; works where Dukascopy/HistData are firewalled)
- `python -m propfloor data synth --symbols EURUSD`  — synthetic data (smoke tests ONLY)
- `python -m propfloor data download --source yahoo --symbols stocks --start 2010-01-01`  — 10+ yr DAILY stocks/indices/oil (`stocks`, `commodities`, `indices`, `fx-majors`, `metals` shorthands); research them with `--timeframes D1`
- `python -m propfloor factory run [--hours N] [--llm] [--agents] [--live]`  — strategy factory (runs forever by default); `--agents` = Claude research-agent team alongside, `--live` = forward-test validated strategies on live data; gates default to 10y real history, 70% profitable years, Monte Carlo robustness
- `python -m propfloor agents run [--roles price_action]`  — one round of the research agents (ideas -> data/ideas/)
- `python -m propfloor live run|status`  — live forward test of validated strategies (data/live, never mixed into data/parquet)
- `python -m propfloor notes`  — regenerate data/RESEARCH_NOTES.md
- `python -m propfloor floor run`              — paper-trading floor step
- `streamlit run propfloor/dashboard/app.py`   — dashboard

## Architecture (package `propfloor/`)
| module | role |
|---|---|
| `config.py` | pydantic models: `ProgramConfig`, `StageConfig`, `DrawdownConfig`, `DailyDrawdownConfig`, `RuleSpec`; `load_program(yaml)` |
| `rules/base.py` | `Rule` ABC (`check(state, event) -> Violation|None`), `@register_rule(name)` registry, `build_rules(specs)`. Entry-point group `propfloor.rules` + `rules.plugins` setting let new rules load without touching core |
| `rules/builtin.py` | max_lot, max_open_trades, news_blackout, no_weekend_holding, max_time_in_trade, min_trading_days (completion gate), consistency_rule (completion gate) |
| `account.py` | `Account`: balance/equity, trailing max-DD floor (peak equity or peak closed balance, optional lock at start balance), daily DD (basis balance/equity, reset hour + tz), event processing, permanent closure on breach |
| `risk.py` | `RiskManager`: worst-case loss = stop + spread + slippage + gap allowance; must fit remaining daily AND trailing buffer; sizing from buffer; soft daily stop; de-risk near target; conservative funded profile |
| `data/` | `loader.py` CSV auto-detect (MT4/MT5/HistData/Dukascopy/generic), `clean.py` (dedupe, gaps, tz->UTC, sanity), `store.py` parquet, `dukascopy.py` (+ `recent()` hourly tick files for live), `histdata.py`, `yahoo.py` (daily 10y+ stocks/indices/futures, recent intraday), `github_mirror.py` (M15 MT exports, server time = NY+7h), `synthetic.py` |
| `strategies/indicators.py` | numpy/numba indicators (SMA, EMA, RSI, MACD, BB, ATR, ADX, Stoch, Donchian, swings) |
| `strategies/blocks.py` | building-block registry + `Ctx` (bars, cached indicators, `peers` for intermarket, `train_end`): ENTRIES, FILTERS, EXIT_SPACE (ATR/swing/bar SL, R TP, time stop incl. -1 = P5 signal-decay, close/chandelier trail, breakeven, partial_r, exit_mode ema20/sma5/bb_mid/rsi50/channel/session_end), SIZING_SPACE, OVERLAY_SPACE |
| `strategies/blocks_ext.py` | catalogue sections A-L, N, O (+ regime filters choppiness/autocorr/hurst/premium-discount) |
| `strategies/blocks_intermarket.py` | section M + H5/H6: gold laggard, JPY risk filter, triangulation, lead-lag, safe-haven filter, basket strength, pair divergence, correlation breakdown |
| `strategies/blocks_meta.py` | section P: vote_ensemble, regime_router (also H4), ml_classifier (walk-forward HistGradientBoosting, yearly refit) |
| `strategies/blocks_stat.py` | user XAUUSD workbook High-feasibility OHLC families: entry `stat_feature` (feature x window -> robust z -> learned(train_mask corr sign)/momentum/fade -> ±z_thr cross) + filter `stat_regime`; 14 causal features (fractal_dim, hurst, perm_entropy, acf_sum, skew, kurt, semivar, vol_of_vol, dd_hazard, range_disagree, path_eff, run_length, sign_entropy, xagg_acf) |
| `strategies/blocks_user.py` | user-ideas mapping "Top 10" entries: dollar_residual_xau (XAU vs USD-basket proxy from EURUSD/GBPUSD/AUDUSD/USDJPY/USDCAD/USDCHF peers, beta on train_mask), ar_innovation (rolling AR(p), ADX fade/follow), gk_vol_surprise, kalman_innovation, daily_pivot, flag_continuation, failed_failure, anchored_vwap, cci_resume, acceptance_break (`confirm_closes` N closes beyond donchian/prev_day/swing/vwap = vwap_reclaim; separate block so existing genome keys are unchanged) |
| `strategies/blocks_cbot.py` | entry logic of the user's gold cBots 6/7 (`ports/cbots.py` replays) as factory entries, exits from EXIT_SPACE, mirrored for shorts: `breakout_runner` (bot 6: bar high > prior `lookback`-bar high with fast>slow seeded EMA on the last closed bar, optional prior-broker-day close > daily EMA200), `macd_dema` (bot 7: 2-bar MACD-histogram state machine, first bar of a leg whose last COMPLETED broker-clock HTF bar (`htf_mult` x bar size) DEMA slope agrees; one signal per leg). Long signals hit 100% of run_bot6/run_bot7 entries on 2012-22 gold (bot 7 on broker-clock H4 bars) |
| `strategies/blocks_concepts.py` | user price-action concepts: `icc` (Indication = close through last confirmed swing high, Correction = min_retr..max_retr pullback of the leg, Continuation = close above prior `trig_n` highs; numba state machine, mirrored shorts), `session_reclaim` (range `range_start..range_end` UTC breaks, retrace into the zone_a..zone_b (~0.70-0.886) fib of the breakout leg, bullish/bearish close back out of the zone; anchors range/level/leg; one per direction per day) |
| `strategies/genome.py` | `StrategyGenome` (entry+filters+exit+params, json-serialisable, hash id), random generation, mutation, crossover |
| `strategies/backtest.py` | numba bar-by-bar engine -> trade list with pnl in R, MAE/MFE in R, order flag, stop distance, timestamps |
| `evaluation/challenge.py` | simulated challenges: many random start dates x Monte Carlo trade reshuffle/bootstrapping through `Account`+`RiskManager`+rules => P(pass) per stage, median days, breach causes, funded survival & payouts |
| `evaluation/stats.py` | Sharpe, deflated Sharpe ratio (Bailey & Lopez de Prado), PSR |
| `evaluation/walkforward.py` | IS / walk-forward folds / OOS holdout; rejection gates (incl. `min_history_years`, `min_positive_years`, robustness); `Universe` skips timeframes finer than a symbol's native data (daily-only stocks -> D1) |
| `evaluation/robustness.py` | Monte Carlo anti-overfitting, run after every other gate passed: trade bootstrap P(profit), random-entry permutation p-value (same exits), +/-15% parameter-neighbourhood plateau, calendar-year breakdown |
| `factory/` | `engine.py` random search + GA loop, `llm.py` optional Claude idea/critique (ANTHROPIC_API_KEY), runs indefinitely; `agents.py` AgentTeam (technical_analyst, price_action, quant, risk_critic) via `claude -p` (WebSearch/WebFetch) or the API, writes idea JSON to data/ideas/; `notes.py` compact data/RESEARCH_NOTES.md (aggregated per entry/symbol/tf niche) + append-only data/FOUND_STRATEGIES.md |
| `live/forward.py` | `ForwardTester`: enrols validated strategies, fetches new real bars (Dukascopy hourly ticks / Yahoo) into data/live, re-runs the unchanged rules and scores only trades entered after enrolment: probation/confirmed/degraded/failed + longevity 0-100 |
| `db.py` | SQLite results DB (strategies, evaluations, trials counter, floor accounts, events) |
| `floor/` | `accounts.py` lifecycle Stage1->Stage2->Funded->Violated, `allocator.py` (top validated, low-correlation), `broker.py` `BrokerAdapter` ABC + `PaperBroker` + MT5/cTrader stubs, `floor.py` orchestrator |
| `dashboard/app.py` | Streamlit: accounts, buffers, pass rates, leaderboard |

## Key modelling decisions
- Units: strategies emit trades in **R** (1R = stop distance). The risk manager converts R to money,
  so the same trade list is re-simulated under any program/sizing cheaply.
- Floating-equity violations use per-trade MAE/MFE plus a flag for which came first, so the trailing
  floor is ratcheted by MFE before MAE only when MFE truly happened first.
- One position at a time per strategy (portfolio-level overlaps happen on the floor).
- Account tactics (catalogue J, P3, P4, Q) are genome `overlay` params applied by `RiskManager.size`
  (profit cap, max trades/day, loss-streak cooldown, fast-pass, buffer-proportional risk, target banking,
  anti-martingale) and by `ChallengeSimulator` (P3 equity-curve filter). P7 rotation = `floor run --rotate-top N`.
- Blocks that fit statistics (time buckets, session drift, ML, P5 hold) use strictly-past / in-sample data only;
  `tests/test_strategies.py::test_all_blocks_run_without_lookahead` and `tests/test_catalog.py` enforce causality.
- Anti-overfitting: every evaluated genome increments the trial counter; DSR uses it. A strategy is
  "validated" only if IS, every walk-forward fold, and the untouched OOS holdout all clear gates.
- Synthetic data is never eligible for validation (data provenance stored per dataset).

## Phases
1. Config + rules engine + Account (trailing/daily DD) — tests
2. Risk manager — tests
3. Data ingest/clean/store + downloaders + synthetic — tests
4. Indicators, blocks, genome, backtester — tests
5. Evaluation: challenge MC, DSR, walk-forward — tests
6. Factory (random + GA + LLM) + SQLite — tests
7. Trading floor + allocator + broker stubs — tests
8. Streamlit dashboard

## Conventions
- Keep core firm-agnostic; firm specifics live only in `programs/*.yaml`.
- New rule = new class with `@register_rule("name")` in any module listed in YAML `plugins:`.
- Tests must not touch the network.
