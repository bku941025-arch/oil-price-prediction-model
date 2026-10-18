# My gas price model looked 2× better than it was. Here's the bug.

*How a three-day publication lag inflated my backtest, and what was left after I fixed it.*

## The setup

PumpCast forecasts retail gasoline prices 1 to 28 days ahead for 77 Canadian cities. The inputs are daily pump prices from Kalibrate, plus wholesale gasoline, crude oil and the exchange rate. The model is a set of Ridge + LightGBM pairs, one per forecast horizon, each predicting how much the price will change.

The benchmark every gas forecast has to beat is "no change": tomorrow's price equals today's. Pump prices are close to a random walk, so this naive guess is surprisingly hard to beat.

## The first result looked great

My first backtest trained on 2016–2023 and tested on 2024 onward. Montréal came out at **4.56 ¢/L** average error vs **4.85** for "no change", 6% better, and other Québec cities were 7–13% better. At 7 days ahead, the model called the direction of the move correctly 67–73% of the time.

The most important input, by far, was the **wholesale gasoline price**: when the New York Harbor price jumps, pump prices follow a few days later. That's a real effect, and it's why I trusted the result.

## The bug: the model could see the future, by three days

For a forecast made on day *t*, the backtest used the wholesale price **dated** day *t*.

But the live pipeline never has that number. FRED publishes the EIA spot price series 2–3 business days late. When the pipeline ran on October 17, the newest wholesale price available was dated October 14, and the newest FRED exchange rate was a week old.

So the backtest let the model see wholesale moves three days before they were public. Since retail follows wholesale with a short lag, that head start was most of the model's apparent edge.

Nothing errored, and the feature engineering was "correct" in the sense that every rolling window looked backward. The leak sat in the **alignment**: a value's date is not the date you learn it.

## The fix

Each market series now carries its publication lag:

```python
FRED_SERIES = {
    'DCOILWTICO': ('wti_usd',  3),   # value dated d is usable from d + 3
    'DGASNYH'   : ('rbob_usd', 3),
    ...
}
```

`known_market()` shifts every series forward by its lag **before** any feature is built, in training, backtesting and live forecasting alike. There's a test that changes all data after a cutoff date and asserts no feature at or before the cutoff moves. I checked that the test can fail by planting a `shift(-3)` leak, which it caught.

I also switched USD/CAD to the Bank of Canada's daily series, which is published the same afternoon, instead of FRED's weekly-updated copy.

## The second leak: taxes nobody had announced yet

Canadian pump prices include fixed per-litre taxes that change on known dates. The carbon charge disappeared on April 1, 2025, and the 10¢ federal excise was suspended on April 20, 2026. The model works on prices net of tax and adds scheduled changes back on top.

The catch: "scheduled" has to mean **announced by the forecast date**. The excise suspension was announced on April 14, 2026, six days before it took effect. Until September 2, the official plan was for the tax to return on September 8; then that was cancelled. A forecast made in August should have expected a 10¢ jump that never came.

`taxes.py` now records when each change was announced, and keeps the cancelled plan, so every backtest forecast sees only the tax schedule known at the time.

## What was left

With both leaks closed, one pooled model across 77 cities:

| | Model | "No change" |
|---|---|---|
| 7 days ahead | 3.49 ¢/L | 3.73 ¢/L |
| 28 days ahead | 7.46 ¢/L | 7.92 ¢/L |
| Average, 1–28 days | **5.06 ¢/L** | **5.34 ¢/L** |

That's about **5% better than "no change", in all 77 cities**. Montréal itself is now only marginally better (4.83 vs 4.85). It's a smaller number than I started with, but this one is real, and the 80% ranges really do cover about 77% of outcomes.

## Lessons

1. **A value's timestamp is not its availability date.** For every input, ask: *on the morning I'd run this, would I have that number?*
2. **Leaks hide in joins and alignment, not in feature formulas.** All my rolling windows were correct.
3. **Test for look-ahead directly.** Perturb the future and assert the past doesn't change, then check the test fails on a planted leak.
4. **"Known in advance" has a date too.** Policy changes, holidays, scheduled events: record when each one became public.
5. **Publish the honest number.** A 5% edge across 77 cities, measured properly, is worth more than a 13% edge that disappears in production.

*Code: [`gasprix/data.py`](../gasprix/data.py) (publication lags), [`gasprix/taxes.py`](../gasprix/taxes.py) (announcement dates), [`tests/test_features.py`](../tests/test_features.py) (look-ahead test). Reproduce the numbers with `python -m gasprix.backtest`.*
