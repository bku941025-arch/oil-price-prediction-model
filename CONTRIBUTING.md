# Contributing to PumpCast

Thanks for helping. The most valuable contributions are usually **tax corrections** and **bug reports**, and neither needs any code.

## Report a missed or wrong tax change

Fuel taxes change often and differ by province. If the price breakdown looks wrong, or a forecast missed a tax change, open a **Tax change** issue with:

- the province (and city, for municipal levies)
- the component (federal excise, carbon charge, provincial fuel tax, transit levy, sales tax)
- the new rate in ¢/L, the date it takes effect, and the **date it was announced**
- a link to an official source (government or regulator site)

The announcement date matters: forecasts made before a change was public must not know about it (see [docs/leakage-writeup.md](docs/leakage-writeup.md)).

To fix it yourself, edit the schedules in [`gasprix/taxes.py`](gasprix/taxes.py), add the announcement date to `ANNOUNCED`, and add a case to `tests/test_taxes.py`.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
playwright install chromium        # only for the browser smoke test
```

| Task | Command |
|---|---|
| Lint | `ruff check .` |
| Unit tests | `pytest --ignore=tests/e2e` |
| Browser smoke test | `pytest tests/e2e` |
| Full pipeline (about 5 min, needs network) | `python pipeline.py` |
| Backtest | `python -m gasprix.backtest` |
| Measure a feature group | `python -m gasprix.backtest --without cycles` |
| Serve the site | `python webapp/app.py` |

## Ground rules for model changes

- **No look-ahead.** Every feature at origin date *t* may use only information published by the end of day *t*. Market series go through `data.known_market()`, and `tests/test_features.py` must keep passing.
- **Measure before keeping.** Run the backtest with and without your change, and include both numbers in the pull request. Keep a change only if it improves the out-of-sample error.
- **Keep the README honest.** If accuracy changes, update the tables in the README.

## Pull requests

CI runs lint, unit tests and a browser smoke test on every pull request. Keep changes focused, and describe what you measured.
