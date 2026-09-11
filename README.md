# Beyond Black-Scholes: Static Arbitrage in the SPY Volatility Surface (2010-2023)

Fourteen years of end-of-day SPY option chains, used to ask one narrow question: **are the quoted
prices consistent with each other?**

Implied volatilities are re-solved from scratch, the three static no-arbitrage conditions (vertical,
butterfly, calendar) are tested both at mid prices and at the prices one could actually trade, and a
raw SVI fit per expiry is used as the benchmark a gradient-boosted model has to beat, with the result
reported whichever way it falls.

- **Notebook on Kaggle:** [kaggle.com/code/mohmdhmedi/beyond-black-scholes](https://www.kaggle.com/code/mohmdhmedi/beyond-black-scholes)
- **Data:** [SPY Options EOD Data (2010-2023)](https://www.kaggle.com/datasets/dudesurfin/spy-options-eod-volatility-surface-2010-2023), 14 yearly parquet files, 596 MB, MIT licence

## What the notebook does

| section | content |
|---|---|
| Data quality audit | every filter counted: missing, zero-bid, crossed, duplicate and expiry-day quotes; a *clean* tier and a *liquid* tier (spread within 20% of mid, non-zero volume) |
| The shape of the market | chain size over time, call/put volume, where volume trades by moneyness and maturity, the cost of the wings |
| Forwards | forward per expiry from put-call parity at the money; why the textbook regression gives *negative* interest rates on American options |
| Implied volatility | Black (1976) on the forward, vectorised Newton-Raphson with a bisection fallback, compared against the vendor's IV |
| The surface | 3D surfaces on a calm day and in March 2020, smiles, term structure, implied versus realised volatility, a week-by-week volatility regime map |
| Static arbitrage | vertical, butterfly (unequal strike spacing) and calendar tests, at mid and at executable bid/ask prices, clean versus liquid, through time |
| SVI | raw SVI per expiry with Lee's wing bound, butterfly check with the Gatheral-Jacquier density, calendar crossings between fitted slices |
| Machine learning | LightGBM against SVI and linear interpolation on the same held-out quotes, split by time (train 2010-2019, test 2020-2023), SHAP |

The interpretation after every chart is generated from that run's numbers, so the text cannot drift
away from the results.

## Findings worth knowing before you open it

- **Put-call parity regression breaks on American options.** Regressing `C - P` on strike to get the
  discount factor gives a negative interest rate in every year of the sample, including 2023 when
  T-bills paid about 5%. The early-exercise premium of in-the-money puts steepens the slope. The notebook
  keeps this as a finding and takes the forward from at-the-money parity instead.
- **Most apparent arbitrage is a statement about a midpoint.** The executable test (buy at the ask,
  sell at the bid) is the one that separates a mispricing from a wide or stale quote.
- **The honest ML result.** The model comparison is set up so that every method sees the same
  information, and its verdict is printed as it comes out, including when the simplest method wins.

## Reusable outputs

The notebook writes these to its output folder (Kaggle: *Add Input -> Notebooks -> Beyond Black-Scholes*):

| file | contents |
|---|---|
| `spy_otm_iv_surface.parquet` | every clean OTM quote on the sampled dates with forward, discount factor, log-moneyness, bid/ask/mid, re-solved IV and a liquidity flag |
| `forwards_discount_factors.csv` | forward and discount factor per (date, expiry) |
| `svi_params.csv` | raw SVI parameters per fitted slice, fit error, butterfly-freedom flags |
| `atm30_iv_weekly.csv` | constant-maturity 30-day ATM implied volatility |
| `arbitrage_rates_by_date.csv` | violation rates per date, per test, clean and liquid, mid and executable |

## Reusable code

[`volsurface.py`](volsurface.py) holds the core functions on their own, with nothing SPY-specific:

```python
from volsurface import implied_vol, butterfly_checks, fit_svi

res = implied_vol(price, F, K, tau, D, is_call)      # vectorised, returns iv + solver diagnostics
bf = butterfly_checks(chain)                           # mid and executable butterfly violations
params, rmse = fit_svi(k, total_variance)              # raw SVI with Lee's bound
```

## Running it

On Kaggle: open the notebook, make sure the dataset is attached (*Add Input*), then *Save Version ->
Save & Run All*. It runs on CPU in about a quarter of an hour.

Locally: `pip install -r requirements.txt`, download the dataset, and set `SPY_DATA_DIR` to the folder
holding the parquet files.

## Limitations

SPY options are American while the pricing model is European (mitigated by using out-of-the-money
contracts); the discount rate is an annual T-bill proxy; quotes are end-of-day snapshots; weekly and
monthly sampling rather than every day. All are discussed in the notebook.

## References

Black & Scholes (1973); Black (1976); Carr & Madan (2005); Gatheral (2004, 2006); Gatheral & Jacquier
(2014); Lee (2004); Roper (2010); Ke et al. (2017); Lundberg & Lee (2017). Full citations in the notebook.

## Licence

MIT, see [LICENSE](LICENSE).
