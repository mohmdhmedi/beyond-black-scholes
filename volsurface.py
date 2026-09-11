"""Reusable pieces of the Beyond Black-Scholes notebook: implied volatility, forwards,
static-arbitrage checks and raw SVI.

Nothing here is specific to SPY. The arbitrage checks expect an option chain in long format with
columns: date, expiry, cp ('C'/'P'), strike, bid, ask, mid, k, dte, rel_spread.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr

SQRT_2PI = np.sqrt(2.0 * np.pi)
TICK = 0.01                      # one cent: smaller violations are treated as rounding
SLICE = ["date", "expiry", "cp"]
K_GRID = np.linspace(-0.25, 0.25, 21)


# ---------------------------------------------------------------- Black (1976) and implied volatility
def black_price(F, K, tau, D, sigma, is_call):
    """Black (1976) price of a European option on the forward F, discounted by D."""
    v = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    call = D * (F * ndtr(d1) - K * ndtr(d2))
    put = D * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(is_call, call, put)


def black_vega(F, K, tau, D, sigma):
    """dPrice/dSigma under Black (1976); identical for calls and puts."""
    v = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    return D * F * np.exp(-0.5 * d1 * d1) / SQRT_2PI * np.sqrt(tau)


@dataclass
class IVResult:
    """Implied volatilities plus solver diagnostics for each contract."""
    iv: np.ndarray            # NaN where no solution exists
    method: np.ndarray        # 'newton', 'bisection', 'no solution', 'not converged'
    newton_steps: np.ndarray


def implied_vol(price, F, K, tau, D, is_call, tol: float = 1e-6, max_newton: int = 40,
                n_bisect: int = 60, lo: float = 1e-4, hi: float = 5.0) -> IVResult:
    """Vectorised implied volatility: Newton-Raphson on vega with a bisection fallback.

    All inputs are 1-d numpy arrays of equal length. tol is an absolute price tolerance.
    Prices outside the no-arbitrage bounds get NaN and method 'no solution'.
    """
    price, F, K, tau, D = (np.asarray(a, dtype=float) for a in (price, F, K, tau, D))
    is_call = np.asarray(is_call, dtype=bool)
    n = price.size
    intrinsic = D * np.maximum(np.where(is_call, F - K, K - F), 0.0)
    upper = np.where(is_call, D * F, D * K)
    valid = np.isfinite(price) & np.isfinite(F) & np.isfinite(D) & (tau > 0) & (price > intrinsic) & (price < upper)

    x = np.clip(np.sqrt(2 * np.pi / np.maximum(tau, 1e-8)) * price / (D * F), 0.05, 1.5)
    steps = np.zeros(n, dtype=np.int16)
    done = np.zeros(n, dtype=bool)
    to_bisect = np.zeros(n, dtype=bool)
    live = valid.copy()
    for _ in range(max_newton):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        diff = black_price(F[idx], K[idx], tau[idx], D[idx], x[idx], is_call[idx]) - price[idx]
        hit = np.abs(diff) < tol
        done[idx[hit]] = True
        vega = black_vega(F[idx], K[idx], tau[idx], D[idx], x[idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            new = x[idx] - diff / vega
        bad = ~hit & ((vega < 1e-8) | ~np.isfinite(new) | (new <= lo) | (new >= hi))
        step = ~hit & ~bad
        x[idx[step]] = new[step]
        steps[idx[~hit]] += 1
        to_bisect[idx[bad]] = True
        live[idx[hit | bad]] = False
    to_bisect |= valid & ~done & ~to_bisect

    idx = np.flatnonzero(to_bisect)
    if idx.size:
        a = np.full(idx.size, lo)
        b = np.full(idx.size, hi)
        for _ in range(n_bisect):
            mid = 0.5 * (a + b)
            above = black_price(F[idx], K[idx], tau[idx], D[idx], mid, is_call[idx]) > price[idx]
            b = np.where(above, mid, b)
            a = np.where(above, a, mid)
        sol = 0.5 * (a + b)
        err = np.abs(black_price(F[idx], K[idx], tau[idx], D[idx], sol, is_call[idx]) - price[idx])
        x[idx] = sol
        done[idx] = err < 10 * tol

    iv = np.where(valid & done, x, np.nan)
    method = np.where(~valid, "no solution",
                      np.where(~done, "not converged", np.where(to_bisect, "bisection", "newton")))
    return IVResult(iv=iv, method=method, newton_steps=steps)


def atm_parity_forward(call_mid, put_mid, strike, spot, D, n_atm: int = 3) -> float:
    """Forward from put-call parity at the n_atm strikes closest to spot: F = K + (C - P) / D.

    At the money C - P is close to zero, so the estimate is insensitive to D and to the
    early-exercise premium of American options.
    """
    strike = np.asarray(strike, dtype=float)
    order = np.argsort(np.abs(strike - spot))[:n_atm]
    f = strike[order] + (np.asarray(call_mid)[order] - np.asarray(put_mid)[order]) / D
    return float(np.median(f))


# ---------------------------------------------------------------- static arbitrage
def _neighbours(df: pd.DataFrame, cols: list, lag: int) -> pd.DataFrame:
    return df.groupby(SLICE, sort=False, observed=True)[cols].shift(lag)


def vertical_checks(df: pd.DataFrame, tick: float = TICK) -> pd.DataFrame:
    """Monotonicity and the (undiscounted, American-valid) spread bound between adjacent strikes."""
    d = df.sort_values(SLICE + ["strike"]).reset_index(drop=True)
    nxt = _neighbours(d, ["strike", "mid", "bid", "ask"], -1)
    dk = nxt["strike"] - d["strike"]
    call = (d["cp"] == "C").to_numpy()
    mono_mid = np.where(call, nxt["mid"] - d["mid"], d["mid"] - nxt["mid"]) > tick
    bound_mid = np.where(call, d["mid"] - nxt["mid"], nxt["mid"] - d["mid"]) - dk > tick
    mono_trd = np.where(call, nxt["bid"] - d["ask"], d["bid"] - nxt["ask"]) > tick
    bound_trd = np.where(call, d["bid"] - nxt["ask"], nxt["bid"] - d["ask"]) - dk > tick
    out = d[["date", "expiry", "cp", "strike", "k", "dte", "rel_spread"]].copy()
    out["mid_violation"] = mono_mid | bound_mid
    out["tradable_violation"] = mono_trd | bound_trd
    return out[nxt["strike"].notna().to_numpy()]


def butterfly_checks(df: pd.DataFrame, tick: float = TICK) -> pd.DataFrame:
    """Convexity in strike for every consecutive strike triple, allowing unequal spacing.

    Mid test:        lam*C1 + (1-lam)*C3 - C2 >= 0,  lam = (K3-K2)/(K3-K1)
    Executable test: buy the wings at the ask, sell the body at the bid.
    """
    d = df.sort_values(SLICE + ["strike"]).reset_index(drop=True)
    prv = _neighbours(d, ["strike", "mid", "ask"], 1)
    nxt = _neighbours(d, ["strike", "mid", "ask"], -1)
    lam = (nxt["strike"] - d["strike"]) / (nxt["strike"] - prv["strike"])
    mid_val = lam * prv["mid"] + (1 - lam) * nxt["mid"] - d["mid"]
    trd_val = lam * prv["ask"] + (1 - lam) * nxt["ask"] - d["bid"]
    out = d[["date", "expiry", "cp", "strike", "k", "dte", "rel_spread"]].copy()
    out["mid_violation"] = (mid_val < -tick).to_numpy()
    out["tradable_violation"] = (trd_val < -tick).to_numpy()
    out["credit"] = -trd_val.to_numpy()
    return out[(prv["strike"].notna() & nxt["strike"].notna()).to_numpy()]


def calendar_checks(otm: pd.DataFrame, rel_tol: float = 0.01) -> pd.DataFrame:
    """Total variance w = iv^2 * tau must not fall with maturity at fixed k (columns: date, expiry, tau, dte, k, w)."""
    rows = []
    for date, day in otm.groupby("date", sort=False):
        curves = []
        for _, g in day.sort_values("k").groupby("expiry"):
            if len(g) >= 3:
                curves.append((g["tau"].iat[0], g["dte"].iat[0], g["k"].to_numpy(), g["w"].to_numpy()))
        curves.sort(key=lambda c: c[0])
        for (_, _, k1, w1), (_, d2, k2, w2) in zip(curves[:-1], curves[1:]):
            lo, hi = max(k1[0], k2[0]), min(k1[-1], k2[-1])
            grid = K_GRID[(K_GRID >= lo) & (K_GRID <= hi)]
            if grid.size:
                v1, v2 = np.interp(grid, k1, w1), np.interp(grid, k2, w2)
                rows += [(date, d2, kk, b < a * (1 - rel_tol)) for kk, a, b in zip(grid, v1, v2)]
    return pd.DataFrame(rows, columns=["date", "dte", "k", "mid_violation"])


# ---------------------------------------------------------------- raw SVI
def svi_w(k, p):
    """Raw SVI total variance w(k) for p = (a, b, rho, m, sigma)."""
    a, b, rho, m, s = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s * s))


def svi_g(k, p):
    """Gatheral-Jacquier density g(k) and w(k): a slice is butterfly-free iff g >= 0 and w > 0."""
    a, b, rho, m, s = p
    x = k - m
    r = np.sqrt(x * x + s * s)
    w = a + b * (rho * x + r)
    w1 = b * (rho + x / r)
    w2 = b * s * s / r ** 3
    g = (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2
    return g, w


def fit_svi(k, w, rho_starts=(-0.7, -0.3, 0.1)):
    """Least-squares raw SVI with bounds, a non-negative-minimum-variance penalty and Lee's wing bound.

    Returns (params, rmse in total variance).
    """
    k, w = np.asarray(k, dtype=float), np.asarray(w, dtype=float)
    wmax = float(w.max())
    lb = np.array([-wmax, 1e-6, -0.999, k.min() - 0.5, 1e-4])
    ub = np.array([wmax, 2.0, 0.999, k.max() + 0.5, 2.0])

    def resid(p):
        a, b, rho, m, s = p
        pen_min = max(0.0, -(a + b * s * np.sqrt(1 - rho * rho)))
        pen_lee = max(0.0, b * (1 + abs(rho)) - 2.0)
        return np.concatenate([svi_w(k, p) - w, [10 * pen_min, 10 * pen_lee]])

    best = None
    m0 = float(k[np.argmin(w)])
    for rho0 in rho_starts:
        b0, s0 = 0.1, 0.1
        a0 = float(w.min()) - b0 * s0 * np.sqrt(1 - rho0 ** 2)
        x0 = np.clip([a0, b0, rho0, m0, s0], lb + 1e-9, ub - 1e-9)
        sol = least_squares(resid, x0, bounds=(lb, ub), method="trf", max_nfev=400)
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x, float(np.sqrt(np.mean((svi_w(k, best.x) - w) ** 2)))
