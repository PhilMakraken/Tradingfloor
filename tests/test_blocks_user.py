"""User-workbook entry blocks (blocks_user): finite output, causality, peer/train isolation, signal counts."""
import numpy as np
import pytest

from propfloor.data.clean import resample
from propfloor.data.synthetic import make_bars
from propfloor.evaluation.walkforward import Universe
from propfloor.strategies import blocks_user as bu
from propfloor.strategies.blocks import ENTRIES, Ctx, sample_params

NAMES = [n for n, b in ENTRIES.items() if b.fn.__module__ == bu.__name__]
PEERS = {"EURUSD": 1.15, "GBPUSD": 1.3, "AUDUSD": 0.7, "USDJPY": 110.0, "USDCAD": 1.3, "USDCHF": 0.95}


@pytest.fixture(scope="module")
def uni():
    data = {"XAUUSD": make_bars("2020-01-01", "2020-07-01", seed=21, price0=1500.0, freq="15min")}
    data |= {s: make_bars("2020-01-01", "2020-07-01", seed=30 + i, price0=p, freq="15min")
             for i, (s, p) in enumerate(PEERS.items())}
    return Universe(data, timeframes=["H1"])


def _truncated(full: Ctx, cut: int) -> Ctx:
    short = Ctx(full.df.iloc[:cut], full.symbol, full.tf)
    short.train_end = full.train_end
    short.peers = {s: Ctx(c.df[c.df.index < full.time[cut]], s, full.tf) for s, c in full.peers.items()}
    return short


def test_all_ten_registered():
    assert set(NAMES) == {"dollar_residual_xau", "ar_innovation", "gk_vol_surprise", "kalman_innovation",
                          "daily_pivot", "flag_continuation", "failed_failure", "anchored_vwap", "cci_resume",
                          "acceptance_break"}


@pytest.mark.parametrize("name", NAMES)
def test_bool_output_and_causal_with_peers(uni, name):
    full = uni.ctx[("XAUUSD", "H1")]
    full.train_end = full.time[int(full.n * 0.6)]
    cut = full.n - 60
    short = _truncated(full, cut)
    rng = np.random.default_rng(5)
    b = ENTRIES[name]
    for _ in range(4):
        p = sample_params(b.space, rng)
        lo, sh = b.fn(full, p)
        assert lo.dtype == bool and sh.dtype == bool and len(lo) == len(sh) == full.n, name
        lo2, sh2 = b.fn(short, p)
        assert (lo[:cut] == lo2[:cut]).all() and (sh[:cut] == sh2[:cut]).all(), (name, p)


def test_intermediate_series_finite_and_causal(uni):
    x = uni.ctx[("XAUUSD", "H1")]
    x.train_end = x.time[int(x.n * 0.6)]
    cut = x.n - 60
    s = _truncated(x, cut)
    u = bu.usd_basket_ret(x, 6)
    z = bu.dollar_residual_z(x, 6)
    assert np.isfinite(u[50:]).mean() > 0.95 and np.isfinite(z[300:]).mean() > 0.95
    assert not np.isinf(z).any()
    np.testing.assert_array_equal(z[:cut], bu.dollar_residual_z(s, 6)[:cut])
    r = np.r_[np.nan, np.diff(np.log(x.c))]
    a = bu._ar_innov(r, 2, 50)
    assert np.isfinite(a[60:]).all()
    np.testing.assert_array_equal(a[:cut], bu._ar_innov(r[:cut], 2, 50))


def test_ar_innovation_matches_direct_ols():
    rng = np.random.default_rng(0)
    r = rng.normal(size=600)
    r[0] = np.nan
    for o in (1, 2, 3):
        w, t = 60, 500
        z = bu._ar_innov(r, o, w)
        X = np.column_stack([r[t - w - 1 - i:t - 1 - i] for i in range(o)])
        y = r[t - w:t]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        sig = np.sqrt(((y - X @ beta) ** 2).sum() / (w - o))
        assert z[t] == pytest.approx((r[t] - beta @ r[t - 1 - np.arange(o)]) / sig, rel=1e-6)


def test_dollar_residual_needs_xau_and_peers_and_uses_train_only(uni):
    x = uni.ctx[("XAUUSD", "H1")]
    p = {"n": 6, "thr": 2.0, "mode": "fade"}
    eur = uni.ctx[("EURUSD", "H1")]
    lo, sh = ENTRIES["dollar_residual_xau"].fn(eur, p)
    assert not lo.any() and not sh.any()
    lone = Ctx(x.df, "XAUUSD", "H1")  # no peers -> no USD proxy -> no signals
    lo, sh = ENTRIES["dollar_residual_xau"].fn(lone, p)
    assert not lo.any() and not sh.any()
    # scrambling the out-of-sample gold path must not change the in-sample residuals (beta fitted in-sample)
    a = Ctx(x.df, "XAUUSD", "H1")
    a.peers, a.train_end = x.peers, x.time[int(x.n * 0.6)]
    d2 = x.df.copy()
    oos = d2.index >= a.train_end
    d2.loc[oos, "close"] = d2.loc[oos, "close"].to_numpy()[::-1]
    b = Ctx(d2, "XAUUSD", "H1")
    b.peers, b.train_end = x.peers, a.train_end
    m = ~oos
    np.testing.assert_array_equal(bu.dollar_residual_z(a, 6)[m], bu.dollar_residual_z(b, 6)[m])
    fl, fs = ENTRIES["dollar_residual_xau"].fn(a, p)
    ol, os_ = ENTRIES["dollar_residual_xau"].fn(a, {**p, "mode": "follow"})
    assert fl.sum() > 0 and (fl == os_).all() and (fs == ol).all()


@pytest.mark.parametrize("name", NAMES)
def test_signal_count_sanity(uni, name):
    """Each block fires on synthetic H1 data at a plausible rate: some signals, not on most bars, never
    long and short on the same bar."""
    x = uni.ctx[("XAUUSD", "H1")]
    rng = np.random.default_rng(1)
    b = ENTRIES[name]
    tot = 0
    for _ in range(6):
        lo, sh = b.fn(x, sample_params(b.space, rng))
        assert not (lo & sh).any(), name
        assert (lo.sum() + sh.sum()) < 0.25 * x.n, name
        tot += lo.sum() + sh.sum()
    assert tot > 0, name


def test_acceptance_break_counts_consecutive_closes():
    df = resample(make_bars("2020-01-01", "2020-03-01", seed=4, freq="15min"), "H1")
    x = Ctx(df, "EURUSD", "H1")
    lo2, _ = ENTRIES["acceptance_break"].fn(x, {"level": "vwap", "confirm_closes": 2})
    lo3, _ = ENTRIES["acceptance_break"].fn(x, {"level": "vwap", "confirm_closes": 3})
    vw, _ = x._cache["vwap"]
    above = x.c > vw
    for t in np.where(lo2)[0]:
        assert above[t] and above[t - 1] and not above[t - 2]
    for t in np.where(lo3)[0]:
        assert above[t] and above[t - 1] and above[t - 2] and not above[t - 3]
    assert lo2.sum() > lo3.sum() > 0
