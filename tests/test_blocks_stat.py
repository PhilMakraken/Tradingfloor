"""Statistical path-feature blocks (blocks_stat): finite output, causality, learned-direction isolation."""
import numpy as np
import pytest

from propfloor.data.clean import resample
from propfloor.data.synthetic import make_bars
from propfloor.strategies import blocks_stat as bs
from propfloor.strategies.blocks import ENTRIES, FILTERS, Ctx


@pytest.fixture(scope="module")
def df():
    return resample(make_bars("2020-01-01", "2020-05-01", seed=11), "H1")


@pytest.mark.parametrize("name", bs.FEATURE_NAMES)
def test_feature_finite_and_causal(df, name):
    w = 40
    full = Ctx(df, "XAUUSD", "H1")
    f = bs.feature(full, name, w)
    z = bs.feature_z(full, name, w)
    assert len(f) == full.n and np.isfinite(f[5 * w:]).mean() > 0.95, name
    assert np.isfinite(z[5 * w + bs.ZWIN_MIN:]).mean() > 0.95, name
    assert not np.isinf(f).any() and not np.isinf(z).any()
    # perturb the future: past values must not change
    cut = full.n - 60
    d2 = df.copy()
    rng = np.random.default_rng(1)
    k = d2.columns.get_loc("close")
    d2.iloc[cut:, k] *= 1 + rng.normal(0, 0.01, full.n - cut)
    d2.iloc[cut:, d2.columns.get_loc("high")] = d2.iloc[cut:][["high", "close"]].max(axis=1)
    d2.iloc[cut:, d2.columns.get_loc("low")] = d2.iloc[cut:][["low", "close"]].min(axis=1)
    alt = Ctx(d2, "XAUUSD", "H1")
    f2, z2 = bs.feature(alt, name, w), bs.feature_z(alt, name, w)
    np.testing.assert_array_equal(f[:cut], f2[:cut])
    np.testing.assert_array_equal(z[:cut], z2[:cut])
    assert not np.allclose(np.nan_to_num(f[cut:]), np.nan_to_num(f2[cut:]))  # the change is seen afterwards


def test_feature_sanity_ranges(df):
    x = Ctx(df, "XAUUSD", "H1")
    pe = bs.feature(x, "perm_entropy", 50)
    se = bs.feature(x, "sign_entropy", 50)
    sv = bs.feature(x, "semivar", 50)
    pf = bs.feature(x, "path_eff", 50)
    dd = bs.feature(x, "dd_hazard", 50)
    for a, lo, hi in ((pe, 0, 1), (se, 0, 1), (sv, -1, 1), (pf, 0, 1), (dd, -np.inf, 0)):
        a = a[np.isfinite(a)]
        assert (a >= lo - 1e-9).all() and (a <= hi + 1e-9).all()
    # a straight line has Katz dimension 1 and path efficiency 1
    lin = df.copy()
    p = np.exp(np.linspace(0, 0.2, len(df)))
    lin["open"] = lin["close"] = lin["high"] = lin["low"] = p
    xl = Ctx(lin, "XAUUSD", "H1")
    assert np.nanmax(np.abs(bs.feature(xl, "fractal_dim", 30) - 1)) < 1e-6
    assert np.nanmin(bs.feature(xl, "path_eff", 30)[40:]) > 0.99


def test_learned_mode_uses_only_training_bars(df):
    p = {"feature": "acf_sum", "window": 40, "z_thr": 1.2, "mode": "learned"}
    a = Ctx(df, "XAUUSD", "H1")
    a.train_end = a.time[int(a.n * 0.6)]
    d2 = df.copy()
    oos = d2.index >= a.train_end
    for col in ("open", "high", "low", "close"):  # scramble the out-of-sample path
        d2.loc[oos, col] = d2.loc[oos, col].to_numpy()[::-1]
    d2.loc[oos, "high"], d2.loc[oos, "low"] = d2.loc[oos, ["open", "high", "close"]].max(axis=1), \
        d2.loc[oos, ["open", "low", "close"]].min(axis=1)
    b = Ctx(d2, "XAUUSD", "H1")
    b.train_end = a.train_end
    assert bs.learned_sign(a, "acf_sum", 40) == bs.learned_sign(b, "acf_sum", 40) != 0
    lo, sh = ENTRIES["stat_feature"].fn(a, p)
    assert lo.dtype == bool and lo.sum() + sh.sum() > 0


def test_modes_and_filter(df):
    x = Ctx(df, "XAUUSD", "H1")
    base = {"feature": "hurst", "window": 40, "z_thr": 1.5}
    ml, ms = ENTRIES["stat_feature"].fn(x, {**base, "mode": "momentum"})
    fl, fs = ENTRIES["stat_feature"].fn(x, {**base, "mode": "fade"})
    assert ml.sum() > 0 and (ml == fs).all() and (ms == fl).all()
    assert not (ml & ms).any()
    hi, _ = FILTERS["stat_regime"].fn(x, {"feature": "hurst", "window": 40, "side": "high"})
    lo, _ = FILTERS["stat_regime"].fn(x, {"feature": "hurst", "window": 40, "side": "low"})
    assert not (hi & lo).any() and hi.sum() > 0 and lo.sum() > 0
