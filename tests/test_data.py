from datetime import date

import numpy as np
import pandas as pd
import pytest

from propfloor.data import dukascopy, histdata
from propfloor.data.clean import clean_bars, resample
from propfloor.data.loader import detect_and_load, symbol_from_name
from propfloor.data.store import DataStore, ingest_raw
from propfloor.data.synthetic import make_bars


def test_symbol_from_name():
    assert symbol_from_name("DAT_ASCII_EURUSD_M1_2018.csv") == "EURUSD"
    assert symbol_from_name("gbp-jpy_h1.csv") == "GBPJPY"
    assert symbol_from_name("data.csv") is None


@pytest.mark.parametrize("name,content,fmt,first_utc", [
    ("DAT_ASCII_EURUSD_M1_2018.csv", "20180102 170000;1.2001;1.2003;1.2000;1.2002;0\n20180102 170100;1.2002;1.2004;1.2001;1.2003;0\n",
     "histdata", "2018-01-02 22:00"),
    ("EURUSD1.csv", "2018.01.02,17:00,1.2001,1.2003,1.2000,1.2002,10\n2018.01.02,17:01,1.2002,1.2004,1.2001,1.2003,12\n",
     "mt4", "2018-01-02 17:00"),
    ("EURUSD_M1_mt5.csv", "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
     "2018.01.02\t17:00:00\t1.2001\t1.2003\t1.2000\t1.2002\t10\t0\t5\n", "mt5", "2018-01-02 17:00"),
    ("EURUSD_Candlestick_1_M_BID.csv", "Gmt time,Open,High,Low,Close,Volume\n02.01.2018 17:00:00.000,1.2001,1.2003,1.2,1.2002,100\n",
     "dukascopy", "2018-01-02 17:00"),
    ("eurusd_generic.csv", "timestamp,open,high,low,close\n2018-01-02T17:00:00Z,1.2001,1.2003,1.2,1.2002\n",
     "generic", "2018-01-02 17:00"),
])
def test_detect_formats(tmp_path, name, content, fmt, first_utc):
    p = tmp_path / name
    p.write_text(content)
    d = detect_and_load(p)
    assert d.fmt == fmt and d.symbol == "EURUSD"
    assert d.df.index[0] == pd.Timestamp(first_utc, tz="UTC")
    assert d.df["close"].iloc[0] == pytest.approx(1.2002)


def test_tick_csv_to_bars(tmp_path):
    p = tmp_path / "EURUSD_ticks.csv"
    p.write_text("time,bid,ask\n2020-01-02 10:00:01,1.1,1.1002\n2020-01-02 10:00:30,1.1004,1.1006\n2020-01-02 10:01:05,1.1,1.1002\n")
    d = detect_and_load(p)
    assert len(d.df) == 2 and d.df["high"].iloc[0] == pytest.approx(1.1005)


def test_clean_dedupe_invalid_weekend():
    df = make_bars("2020-01-06", "2020-01-08", seed=1)
    dirty = pd.concat([df, df.iloc[:10]])
    dirty.iloc[100, 0] = -1
    sat = pd.DataFrame({"open": [1.1], "high": [1.1], "low": [1.1], "close": [1.1], "volume": [1]},
                       index=pd.DatetimeIndex([pd.Timestamp("2020-01-11 12:00", tz="UTC")]))
    c, rep = clean_bars(pd.concat([dirty, sat]))
    assert rep.duplicates == 10 and rep.invalid == 1 and rep.weekend_bars == 1
    assert c.index.is_monotonic_increasing and not c.index.duplicated().any()
    assert (c["high"] >= c[["open", "close"]].max(axis=1)).all()


def test_gap_detection():
    df = make_bars("2020-01-06", "2020-01-08", seed=1)
    holed = df[(df.index < "2020-01-06 10:00") | (df.index > "2020-01-06 14:00")]
    _, rep = clean_bars(holed)
    assert len(rep.gaps) == 1 and rep.gaps[0][2] > 200


def test_resample_h1_and_daily():
    df = make_bars("2020-01-06", "2020-01-10", seed=2)
    h1 = resample(df, "H1")
    assert h1.index[1] - h1.index[0] == pd.Timedelta("1h")
    g = df[(df.index >= h1.index[3]) & (df.index < h1.index[3] + pd.Timedelta("1h"))]
    assert h1["high"].iloc[3] == g["high"].max() and h1["close"].iloc[3] == g["close"].iloc[-1]
    d1 = resample(df, "D1")
    assert all(t.tz_convert("America/New_York").hour == 17 for t in d1.index)


def test_store_roundtrip_and_synthetic_guard(tmp_path):
    st = DataStore(tmp_path)
    df = make_bars("2020-01-06", "2020-01-08", seed=3)
    st.write("EURUSD", df, "test", synthetic=True)
    assert st.is_synthetic("EURUSD") and st.symbols() == ["EURUSD"]
    assert len(st.load("EURUSD")) == len(df)
    with pytest.raises(ValueError):
        st.write("EURUSD", df, "real", synthetic=False)


def test_ingest_raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    df = make_bars("2020-01-06", "2020-01-07", seed=4)
    lines = [f"{t.tz_convert('Etc/GMT+5').strftime('%Y%m%d %H%M%S')};{r.open};{r.high};{r.low};{r.close};0" for t, r in df.iterrows()]
    (raw / "DAT_ASCII_GBPUSD_M1_2020.csv").write_text("\n".join(lines))
    st = DataStore(tmp_path / "pq")
    ingest_raw(raw, st)
    out = st.load("GBPUSD")
    assert len(out) == len(df) and np.allclose(out["close"].to_numpy(), df["close"].to_numpy())


def test_dukascopy_codec_roundtrip():
    df = make_bars("2020-01-06", "2020-01-06 02:00", seed=5)
    df["volume"] = 1.5
    raw = dukascopy.encode_bi5(df, date(2020, 1, 6), "EURUSD")
    out = dukascopy.decode_bi5(raw, date(2020, 1, 6), "EURUSD")
    assert len(out) == len(df) and np.allclose(out["close"], df["close"], atol=1e-5)
    assert out.index[0] == df.index[0]


def test_histdata_parsers():
    assert histdata.parse_token('<input type="hidden" id="tk" value="abc123" />') == "abc123"
    out = histdata.parse_ascii("20180102 170000;1.2;1.3;1.1;1.25;0\n")
    assert out.index[0] == pd.Timestamp("2018-01-02 22:00", tz="UTC")


def test_github_mirror_parse_scale_and_timezone():
    from propfloor.data import github_mirror

    text = "Date,open,high,low,close,tick_volume\n2020-01-06 00:00:00,112000.0,112010.0,111990.0,112005.0,10\n" \
           "2020-07-06 00:00:00,112000.0,112010.0,111990.0,112005.0,10\n"
    df = github_mirror.parse(text, "EURUSD")
    assert df["close"].iloc[0] == pytest.approx(1.12005)
    # server midnight = 17:00 New York (EST in winter, EDT in summer)
    assert df.index[0] == pd.Timestamp("2020-01-05 22:00", tz="UTC")
    assert df.index[1] == pd.Timestamp("2020-07-05 21:00", tz="UTC")
    jpy = github_mirror.parse(text.replace("112", "108"), "USDJPY")
    assert 50 < jpy["close"].iloc[0] < 250
