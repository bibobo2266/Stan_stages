"""
Weinstein 四階段判斷 — 共用核心

app.py 和 scripts/snapshot.py 都 import 這支，確保畫面上看到的階段
和存進歷史的階段用完全同一套邏輯。任何一邊自己算都會讓階段轉換失真。
"""
import io

import numpy as np
import pandas as pd
import requests

REPO = "bibobo2266/minervini_picks"
BRANCH = "main"
PRICES_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data/prices.parquet"
UNIVERSE_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data/universe.parquet"

MA_WEEKS = 30        # Weinstein 的 30 週均線
RS_WEEKS = 13        # 相對強度回看
RANGE_WEEKS = 52     # 52 週高低區間
TOP_ZONE = 0.60      # 站上均線但均線未上彎：區間 60% 以上=頭部，以下=轉折
VOL_MULT = 1.5       # 突破量 vs 10 週均量

STAGE_NAME = {1: "打底", 2: "上升", 3: "頭部", 4: "下跌"}


def fetch_parquet(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def load_frames(min_wan: float = 5000, min_days: int = 250):
    """回傳 (寬矩陣 dict, universe DataFrame)。已套流動性門檻。"""
    df = fetch_parquet(PRICES_URL)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["close"] > 0]

    P = {}
    for k, col in [("c", "close"), ("h", "max"), ("l", "min"),
                   ("v", "Trading_Volume"), ("mo", "Trading_money")]:
        # ffill：停牌週不該讓 rolling 整段變 NaN
        P[k] = df.pivot(index="date", columns="stock_id", values=col).sort_index().ffill()

    keep = (P["mo"].tail(60).mean() > min_wan * 1e4) & (P["c"].notna().sum() >= min_days)
    ids = keep[keep].index
    P = {k: v[ids] for k, v in P.items()}

    uni = fetch_parquet(UNIVERSE_URL).drop_duplicates("stock_id").set_index("stock_id")
    return P, uni


def market_index(P: dict) -> pd.Series:
    """等權市場指數。
    用母體每日報酬的算術平均（= 每日再平衡的等權組合），這是一個真的可投資的組合。
    不用中位數：中位數複利會產生假的負漂移，不是任何組合的報酬。
    不用成交值加權：那會過度加權當下最熱的股票，指數本身變成追高。"""
    r = P["c"].pct_change()
    return (1 + r.mean(axis=1).fillna(0)).cumprod()


def weekly(P: dict, idx: pd.Series):
    """日線寬矩陣 → 週線。丟掉還沒收盤的當週：
    半根週線的量是半根的量，會讓爆量幾乎永遠 False。"""
    wc = P["c"].resample("W-FRI").last()
    wh = P["h"].resample("W-FRI").max()
    wl = P["l"].resample("W-FRI").min()
    wv = P["v"].resample("W-FRI").sum()
    wi = idx.resample("W-FRI").last()

    last_data = P["c"].index[-1]
    partial = wc.index[-1] > last_data
    if partial and len(wc) > 1:
        wc, wh, wl, wv, wi = wc.iloc[:-1], wh.iloc[:-1], wl.iloc[:-1], wv.iloc[:-1], wi.iloc[:-1]
    return dict(c=wc, h=wh, l=wl, v=wv, i=wi), bool(partial)


def classify(W: dict) -> pd.DataFrame:
    """全母體一次判斷。回傳每檔一列。"""
    wc, wh, wl, wv, wi = W["c"], W["h"], W["l"], W["v"], W["i"]
    ma = wc.rolling(MA_WEEKS, min_periods=MA_WEEKS - 2).mean()

    price = wc.iloc[-1]
    ma_now = ma.iloc[-1]
    slope = ma.iloc[-1] - ma.iloc[-5]          # 4 週斜率
    above = price > ma_now
    up = slope > 0

    n = min(len(wc), RANGE_WEEKS)
    hi, lo = wh.tail(n).max(), wl.tail(n).min()
    pos = ((price - lo) / (hi - lo)).where(hi > lo, 0.5)

    k = RS_WEEKS + 1
    rs = (wc.iloc[-1] / wc.iloc[-k] - 1) - (wi.iloc[-1] / wi.iloc[-k] - 1)

    prior_high = wh.iloc[-7:-1].max()
    vol_avg = wv.iloc[-11:-1].mean()
    breakout = price > prior_high
    vol_surge = wv.iloc[-1] > VOL_MULT * vol_avg

    stage = pd.Series(1, index=wc.columns, dtype=int)
    note = pd.Series("", index=wc.columns, dtype=object)
    stage[above & up] = 2
    stage[(~above) & (slope < 0)] = 4
    # 站回均線但均線還沒上彎：在區間高檔是出貨，在低檔是第一階段末端的轉折
    top = above & (~up) & (pos >= TOP_ZONE)
    turn = above & (~up) & (pos < TOP_ZONE)
    stage[top] = 3
    stage[turn] = 1
    note[turn] = "轉折"
    note[(~above) & (slope >= 0)] = "回檔"

    out = pd.DataFrame({
        "階段": stage, "註記": note,
        "收盤": price.round(2), "MA30W": ma_now.round(2),
        "RS%": (rs * 100).round(1),
        "乖離%": ((price / ma_now - 1) * 100).round(1),
        "區間位置": (pos * 100).round(0),
        "突破前高": breakout, "爆量": vol_surge,
    })
    return out[out["MA30W"].notna()]


def market_stage(idx: pd.Series):
    """對市場指數本身跑同一套階段判斷。"""
    w = idx.resample("W-FRI").last().dropna()
    if len(w) < MA_WEEKS + 5:
        return None, None
    ma = w.rolling(MA_WEEKS).mean()
    price, ma_now = float(w.iloc[-1]), float(ma.iloc[-1])
    slope = float(ma.iloc[-1] - ma.iloc[-5])
    above, up = price > ma_now, slope > 0
    if above and up:
        s = 2
    elif not above and slope < 0:
        s = 4
    elif above:
        s = 3
    else:
        s = 1
    return s, round((price / ma_now - 1) * 100, 1)


def extras(P: dict) -> pd.DataFrame:
    """第一階段用得到的補充欄位：打底多久、量能是否乾涸、RS 排名是否改善。"""
    c, v, mo = P["c"], P["v"], P["mo"]
    wc = c.resample("W-FRI").last()

    # 打底週數：連續多少週的振幅維持在 35% 以內
    base_wk = pd.Series(0, index=c.columns, dtype=int)
    for k in range(8, min(len(wc), 105), 2):
        seg = wc.tail(k)
        ok = (seg.max() / seg.min() - 1) <= 0.35
        base_wk[ok] = k

    rs_now = (0.6 * (c.iloc[-1] / c.iloc[-127] - 1)
              + 0.4 * (c.iloc[-1] / c.iloc[-64] - 1)).rank(pct=True) * 100
    rs_old = (0.6 * (c.iloc[-64] / c.iloc[-190] - 1)
              + 0.4 * (c.iloc[-64] / c.iloc[-127] - 1)).rank(pct=True) * 100

    return pd.DataFrame({
        "打底週": base_wk,
        "量能比": (v.tail(20).mean() / v.tail(120).mean()).round(2),
        "RS改善": (rs_now - rs_old).round(0),
        "均額億": (mo.tail(60).mean() / 1e8).round(2),
    })


def ma_turn_weeks(wc: pd.DataFrame, max_k: int = 20) -> pd.Series:
    """扣抵值：假設價格持平，30 週均線的 4 週斜率還要幾週才會由負轉正。
    這是確定性的未來資訊——即將滾出視窗的舊 K 棒已經寫好了，不是預測。
    用 4 週斜率而不是週對週，才跟 classify() 判階段的定義一致
    （週對週幾乎所有站上均線的股票都會是 1，沒有鑑別度）。
    回傳 0 = 斜率已經是正的；max_k+1 = 這段期間內都不會翻。"""
    arr = wc.tail(MA_WEEKS + max_k + 8).to_numpy(dtype=float)
    cols = wc.columns
    price = arr[-1]

    # 目前的 MA 序列（最後 5 期就夠算 4 週斜率）
    ma_hist = []
    for j in range(5, 0, -1):
        ma_hist.append(np.nanmean(arr[-MA_WEEKS - j + 1: len(arr) - j + 1], axis=0))
    ma_hist.append(np.nanmean(arr[-MA_WEEKS:], axis=0))
    ma_hist = ma_hist[-5:]                      # ma[-5] … ma[-1]

    out = np.full(len(cols), max_k + 1, dtype=float)
    out[ma_hist[-1] > ma_hist[0]] = 0           # 4 週斜率已經為正

    win = arr[-MA_WEEKS:].copy()
    seq = list(ma_hist)
    for k in range(1, max_k + 1):
        win = np.vstack([win[1:], price])
        seq.append(np.nanmean(win, axis=0))
        seq = seq[-5:]
        turned = (seq[-1] > seq[0]) & (out > max_k)
        out[turned] = k
    return pd.Series(out, index=cols)
