"""
Weinstein 4-Stage Scanner — Taiwan stocks & ETFs
Single-file Streamlit app. Deploy free on Streamlit Cloud.

Setup:
  1. Get a free token at https://finmindtrade.com  (Login > API Token)
  2. Put it in Streamlit secrets as FINMIND_TOKEN (or paste in the sidebar)
  3. requirements.txt  ->  streamlit\nFinMind\npandas
"""

import datetime as dt
import pandas as pd
import streamlit as st
from FinMind.data import DataLoader

# ------------------------------------------------------------------ config
st.set_page_config(page_title="階段掃描 · Stage Scanner", page_icon="◧", layout="wide")

MA_WEEKS = 30            # Weinstein's 30-week MA
VOL_MULT = 1.5           # breakout volume vs 10-week avg
RS_WEEKS = 13            # relative strength lookback
RANGE_WEEKS = 52         # window for "where in the range" (Stage 3 vs turning)
TOP_ZONE = 0.60          # above MA + flat/falling MA: >=this = real top, below = turning
MA_DAYS = 6              # daily entry filter: close > 6-day MA
DMI_LEN = 14             # daily DMI length
LOOKBACK_DAYS = 560      # ~80 weekly bars: 30w MA + 52w range + buffer
INST_DAYS = 90           # institutional net-buy window (the label says 3 months)

STAGE_META = {
    2: ("上升 Advancing", "買 / 觀察", "#1f9d55", "站上30週線、均線上彎、放量突破。錢在這裡賺。"),
    1: ("打底 Basing",    "觀望",     "#8a8f98", "橫盤、量縮，聰明錢吸貨。含剛翻多的轉折股，等轉2。"),
    3: ("頭部 Topping",   "減碼",     "#d9a441", "高檔震盪、量大不漲。出貨警訊。"),
    4: ("下跌 Declining", "避開 / 出", "#d9534f", "跌破30週線、均線下彎。錢在這裡賠。"),
}

# ------------------------------------------------------------------ data
def _api(token: str) -> DataLoader:
    api = DataLoader()
    api.login_by_token(api_token=token)
    return api


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_universe(token: str) -> pd.DataFrame:
    api = _api(token)
    df = api.taiwan_stock_info()
    return df[["stock_id", "stock_name", "type", "industry_category"]].drop_duplicates("stock_id")


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def price_mode(token: str) -> str:
    """Probe once: 'adj' if adjusted (dividend-restored) prices are available, else 'raw'.

    This decides which benchmark we may fairly compare against — mixing raw prices
    with a total-return index understates every stock during dividend season.
    """
    api = _api(token)
    probe = (dt.date.today() - dt.timedelta(days=40)).isoformat()
    try:
        df = api.taiwan_stock_daily_adj(stock_id="2330", start_date=probe)
        if not df.empty:
            return "adj"
    except Exception:
        pass
    return "raw"


@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def load_prices(token: str, sid: str, start: str, mode: str) -> pd.DataFrame:
    api = _api(token)
    df = pd.DataFrame()
    if mode == "adj":
        try:
            df = api.taiwan_stock_daily_adj(stock_id=sid, start_date=start)
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        df = api.taiwan_stock_daily(stock_id=sid, start_date=start)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def inst_netbuy(token: str, sid: str, start: str) -> float:
    """Net institutional buy (shares) over the period. >0 = accumulating."""
    api = _api(token)
    try:
        df = api.taiwan_stock_institutional_investors(stock_id=sid, start_date=start)
    except Exception:
        return 0.0
    if df.empty:
        return 0.0
    return float(df["buy"].sum() - df["sell"].sum())


@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def load_benchmark(token: str, start: str, mode: str):
    """TAIEX daily close + a label. Matched to `mode` so RS is apples-to-apples.

    adj stock prices -> total-return index;  raw stock prices -> price index.
    Returns (Series, label, matched).
    """
    api = _api(token)

    def _tr():
        df = api.taiwan_stock_total_return_index(index_id="TAIEX", start_date=start)
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()["price"]

    def _px():
        df = api.taiwan_stock_daily(stock_id="TAIEX", start_date=start)
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()["close"]

    first, second = (_tr, _px) if mode == "adj" else (_px, _tr)
    labels = ("報酬指數", "價格指數") if mode == "adj" else ("價格指數", "報酬指數")
    for fn, lab, matched in ((first, labels[0], True), (second, labels[1], False)):
        try:
            s = fn()
        except Exception:
            s = None
        if s is not None and not s.empty:
            return s, lab, matched
    return pd.Series(dtype=float), "無", False


# Big/mid-cap priority list (verified real tickers, ≈cap order).
BIGCAP = [
    "2330","2317","2454","2308","2382","2891","2412","2303","2881","3711",
    "2882","2886","1216","2884","2357","3034","2892","2885","2890","3231",
    "2345","2379","2603","3037","2880","5880","2887","2883","1303","2002",
    "1301","2327","3008","2395","3045","4938","2409","2301","2408","6505",
    "5871","2207","1101","2618","2610","2615","9910","2801","2823","2474",
    "6669","3661","3017","2376","2356","2360","3702","2385","6415","3005",
    "2377","4904","2337","6446","1476","2049","1590","9945","2542","2455",
    "2353","3443","2451","8046","2324","6488","3533","5269","2368","3653",
    "2347","2344","2371","3529","2492","2312","1102","2404","1802","9917",
    "2809","2812","6412","3406","2439","2201","1326","2105","2633","2354",
    "2338","3260","2504","4958","3019","8210","2481","6213","1229","2231",
    "1503","3044",
]


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_ranking() -> pd.DataFrame:
    """Static big-cap priority list; rank = position."""
    return pd.DataFrame({"stock_id": BIGCAP,
                         "market_value": list(range(len(BIGCAP), 0, -1))})


# ------------------------------------------------------------------ logic
def weekly(df: pd.DataFrame):
    """Daily -> weekly OHLCV (Fri close). Drops the still-running week.

    A partial week has partial volume, so leaving it in makes 爆量 almost
    always False and 突破前高 read off a half-formed bar.
    Returns (weekly_df, dropped_partial).
    """
    o = df["open"].resample("W-FRI").first()
    h = df["max"].resample("W-FRI").max()
    l = df["min"].resample("W-FRI").min()
    c = df["close"].resample("W-FRI").last()
    v = df["Trading_Volume"].resample("W-FRI").sum()
    w = pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v}).dropna()
    if w.empty:
        return w, False
    label = w.index[-1].date()          # that week's Friday
    last_data = df.index[-1].date()
    today = dt.date.today()
    # incomplete only if the week's Friday hasn't happened yet (or is today,
    # pre-close). A Friday holiday leaves label > last_data but label < today.
    partial = label > last_data and label >= today
    if partial:
        w = w.iloc[:-1]
    return w, partial


def dmi(df: pd.DataFrame, n: int = DMI_LEN):
    """Wilder DI+/DI- on daily bars."""
    h, l, c = df["max"], df["min"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0).fillna(0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0).fillna(0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    return pdi, mdi


def daily_signals(px: pd.DataFrame) -> dict:
    """Timing filters on the daily chart. Deliberately kept out of classify():
    the stage is a weekly judgement, these only say 'is this week entryable'."""
    out = dict(ma6=None, di_gap=None)
    if len(px) < max(MA_DAYS, DMI_LEN * 3) + 5:
        return out
    ma6 = px["close"].rolling(MA_DAYS).mean().iloc[-1]
    pdi, mdi = dmi(px)
    out["ma6"] = bool(px["close"].iloc[-1] > ma6)
    out["di_gap"] = round(float(pdi.iloc[-1] - mdi.iloc[-1]), 1)
    return out


def classify(w: pd.DataFrame, bench_w: pd.Series):
    """Return (stage, detail) or (None, None) if not enough data."""
    if len(w) < MA_WEEKS + 5:
        return None, None
    ma = w["c"].rolling(MA_WEEKS).mean()
    price = float(w["c"].iloc[-1])
    ma_now = float(ma.iloc[-1])
    slope = float(ma.diff(4).iloc[-1])       # 4-week slope of the MA
    above = price > ma_now
    slope_up = slope > 0
    slope_dn = slope < 0

    # breakout: last completed week's close above the prior 6 weeks' high, on volume
    prior_high = w["h"].iloc[-7:-1].max()
    vol_avg = w["v"].iloc[-11:-1].mean()
    breakout = bool(price > prior_high)
    vol_surge = bool(w["v"].iloc[-1] > VOL_MULT * vol_avg)

    # where in the 52-week range — separates a real top from a fresh turn
    n = min(len(w), RANGE_WEEKS)
    hi, lo = float(w["h"].iloc[-n:].max()), float(w["l"].iloc[-n:].min())
    pos = (price - lo) / (hi - lo) if hi > lo else 0.5

    # relative strength vs the benchmark
    rs = None
    if bench_w is not None and len(bench_w) > RS_WEEKS + 1:
        j = w.join(bench_w.rename("bm"), how="inner")
        if len(j) > RS_WEEKS + 1:
            k = -(RS_WEEKS + 1)
            rs = (j["c"].iloc[-1] / j["c"].iloc[k] - 1) - (j["bm"].iloc[-1] / j["bm"].iloc[k] - 1)

    note = ""
    if above and slope_up:
        stage = 2
    elif not above and slope_dn:
        stage = 4
    elif above and not slope_up:
        # price reclaimed the MA but the MA hasn't turned up yet.
        # High in the range = distribution. Low in the range = 1->2 turn.
        if pos >= TOP_ZONE:
            stage = 3
        else:
            stage, note = 1, "轉折"
    else:
        stage, note = 1, "回檔"      # below MA but MA still rising

    detail = dict(price=round(price, 2), ma=round(ma_now, 2),
                  above=above, slope_up=slope_up, breakout=breakout,
                  vol_surge=vol_surge, pos=round(pos * 100),
                  rs=None if rs is None else round(rs * 100, 1), note=note)
    return stage, detail


def market_stage(bench_daily: pd.Series):
    """Run the same stage logic on the index itself."""
    if bench_daily is None or bench_daily.empty:
        return None
    c = bench_daily.resample("W-FRI").last().dropna()
    w = pd.DataFrame({"o": c, "h": c, "l": c, "c": c, "v": 1.0})
    stage, _ = classify(w, None)
    return stage


# ------------------------------------------------------------------ ui
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1100px;}
  h1 {font-weight: 700; letter-spacing:-.5px;}
  .pill {display:inline-block;padding:2px 10px;border-radius:999px;
         font-size:.78rem;font-weight:600;color:#fff;}
  .muted {color:#8a8f98;font-size:.85rem;}
  div[data-testid="stMetricValue"] {font-size:1.4rem;}
</style>
""", unsafe_allow_html=True)

st.title("階段掃描器")
st.markdown('<span class="muted">Weinstein 四階段 · 台股 / ETF · 週線判斷</span>',
            unsafe_allow_html=True)

with st.sidebar:
    st.subheader("設定")
    token = st.text_input("FinMind Token", type="password",
                          value=st.secrets.get("FINMIND_TOKEN", ""),
                          help="免費申請：finmindtrade.com")
    asset = st.radio("標的", ["股票", "ETF"], horizontal=True)
    rank_by = st.selectbox(
        "掃描範圍", ["大型股優先", "代號順序"],
        help="大型股優先=先掃約120檔法人愛的大中型股。")
    want_stages = st.multiselect("顯示階段", [2, 1, 3, 4], default=[2, 4],
                                 format_func=lambda s: STAGE_META[s][0])
    st.divider()
    mkt_gate = st.checkbox("大盤 Stage 4 時不顯示 Stage 2", value=True,
                           help="Weinstein 原則：大盤走空時不做多。")
    need_ma6 = st.checkbox("只留 收盤 > 6日均", value=False)
    need_di = st.checkbox("只留 DI+ > DI-", value=False)
    inst_only = st.checkbox("只留法人近3月淨買", value=False,
                            help="過濾掉法人沒買的。台股法人主導。")
    max_scan = st.slider("掃描檔數上限", 30, 400, 120, 10,
                         help="檔數越多越慢。開法人過濾會再多打一輪 API。")
    go = st.button("開始掃描", type="primary", use_container_width=True)

if not token:
    st.info("在左側填入 FinMind Token 後即可開始。免費申請：finmindtrade.com")
    st.stop()

if not want_stages:
    st.warning("請至少選一個階段。")
    st.stop()

# ------------------------------------------------------------------ run
if go:
    start = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    inst_start = (dt.date.today() - dt.timedelta(days=INST_DAYS)).isoformat()
    try:
        mode = price_mode(token)
        uni = load_universe(token)
    except Exception as e:
        st.error(f"讀取股票清單失敗：{e}. 檢查 Token 是否正確。")
        st.stop()

    is_etf = uni["type"].str.contains("etf|ETF", case=False, na=False) | \
             uni["stock_id"].str.match(r"^00\d{2,4}")
    pool = uni[is_etf] if asset == "ETF" else uni[~is_etf]
    pool = pool[pool["stock_id"].str.match(r"^\d{4,6}$")]

    if rank_by != "代號順序":
        rk = load_ranking()
        pool = pool.merge(rk, on="stock_id", how="left") \
                   .sort_values("market_value", ascending=False, na_position="last")

    pool = pool.head(max_scan)

    bench_d, bench_lab, matched = load_benchmark(token, start, mode)
    bench_w = bench_d.resample("W-FRI").last().dropna() if not bench_d.empty else pd.Series(dtype=float)
    mstage = market_stage(bench_d)

    # market banner
    if mstage:
        mn, ma_, mc, _ = STAGE_META[mstage]
        st.markdown(f'大盤　<span class="pill" style="background:{mc}">{mn}</span>',
                    unsafe_allow_html=True)
    if not matched:
        st.warning(f"RS 基準用的是 {bench_lab}，與 {'還原' if mode == 'adj' else '未還原'}"
                   f"股價不同基準 — 除息旺季 RS 會偏低，請當參考值看。")

    gate_on = bool(mkt_gate and mstage == 4 and 2 in want_stages)

    rows, partial_hits, prog = [], 0, st.progress(0.0, text="掃描中…")
    for i, (_, r) in enumerate(pool.iterrows(), 1):
        prog.progress(i / len(pool), text=f"掃描中… {r.stock_id} {r.stock_name}")
        try:
            px = load_prices(token, r.stock_id, start, mode)
            if px.empty:
                continue
            w, partial = weekly(px)
            partial_hits += int(partial)
            stage, d = classify(w, bench_w)
            if stage is None or stage not in want_stages:
                continue
            if gate_on and stage == 2:
                continue
            sig = daily_signals(px)
            if need_ma6 and sig["ma6"] is not True:
                continue
            if need_di and not (sig["di_gap"] is not None and sig["di_gap"] > 0):
                continue
            if inst_only and inst_netbuy(token, r.stock_id, inst_start) <= 0:
                continue
            rows.append(dict(代號=r.stock_id, 名稱=r.stock_name, _stage=stage,
                             收盤=d["price"], MA30W=d["ma"], RS=d["rs"],
                             區間位置=d["pos"], 突破前高=d["breakout"],
                             爆量=d["vol_surge"],
                             MA6=sig["ma6"],
                             DI差=sig["di_gap"],
                             註記=d["note"]))
        except Exception:
            continue
    prog.empty()

    if gate_on:
        st.warning("大盤處於 Stage 4，已隱藏 Stage 2 名單。要看的話取消左側勾選。")

    if not rows:
        st.info("這批沒有符合的標的。放寬條件或提高掃描檔數再試。")
        st.stop()

    df = pd.DataFrame(rows)
    # Stage 2 must beat the index: drop laggards (RS<0). Only when RS is trustworthy.
    if matched:
        df = df[~((df["_stage"] == 2) & (df["RS"].fillna(-1) < 0))]
    if df.empty:
        st.info("這批沒有符合的標的。放寬條件或提高掃描檔數再試。")
        st.stop()
    df = df.sort_values(["_stage", "RS"], ascending=[True, False], na_position="last")

    counts = df["_stage"].value_counts()
    chips = "  ".join(
        f'<span class="pill" style="background:{STAGE_META[s][2]}">'
        f'{STAGE_META[s][0].split()[0]} {counts.get(s,0)}</span>'
        for s in [2, 3, 1, 4] if s in want_stages)
    st.markdown(chips, unsafe_allow_html=True)
    bar = "已排除未完成的本週" if partial_hits else "本週已收盤"
    st.caption(f"掃描 {len(pool)} 檔 · 命中 {len(df)} 檔 · {bar} · "
               f"股價{'還原' if mode == 'adj' else '未還原'} · RS基準 {bench_lab} · "
               f"{dt.date.today():%Y-%m-%d}")

    for s in [2, 3, 1, 4]:
        if s not in want_stages:
            continue
        sub = df[df["_stage"] == s].drop(columns="_stage")
        if sub.empty:
            continue
        name, action, color, note = STAGE_META[s]
        stage_tag = name.split()[0]
        st.markdown(f"### {name} · **{action}**")
        st.caption(note)
        st.dataframe(
            sub, hide_index=True, use_container_width=True,
            column_config={
                "RS": st.column_config.NumberColumn("RS%", help=f"vs 加權，{RS_WEEKS}週。正=贏大盤", format="%.1f"),
                "區間位置": st.column_config.NumberColumn("區間位置", help="在52週高低區間的位置%，0=底 100=頂", format="%d"),
                "突破前高": st.column_config.CheckboxColumn("突破前高"),
                "爆量": st.column_config.CheckboxColumn(f"爆量>{VOL_MULT}x"),
                "MA6": st.column_config.CheckboxColumn("C>MA6", help="日線：收盤 > 6日均"),
                "DI差": st.column_config.NumberColumn("DI+−DI-", help="日線 DMI(14)。正=多方掌控", format="%.1f"),
                "註記": st.column_config.TextColumn("註記", help="轉折=剛站上均線但均線未上彎；回檔=跌破均線但均線仍上彎"),
            })
        csv = sub.assign(階段=stage_tag).to_csv(index=False).encode("utf-8-sig")
        st.download_button(f"下載 {stage_tag} CSV", csv,
                           file_name=f"stage_{stage_tag}_{dt.date.today().isoformat()}.csv",
                           mime="text/csv", key=f"dl{s}")

    st.caption("僅供研究，非投資建議。TradingView / 券商 App 覆核後再下單。")
else:
    st.markdown("← 左側設定條件，按 **開始掃描**。")
    st.markdown(f"""
    <div class="muted">
    <b>階段判斷（週線）</b>　收盤 vs {MA_WEEKS}週均線 + 均線方向。<br>
    站上但均線未上彎：在52週區間高檔({int(TOP_ZONE*100)}%以上)算頭部，低檔算<b>轉折</b>。<br>
    <b>Stage 2 加看</b>　突破前6週高點、量 > 10週均量×{VOL_MULT}、RS vs 加權。<br>
    <b>進場擇時（日線，不影響階段）</b>　收盤 > {MA_DAYS}日均、DMI({DMI_LEN}) DI+ > DI-。<br>
    <b>大盤濾網</b>　大盤 Stage 4 時預設不出 Stage 2 名單。<br>
    未收盤的當週一律排除，所有數字都以最後一根完整週線為準。
    </div>
    """, unsafe_allow_html=True)
