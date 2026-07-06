"""
Weinstein 4-Stage Scanner — Taiwan stocks & ETFs
Single-file Streamlit app. Deploy free on Streamlit Cloud.

Setup:
  1. Get a free token at https://finmindtrade.com  (Login > API Token)
  2. Put it in Streamlit secrets as FINMIND_TOKEN (or paste in the sidebar)
  3. requirements.txt  ->  streamlit\nFinMind\npandas\nnumpy
"""

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from FinMind.data import DataLoader

# ------------------------------------------------------------------ config
st.set_page_config(page_title="階段掃描 · Stage Scanner", page_icon="◧", layout="wide")

MA_WEEKS = 30            # Weinstein's 30-week MA
VOL_MULT = 1.5          # breakout volume vs 10-week avg
LOOKBACK_DAYS = 320     # ~ enough weekly bars for a 30w MA + buffer

STAGE_META = {
    2: ("上升 Advancing", "買 / 觀察", "#1f9d55", "站上30週線、均線上彎、放量突破。錢在這裡賺。"),
    1: ("打底 Basing",    "觀望",     "#8a8f98", "橫盤、量縮，聰明錢吸貨。等轉2。"),
    3: ("頭部 Topping",   "減碼",     "#d9a441", "高檔震盪、量大不漲。出貨警訊。"),
    4: ("下跌 Declining", "避開 / 出", "#d9534f", "跌破30週線、均線下彎。錢在這裡賠。"),
}

# ------------------------------------------------------------------ data
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_universe(token: str) -> pd.DataFrame:
    api = DataLoader(); api.login_by_token(api_token=token)
    df = api.taiwan_stock_info()
    return df[["stock_id", "stock_name", "type", "industry_category"]].drop_duplicates("stock_id")

@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def load_prices(token: str, sid: str, start: str) -> pd.DataFrame:
    api = DataLoader(); api.login_by_token(api_token=token)
    df = api.taiwan_stock_daily(stock_id=sid, start_date=start)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_snapshot(token: str) -> pd.DataFrame:
    """Whole-market last trading day: used to rank the universe cheaply."""
    api = DataLoader(); api.login_by_token(api_token=token)
    for back in range(0, 7):
        d = (dt.date.today() - dt.timedelta(days=back)).isoformat()
        df = api.taiwan_stock_daily(start_date=d, end_date=d)
        if not df.empty:
            df["turnover"] = df["close"] * df["Trading_Volume"]
            return df[["stock_id", "close", "Trading_Volume", "turnover"]]
    return pd.DataFrame()

@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def load_benchmark(token: str, start: str) -> pd.Series:
    api = DataLoader(); api.login_by_token(api_token=token)
    df = api.taiwan_stock_total_return_index(index_id="TAIEX", start_date=start)
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    w = df.set_index("date").sort_index()["price"].resample("W-FRI").last()
    return w

# ------------------------------------------------------------------ logic
def weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Daily -> weekly OHLCV (Fri close)."""
    o = df["open"].resample("W-FRI").first()
    h = df["max"].resample("W-FRI").max()
    l = df["min"].resample("W-FRI").min()
    c = df["close"].resample("W-FRI").last()
    v = df["Trading_Volume"].resample("W-FRI").sum()
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v}).dropna()

def classify(w: pd.DataFrame, bench: pd.Series):
    """Return (stage, detail dict) or (None, None) if not enough data."""
    if len(w) < MA_WEEKS + 5:
        return None, None
    ma = w["c"].rolling(MA_WEEKS).mean()
    ma_slope = ma.diff(4)                      # 4-week slope of the MA
    price = w["c"].iloc[-1]
    ma_now = ma.iloc[-1]
    slope = ma_slope.iloc[-1]
    above = price > ma_now

    # breakout: this week's close above prior 6-week high, on volume
    prior_high = w["h"].iloc[-7:-1].max()
    vol_avg = w["v"].iloc[-11:-1].mean()
    breakout = price > prior_high
    vol_surge = w["v"].iloc[-1] > VOL_MULT * vol_avg

    # relative strength vs TAIEX over ~13 weeks
    rs = None
    if len(bench) > 14:
        j = w.join(bench.rename("bm"), how="inner")
        if len(j) > 14:
            stock_ret = j["c"].iloc[-1] / j["c"].iloc[-14] - 1
            bm_ret = j["bm"].iloc[-1] / j["bm"].iloc[-14] - 1
            rs = stock_ret - bm_ret

    slope_up = slope > 0
    slope_dn = slope < 0

    if above and slope_up:
        stage = 2
    elif not above and slope_dn:
        stage = 4
    elif above and not slope_up:
        stage = 3
    else:
        stage = 1

    detail = dict(
        price=round(price, 2), ma=round(ma_now, 2),
        above=above, slope_up=slope_up,
        breakout=breakout, vol_surge=vol_surge,
        rs=None if rs is None else round(rs * 100, 1),
    )
    return stage, detail

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
        "先排序，取前 N 檔", ["流動性 (成交值)", "價格 (高→低)", "代號順序"],
        help="流動性=只掃有量的名字，最實用。")
    want_stages = st.multiselect("顯示階段", [2, 1, 3, 4], default=[2, 4],
                                 format_func=lambda s: STAGE_META[s][0])
    max_scan = st.slider("掃描檔數上限", 30, 400, 120, 10,
                         help="檔數越多越慢。")
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
    try:
        uni = load_universe(token)
    except Exception as e:
        st.error(f"讀取股票清單失敗：{e}. 檢查 Token 是否正確。")
        st.stop()

    is_etf = uni["type"].str.contains("etf|ETF", case=False, na=False) | \
             uni["stock_id"].str.match(r"^00\d{2,4}")
    pool = uni[is_etf] if asset == "ETF" else uni[~is_etf]
    pool = pool[pool["stock_id"].str.match(r"^\d{4,6}$")]

    if rank_by != "代號順序":
        snap = load_snapshot(token)
        if not snap.empty:
            pool = pool.merge(snap, on="stock_id", how="left")
            col = "turnover" if "流動性" in rank_by else "close"
            pool = pool.sort_values(col, ascending=False, na_position="last")

    pool = pool.head(max_scan)

    bench = load_benchmark(token, start)

    rows, prog = [], st.progress(0.0, text="掃描中…")
    for i, (_, r) in enumerate(pool.iterrows(), 1):
        prog.progress(i / len(pool), text=f"掃描中… {r.stock_id} {r.stock_name}")
        try:
            px = load_prices(token, r.stock_id, start)
            if px.empty:
                continue
            stage, d = classify(weekly(px), bench)
            if stage is None or stage not in want_stages:
                continue
            rows.append(dict(代號=r.stock_id, 名稱=r.stock_name, _stage=stage,
                             收盤=d["price"], MA30W=d["ma"],
                             突破=d["breakout"], 放量=d["vol_surge"], RS=d["rs"]))
        except Exception:
            continue
    prog.empty()

    if not rows:
        st.info("這批沒有符合的標的。放寬階段或提高掃描檔數再試。")
        st.stop()

    df = pd.DataFrame(rows).sort_values(["_stage", "RS"],
                                        ascending=[True, False], na_position="last")

    # summary line
    counts = df["_stage"].value_counts()
    chips = "  ".join(
        f'<span class="pill" style="background:{STAGE_META[s][2]}">'
        f'{STAGE_META[s][0].split()[0]} {counts.get(s,0)}</span>'
        for s in [2, 3, 1, 4] if s in want_stages)
    st.markdown(chips, unsafe_allow_html=True)
    st.caption(f"掃描 {len(pool)} 檔 · 命中 {len(df)} 檔 · "
               f"{dt.date.today():%Y-%m-%d} 週線")

    # per-stage tables
    for s in [2, 3, 1, 4]:
        if s not in want_stages:
            continue
        sub = df[df["_stage"] == s].drop(columns="_stage")
        if sub.empty:
            continue
        name, action, color, note = STAGE_META[s]
        st.markdown(f"### {name} · **{action}**")
        st.caption(note)
        st.dataframe(
            sub, hide_index=True, use_container_width=True,
            column_config={
                "RS": st.column_config.NumberColumn("RS%", help="vs 加權，13週。正=贏大盤", format="%.1f"),
                "突破": st.column_config.CheckboxColumn("突破前高"),
                "放量": st.column_config.CheckboxColumn(f"量>{VOL_MULT}x"),
            })

    st.caption("僅供研究，非投資建議。TradingView / 券商 App 覆核後再下單。")
else:
    st.markdown("← 左側設定條件，按 **開始掃描**。")
    st.markdown("""
    <div class="muted">
    <b>判斷規則</b>　收盤 vs 30週均線 + 均線方向 → 定階段。<br>
    Stage 2 加看：突破前6週高點、量 > 10週均量×1.5、RS vs 加權。<br>
    只對 Stage 2 進場、Stage 4 出場，1/3 當警訊。
    </div>
    """, unsafe_allow_html=True)
