"""
Weinstein 四階段掃描器 — 台股

v2 — 資料改讀 minervini_picks 的 data/prices.parquet（全市場，每日 17:00 自動更新）
     母體從硬編大型股清單改成流動性門檻；階段快照改由 GitHub Actions 每週
     commit 進 repo，「階段轉換」不再需要手動下載／上傳。

這支 app 的位置：抓第一階段 → 第二階段的轉折，以及大盤環境。
Minervini 的趨勢模板有四條在確認「已經是第二階段」，所以它結構上撈不到
這一段。兩者互補，不是重複。
"""
import datetime as dt
import io

import numpy as np
import pandas as pd
import requests
import streamlit as st

import stages_core as S

st.set_page_config(page_title="階段掃描 · Stage Scanner", page_icon="◧", layout="wide")

SELF_REPO = "bibobo2266/Stan_stages"
STAGES_URL = f"https://raw.githubusercontent.com/{SELF_REPO}/main/data/stages.parquet"

STAGE_META = {
    2: ("上升 Advancing", "買 / 觀察", "#1f9d55", "站上30週線、均線上彎。錢在這裡賺。"),
    1: ("打底 Basing", "觀望 / 轉折可佈局", "#8a8f98", "橫盤、量縮，聰明錢吸貨。含剛翻多的轉折股。"),
    3: ("頭部 Topping", "減碼", "#d9a441", "高檔震盪、量大不漲。出貨警訊。"),
    4: ("下跌 Declining", "避開 / 出", "#d9534f", "跌破30週線、均線下彎。錢在這裡賠。"),
}


@st.cache_data(ttl=60 * 60 * 4, show_spinner="讀取行情資料…")
def build(liq: float):
    P, uni = S.load_frames(min_wan=liq)
    idx = S.market_index(P)
    W, partial = S.weekly(P, idx)
    d = S.classify(W)
    d = d.join(S.extras(P)).join(S.ma_turn_weeks(W["c"]).rename("翻揚週"))
    d = d.join(uni[["stock_name", "industry_category"]])
    d = d.rename(columns={"stock_name": "名稱", "industry_category": "產業"})
    ms, mbias = S.market_stage(idx)
    return d, ms, mbias, W["c"].index[-1].date(), partial, P["c"].shape[1]


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def load_history():
    try:
        r = requests.get(STAGES_URL, timeout=60)
        r.raise_for_status()
        h = pd.read_parquet(io.BytesIO(r.content))
        return h.sort_values(["week", "stock_id"])
    except Exception:
        return pd.DataFrame()


st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1150px;}
h1 {font-weight:700; letter-spacing:-.5px;}
.pill {display:inline-block;padding:2px 10px;border-radius:999px;
       font-size:.78rem;font-weight:600;color:#fff;}
.muted {color:#8a8f98;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

st.title("階段掃描器")
st.markdown('<span class="muted">Weinstein 四階段 · 全市場週線判斷 · '
            '抓第一階段→第二階段的轉折</span>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("設定")
    liq = st.number_input("60日均額門檻（萬元）", 500, 100000, 5000, 500,
                          help="流動性過濾。第一階段的股票天生冷門，"
                               "用成交值排行取樣會全部漏掉。")
    want = st.multiselect("顯示階段", [2, 1, 3, 4], default=[1, 2],
                          format_func=lambda s: STAGE_META[s][0])
    st.divider()
    mkt_gate = st.checkbox("大盤 Stage 4 時隱藏 Stage 2", value=True,
                           help="Weinstein 原則：大盤走空時不做多。")
    turn_only = st.checkbox("打底區只看「轉折」", value=True,
                            help="轉折＝站回30週線但均線未上彎、且在52週區間低檔。"
                                 "這是第一階段末端，最早的進場點。")
    max_turn = st.slider("均線翻揚倒數上限（週）", 0, 21, 8,
                         help="扣抵值算出來的：價格持平下，30週均線還要幾週"
                              "才會由平轉揚。0 = 已經上彎。21 = 不限。")
    min_rs = st.slider("最低 RS%（vs 等權市場）", -30, 30, 0)
    go = st.button("開始掃描", type="primary", use_container_width=True)

if not go:
    st.info("← 左側設定後按「開始掃描」。資料每個交易日 17:00 自動更新，掃描約 5 秒。")
    st.markdown("""
<div class="muted">
<b>階段判斷（週線）</b> 收盤 vs 30週均線 + 均線 4 週斜率。<br>
站上但均線未上彎：52週區間 60% 以上算頭部，以下算<b>轉折</b>（第一階段末端）。<br>
<b>RS</b> vs 等權市場指數（母體每日報酬算術平均）。用等權而不是市值加權，
是因為市值加權會被權值股主導，RS 就變成「有沒有贏台積電」。<br>
<b>翻揚週</b> 扣抵值算的：假設價格持平，均線還要幾週由平轉揚。這是確定性的
未來資訊——即將滾出視窗的舊 K 棒已經寫好了。<br>
<b>階段轉換</b> GitHub Actions 每週自動存快照並 commit，不需手動上傳。<br>
<b>注意</b> parquet 是未還原股價，除息旺季個股會有除息缺口；但 RS 的基準也是
同一批未還原價格算出來的，偏誤大致互相抵消。
</div>
""", unsafe_allow_html=True)
    st.stop()

d, ms, mbias, wk, partial, pool_n = build(liq)

# ---- 大盤橫幅 ----
if ms:
    mn, ma_, mc, _ = STAGE_META[ms]
    st.markdown(f'大盤 <span class="pill" style="background:{mc}">{mn}</span> '
                f'<span class="muted">乖離 {mbias:+.1f}% · 建議 {ma_}</span>',
                unsafe_allow_html=True)
gate_on = bool(mkt_gate and ms == 4 and 2 in want)
if gate_on:
    st.warning("大盤處於 Stage 4，已隱藏 Stage 2 名單。要看的話取消左側勾選。")

st.caption(f"母體 {pool_n} 檔 · 週線收盤 {wk} · "
           f"{'已排除未完成的當週' if partial else '當週已收盤'}")

# ---- 階段轉換（自動，不需上傳）----
hist = load_history()
if not hist.empty and hist["week"].nunique() >= 2:
    weeks = sorted(hist["week"].astype(str).unique())
    cur, prev = weeks[-1], weeks[-2]
    a = hist[hist["week"].astype(str) == prev].set_index("stock_id")["stage"]
    b = hist[hist["week"].astype(str) == cur].set_index("stock_id")["stage"]
    j = pd.concat([a.rename("舊"), b.rename("新")], axis=1).dropna()
    chg = j[j["舊"] != j["新"]].copy()
    chg["轉換"] = (chg["舊"].astype(int).map(S.STAGE_NAME) + "→"
                 + chg["新"].astype(int).map(S.STAGE_NAME))
    chg = chg.join(d[["名稱", "產業", "收盤", "RS%"]])

    st.markdown(f"### 階段轉換 <span class='muted'>{prev} → {cur}</span>",
                unsafe_allow_html=True)
    key = chg[chg["轉換"] == "打底→上升"]
    c1, c2 = st.columns([1, 3])
    c1.metric("🚀 打底→上升", len(key), help="最值得注意的買進訊號")
    c2.metric("⛔ 上升→頭部／下跌",
              int(chg["轉換"].isin(["上升→頭部", "上升→下跌"]).sum()),
              help="出場訊號")
    st.dataframe(chg.reset_index()[["stock_id", "名稱", "轉換", "收盤", "RS%", "產業"]],
                 hide_index=True, use_container_width=True, height=260)
else:
    st.info("階段轉換需要至少兩週的歷史快照。Actions 每週會自動累積，"
            "第二週開始就會出現。")

# ---- 名單 ----
st.divider()
for s in [1, 2, 3, 4]:
    if s not in want or (gate_on and s == 2):
        continue
    sub = d[d["階段"] == s].copy()
    if s == 1 and turn_only:
        sub = sub[sub["註記"] == "轉折"]
    if s in (1, 2):
        sub = sub[(sub["RS%"] >= min_rs) & (sub["翻揚週"] <= max_turn)]
    if sub.empty:
        continue

    name, action, color, note = STAGE_META[s]
    st.markdown(f"### {name} · **{action}** <span class='muted'>{len(sub)} 檔</span>",
                unsafe_allow_html=True)
    st.caption(note)

    sub = sub.sort_values(["翻揚週", "RS%"], ascending=[True, False]) if s == 1 \
        else sub.sort_values(["突破前高", "爆量", "RS%"], ascending=False)
    cols = ["名稱", "產業", "收盤", "MA30W", "RS%", "乖離%", "區間位置",
            "打底週", "量能比", "RS改善", "翻揚週", "突破前高", "爆量", "均額億"]
    st.dataframe(sub[cols].reset_index(), hide_index=True, use_container_width=True,
                 height=min(460, 60 + 35 * len(sub)), column_config={
        "RS%": st.column_config.NumberColumn("RS%", format="%.1f",
            help="vs 等權市場指數，13週。正=贏市場"),
        "乖離%": st.column_config.NumberColumn("乖離%", format="%.1f"),
        "區間位置": st.column_config.NumberColumn("區間位置", format="%d",
            help="52週高低區間位置，0=底 100=頂"),
        "打底週": st.column_config.NumberColumn("打底週", format="%d",
            help="振幅維持在 35% 以內已持續幾週。越久，突破時爆發力越強"),
        "量能比": st.column_config.NumberColumn("量能比", format="%.2f",
            help="近20日均量 / 近120日均量。<1 = 量能乾涸，第一階段末端的特徵"),
        "RS改善": st.column_config.NumberColumn("RS改善", format="%.0f",
            help="RS 百分位近三個月的變化。重點不是 RS 高，是排名正在爬"),
        "翻揚週": st.column_config.NumberColumn("翻揚週", format="%.0f",
            help="扣抵值：價格持平下，30週均線還要幾週由平轉揚。0=已上彎"),
        "突破前高": st.column_config.CheckboxColumn("突破前高"),
        "爆量": st.column_config.CheckboxColumn(f"爆量>{S.VOL_MULT}x"),
        "均額億": st.column_config.NumberColumn("均額億", format="%.2f"),
    })
    st.text_area("複製代號", " ".join(sub.index.tolist()), height=68, key=f"cp{s}")
    st.download_button(f"下載 {name.split()[0]} CSV",
                       sub[cols].reset_index().to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"stage_{s}_{dt.date.today().isoformat()}.csv",
                       mime="text/csv", key=f"dl{s}")
    st.divider()

# ---- 轉換訊號的事後驗證 ----
if not hist.empty and hist["week"].nunique() >= 6:
    with st.expander("「打底→上升」訊號的事後表現（驗證這個訊號有沒有用）"):
        weeks = sorted(hist["week"].astype(str).unique())
        rows = []
        for i in range(1, len(weeks) - 1):
            a = hist[hist["week"].astype(str) == weeks[i - 1]].set_index("stock_id")["stage"]
            b = hist[hist["week"].astype(str) == weeks[i]].set_index("stock_id")
            hit = b[(b["stage"] == 2) & (b.index.map(a).fillna(0) == 1)]
            for h in (4, 8, 12):
                if i + h >= len(weeks):
                    continue
                fut = hist[hist["week"].astype(str) == weeks[i + h]].set_index("stock_id")["close"]
                ov = hit.index.intersection(fut.index)
                if len(ov) == 0:
                    continue
                r = (fut[ov] / hit.loc[ov, "close"] - 1) * 100
                rows.append({"訊號週": weeks[i], "檔數": len(ov), "之後": f"{h}週",
                             "中位報酬%": round(float(r.median()), 1),
                             "勝率%": round(float((r > 0).mean() * 100))})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("樣本累積越多越有參考價值。若中位報酬長期為負，"
                       "代表這個訊號在當前市場結構下無效，該調整條件。")
        else:
            st.caption("歷史還不夠長，再累積幾週。")

st.caption("僅供研究，非投資建議。")
