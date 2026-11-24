#!/usr/bin/env python3
"""Stan 階段週報 — 從 data/stages.parquet 產出每週報表。

放在 Stan_stages 而不是 minervini_picks：
    stages.parquet 就在同一個 repo，checkout 完就有，不用跨 repo 抓 raw、
    也不用留緩衝等對面的 commit 與快取更新。workflow 直接串在
    Weekly snapshot 後面（workflow_run），時序不用猜。

報表的重點是「這一週變了什麼」，不是「現在有哪些股票」：
    現況隨時可以打開 app 看，週報要講的是階段轉換 —— 誰剛從打底轉上升
    （最值得注意的進場候選），誰從上升掉出去（該減碼的警示）。

輸出：
    out/summary.txt            信件正文
    out/stan_transitions.csv   階段轉換
    out/stan_breakouts.csv     本週突破
    out/stan_stage2.csv        目前處於上升期的全部標的
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stages_core as S  # noqa: E402

SRC = "data/stages.parquet"
OUT = "out"
NAME = S.STAGE_NAME              # {1:打底, 2:上升, 3:頭部, 4:下跌}


def load_names(ids):
    """股票名稱來自 minervini_picks 的 universe。抓不到就只顯示代號，
    報表不該因為名稱查不到而整份掛掉。"""
    try:
        local = os.getenv("STAN_UNIVERSE_FILE", "")
        u = pd.read_parquet(local) if local and os.path.exists(local) else S.fetch_parquet(S.UNIVERSE_URL)
        u["stock_id"] = u["stock_id"].astype(str)
        return dict(zip(u["stock_id"], u["stock_name"]))
    except Exception as e:
        print(f"::warning::取不到股票名稱（{type(e).__name__}），只輸出代號")
        return {}


def fmt(df, nm, cols, n=None):
    if df.empty:
        return ["（無）"]
    d = df.head(n) if n else df
    head = "| 代號 | 名稱 | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|---|" + "---|" * len(cols)
    out = [head, sep]
    for _, r in d.iterrows():
        vals = []
        for key, _lab in cols:
            v = r[key]
            if isinstance(v, float):
                v = f"{v:.1f}"
            elif isinstance(v, bool):
                v = "✓" if v else ""
            vals.append(str(v))
        out.append(f"| {r['stock_id']} | {nm.get(r['stock_id'], '')} | "
                   + " | ".join(vals) + " |")
    return out


# ---------------------------------------------------------------------------
# 2026-11-24：HTML 版（手機可讀、有敘事）。純文字 summary.txt 保留當備援與稽核。
C = {"ink": "#1f2937", "mute": "#6b7280", "line": "#e5e7eb", "bg": "#f9fafb",
     "green": "#047857", "greenbg": "#ecfdf5", "red": "#b91c1c", "redbg": "#fef2f2",
     "amber": "#92400e", "amberbg": "#fffbeb", "blue": "#1d4ed8", "bluebg": "#eff6ff"}
MKT_TEXT = {1: "打底：大盤在 30 週均線附近整理，方向還沒出來。",
            2: "上升：大盤站上往上彎的 30 週均線，順勢做多的環境。",
            3: "頭部：大盤上漲動能轉弱、均線走平，該收斂、不追高。",
            4: "下跌：大盤在往下彎的 30 週均線之下，以保本為先。"}


def _e(x):
    import html
    return html.escape(str(x))


def _p(t, small=False):
    return (f"<p style='margin:4px 0;font-size:{'13px' if small else '15px'};"
            f"color:{C['mute'] if small else C['ink']};line-height:1.65'>{t}</p>")


def _box(title, inner, fg, bg):
    return (f"<div style='background:{bg};border-left:4px solid {fg};padding:10px 14px;margin:14px 0'>"
            f"<div style='color:{fg};font-weight:700;margin-bottom:4px'>{title}</div>{inner}</div>")


def _h(t):
    return (f"<h3 style='font-size:18px;margin:24px 0 6px;border-bottom:1px solid {C['line']};"
            f"padding-bottom:4px'>{t}</h3>")


def _tbl(df, nm, cols, n):
    if df.empty:
        return _p("（無）")
    th = "".join(f"<th style='text-align:left;padding:6px;border-bottom:2px solid {C['line']};"
                 f"color:{C['mute']};font-size:13px'>{h}</th>" for h in ["股票"] + [c[1] for c in cols])
    rows = []
    for _, r in df.head(n).iterrows():
        cells = [f"<b>{_e(r['stock_id'])}</b> {_e(nm.get(r['stock_id'], ''))}"]
        for key, _lab, kind in cols:
            v = r[key]
            if kind == "pct":
                v = f"{float(v):+.1f}%"
            elif kind == "pos":
                v = f"{float(v):.0f}%"
            elif kind == "num":
                v = f"{float(v):,.2f}"
            elif kind == "int":
                v = f"{int(v)}"
            elif kind == "bool":
                v = "✓" if bool(v) else ""
            cells.append(_e(v))
        rows.append("<tr>" + "".join(f"<td style='padding:6px;border-bottom:1px solid {C['line']};font-size:15px;white-space:nowrap'>{c}</td>"
                                     for c in cells) + "</tr>")
    more = _p(f"只列前 {n} 檔，共 {len(df)} 檔，完整清單在附件 CSV。", True) if len(df) > n else ""
    return f"<table style='border-collapse:collapse;width:100%'><tr>{th}</tr>{''.join(rows)}</table>" + more


def write_html(cur, prev, now, mk, dist, tr, bo, st2, nm, top):
    total = max(len(now), 1)
    share = {k: int(dist.get(k, 0)) for k in (1, 2, 3, 4)}
    up = tr[(tr.prev_stage == 1) & (tr.stage == 2)].sort_values("rs", ascending=False) if not tr.empty else tr
    down = tr[(tr.prev_stage == 2) & (tr.stage != 2)].sort_values("rs") if not tr.empty else tr
    head = (f"<div style='font-size:22px;font-weight:700'>Stan 階段週報｜{_e(cur)}</div>"
            f"<div style='color:{C['mute']};font-size:13px;margin-top:2px'>Weinstein 四階段（以 30 週均線判斷）｜"
            f"母體 {len(now)} 檔{('｜對比上週 ' + _e(prev)) if prev else ''}</div>")
    color = {1: (C["amber"], C["amberbg"]), 2: (C["green"], C["greenbg"]),
             3: (C["amber"], C["amberbg"]), 4: (C["red"], C["redbg"])}.get(mk, (C["mute"], C["bg"]))
    verdict = _box(f"大盤：第 {mk} 期（{NAME.get(mk, '?')}）", _p(MKT_TEXT.get(mk, "")) + _p(
        "　".join(f"{k} {NAME[k]} {share[k]} 檔（{share[k] / total * 100:.0f}%）" for k in (1, 2, 3, 4))), *color)
    def names(df, k=5):
        return "、".join(f"{r.stock_id} {nm.get(r.stock_id, '')}".strip() for r in df.head(k).itertuples())
    lead = _box("給忙的你", "".join(_p(x) for x in [
        f"打底轉上升（進場候選）：<b>{len(up)} 檔</b>" + (f"，RS 最強的是 {_e(names(up))}" if len(up) else "") + "。",
        f"掉出上升期（減碼警示）：<b>{len(down)} 檔</b>" + (f"，最弱的是 {_e(names(down))}" if len(down) else "") + "。",
        f"上升期且本週突破：<b>{len(bo)} 檔</b>；目前在上升期的一共 {len(st2)} 檔。",
    ]), C["blue"], C["bluebg"])
    concept = _box("先看懂這封信", "".join(_p(x) for x in [
        "Weinstein 用 30 週均線把一檔股票分成四個階段：<b>1 打底</b>（均線走平、價格在附近盤整）、"
        "<b>2 上升</b>（價格站上往上彎的均線）、<b>3 頭部</b>（漲不動、均線走平）、<b>4 下跌</b>（跌破往下彎的均線）。"
        "他的做法很單純：只在第 2 期持有，第 1 期轉第 2 期是最好的進場點，掉出第 2 期就離場觀望。",
        "<b>RS</b>（相對強度）：過去 13 週這檔的漲跌幅，減掉加權指數同期的漲跌幅。+20% 代表比大盤多漲 20%。",
        "<b>區間位置</b>：現價落在過去 52 週最低到最高之間的哪裡，0% 是最低點、100% 是最高點。",
        "<b>打底週數</b>：股價連續幾週都在 35% 的振幅內盤整，打得越久，突破後的行情通常越有份量。",
        "<b>突破</b>：本週收盤高過前 6 週的最高點。<b>爆量</b>：本週成交量超過前 10 週平均的 1.5 倍。",
    ]), C["mute"], C["bg"])
    s1 = _h(f"① 打底 → 上升：{len(up)} 檔（進場候選）") + _p(
        "剛從盤整區往上走、30 週均線也開始上彎。Weinstein 認為這是風險報酬最好的位置；有爆量突破的更可靠。", True) + _tbl(
        up, nm, [("close", "收盤", "num"), ("rs", "RS", "pct"), ("pos", "區間位置", "pos"),
                 ("base_wk", "打底週數", "int"), ("breakout", "突破", "bool")], top)
    s2 = _h(f"② 掉出上升期：{len(down)} 檔（減碼警示）") + _p(
        "原本在第 2 期（上升），這週掉到「掉到」欄寫的階段，代表 30 週均線的多頭結構失效。手上有的話，照 Weinstein 的做法是先離場觀望。", True) + _tbl(
        down.assign(去向=down["stage"].map(lambda x: "→" + NAME[int(x)]) if not down.empty else []),
        nm, [("去向", "掉到", "str"), ("close", "收盤", "num"), ("rs", "RS", "pct"),
                   ("rs_chg", "RS 排名變化", "int")], top)
    s3 = _h(f"③ 上升期且本週突破：{len(bo)} 檔") + _p(
        "已經在第 2 期、這週又創 6 週新高的股票，屬於趨勢延續。", True) + _tbl(
        bo, nm, [("close", "收盤", "num"), ("rs", "RS", "pct"), ("pos", "區間位置", "pos"),
                 ("vol_ratio", "量比", "num"), ("vol_surge", "爆量", "bool")], top)
    s4 = _h(f"④ 上升期強勢榜（RS 前 {top}，共 {len(st2)} 檔）") + _p(
        "目前在第 2 期、相對大盤最強的股票。「均線上彎週數」0 代表 30 週均線已經往上。", True) + _tbl(
        st2, nm, [("close", "收盤", "num"), ("rs", "RS", "pct"), ("pos", "區間位置", "pos"),
                  ("ma_turn_wk", "均線上彎週數", "int")], top)
    other = tr[~tr.index.isin(up.index) & ~tr.index.isin(down.index)] if not tr.empty else tr
    s5 = _h(f"⑤ 其餘階段轉換：{len(other)} 檔") + (_p("　".join(
        f"{_e(r.stock_id)} {_e(nm.get(r.stock_id, ''))}（{_e(r['轉換'])}）" for _, r in other.head(30).iterrows()), True)
        if len(other) else _p("（無）", True))
    warn = _box("容易誤讀的地方", "".join(_p(x) for x in [
        "階段是用週線判斷的，反應比日線慢；一週內的大漲大跌不會馬上改變階段。",
        "打底轉上升是「候選」，不是買進指令。進場前還要看量能、族群與大盤階段。",
        "大盤在第 4 期（下跌）時，個股的打底轉上升失敗率會明顯變高。",
    ]), C["amber"], C["amberbg"])
    secret = _box("我沒有告訴你的事", "".join(_p(x) for x in [
        "這份週報只描述狀態，沒有附回測勝率；四階段的判斷規則是照 Weinstein 原典寫成程式，沒有用歷史資料調參數。",
        "母體是我們整理過的股票清單，不是全市場。",
    ]), C["red"], C["redbg"])
    foot = _p(f"資料：Stan_stages／stages.parquet，快照週 {_e(cur)}。僅供研究參考，不構成投資建議。", True)
    page = (head + verdict + lead + concept + s1 + s2 + s3 + s4 + s5 + warn + secret + foot)
    html_doc = ("<!DOCTYPE html><html lang='zh-Hant'><head><meta charset='utf-8'></head>"
                f"<body style='font-family:-apple-system,\"PingFang TC\",\"Noto Sans TC\",sans-serif;color:{C['ink']};"
                f"max-width:760px;margin:0 auto;padding:12px'>{page}</body></html>")
    with open(f"{OUT}/summary.html", "w", encoding="utf-8") as f:
        f.write(html_doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20,
                    help="強勢榜最多列幾檔")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    s = pd.read_parquet(SRC)
    s["stock_id"] = s["stock_id"].astype(str)
    weeks = sorted(s["week"].unique())
    cur = weeks[-1]
    prev = weeks[-2] if len(weeks) >= 2 else None
    now = s[s["week"] == cur].copy()
    nm = load_names(now["stock_id"])

    L = [f"# Stan 階段週報 {cur}", ""]

    mk = int(now["market_stage"].iloc[0]) if len(now) else 0
    dist = now["stage"].value_counts().sort_index()
    L += [f"**大盤：第 {mk} 期（{NAME.get(mk, '?')}）**　母體 {len(now)} 檔", ""]
    L += ["| 階段 | 檔數 | 占比 |", "|---|---|---|"]
    for k in (1, 2, 3, 4):
        c = int(dist.get(k, 0))
        L.append(f"| {k} {NAME[k]} | {c} | {c/max(len(now),1)*100:.0f}% |")
    L.append("")

    # --- 階段轉換 ---
    if prev is None:
        L += ["## 階段轉換", "", "（只有一週快照，無法比較）", ""]
        tr = pd.DataFrame()
    else:
        old = s[s["week"] == prev][["stock_id", "stage"]].rename(
            columns={"stage": "prev_stage"})
        tr = now.merge(old, on="stock_id", how="inner")
        tr = tr[tr["stage"] != tr["prev_stage"]].copy()
        tr["轉換"] = tr.apply(
            lambda r: f"{int(r.prev_stage)}{NAME[int(r.prev_stage)]}"
                      f" → {int(r.stage)}{NAME[int(r.stage)]}", axis=1)
        L += [f"## 階段轉換（對比 {prev}）", ""]

        up = tr[(tr.prev_stage == 1) & (tr.stage == 2)].sort_values(
            "rs", ascending=False)
        L += [f"### 打底 → 上升：{len(up)} 檔（進場候選）", ""]
        L += fmt(up, nm, [("close", "收盤"), ("ma30w", "30週均"),
                          ("rs", "RS"), ("pos", "區間位置"),
                          ("base_wk", "打底週數"), ("breakout", "突破")],
                 n=args.top)
        if len(up) > args.top:
            L.append(f"\n（依 RS 排序只列前 {args.top} 檔，完整清單見 CSV）")
        L.append("")

        # 掉出上升期一律算警示，不只 2→3/2→4。2→1（回到打底）同樣代表
        # 30 週均線的多頭結構已經失效，Weinstein 的做法是離場觀望。
        down = tr[(tr.prev_stage == 2) & (tr.stage != 2)].sort_values("rs")
        L += [f"### 掉出上升期：{len(down)} 檔（減碼警示）", ""]
        L += fmt(down, nm, [("轉換", "轉換"), ("close", "收盤"),
                            ("ma30w", "30週均"), ("rs", "RS"),
                            ("rs_chg", "RS變化")], n=args.top)
        if len(down) > args.top:
            L.append(f"\n（依 RS 由弱到強只列前 {args.top} 檔，完整清單見 CSV）")
        L.append("")

        other = tr[~tr.index.isin(up.index) & ~tr.index.isin(down.index)]
        L += [f"### 其餘轉換：{len(other)} 檔（明細見 CSV）", ""]
        if not other.empty:
            L += ["　".join(f"{r.stock_id}{nm.get(r.stock_id,'')}"
                            f"({r['轉換']})" for _, r in other.head(30).iterrows())]
            if len(other) > 30:
                L.append(f"　…另外 {len(other)-30} 檔")
        else:
            L.append("（無）")
        L.append("")

    # --- 本週突破 ---
    bo = now[(now.stage == 2) & (now.breakout)].sort_values(
        "rs", ascending=False)
    L += [f"## 上升期且本週突破：{len(bo)} 檔", ""]
    L += fmt(bo, nm, [("close", "收盤"), ("rs", "RS"), ("pos", "區間位置"),
                      ("vol_ratio", "量比"), ("vol_surge", "爆量"),
                      ("base_wk", "打底週數")], n=args.top)
    if len(bo) > args.top:
        L.append(f"\n（只列 RS 前 {args.top} 檔，完整清單見 CSV）")
    L.append("")

    # --- 上升期強勢榜 ---
    st2 = now[now.stage == 2].sort_values("rs", ascending=False)
    L += [f"## 上升期強勢榜（RS 前 {args.top}，共 {len(st2)} 檔）", ""]
    L += fmt(st2, nm, [("close", "收盤"), ("rs", "RS"), ("rs_chg", "RS變化"),
                       ("pos", "區間位置"), ("ma_turn_wk", "均線上彎週數")],
             n=args.top)
    L += ["", "---", "",
          f"資料：Stan_stages/data/stages.parquet　快照週 {cur}"
          + (f"（上一筆 {prev}）" if prev else ""),
          "階段定義：1 打底、2 上升、3 頭部、4 下跌（Weinstein 30 週均線）"]

    body = "\n".join(L)
    write_html(cur, prev, now, mk, dist, tr if prev is not None else pd.DataFrame(), bo, st2, nm, args.top)
    with open(f"{OUT}/summary.txt", "w") as f:
        f.write(body.rstrip() + "\n")
    if not tr.empty:
        tr.drop(columns=["note"], errors="ignore").to_csv(
            f"{OUT}/stan_transitions.csv", index=False)
    bo.to_csv(f"{OUT}/stan_breakouts.csv", index=False)
    st2.to_csv(f"{OUT}/stan_stage2.csv", index=False)
    print(body)


if __name__ == "__main__":
    main()
