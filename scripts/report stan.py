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
        u = S.fetch_parquet(S.UNIVERSE_URL)
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
