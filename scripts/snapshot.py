"""
每週快照：把全母體的階段存進 data/stages.parquet

由 GitHub Actions 每週執行並 commit。這樣「階段轉換」完全自動，不用手動
下載／上傳 CSV——舊版存在 Streamlit Cloud 的本機磁碟，重開就沒了，
等於把整個 app 最有價值的功能變成手動作業。

用法：
    python scripts/snapshot.py
    python scripts/snapshot.py --dry-run
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stages_core as S  # noqa: E402

OUT = "data/stages.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--liq", type=float, default=5000, help="60日均額門檻（萬元）")
    args = ap.parse_args()

    P, uni = S.load_frames(min_wan=args.liq)
    idx = S.market_index(P)
    W, partial = S.weekly(P, idx)
    d = S.classify(W)
    ms, mbias = S.market_stage(idx)

    week = W["c"].index[-1].date().isoformat()
    print(f"母體 {P['c'].shape[1]} 檔 · 週線收盤日 {week} · "
          f"{'已排除未完成的當週' if partial else '當週已收盤'}")
    print(f"大盤階段 {ms} {S.STAGE_NAME.get(ms, '?')} 乖離 {mbias}%")
    print("階段分佈:", d["階段"].value_counts().sort_index().to_dict(),
          "| 轉折", int((d["註記"] == "轉折").sum()))

    snap = pd.DataFrame({
        "week": week,
        "stock_id": d.index,
        "stage": d["階段"].values,
        "note": d["註記"].values,
        "close": d["收盤"].values,
        "ma30w": d["MA30W"].values,
        "rs": d["RS%"].values,
        "pos": d["區間位置"].values,
        "market_stage": ms,
    })

    if args.dry_run:
        print(snap.head(10).to_string(index=False))
        return

    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUT):
        old = pd.read_parquet(OUT)
        if week in set(old["week"].astype(str)):
            print(f"{week} 已存在，跳過")
            return
        snap = pd.concat([old, snap], ignore_index=True)

    snap = snap.drop_duplicates(subset=["week", "stock_id"], keep="last")
    snap = snap.sort_values(["week", "stock_id"]).reset_index(drop=True)
    snap.to_parquet(OUT, index=False, compression="zstd")
    print(f"已寫入 {snap['week'].nunique()} 週 / {len(snap):,} 列 / "
          f"{os.path.getsize(OUT) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
