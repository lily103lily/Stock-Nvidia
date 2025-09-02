#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉回買（Dip Buy）— 適合長期投資的進場時機判斷（即時價版）

邏輯總覽
1) 長期多頭篩選（必須先通過）：
   - SMA50 > SMA200，且 Close > SMA200
2) 三種「拉回後回升」訊號，命中達到 --min-hits 即判定可進場：
   A) Reclaim20：昨日在 SMA20 下方，今天收上 SMA20 且收紅
   B) Bounce50 ：今日低點觸/破 SMA50 後收回到 SMA50 之上且收紅（或距離 SMA50 ≤ +1%）
   C) RSI 回升 ：價格在 SMA200 上方，RSI(14) 昨天 ≤ 42，今天上升（> 昨天）
3) 若不滿足進場，顯示「建議觀察價位」：
   - 價格 < SMA50 → 觀察 SMA50（回到中期趨勢）
   - SMA50 ≤ 價格 < SMA20 → 觀察 SMA20（收復短期趨勢）
   - 價格 ≥ SMA20 → 觀察 20 日高（等待突破/回測）

注意：盤中執行時以即時價覆寫今日日K（Close/High/Low），指標會隨行情變動。
"""

# --- Windows/中文路徑 SSL 憑證修復（避免 curl:77） ---
import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["YF_USE_CURL_CFFI"] = "0"

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# -------- 可調預設（可被 CLI 覆蓋） --------
PERIOD   = "3y"   # 拉長到 3 年，用於長期趨勢
INTERVAL = "1d"

SMA20  = 20
SMA50  = 50
SMA200 = 200
RSI_PERIOD = 14

# -------- 指標工具 --------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0); down = -d.clip(upper=0)
    avg_gain = up.rolling(period).mean()
    avg_loss = down.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - 100/(1+rs)

def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)

# -------- 取價 & 覆寫 --------
def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"無法取得 {symbol} 歷史資料")
    return df.dropna()

def fetch_live_price(symbol: str) -> float | None:
    try:
        h1m = yf.Ticker(symbol).history(period="1d", interval="1m")
        if h1m is None or h1m.empty:
            return None
        p = float(h1m["Close"].iloc[-1])
        return None if (p <= 0 or math.isnan(p)) else p
    except Exception:
        return None

def apply_live_override(df_daily: pd.DataFrame, live_price: float) -> pd.DataFrame:
    df = df_daily.copy()
    if df.empty or live_price is None: return df
    idx = df.index[-1]
    df.at[idx, "Close"] = live_price
    df.at[idx, "High"]  = max(float(df.at[idx,"High"]), live_price)
    df.at[idx, "Low"]   = min(float(df.at[idx,"Low"]),  live_price)
    return df

# -------- 結構 --------
@dataclass
class Signal:
    name: str
    triggered: bool
    details: Dict[str, float]

# -------- 計算/判斷 --------
def evaluate_dip(df: pd.DataFrame, min_hits: int = 1) -> Tuple[bool, List[Signal], Dict[str,float]]:
    c = df["Close"]; o = df["Open"]; h = df["High"]; l = df["Low"]

    df["SMA20"]  = sma(c, SMA20)
    df["SMA50"]  = sma(c, SMA50)
    df["SMA200"] = sma(c, SMA200)
    df["RSI"]    = rsi(c, RSI_PERIOD)
    df["HH20"]   = c.rolling(20).max()

    last = df.iloc[-1]; prev = df.iloc[-2]

    # 長期多頭篩選
    trend_ok = bool((last["SMA50"] > last["SMA200"]) and (last["Close"] > last["SMA200"]))

    sigs: List[Signal] = []

    # A) Reclaim20：昨日在 SMA20 下方，今天收上 SMA20 且收紅
    reclaim20 = bool((prev["Close"] <= prev["SMA20"]) and (last["Close"] > last["SMA20"]) and (last["Close"] > last["Open"]))
    sigs.append(Signal("Reclaim20：收復 SMA20 並收紅", reclaim20, {
        "close": float(last["Close"]), "sma20": float(last["SMA20"])
    }))

    # B) Bounce50：今日低點觸/破 SMA50，收回到 SMA50 之上且收紅（或距離 SMA50 ≤ +1%）
    touched50 = bool(l.iloc[-1] <= last["SMA50"] * 1.001)  # 容許 0.1% 誤差
    rec_above50 = bool(last["Close"] >= last["SMA50"] and last["Close"] > last["Open"])
    near50 = bool(last["Close"] <= last["SMA50"] * 1.01)  # 收盤不離 50 太遠
    bounce50 = bool((touched50 and rec_above50) or (rec_above50 and near50))
    sigs.append(Signal("Bounce50：SMA50 附近回彈收紅", bounce50, {
        "close": float(last["Close"]), "sma50": float(last["SMA50"]), "low": float(last["Low"])
    }))

    # C) RSI 回升：價格在 SMA200 上，RSI 昨 ≤42，今上升
    rsi_rise = bool((last["Close"] > last["SMA200"]) and (prev["RSI"] <= 42) and (last["RSI"] > prev["RSI"]))
    sigs.append(Signal("RSI 回升：多頭回檔後轉強", rsi_rise, {
        "RSI": float(last["RSI"]), "RSI_prev": float(prev["RSI"]), "sma200": float(last["SMA200"])
    }))

    hits = sum(1 for s in sigs if s.triggered)
    ok = bool(trend_ok and (hits >= min_hits))

    tips = {
        "trend_ok": float(1.0 if trend_ok else 0.0),
        "close": float(last["Close"]),
        "open": float(last["Open"]),
        "sma20": float(last["SMA20"]),
        "sma50": float(last["SMA50"]),
        "sma200": float(last["SMA200"]),
        "rsi": float(last["RSI"]),
        "hh20": float(last["HH20"]),
        "pct_to_sma20": float((last["Close"]/last["SMA20"] - 1)*100) if not math.isnan(last["SMA20"]) else float("nan"),
        "pct_to_sma50": float((last["Close"]/last["SMA50"] - 1)*100) if not math.isnan(last["SMA50"]) else float("nan"),
        "pct_to_sma200": float((last["Close"]/last["SMA200"] - 1)*100) if not math.isnan(last["SMA200"]) else float("nan"),
    }
    return ok, sigs, tips

# -------- 輔助：部位/掛單 --------
def compute_position(price: float, capital_usd: float, fx: float, fractional: bool):
    if price <= 0 or capital_usd <= 0:
        return {"whole_shares":0,"whole_cost_usd":0,"whole_cost_twd":0,"frac_shares":0.0,"frac_cost_usd":0.0,"frac_cost_twd":0.0}
    whole = int(capital_usd // price)
    whole_cost = whole * price
    frac_shares = capital_usd / price if fractional else float(whole)
    frac_cost = capital_usd if fractional else whole_cost
    return {
        "whole_shares": whole,
        "whole_cost_usd": whole_cost,
        "whole_cost_twd": whole_cost*fx,
        "frac_shares": frac_shares,
        "frac_cost_usd": frac_cost,
        "frac_cost_twd": frac_cost*fx
    }

def order_band(price: float, pct: float):
    if price <=0 or pct<=0: return (float("nan"), float("nan"))
    return (price*(1-pct), price*(1+pct))

# -------- 主程式 --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NVDA")
    ap.add_argument("--capital-usd", type=float, default=1000.0)
    ap.add_argument("--fx", type=float, default=32.0)
    ap.add_argument("--fractional", default="yes")
    ap.add_argument("--band-pct", type=float, default=0.01)
    ap.add_argument("--min-hits", type=int, default=1)
    args = ap.parse_args()
    fractional = str(args.fractional).lower() in ("y","yes","true","1")

    # 日線 + 即時價覆寫
    df = fetch_history(args.symbol, PERIOD, INTERVAL)
    live = fetch_live_price(args.symbol)
    if live:
        print(f"\n💹 {args.symbol} 即時股價：${live:.2f}  (≈ NT${live*args.fx:,.0f})")
        df = apply_live_override(df, live)
        ref_price = live
    else:
        ref_price = float(df["Close"].iloc[-1])
        print(f"\n⚠️ 未取到即時價，改用最近收盤：${ref_price:.2f}")

    ok, sigs, tips = evaluate_dip(df, min_hits=args.min_hits)
    yes = [s for s in sigs if s.triggered]

    print(f"\n📊 {args.symbol} 即時版・拉回買（長期投資）判斷\n" + "-"*62)

    # 長期多頭篩選說明
    if tips["trend_ok"] < 0.5:
        print("🚧 長期多頭篩選未通過（SMA50 ≤ SMA200 或 價格 ≤ SMA200），建議先觀望。")

    if ok:
        print("✅ 可進場（拉回買）")
        print("原因：")
        for s in yes:
            print(f"  👉 {s.name} ✨")

        # 推薦進場價 = 即時價（已回升/收復）
        lo, hi = order_band(ref_price, args.band_pct)
        print(f"\n📌 推薦進場價（即時）：${ref_price:.2f}  (≈ NT${ref_price*args.fx:,.0f})")
        print(f"🧾 建議掛單區間（±{args.band_pct*100:.1f}%）：${lo:.2f} ~ ${hi:.2f}")

        pos = compute_position(ref_price, args.capital_usd, args.fx, fractional)
        print(f"\n💰 可投入資金：${args.capital_usd:,.2f}  |  匯率：≈ {args.fx:.2f} TWD/USD")
        print(f"🧱 整股：{pos['whole_shares']} 股，需 ${pos['whole_cost_usd']:,.2f}（≈ NT${pos['whole_cost_twd']:,.0f}）")
        if fractional:
            print(f"🧩 碎股：{pos['frac_shares']:.2f} 股，交易金額 ${pos['frac_cost_usd']:,.2f}（≈ NT${pos['frac_cost_twd']:,.0f}）")

        # 指標快照
        print("\n📌 指標快照：")
        print(f"  🔹 Close: ${tips['close']:.2f} | SMA20: ${tips['sma20']:.2f} | SMA50: ${tips['sma50']:.2f} | SMA200: ${tips['sma200']:.2f}")
        print(f"  🔹 RSI(14): {tips['rsi']:.2f} | 與 SMA20 乖離: {tips['pct_to_sma20']:+.2f}% | 與 SMA50 乖離: {tips['pct_to_sma50']:+.2f}%")
    else:
        print("⌛ 暫不進場（等待更好的拉回訊號）")
        # 建議觀察價位規則
        price  = tips["close"]; s20 = tips["sma20"]; s50 = tips["sma50"]; s200 = tips["sma200"]; hh20 = tips["hh20"]

        if price < s50:
            label, val, why = "SMA50（回到中期趨勢）", s50, "價格在 SMA50 下方，先觀察回到 SMA50 再說。"
        elif s50 <= price < s20:
            label, val, why = "SMA20（收復短期趨勢）", s20, "價格介於 SMA50 與 SMA20，收復 SMA20 後較穩健。"
        else:
            label, val, why = "20 日高（等待突破/回測）", hh20, "價格已在 SMA20 上方，觀察 20 日高的突破或回測站穩。"

        lo, hi = order_band(val, 0.005)  # ±0.5%
        print("⚠️ 參考（已用即時價）：")
        print(f"  📉 與 SMA20 乖離：{tips['pct_to_sma20']:+.2f}%、與 SMA50 乖離：{tips['pct_to_sma50']:+.2f}%、與 SMA200：{tips['pct_to_sma200']:+.2f}%")
        print(f"  🎯 RSI(14)：{tips['rsi']:.2f}")
        print(f"\n👀 建議觀察價位：{label} ≈ ${val:.2f}  (≈ NT${val*args.fx:,.0f})")
        print(f"🧾 建議掛單區間（±0.5%）：${lo:.2f} ~ ${hi:.2f}")
        print(f"🗒️ 說明：{why}")

        # 也提供以即時價估算的部位
        pos = compute_position(price, args.capital_usd, args.fx, fractional)
        print(f"\n🧮 參考（以即時價 ${price:.2f}）：")
        print(f"  🧱 整股：{pos['whole_shares']} 股，需 ${pos['whole_cost_usd']:,.2f}（≈ NT${pos['whole_cost_twd']:,.0f}）")
        if fractional:
            print(f"  🧩 碎股：{pos['frac_shares']:.2f} 股，交易金額 ${pos['frac_cost_usd']:,.2f}（≈ NT${pos['frac_cost_twd']:,.0f}）")

    print("\n📝 提示：此腳本偏向『長期多頭中的拉回佈局』；若你希望更嚴格，可把 --min-hits 設為 2。")

if __name__ == "__main__":
    main()
