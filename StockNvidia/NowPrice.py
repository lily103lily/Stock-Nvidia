#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
即時價格版：日線短線進場判斷（1–3 個月視角）

變更重點：
- 「暫不進場」時，顯示：建議觀察價位 = SMA20 或 20 日高（而不是重複即時價）
- 規則：
  1) 價格 < SMA20 → 建議觀察 SMA20（回到趨勢線）
  2) SMA20 ≤ 價格 < 20D 高 → 建議觀察 20D 高（等待突破/回測）
  3) 價格 ≥ 20D 高 但未觸發其它訊號 → 建議觀察「昨日 20D 高」的回測

用法：
  pip install yfinance pandas numpy certifi
  python entry_check_realtime.py --symbol NVDA --capital-usd 937.5 --fx 32 --fractional yes --band-pct 0.01
"""

# --- Windows/中文路徑 SSL 憑證修復（避免 curl: (77)）---
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

# -------- 可調預設參數（可用 CLI 覆蓋） --------
SMA_SHORT = 20
SMA_LONG  = 50
SMA_TREND = 200
RSI_PERIOD = 14
RSI_BUY_LOW = 30
RSI_BUY_HIGH = 42
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BREAKOUT_LOOKBACK = 20
RETEST_TOLERANCE_PCT = 1.0  # ±1%
PERIOD = "1y"
INTERVAL = "1d"

# -------- 指標工具 --------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = up.rolling(period).mean()
    avg_loss = down.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100.0 - (100.0 / (1.0 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)

# -------- 取價與覆寫 --------
def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"取不到 {symbol} 的資料（period={period}, interval={interval}）")
    return df.dropna()

def fetch_live_price(symbol: str) -> float | None:
    try:
        h1m = yf.Ticker(symbol).history(period="1d", interval="1m")
        if h1m is None or h1m.empty:
            return None
        last = float(h1m["Close"].iloc[-1])
        if math.isnan(last) or last <= 0:
            return None
        return last
    except Exception:
        return None

def apply_live_override(df_daily: pd.DataFrame, live_price: float) -> pd.DataFrame:
    """用即時價覆寫最後一根日K的 Close，並擴大當日高/低。"""
    df = df_daily.copy()
    if df.empty or live_price is None:
        return df
    last_idx = df.index[-1]
    df.at[last_idx, "Close"] = live_price
    df.at[last_idx, "High"]  = max(float(df.at[last_idx, "High"]), live_price)
    df.at[last_idx, "Low"]   = min(float(df.at[last_idx, "Low"]),  live_price)
    return df

# -------- 資料結構 --------
@dataclass
class Signal:
    name: str
    triggered: bool
    details: Dict[str, float]

# -------- 判斷邏輯（用覆寫後的 df） --------
def evaluate_with_df(df: pd.DataFrame) -> Tuple[List[Signal], Dict[str, float]]:
    c = df["Close"]
    o = df["Open"]

    df["SMA_S"] = sma(c, SMA_SHORT)
    df["SMA_L"] = sma(c, SMA_LONG)
    df["SMA_T"] = sma(c, SMA_TREND)
    df["RSI"]   = rsi(c, RSI_PERIOD)
    macd_line, macd_sig, _ = macd(c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df["MACD"]     = macd_line
    df["MACD_SIG"] = macd_sig
    df["HH"]       = c.rolling(BREAKOUT_LOOKBACK).max()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    sigs: List[Signal] = []

    # 1) SMA 黃金交叉 + 價格在兩均線上方
    gc_today = bool(cross_up(df["SMA_S"], df["SMA_L"]).iloc[-1])
    gc_valid = gc_today and (last["Close"] > last["SMA_S"] > last["SMA_L"])
    sigs.append(Signal(
        "SMA 黃金交叉",
        gc_valid,
        {"close": float(last["Close"]), "sma20": float(last["SMA_S"]), "sma50": float(last["SMA_L"])}
    ))

    # 2) RSI 拉回回升：趨勢向上 + RSI在買區 + RSI回升
    in_uptrend = bool(last["Close"] > last["SMA_T"])
    rsi_in_zone = bool(RSI_BUY_LOW <= last["RSI"] <= RSI_BUY_HIGH)
    rsi_rising = bool(last["RSI"] > prev["RSI"])
    rsi_ok = in_uptrend and rsi_in_zone and rsi_rising
    sigs.append(Signal(
        "RSI 拉回回升",
        rsi_ok,
        {"RSI": float(last["RSI"]), "SMA200": float(last["SMA_T"])}
    ))

    # 3) MACD 金叉
    macd_up = bool(cross_up(df["MACD"], df["MACD_SIG"]).iloc[-1])
    sigs.append(Signal(
        "MACD 金叉",
        macd_up,
        {"MACD": float(last["MACD"]), "Signal": float(last["MACD_SIG"])}
    ))

    # 4) 20日突破 + 回測（昨突破、今回測±1%並收紅）
    tol = RETEST_TOLERANCE_PCT / 100.0
    hh_y = float(df["HH"].iloc[-2])
    broke_y = bool(prev["Close"] > hh_y * (1 + 0.001))
    near_retest_today = bool(abs(last["Low"] - hh_y) <= hh_y * tol or abs(last["Close"] - hh_y) <= hh_y * tol)
    green_today = bool(last["Close"] > last["Open"])
    br_retest = broke_y and near_retest_today and green_today
    sigs.append(Signal(
        "20日突破 + 回測成功",
        br_retest,
        {"HH_yesterday": hh_y, "close": float(last["Close"]), "low": float(last["Low"])}
    ))

    tips = {
        "last_close": float(last["Close"]),        # 已是即時價覆寫後
        "sma20": float(last["SMA_S"]),
        "sma50": float(last["SMA_L"]),
        "sma200": float(last["SMA_T"]),
        "hh_today": float(df["HH"].iloc[-1]),
        "hh_yesterday": hh_y,
        "pct_to_20D_high": float((last["Close"]/float(df["HH"].iloc[-1]) - 1)*100) if not math.isnan(df["HH"].iloc[-1]) else float("nan"),
        "pct_to_SMA20": float((last["Close"]/float(last["SMA_S"]) - 1)*100) if not math.isnan(last["SMA_S"]) else float("nan"),
        "pct_to_SMA50": float((last["Close"]/float(last["SMA_L"]) - 1)*100) if not math.isnan(last["SMA_L"]) else float("nan"),
        "RSI": float(last["RSI"]),
        "in_uptrend": float(1.0 if in_uptrend else 0.0),
    }
    return sigs, tips

# -------- 輔助：交易金額與股數、掛單區間 --------
def compute_position(last_price: float, capital_usd: float, fx_twd_per_usd: float, fractional: bool):
    if last_price <= 0 or capital_usd <= 0:
        return {"whole_shares": 0, "whole_cost_usd": 0.0, "whole_cost_twd": 0.0,
                "frac_shares": 0.0, "frac_cost_usd": 0.0, "frac_cost_twd": 0.0}
    whole_shares = int(capital_usd // last_price)
    whole_cost_usd = whole_shares * last_price
    whole_cost_twd = whole_cost_usd * fx_twd_per_usd
    if fractional:
        frac_shares = capital_usd / last_price
        frac_cost_usd = capital_usd
    else:
        frac_shares = float(whole_shares)
        frac_cost_usd = whole_cost_usd
    frac_cost_twd = frac_cost_usd * fx_twd_per_usd
    return {"whole_shares": whole_shares, "whole_cost_usd": whole_cost_usd, "whole_cost_twd": whole_cost_twd,
            "frac_shares": frac_shares, "frac_cost_usd": frac_cost_usd, "frac_cost_twd": frac_cost_twd}

def order_band(price: float, pct: float):
    if price <= 0 or pct <= 0:
        return (float("nan"), float("nan"))
    return (price*(1-pct), price*(1+pct))

# -------- 主程式 --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NVDA", help="美股代號，例如 NVDA/AAPL/MSFT")
    ap.add_argument("--capital-usd", type=float, default=1000.0, help="可投入資金 (USD)")
    ap.add_argument("--fx", type=float, default=32.0, help="匯率（TWD / USD）")
    ap.add_argument("--fractional", default="yes", help="是否使用碎股 yes/no（預設 yes）")
    ap.add_argument("--band-pct", type=float, default=0.01, help="建議掛單區間百分比（預設 0.01=±1%）")
    args = ap.parse_args()

    fractional = str(args.fractional).lower() in ("y", "yes", "true", "1")

    # 1) 基礎日線
    df_daily = fetch_history(args.symbol, PERIOD, INTERVAL)

    # 2) 即時價
    live_price = fetch_live_price(args.symbol)
    if live_price:
        print(f"\n💹 {args.symbol} 即時股價：${live_price:.2f} 美元  (≈ NT${live_price*args.fx:,.0f})")
        df_used = apply_live_override(df_daily, live_price)
        ref_price = live_price
    else:
        last_close = float(df_daily["Close"].iloc[-1])
        print(f"\n⚠️ 取不到即時價，改用最新日線收盤價：${last_close:.2f}")
        df_used = df_daily
        ref_price = last_close

    # 3) 用覆寫後的 df 判斷
    sigs, tips = evaluate_with_df(df_used)
    yes = [s for s in sigs if s.triggered]

    print(f"\n📊 {args.symbol} 即時版・日線短線進場判斷\n" + "-"*60)
    if yes:
        print("✅ 可進場")
        print("原因：")
        for s in yes:
            print(f"  👉 {s.name} ✨")

        lo, hi = order_band(ref_price, args.band_pct)
        print(f"\n📌 推薦進場價（即時）：${ref_price:.2f}  (≈ NT${ref_price*args.fx:,.0f})")
        print(f"🧾 建議掛單區間（±{args.band_pct*100:.1f}%）：${lo:.2f} ~ ${hi:.2f}")

        pos = compute_position(ref_price, args.capital_usd, args.fx, fractional)
        print(f"\n💰 可投入資金：${args.capital_usd:,.2f}  |  匯率：≈ {args.fx:.2f} TWD/USD")
        print(f"🧱 整股模式：{pos['whole_shares']} 股")
        print(f"   💳 交易金額（整股）：${pos['whole_cost_usd']:,.2f}  ≈  NT${pos['whole_cost_twd']:,.0f}")
        if fractional:
            print(f"🧩 碎股模式：{pos['frac_shares']:.2f} 股（用滿資金）")
            print(f"   💳 交易金額（碎股）：${pos['frac_cost_usd']:,.2f}  ≈  NT${pos['frac_cost_twd']:,.0f}")

        print("\n📌 指標快照：")
        for s in yes:
            for k, v in s.details.items():
                try:
                    print(f"  🔹 {s.name} | {k}: {v:.2f}")
                except Exception:
                    print(f"  🔹 {s.name} | {k}: {v}")
    else:
        # ====== 這裡是你要的改動：顯示「建議觀察價位 = SMA20 或 20 日高」 ======
        price   = tips.get("last_close", float("nan"))
        sma20   = tips.get("sma20", float("nan"))
        hh_today = tips.get("hh_today", float("nan"))
        hh_y     = tips.get("hh_yesterday", float("nan"))
        rsi     = tips.get("RSI", float("nan"))
        pct20   = tips.get("pct_to_20D_high", float("nan"))
        pctS20  = tips.get("pct_to_SMA20", float("nan"))
        pctS50  = tips.get("pct_to_SMA50", float("nan"))

        print("⌛ 暫不進場")
        print("⚠️ 接近條件 / 參考（已用即時價）：")
        print(f"  📈 距 20日高 變動：{pct20:+.2f}%")
        print(f"  📉 與 SMA20 乖離：{pctS20:+.2f}%、SMA50：{pctS50:+.2f}%")
        print(f"  🎯 RSI(14)：{rsi:.2f}（買區 {RSI_BUY_LOW}~{RSI_BUY_HIGH}、≥50 動能更佳）")

        # 規則化的觀察位建議
        suggest_label = ""
        suggest_price = float("nan")
        explain = ""

        if not math.isnan(price) and not math.isnan(sma20) and price < sma20:
            suggest_label = "SMA20（回到趨勢線）"
            suggest_price = sma20
            explain = "價格在 SMA20 下方，建議等待回到趨勢線再考慮進場。"
        elif (not math.isnan(price) and not math.isnan(sma20) and not math.isnan(hh_today)
              and price >= sma20 and price < hh_today):
            suggest_label = "20 日高（等待突破/回測）"
            suggest_price = hh_today
            explain = "價格介於 SMA20 與 20 日高之間，建議觀察 20 日高的突破或回測站穩。"
        elif not math.isnan(price) and not math.isnan(hh_y) and price >= hh_today:
            suggest_label = "昨日 20 日高（回測確認）"
            suggest_price = hh_y
            explain = "價格已在 20 日高上方，但訊號未成形，建議等待昨日 20 日高的回測確認。"
        else:
            # 後備：若有值就至少列出兩個參考位
            if not math.isnan(sma20):
                suggest_label = "SMA20（回到趨勢線）"
                suggest_price = sma20
                explain = "提供趨勢線參考。"

        if not math.isnan(suggest_price):
            lo, hi = order_band(suggest_price, 0.005)  # ±0.5% 建議掛單區間
            print(f"\n👀 建議觀察價位：{suggest_label} ≈ ${suggest_price:.2f}  (≈ NT${suggest_price*args.fx:,.0f})")
            print(f"🧾 建議掛單區間（±0.5%）：${lo:.2f} ~ ${hi:.2f}")
            if explain:
                print(f"🗒️ 說明：{explain}")
        else:
            print("\n👀 暫無明確觀察價位（資料不足）")

        # 仍提供用「即時價」估算的資金/股數，方便你衡量部位
        if not math.isnan(price) and price > 0:
            pos = compute_position(price, args.capital_usd, args.fx, fractional)
            print(f"\n🧮 參考（以即時價 ${price:.2f}）：")
            print(f"  🧱 整股：{pos['whole_shares']} 股，需 ${pos['whole_cost_usd']:,.2f}（≈ NT${pos['whole_cost_twd']:,.0f}）")
            if fractional:
                print(f"  🧩 碎股：{pos['frac_shares']:.2f} 股，交易金額 ${pos['frac_cost_usd']:,.2f}（≈ NT${pos['frac_cost_twd']:,.0f}）")

    print("\n📝 提示：若要自動刷新，可用系統排程每 5–10 分鐘執行一次本腳本。")

if __name__ == "__main__":
    main()
