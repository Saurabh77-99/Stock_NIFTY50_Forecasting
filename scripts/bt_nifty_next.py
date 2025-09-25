import yfinance as yf
import pandas as pd
import talib
import numpy as np
from datetime import datetime, timedelta

nifty_symbol = "^NSEI"
ma_fast = 50
ma_slow = 200
macd_fast = 12
macd_slow = 26
macd_signal = 9
atr_period = 14

def predict_next_week(start_days_back=500):
    """
    Downloads recent NIFTY data, computes indicators, and outputs suggested
    entry / stop / target levels for the next week for LONG and SHORT
    based on the same MACD+MA+ATR logic you used in backtest.

    Returns:
        dict with 'long' and 'short' entries (or None if no immediate trigger).
    Side-effect:
        prints summary and writes 'nifty_next_week_signal.csv' with levels.
    """
    # download enough history to compute MA200, MACD, ATR
    end = datetime.now().date() + timedelta(days=1)
    start = (datetime.now().date() - timedelta(days=start_days_back)).isoformat()
    df = yf.download(nifty_symbol, start=start, end=end.isoformat(), interval="1d", auto_adjust=True)
    if df.empty:
        print("No data downloaded for NIFTY. Check connection or ticker.")
        return None

    # handle MultiIndex if returned
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.dropna()
    if len(df) < ma_slow + 5:
        print("Not enough data to compute indicators.")
        return None

    close = np.array(df['Close']).flatten()
    high = np.array(df['High']).flatten()
    low = np.array(df['Low']).flatten()

    # indicators
    df['MA_Fast'] = talib.SMA(close, ma_fast)
    df['MA_Slow'] = talib.SMA(close, ma_slow)
    df['ATR'] = talib.ATR(high, low, close, timeperiod=atr_period)

    macd, signal, hist = talib.MACD(close,
                                   fastperiod=macd_fast,
                                   slowperiod=macd_slow,
                                   signalperiod=macd_signal)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    df = df.dropna()
    if len(df) < 3:
        print("Not enough indicator rows after dropna.")
        return None

    last = df.iloc[-1]
    prev = df.iloc[-1 - 1]
    prev2 = df.iloc[-2 - 1] if len(df) >= 3 else prev

    close_last = last['Close']
    ma_f_last = last['MA_Fast']
    ma_s_last = last['MA_Slow']
    atr_last = last['ATR']
    macd_last = last['MACD']
    macd_sig_last = last['MACD_Signal']
    macd_hist_last = last['MACD_Hist']

    prev_macd = prev['MACD']
    prev_macd_sig = prev['MACD_Signal']
    prev_macd_hist = prev['MACD_Hist']

    prev2_macd_hist = prev2['MACD_Hist'] if 'MACD_Hist' in prev2 else prev_macd_hist

    # Prepare outputs
    recommendations = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "last_close": round(close_last, 2),
                       "atr": round(atr_last, 4),
                       "long": None,
                       "short": None,
                       "signal_summary": []}

    # Determine immediate signals using the same rules as backtest (but for next week we'll
    # return trigger levels rather than force an immediate entry).
    price_above_mas = close_last > ma_f_last and close_last > ma_s_last
    ma_bullish = ma_f_last > ma_s_last
    macd_bullish_cross = (prev_macd <= prev_macd_sig) and (macd_last > macd_sig_last)
    macd_hist_positive = (macd_hist_last > 0) and (prev_macd_hist <= 0)
    macd_momentum_bull = (macd_hist_last > prev_macd_hist > prev2_macd_hist) and (macd_hist_last > 0)

    price_below_mas = close_last < ma_f_last and close_last < ma_s_last
    ma_bearish = ma_f_last < ma_s_last
    macd_bearish_cross = (prev_macd >= prev_macd_sig) and (macd_last < macd_sig_last)
    macd_hist_negative = (macd_hist_last < 0) and (prev_macd_hist >= 0)
    macd_momentum_bear = (macd_hist_last < prev_macd_hist < prev2_macd_hist) and (macd_hist_last < 0)

    # Build suggested trigger levels using ATR-based buffers
    # Entry triggers: Conservative — wait for close to confirm across MA or MACD.
    # We provide two trigger types:
    #  - Immediate "momentum" trigger (if MACD crossover + histogram confirmation happened today)
    #  - "Watch" triggers (price breaks above/below a level — useful for next week)

    # LONG suggestion
    long_conditions_met = price_above_mas and ma_bullish and ((macd_bullish_cross and macd_hist_positive) or macd_momentum_bull)
    if long_conditions_met:
        # Suggest aggressive immediate entry at last close (momentum present)
        entry = close_last
        sl = entry - 1.5 * atr_last
        tp = entry + 2.5 * (entry - sl)
        recommendations['long'] = {
            "type": "AGGRESSIVE (momentum present)",
            "entry": round(entry, 2),
            "stop_loss": round(sl, 2),
            "take_profit": round(tp, 2),
            "rr": "1:2.5"
        }
        recommendations['signal_summary'].append("LONG: momentum conditions met — consider aggressive entry.")
    else:
        # Provide watch levels: a break above MA_Fast or a close above last swing high (we use MA_Fast + small buffer)
        watch_entry = ma_f_last * 1.001  # small buffer above MA_Fast to avoid false breakouts
        sl = watch_entry - 1.5 * atr_last
        tp = watch_entry + 2.5 * (watch_entry - sl)
        recommendations['long'] = {
            "type": "WATCH (wait for confirmation)",
            "watch_trigger": round(watch_entry, 2),
            "watch_reason": "Close above MA_Fast + small buffer and MACD bullish confirmation",
            "stop_loss_if_entered": round(sl, 2),
            "take_profit_if_entered": round(tp, 2),
            "note": "Enter only when price closes above trigger and MACD is not strongly bearish."
        }
        recommendations['signal_summary'].append("LONG: no immediate momentum — watch for a close above MA_Fast as entry trigger.")

    # SHORT suggestion
    short_conditions_met = price_below_mas and ma_bearish and ((macd_bearish_cross and macd_hist_negative) or macd_momentum_bear)
    if short_conditions_met:
        entry = close_last
        sl = entry + 1.5 * atr_last
        tp = entry - 2.5 * (sl - entry)
        recommendations['short'] = {
            "type": "AGGRESSIVE (momentum present)",
            "entry": round(entry, 2),
            "stop_loss": round(sl, 2),
            "take_profit": round(tp, 2),
            "rr": "1:2.5"
        }
        recommendations['signal_summary'].append("SHORT: momentum conditions met — consider aggressive short entry.")
    else:
        watch_entry = ma_f_last * 0.999  # small buffer below MA_Fast
        sl = watch_entry + 1.5 * atr_last
        tp = watch_entry - 2.5 * (sl - watch_entry)
        recommendations['short'] = {
            "type": "WATCH (wait for confirmation)",
            "watch_trigger": round(watch_entry, 2),
            "watch_reason": "Close below MA_Fast + small buffer and MACD bearish confirmation",
            "stop_loss_if_entered": round(sl, 2),
            "take_profit_if_entered": round(tp, 2),
            "note": "Enter only when price closes below trigger and MACD is not strongly bullish."
        }
        recommendations['signal_summary'].append("SHORT: no immediate momentum — watch for a close below MA_Fast as entry trigger.")

    # Extra contextual info
    recommendations['market_structure'] = {
        "close_vs_MA_fast": f"{round(close_last - ma_f_last, 2)} points from MA_Fast",
        "ma_fast_vs_ma_slow": f"{round(ma_f_last - ma_s_last, 2)} points (MA_Fast - MA_Slow)",
        "macd_hist_last": round(macd_hist_last, 6)
    }

    rows = []
    for side in ['long', 'short']:
        info = recommendations[side]
        if info is None:
            continue
        row = {
            "side": side.upper(),
            "type": info.get("type"),
            "entry": info.get("entry") or info.get("watch_trigger"),
            "stop_loss": info.get("stop_loss") or info.get("stop_loss_if_entered"),
            "take_profit": info.get("take_profit") or info.get("take_profit_if_entered"),
            "note": info.get("note") or info.get("watch_reason")
        }
        rows.append(row)
    df_signals = pd.DataFrame(rows)
    csv_filename = "nifty_next_week_signal.csv"
    df_signals.to_csv(csv_filename, index=False)

    print("\n=== NIFTY NEXT-WEEK SIGNALS ===")
    print("Generated at:", recommendations['generated_at'])
    print("Last Close:", recommendations['last_close'], "| ATR:", recommendations['atr'])
    print("\nSignal summary:")
    for s in recommendations['signal_summary']:
        print("- ", s)
    print("\nLONG recommendation:")
    print(recommendations['long'])
    print("\nSHORT recommendation:")
    print(recommendations['short'])
    print(f"\nSaved quick signal CSV -> {csv_filename}")
    print("\nNotes: 1) 'AGGRESSIVE' means the strategy's entry conditions are live today.\n2) 'WATCH' gives a trigger price to wait for next-week confirmation.\n3) Use position sizing and consider brokerage/slippage.")
    return recommendations

if __name__ == "__main__":
    preds = predict_next_week()
