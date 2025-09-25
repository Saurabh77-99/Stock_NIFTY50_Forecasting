import yfinance as yf
import pandas as pd
import talib
import numpy as np
from datetime import datetime, timedelta

nifty_symbol = "^NSEI"
start_date = "2014-01-01"
end_date = "2025-09-25"

ma_fast = 50
ma_slow = 200
macd_fast = 12
macd_slow = 26
macd_signal = 9
atr_period = 14

def backtest_nifty():
    try:
        print(f"Fetching NIFTY 50 data...")
        df = yf.download(nifty_symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True)
        
        if df.empty:
            print(f"No data for NIFTY 50")
            return []
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure we have the required columns
        if 'Close' not in df.columns:
            print(f"No 'Close' column found for NIFTY 50")
            return []
        
        # Clean the data
        df = df.dropna()
        
        if len(df) < ma_slow:
            print(f"Not enough data for NIFTY 50")
            return []
        
        # Convert to numpy arrays
        close_prices = np.array(df['Close']).flatten()
        high_prices = np.array(df['High']).flatten()
        low_prices = np.array(df['Low']).flatten()
        
        # Calculate technical indicators - ONLY MACD and MA
        df['MA_Fast'] = talib.SMA(close_prices, ma_fast)
        df['MA_Slow'] = talib.SMA(close_prices, ma_slow)
        df['ATR'] = talib.ATR(high_prices, low_prices, close_prices, atr_period)
        
        # MACD calculation
        macd, signal, histogram = talib.MACD(close_prices, 
                                           fastperiod=macd_fast, 
                                           slowperiod=macd_slow, 
                                           signalperiod=macd_signal)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) < 2:
            print(f"Not enough valid data points for NIFTY 50")
            return []
        
        trades = []
        position = None
        entry_price = None
        entry_date = None
        stop_loss = None
        take_profit = None
        
        print(f"Starting backtest for NIFTY 50 with {len(df)} valid data points")
        
        for i in range(2, len(df)):  # Start from index 2 to have enough historical data
            try:
                # Current values
                close = df['Close'].iloc[i]
                ma_fast_val = df['MA_Fast'].iloc[i]
                ma_slow_val = df['MA_Slow'].iloc[i]
                atr = df['ATR'].iloc[i]
                macd_val = df['MACD'].iloc[i]
                macd_signal_val = df['MACD_Signal'].iloc[i]
                macd_hist = df['MACD_Histogram'].iloc[i]
                
                # Previous values for crossover detection
                prev_ma_fast = df['MA_Fast'].iloc[i-1]
                prev_ma_slow = df['MA_Slow'].iloc[i-1]
                prev_macd = df['MACD'].iloc[i-1]
                prev_macd_signal = df['MACD_Signal'].iloc[i-1]
                prev_macd_hist = df['MACD_Histogram'].iloc[i-1]
                
                # Two periods ago for trend confirmation
                prev2_macd_hist = df['MACD_Histogram'].iloc[i-2]
                
                # Entry conditions - Check for both LONG and SHORT when no position
                if position is None:
                    
                    # LONG entry conditions - Enhanced MACD + MA strategy
                    # 1. Price must be above both moving averages (trend filter)
                    price_above_mas = close > ma_fast_val and close > ma_slow_val
                    
                    # 2. Fast MA above Slow MA (uptrend confirmation)
                    ma_bullish = ma_fast_val > ma_slow_val
                    
                    # 3. MACD bullish crossover (MACD line crosses above signal line)
                    macd_bullish_cross = prev_macd <= prev_macd_signal and macd_val > macd_signal_val
                    
                    # 4. MACD histogram turning positive (momentum confirmation)
                    macd_hist_positive = macd_hist > 0 and prev_macd_hist <= 0
                    
                    # 5. Alternative: Strong MACD momentum (histogram increasing for 2 periods)
                    macd_momentum_bull = macd_hist > prev_macd_hist > prev2_macd_hist and macd_hist > 0
                    
                    # Entry condition for LONG: Either bullish crossover with positive histogram OR strong momentum
                    long_entry_signal = price_above_mas and ma_bullish and (
                        (macd_bullish_cross and macd_hist_positive) or macd_momentum_bull
                    )
                    
                    # SHORT entry conditions - Enhanced MACD + MA strategy
                    # 1. Price must be below both moving averages (trend filter)
                    price_below_mas = close < ma_fast_val and close < ma_slow_val
                    
                    # 2. Fast MA below Slow MA (downtrend confirmation)
                    ma_bearish = ma_fast_val < ma_slow_val
                    
                    # 3. MACD bearish crossover (MACD line crosses below signal line)
                    macd_bearish_cross = prev_macd >= prev_macd_signal and macd_val < macd_signal_val
                    
                    # 4. MACD histogram turning negative (momentum confirmation)
                    macd_hist_negative = macd_hist < 0 and prev_macd_hist >= 0
                    
                    # 5. Alternative: Strong MACD momentum (histogram decreasing for 2 periods)
                    macd_momentum_bear = macd_hist < prev_macd_hist < prev2_macd_hist and macd_hist < 0
                    
                    # Entry condition for SHORT: Either bearish crossover with negative histogram OR strong momentum
                    short_entry_signal = price_below_mas and ma_bearish and (
                        (macd_bearish_cross and macd_hist_negative) or macd_momentum_bear
                    )
                    
                    # Execute LONG entry
                    if long_entry_signal:
                        position = "LONG"
                        entry_price = close
                        entry_date = df.index[i]
                        # Dynamic stop loss based on ATR
                        stop_loss = entry_price - (1.5 * atr)
                        # Take profit at 2.5:1 risk-reward ratio
                        take_profit = entry_price + (2.5 * (entry_price - stop_loss))
                        
                    # Execute SHORT entry (only if LONG didn't trigger)
                    elif short_entry_signal:
                        position = "SHORT"
                        entry_price = close
                        entry_date = df.index[i]
                        # Dynamic stop loss based on ATR
                        stop_loss = entry_price + (1.5 * atr)
                        # Take profit at 2.5:1 risk-reward ratio
                        take_profit = entry_price - (2.5 * (stop_loss - entry_price))
                
                # Check for exit conditions if in a position
                if position is not None:
                    exit_price = None
                    exit_reason = None
                    
                    # 1. Stop loss hit (highest priority)
                    if (position == "LONG" and close <= stop_loss) or \
                       (position == "SHORT" and close >= stop_loss):
                        exit_price = stop_loss
                        exit_reason = "STOP LOSS"
                    
                    # 2. Take profit hit
                    elif (position == "LONG" and close >= take_profit) or \
                         (position == "SHORT" and close <= take_profit):
                        exit_price = take_profit
                        exit_reason = "TAKE PROFIT"
                    
                    # 3. MACD signal reversal (trend change)
                    elif position == "LONG" and (
                        (prev_macd >= prev_macd_signal and macd_val < macd_signal_val) or  # MACD bearish cross
                        (close < ma_fast_val and ma_fast_val < ma_slow_val)  # Price below MAs and MA bearish
                    ):
                        exit_price = close
                        exit_reason = "SIGNAL REVERSAL"
                    
                    elif position == "SHORT" and (
                        (prev_macd <= prev_macd_signal and macd_val > macd_signal_val) or  # MACD bullish cross
                        (close > ma_fast_val and ma_fast_val > ma_slow_val)  # Price above MAs and MA bullish
                    ):
                        exit_price = close
                        exit_reason = "SIGNAL REVERSAL"
                    
                    # 4. Trailing stop (only for profitable positions)
                    elif position == "LONG" and close > entry_price * 1.02:  # 2% profit
                        new_stop = close - (1.2 * atr)  # Tighter trailing stop
                        if new_stop > stop_loss:
                            stop_loss = new_stop
                    
                    elif position == "SHORT" and close < entry_price * 0.98:  # 2% profit
                        new_stop = close + (1.2 * atr)  # Tighter trailing stop
                        if new_stop < stop_loss:
                            stop_loss = new_stop
                    
                    # Execute exit if conditions are met
                    if exit_price is not None:
                        exit_date = df.index[i]
                        
                        # Calculate profit points
                        if position == "LONG":
                            profit_points = exit_price - entry_price
                        else:  # SHORT
                            profit_points = entry_price - exit_price
                        
                        # Calculate percentage return
                        percentage_return = (profit_points / entry_price) * 100
                        
                        trades.append({
                            "Trade_No": len(trades) + 1,
                            "Entry_Date": entry_date.strftime("%Y-%m-%d"),
                            "Entry_Time": entry_date.strftime("%H:%M:%S"),
                            "Entry_Price": round(entry_price, 2),
                            "Exit_Date": exit_date.strftime("%Y-%m-%d"),
                            "Exit_Time": exit_date.strftime("%H:%M:%S"),
                            "Exit_Price": round(exit_price, 2),
                            "Position_Type": position,
                            "Profit_Points": round(profit_points, 2),
                            "Percentage_Return": round(percentage_return, 2),
                            "Days_Held": (exit_date - entry_date).days,
                            "Exit_Reason": exit_reason,
                            "Stop_Loss": round(stop_loss, 2),
                            "Take_Profit": round(take_profit, 2) if position == "LONG" else round(take_profit, 2),
                            "ATR_at_Entry": round(atr, 2)
                        })
                        
                        position = None
                    
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue
        
        # Close any open position at the end
        if position is not None:
            exit_price = df['Close'].iloc[-1]
            exit_date = df.index[-1]
            
            # Calculate profit points
            if position == "LONG":
                profit_points = exit_price - entry_price
            else:  # SHORT
                profit_points = entry_price - exit_price
            
            # Calculate percentage return
            percentage_return = (profit_points / entry_price) * 100
            
            trades.append({
                "Trade_No": len(trades) + 1,
                "Entry_Date": entry_date.strftime("%Y-%m-%d"),
                "Entry_Time": entry_date.strftime("%H:%M:%S"),
                "Entry_Price": round(entry_price, 2),
                "Exit_Date": exit_date.strftime("%Y-%m-%d"),
                "Exit_Time": exit_date.strftime("%H:%M:%S"),
                "Exit_Price": round(exit_price, 2),
                "Position_Type": position,
                "Profit_Points": round(profit_points, 2),
                "Percentage_Return": round(percentage_return, 2),
                "Days_Held": (exit_date - entry_date).days,
                "Exit_Reason": "END OF DATA",
                "Stop_Loss": round(stop_loss, 2),
                "Take_Profit": round(take_profit, 2),
                "ATR_at_Entry": round(atr, 2)
            })
        
        print(f"Found {len(trades)} trades for NIFTY 50")
        return trades
        
    except Exception as e:
        print(f"Error with NIFTY 50: {e}")
        import traceback
        traceback.print_exc()
        return []

# Main execution
print("Starting NIFTY 50 MACD + MA Strategy Backtest...")
print("=" * 60)
print("Strategy Details:")
print(f"- Index: NIFTY 50 (^NSEI)")
print(f"- Fast MA: {ma_fast} periods")
print(f"- Slow MA: {ma_slow} periods") 
print(f"- MACD: {macd_fast}, {macd_slow}, {macd_signal}")
print(f"- Risk-Reward Ratio: 1:2.5")
print("- Position Types: BOTH LONG AND SHORT")
print("=" * 60)

trades = backtest_nifty()

print("\n" + "=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)

if trades:
    # Create DataFrame
    df_trades = pd.DataFrame(trades)
    
    # Save to CSV
    csv_filename = "nifty50_trades_long_short_detailed.csv"
    df_trades.to_csv(csv_filename, index=False)
    print(f"\nDetailed trades saved to {csv_filename}")
    print(f"Total trades found: {len(df_trades)}")
    
    # Summary statistics
    total_profit_points = df_trades['Profit_Points'].sum()
    winning_trades = len(df_trades[df_trades['Profit_Points'] > 0])
    losing_trades = len(df_trades[df_trades['Profit_Points'] < 0])
    break_even_trades = len(df_trades[df_trades['Profit_Points'] == 0])
    win_rate = (winning_trades / len(df_trades)) * 100 if len(df_trades) > 0 else 0
    
    # Calculate metrics
    avg_win_points = df_trades[df_trades['Profit_Points'] > 0]['Profit_Points'].mean() if winning_trades > 0 else 0
    avg_loss_points = abs(df_trades[df_trades['Profit_Points'] < 0]['Profit_Points'].mean()) if losing_trades > 0 else 0
    avg_win_percentage = df_trades[df_trades['Percentage_Return'] > 0]['Percentage_Return'].mean() if winning_trades > 0 else 0
    avg_loss_percentage = abs(df_trades[df_trades['Percentage_Return'] < 0]['Percentage_Return'].mean()) if losing_trades > 0 else 0
    
    # Profit factor
    total_winning_points = df_trades[df_trades['Profit_Points'] > 0]['Profit_Points'].sum()
    total_losing_points = abs(df_trades[df_trades['Profit_Points'] < 0]['Profit_Points'].sum())
    profit_factor = total_winning_points / total_losing_points if total_losing_points > 0 else float('inf')
    
    # Average holding period
    avg_holding_days = df_trades['Days_Held'].mean()
    
    print(f"\nNIFTY 50 STRATEGY RESULTS:")
    print(f"Total Trades: {len(df_trades)}")
    print(f"Winning Trades: {winning_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades: {losing_trades}")
    print(f"Break-even Trades: {break_even_trades}")
    print(f"Total Profit Points: {total_profit_points:.2f}")
    print(f"Average Winning Trade: {avg_win_points:.2f} points ({avg_win_percentage:.2f}%)")
    print(f"Average Losing Trade: {avg_loss_points:.2f} points ({avg_loss_percentage:.2f}%)")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Holding Period: {avg_holding_days:.1f} days")
    
    # Yearly analysis
    df_trades['Entry_Year'] = pd.to_datetime(df_trades['Entry_Date']).dt.year
    yearly_analysis = df_trades.groupby('Entry_Year').agg({
        'Profit_Points': ['sum', 'count', 'mean'],
        'Percentage_Return': 'mean'
    }).round(2)
    yearly_analysis.columns = ['Total_Points', 'Trade_Count', 'Avg_Points', 'Avg_Percentage']
    print("\nYearly Performance:")
    print(yearly_analysis)
    
    # Position type analysis
    position_analysis = df_trades.groupby('Position_Type').agg({
        'Profit_Points': ['sum', 'count', 'mean'],
        'Percentage_Return': 'mean'
    }).round(2)
    position_analysis.columns = ['Total_Points', 'Trade_Count', 'Avg_Points', 'Avg_Percentage']
    print("\nLong vs Short Performance:")
    print(position_analysis)
    
    # Exit reason analysis
    exit_reason_analysis = df_trades.groupby('Exit_Reason').agg({
        'Profit_Points': ['sum', 'count', 'mean']
    }).round(2)
    exit_reason_analysis.columns = ['Total_Points', 'Trade_Count', 'Avg_Points']
    print("\nExit Reason Analysis:")
    print(exit_reason_analysis)
    
    # Show first few trades as sample
    print(f"\nSample Trades (First 10):")
    print(df_trades[['Trade_No', 'Entry_Date', 'Entry_Price', 'Exit_Date', 'Exit_Price', 
                     'Position_Type', 'Profit_Points', 'Percentage_Return', 'Exit_Reason']].head(10).to_string(index=False))
    
    print(f"\nLargest Winners:")
    top_winners = df_trades.nlargest(3, 'Profit_Points')[['Trade_No', 'Entry_Date', 'Entry_Price', 
                                                          'Exit_Price', 'Position_Type', 'Profit_Points', 
                                                          'Percentage_Return', 'Days_Held']]
    print(top_winners.to_string(index=False))
    
    print(f"\nLargest Losers:")
    top_losers = df_trades.nsmallest(3, 'Profit_Points')[['Trade_No', 'Entry_Date', 'Entry_Price', 
                                                          'Exit_Price', 'Position_Type', 'Profit_Points', 
                                                          'Percentage_Return', 'Days_Held']]
    print(top_losers.to_string(index=False))
    
else:
    print("No trades found.")

print(f"\n{'='*60}")
print("CSV FILE COLUMNS EXPLANATION:")
print("- Trade_No: Sequential trade number")
print("- Entry_Date/Time: When the trade was entered")
print("- Entry_Price: NIFTY price at entry")
print("- Exit_Date/Time: When the trade was exited")
print("- Exit_Price: NIFTY price at exit")
print("- Position_Type: LONG or SHORT")
print("- Profit_Points: Actual points gained/lost")
print("- Percentage_Return: % return on the trade")
print("- Days_Held: Number of days trade was held")
print("- Exit_Reason: Why the trade was closed")
print("- Stop_Loss: Stop loss level set")
print("- Take_Profit: Take profit level set")
print("- ATR_at_Entry: ATR value when trade entered")
print(f"{'='*60}")