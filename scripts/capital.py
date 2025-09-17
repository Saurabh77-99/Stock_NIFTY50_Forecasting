import yfinance as yf
import pandas as pd
import talib
import numpy as np
from datetime import datetime, timedelta

stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "KOTAKBANK.NS", "LT.NS", "ITC.NS", "SBIN.NS"
]
start_date = "2014-01-01"
end_date = "2025-09-18"

# Optimized parameters - focusing on MACD and MA only
ma_fast = 50    # Reduced from 200 for more responsiveness
ma_slow = 200   # Reduced from 500 for more responsiveness
macd_fast = 12
macd_slow = 26
macd_signal = 9
atr_period = 14
risk_per_trade = 0.015  # Increased to 1.5% for better position sizes
initial_capital = 1000000  # ₹10,00,000 initial capital

all_trades = []
equity_curve = []

def calculate_position_size(entry_price, stop_loss, capital):
    risk_amount = capital * risk_per_trade
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share == 0:
        return 0
    return int(risk_amount / risk_per_share)

def backtest_stock(stock):
    try:
        print(f"Fetching {stock}...")
        df = yf.download(stock, start=start_date, end=end_date, interval="1d", auto_adjust=True)
        
        if df.empty:
            print(f"No data for {stock}")
            return []
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure we have the required columns
        if 'Close' not in df.columns:
            print(f"No 'Close' column found for {stock}")
            return []
        
        # Clean the data
        df = df.dropna()
        
        if len(df) < ma_slow:
            print(f"Not enough data for {stock}")
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
            print(f"Not enough valid data points for {stock}")
            return []
        
        trades = []
        position = None
        entry_price = None
        entry_date = None
        stop_loss = None
        take_profit = None
        position_size = 0
        capital = initial_capital
        
        print(f"Starting backtest for {stock} with {len(df)} valid data points")
        
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
                
                # LONG entry conditions - Enhanced MACD + MA strategy
                if position is None:
                    # 1. Price must be above both moving averages (trend filter)
                    price_above_mas = close > ma_fast_val and close > ma_slow_val
                    
                    # 2. Fast MA above Slow MA (uptrend confirmation)
                    ma_bullish = ma_fast_val > ma_slow_val
                    
                    # 3. MACD bullish crossover (MACD line crosses above signal line)
                    macd_bullish_cross = prev_macd <= prev_macd_signal and macd_val > macd_signal_val
                    
                    # 4. MACD histogram turning positive (momentum confirmation)
                    macd_hist_positive = macd_hist > 0 and prev_macd_hist <= 0
                    
                    # 5. Alternative: Strong MACD momentum (histogram increasing for 2 periods)
                    macd_momentum = macd_hist > prev_macd_hist > prev2_macd_hist and macd_hist > 0
                    
                    # Entry condition: Either bullish crossover with positive histogram OR strong momentum
                    entry_signal = price_above_mas and ma_bullish and (
                        (macd_bullish_cross and macd_hist_positive) or macd_momentum
                    )
                    
                    if entry_signal:
                        position = "LONG"
                        entry_price = close
                        entry_date = df.index[i]
                        # Dynamic stop loss based on ATR
                        stop_loss = entry_price - (1.5 * atr)
                        # Take profit at 2.5:1 risk-reward ratio
                        take_profit = entry_price + (2.5 * (entry_price - stop_loss))
                        position_size = calculate_position_size(entry_price, stop_loss, capital)
                
                # SHORT entry conditions - Enhanced MACD + MA strategy
                elif position is None:
                    # 1. Price must be below both moving averages (trend filter)
                    price_below_mas = close < ma_fast_val and close < ma_slow_val
                    
                    # 2. Fast MA below Slow MA (downtrend confirmation)
                    ma_bearish = ma_fast_val < ma_slow_val
                    
                    # 3. MACD bearish crossover (MACD line crosses below signal line)
                    macd_bearish_cross = prev_macd >= prev_macd_signal and macd_val < macd_signal_val
                    
                    # 4. MACD histogram turning negative (momentum confirmation)
                    macd_hist_negative = macd_hist < 0 and prev_macd_hist >= 0
                    
                    # 5. Alternative: Strong MACD momentum (histogram decreasing for 2 periods)
                    macd_momentum = macd_hist < prev_macd_hist < prev2_macd_hist and macd_hist < 0
                    
                    # Entry condition: Either bearish crossover with negative histogram OR strong momentum
                    entry_signal = price_below_mas and ma_bearish and (
                        (macd_bearish_cross and macd_hist_negative) or macd_momentum
                    )
                    
                    if entry_signal:
                        position = "SHORT"
                        entry_price = close
                        entry_date = df.index[i]
                        # Dynamic stop loss based on ATR
                        stop_loss = entry_price + (1.5 * atr)
                        # Take profit at 2.5:1 risk-reward ratio
                        take_profit = entry_price - (2.5 * (stop_loss - entry_price))
                        position_size = calculate_position_size(entry_price, stop_loss, capital)
                
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
                        if position == "LONG":
                            pnl = (exit_price - entry_price) * position_size
                        else:
                            pnl = (entry_price - exit_price) * position_size
                        
                        trades.append([stock, entry_date, exit_date, entry_price, 
                                      exit_price, position, pnl, position_size, exit_reason])
                        
                        # Update capital
                        capital += pnl
                        position = None
                
                # Record equity curve
                equity_curve.append([df.index[i], capital])
                    
            except Exception as e:
                print(f"Error processing row {i} for {stock}: {e}")
                continue
        
        # Close any open position at the end
        if position is not None:
            exit_price = df['Close'].iloc[-1]
            exit_date = df.index[-1]
            if position == "LONG":
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size
            
            trades.append([stock, entry_date, exit_date, entry_price, 
                          exit_price, position, pnl, position_size, "END OF DATA"])
            
            # Update capital
            capital += pnl
        
        print(f"Found {len(trades)} trades for {stock}")
        print(f"Final capital for {stock}: ₹{capital:,.2f}")
        return trades
        
    except Exception as e:
        print(f"Error with {stock}: {e}")
        import traceback
        traceback.print_exc()
        return []

# Main execution
print("Starting Improved MACD + MA Strategy Backtest...")
print("=" * 60)
print("Strategy Details:")
print(f"- Fast MA: {ma_fast} periods")
print(f"- Slow MA: {ma_slow} periods") 
print(f"- MACD: {macd_fast}, {macd_slow}, {macd_signal}")
print(f"- Risk per trade: {risk_per_trade*100}%")
print(f"- Risk-Reward Ratio: 1:2.5")
print("=" * 60)

for stock in stocks:
    print(f"\nProcessing {stock}...")
    trades = backtest_stock(stock)
    all_trades.extend(trades)
    print(f"Completed {stock}: {len(trades)} trades found")
    print("-" * 40)

print("\n" + "=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)

if all_trades:
    df_trades = pd.DataFrame(all_trades, columns=[
        "Stock", "Entry_Date", "Exit_Date", "Entry_Price", "Exit_Price", 
        "Position", "PnL", "Quantity", "Exit_Reason"
    ])
    df_trades.to_csv("improved_macd_ma_trades.csv", index=False)
    print(f"\nTrades saved to improved_macd_ma_trades.csv")
    print(f"Total trades found: {len(df_trades)}")
    
    # Save equity curve
    df_equity = pd.DataFrame(equity_curve, columns=["Date", "Equity"])
    df_equity = df_equity.drop_duplicates("Date")
    df_equity.to_csv("improved_equity_curve.csv", index=False)
    print("Equity curve saved to improved_equity_curve.csv")
    
    # Summary statistics
    total_pnl = df_trades['PnL'].sum()
    winning_trades = len(df_trades[df_trades['PnL'] > 0])
    losing_trades = len(df_trades[df_trades['PnL'] < 0])
    break_even_trades = len(df_trades[df_trades['PnL'] == 0])
    win_rate = (winning_trades / len(df_trades)) * 100 if len(df_trades) > 0 else 0
    
    # Calculate risk-adjusted metrics
    avg_win = df_trades[df_trades['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
    avg_loss = abs(df_trades[df_trades['PnL'] < 0]['PnL'].mean()) if losing_trades > 0 else 0
    profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else float('inf')
    
    # Max drawdown calculation
    if not df_equity.empty:
        df_equity['Peak'] = df_equity['Equity'].cummax()
        df_equity['Drawdown'] = (df_equity['Equity'] - df_equity['Peak']) / df_equity['Peak'] * 100
        max_drawdown = df_equity['Drawdown'].min()
    else:
        max_drawdown = 0
    
    final_capital = initial_capital + total_pnl
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    # Calculate annualized return
    years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    annualized_return = ((final_capital / initial_capital) ** (1/years) - 1) * 100
    
    print(f"\nIMPROVED STRATEGY RESULTS:")
    print(f"Initial Capital: ₹{initial_capital:,.2f}")
    print(f"Final Capital: ₹{final_capital:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Return: {annualized_return:.2f}%")
    print(f"Total P&L: ₹{total_pnl:,.2f}")
    print(f"Winning trades: {winning_trades} ({win_rate:.2f}%)")
    print(f"Losing trades: {losing_trades}")
    print(f"Break-even trades: {break_even_trades}")
    print(f"Average winning trade: ₹{avg_win:,.2f}")
    print(f"Average losing trade: ₹{avg_loss:,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Risk-adjusted metrics
    if not df_equity.empty and len(df_equity) > 1:
        returns = df_equity['Equity'].pct_change().dropna()
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Yearly P&L
    df_trades['Year'] = pd.to_datetime(df_trades['Entry_Date']).dt.year
    yearly_pnl = df_trades.groupby('Year')['PnL'].sum().round(2).reset_index()
    print("\nYearly P&L:")
    print(yearly_pnl.to_string(index=False))
    
    # Stock-wise P&L
    stock_pnl = df_trades.groupby('Stock').agg({
        'PnL': ['sum', 'count', 'mean']
    }).round(2)
    stock_pnl.columns = ['Total_PnL', 'Trade_Count', 'Avg_PnL']
    stock_pnl = stock_pnl.sort_values('Total_PnL', ascending=False)
    print("\nStock-wise Performance:")
    print(stock_pnl)
    
    # Exit reason analysis
    exit_reason = df_trades.groupby('Exit_Reason').agg({
        'PnL': ['sum', 'count', 'mean']
    }).round(2)
    exit_reason.columns = ['Total_PnL', 'Trade_Count', 'Avg_PnL']
    print("\nExit Reason Analysis:")
    print(exit_reason)
    
    # Position type analysis
    position_analysis = df_trades.groupby('Position').agg({
        'PnL': ['sum', 'count', 'mean'],
    }).round(2)
    position_analysis.columns = ['Total_PnL', 'Trade_Count', 'Avg_PnL']
    print("\nLong vs Short Performance:")
    print(position_analysis)
    
else:
    print("No trades found.")

print(f"\n{'='*60}")
print("STRATEGY IMPROVEMENTS IMPLEMENTED:")
print("1. Simplified to MACD + Moving Average only (removed RSI)")
print("2. Enhanced MACD signals with histogram confirmation")
print("3. Improved MA parameters (50/200 vs 200/500)")
print("4. Better risk-reward ratio (1:2.5)")
print("5. Dynamic ATR-based stops and trailing stops")
print("6. Multiple confirmation signals for entries")
print("7. Enhanced exit conditions with signal reversal")
print("8. Increased position sizing (1.5% risk)")
print(f"{'='*60}")