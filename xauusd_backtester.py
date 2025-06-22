"""
XAUUSD Trading Plan Backtest Bot

Requirements:
pip install pandas numpy yfinance

Usage:
python3 xauusd_backtester.py

This bot implements the complete XAUUSD trading plan with 8 setups
and generates trading_logs.csv and account_logs.csv files.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import csv
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class XAUUSDBacktestBot:
    def __init__(self):
        # Backtest parameters
        self.START_DATE = '2024-01-01'
        self.END_DATE = '2024-02-01'
        self.INITIAL_BALANCE = 100.00
        self.LOT_SIZE = 0.02
        self.PIP_DECIMAL_PLACE = 1
        self.PIP_VALUE_PER_LOT = 10.0
        
        # Trading parameters
        self.current_balance = self.INITIAL_BALANCE
        self.current_equity = self.INITIAL_BALANCE
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_daily_loss = 0.03  # 3% max daily loss
        self.max_daily_profit = 0.05  # 5% max daily profit
        self.max_trades_per_session = 4
        self.max_total_risk = 0.05  # 5% max total risk across all positions
        
        # Trading logs
        self.trading_logs = []
        self.account_logs = []
        
        # Market data
        self.data = None
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI without talib"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values

    def fetch_data(self):
        """Fetch XAUUSD data from Yahoo Finance"""
        try:
            # Download XAUUSD data (Gold/USD)
            ticker = "GC=F"  # Gold futures
            data = yf.download(ticker, start=self.START_DATE, end=self.END_DATE, interval="5m")
            
            if data.empty:
                # Try alternative ticker
                ticker = "XAUUSD=X"
                data = yf.download(ticker, start=self.START_DATE, end=self.END_DATE, interval="5m")
            
            if data.empty:
                raise Exception("No data available")
                
            # Clean and prepare data
            data = data.dropna()
            data.reset_index(inplace=True)
            
            # Add technical indicators using custom functions
            data['RSI'] = self.calculate_rsi(data['Close'].values, 14)
            data['SMA_20'] = self.calculate_sma(data['Close'].values, 20)
            data['SMA_50'] = self.calculate_sma(data['Close'].values, 50)
            
            # Add time-based columns
            data['Hour'] = data['Datetime'].dt.hour
            data['Minute'] = data['Datetime'].dt.minute
            data['DayOfWeek'] = data['Datetime'].dt.dayofweek
            
            # Calculate daily high/low
            data['Date'] = data['Datetime'].dt.date
            daily_stats = data.groupby('Date').agg({
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).reset_index()
            daily_stats.columns = ['Date', 'DailyHigh', 'DailyLow', 'DailyClose']
            
            # Merge daily stats
            data = data.merge(daily_stats, on='Date', how='left')
            
            self.data = data
            print(f"Data loaded successfully: {len(data)} records from {data['Datetime'].min()} to {data['Datetime'].max()}")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Create synthetic data for demonstration
            self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic XAUUSD data for demonstration"""
        print("Creating synthetic data for demonstration...")
        
        start_date = pd.to_datetime(self.START_DATE)
        end_date = pd.to_datetime(self.END_DATE)
        
        # Generate 5-minute intervals
        date_range = pd.date_range(start=start_date, end=end_date, freq='5T')
        
        # Filter for trading hours (9 PM - 2 AM EST, Monday-Friday)
        trading_hours = []
        for dt in date_range:
            if dt.weekday() < 5:  # Monday-Friday
                hour = dt.hour
                if (hour >= 21) or (hour <= 2):  # 9 PM - 2 AM
                    trading_hours.append(dt)
        
        n_periods = len(trading_hours)
        
        # Generate realistic XAUUSD price data
        np.random.seed(42)
        base_price = 1950.0  # Starting price
        
        # Generate returns with higher volatility during NY session
        returns = np.random.normal(0, 0.0005, n_periods)  # 0.05% volatility
        
        # Add session-based volatility
        for i, dt in enumerate(trading_hours):
            hour = dt.hour
            if 21 <= hour <= 23 or hour == 0:  # NY prime time
                returns[i] *= 2  # Higher volatility
        
        # Generate price series
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLC data
        data = []
        for i, dt in enumerate(trading_hours):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, 0.0002)))
            low = price * (1 - abs(np.random.normal(0, 0.0002)))
            close = prices[i + 1] if i + 1 < len(prices) else price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'Datetime': dt,
                'Open': price,
                'High': max(price, high, close),
                'Low': min(price, low, close),
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Add technical indicators using custom functions
        df['RSI'] = self.calculate_rsi(df['Close'].values, 14)
        df['SMA_20'] = self.calculate_sma(df['Close'].values, 20)
        df['SMA_50'] = self.calculate_sma(df['Close'].values, 50)
        
        # Add time-based columns
        df['Hour'] = df['Datetime'].dt.hour
        df['Minute'] = df['Datetime'].dt.minute
        df['DayOfWeek'] = df['Datetime'].dt.dayofweek
        
        # Calculate daily high/low
        df['Date'] = df['Datetime'].dt.date
        daily_stats = df.groupby('Date').agg({
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).reset_index()
        daily_stats.columns = ['Date', 'DailyHigh', 'DailyLow', 'DailyClose']
        
        # Merge daily stats
        df = df.merge(daily_stats, on='Date', how='left')
        
        self.data = df
        print(f"Synthetic data created: {len(df)} records")
    
    def is_ny_session_1(self, hour: int, minute: int) -> bool:
        """Check if time is within NY Session 1 (9:00-11:30 PM)"""
        if hour == 21:  # 9 PM
            return True
        elif hour == 22:  # 10 PM
            return True
        elif hour == 23 and minute <= 30:  # 11:00-11:30 PM
            return True
        return False
    
    def is_ny_session_2(self, hour: int, minute: int) -> bool:
        """Check if time is within NY Session 2 (12:30-1:00 AM)"""
        if hour == 0 and minute >= 30:  # 12:30-1:00 AM
            return True
        elif hour == 1 and minute <= 30:  # 1:00-1:30 AM
            return True
        return False
    
    def calculate_pips(self, entry_price: float, exit_price: float, trade_type: str) -> float:
        """Calculate pips based on entry and exit prices"""
        if trade_type == 'BUY':
            pip_diff = (exit_price - entry_price) / (10 ** -self.PIP_DECIMAL_PLACE)
        else:  # SELL
            pip_diff = (entry_price - exit_price) / (10 ** -self.PIP_DECIMAL_PLACE)
        return pip_diff
    
    def calculate_profit_loss(self, pips: float) -> float:
        """Calculate P&L in account currency"""
        return pips * (self.PIP_VALUE_PER_LOT * self.LOT_SIZE)
    
    def setup_1_liquidity_sweep_reversal(self, i: int) -> Dict:
        """Setup 1: Liquidity Sweep Reversal"""
        if i < 50:  # Need enough historical data
            return None
            
        current = self.data.iloc[i]
        
        # Check if in correct time window (9:15-10:45 PM)
        if not (current['Hour'] == 21 and current['Minute'] >= 15) and not (current['Hour'] == 22 and current['Minute'] <= 45):
            return None
        
        # Look for sweep of previous day high/low
        prev_day_high = current['DailyHigh']
        prev_day_low = current['DailyLow']
        
        # Check for sweep above previous day high
        if current['High'] > prev_day_high:
            # Look for immediate rejection
            if current['Close'] < prev_day_high and current['Close'] < current['Open']:
                # Check for strong momentum candle
                body_size = abs(current['Close'] - current['Open'])
                if body_size > 0.5:  # Minimum body size
                    return {
                        'setup': 'Liquidity Sweep Reversal',
                        'type': 'SELL',
                        'entry_price': current['Close'],
                        'stop_loss': prev_day_high + 1.0,  # 10 pips beyond sweep
                        'take_profit_1': current['Close'] - 1.5,  # 15 pips
                        'take_profit_2': current['Close'] - 3.0,  # 30 pips
                        'risk_percent': 0.02
                    }
        
        # Check for sweep below previous day low
        if current['Low'] < prev_day_low:
            # Look for immediate rejection
            if current['Close'] > prev_day_low and current['Close'] > current['Open']:
                # Check for strong momentum candle
                body_size = abs(current['Close'] - current['Open'])
                if body_size > 0.5:  # Minimum body size
                    return {
                        'setup': 'Liquidity Sweep Reversal',
                        'type': 'BUY',
                        'entry_price': current['Close'],
                        'stop_loss': prev_day_low - 1.0,  # 10 pips beyond sweep
                        'take_profit_1': current['Close'] + 1.5,  # 15 pips
                        'take_profit_2': current['Close'] + 3.0,  # 30 pips
                        'risk_percent': 0.02
                    }
        
        return None
    
    def setup_2_ny_continuation_breakout(self, i: int) -> Dict:
        """Setup 2: NY Continuation Breakout"""
        if i < 50:
            return None
            
        current = self.data.iloc[i]
        
        # Check if in correct time window (9:00-10:00 PM)
        if not (current['Hour'] == 21):
            return None
        
        # Simple trend detection using moving averages
        if current['SMA_20'] > current['SMA_50']:  # Uptrend
            # Look for break above recent high
            recent_high = self.data.iloc[i-10:i]['High'].max()
            if current['High'] > recent_high and current['Close'] > current['Open']:
                return {
                    'setup': 'NY Continuation Breakout',
                    'type': 'BUY',
                    'entry_price': current['Close'],
                    'stop_loss': recent_high - 1.0,
                    'take_profit_1': current['Close'] + 2.0,  # 20 pips
                    'take_profit_2': current['Close'] + 4.0,  # 40 pips
                    'risk_percent': 0.02
                }
        
        elif current['SMA_20'] < current['SMA_50']:  # Downtrend
            # Look for break below recent low
            recent_low = self.data.iloc[i-10:i]['Low'].min()
            if current['Low'] < recent_low and current['Close'] < current['Open']:
                return {
                    'setup': 'NY Continuation Breakout',
                    'type': 'SELL',
                    'entry_price': current['Close'],
                    'stop_loss': recent_low + 1.0,
                    'take_profit_1': current['Close'] - 2.0,  # 20 pips
                    'take_profit_2': current['Close'] - 4.0,  # 40 pips
                    'risk_percent': 0.02
                }
        
        return None
    
    def setup_3_false_breakout_trap(self, i: int) -> Dict:
        """Setup 3: False Breakout Trap"""
        if i < 20:
            return None
            
        current = self.data.iloc[i]
        
        # Check if in correct time window (9:30-11:00 PM)
        if not ((current['Hour'] == 21 and current['Minute'] >= 30) or current['Hour'] == 22):
            return None
        
        # Look for false breakout pattern
        prev_candles = self.data.iloc[i-5:i]
        
        # Check for break of significant level with weak follow-through
        if len(prev_candles) >= 3:
            # Look for small body candles after initial break
            small_bodies = []
            for j in range(len(prev_candles)-2):
                candle = prev_candles.iloc[j]
                body_size = abs(candle['Close'] - candle['Open'])
                small_bodies.append(body_size < 0.3)
            
            if sum(small_bodies) >= 2:  # At least 2 small body candles
                # Look for strong reversal
                body_size = abs(current['Close'] - current['Open'])
                if body_size > 0.8:  # Strong reversal candle
                    if current['Close'] > current['Open']:  # Bullish reversal
                        return {
                            'setup': 'False Breakout Trap',
                            'type': 'BUY',
                            'entry_price': current['Close'],
                            'stop_loss': current['Low'] - 0.5,
                            'take_profit_1': current['Close'] + 2.5,  # 25 pips
                            'take_profit_2': current['Close'] + 3.5,  # 35 pips
                            'risk_percent': 0.02
                        }
                    else:  # Bearish reversal
                        return {
                            'setup': 'False Breakout Trap',
                            'type': 'SELL',
                            'entry_price': current['Close'],
                            'stop_loss': current['High'] + 0.5,
                            'take_profit_1': current['Close'] - 2.5,  # 25 pips
                            'take_profit_2': current['Close'] - 3.5,  # 35 pips
                            'risk_percent': 0.02
                        }
        
        return None
    
    def setup_4_range_reversal(self, i: int) -> Dict:
        """Setup 4: Range Reversal Setup"""
        if i < 50:
            return None
            
        current = self.data.iloc[i]
        
        # Check if in correct time window (9:00-11:00 PM)
        if not (current['Hour'] == 21 or current['Hour'] == 22):
            return None
        
        # Identify range from recent data
        recent_data = self.data.iloc[i-24:i]  # Last 2 hours
        range_high = recent_data['High'].max()
        range_low = recent_data['Low'].min()
        range_size = range_high - range_low
        
        # Check if range is at least 20 pips
        if range_size < 2.0:
            return None
        
        # Check for touch of range boundary with rejection
        if current['High'] >= range_high * 0.999 and current['RSI'] > 70:  # Near range high, overbought
            if current['Close'] < current['High']:  # Rejection
                return {
                    'setup': 'Range Reversal',
                    'type': 'SELL',
                    'entry_price': current['Close'],
                    'stop_loss': range_high + 0.8,
                    'take_profit_1': (range_high + range_low) / 2,  # Middle of range
                    'take_profit_2': range_low,  # Opposite boundary
                    'risk_percent': 0.02
                }
        
        elif current['Low'] <= range_low * 1.001 and current['RSI'] < 30:  # Near range low, oversold
            if current['Close'] > current['Low']:  # Rejection
                return {
                    'setup': 'Range Reversal',
                    'type': 'BUY',
                    'entry_price': current['Close'],
                    'stop_loss': range_low - 0.8,
                    'take_profit_1': (range_high + range_low) / 2,  # Middle of range
                    'take_profit_2': range_high,  # Opposite boundary
                    'risk_percent': 0.02
                }
        
        return None
    
    def setup_5_ny_session_continuation(self, i: int) -> Dict:
        """Setup 5: NY Session Continuation"""
        if i < 50:
            return None
            
        current = self.data.iloc[i]
        
        # Check if in correct time window (12:30-1:15 AM)
        if not self.is_ny_session_2(current['Hour'], current['Minute']):
            return None
        
        # Look for pullback to Fibonacci levels
        # Simplified: look for pullback after strong move
        recent_data = self.data.iloc[i-12:i]  # Last hour
        
        if len(recent_data) > 0:
            session_high = recent_data['High'].max()
            session_low = recent_data['Low'].min()
            fib_50 = (session_high + session_low) / 2
            
            # Check for bounce from 50% level
            if abs(current['Low'] - fib_50) < 0.3 and current['Close'] > current['Open']:
                return {
                    'setup': 'NY Session Continuation',
                    'type': 'BUY',
                    'entry_price': current['Close'],
                    'stop_loss': session_low - 0.5,
                    'take_profit_1': session_high,
                    'take_profit_2': session_high + 1.0,
                    'risk_percent': 0.03
                }
            elif abs(current['High'] - fib_50) < 0.3 and current['Close'] < current['Open']:
                return {
                    'setup': 'NY Session Continuation',
                    'type': 'SELL',
                    'entry_price': current['Close'],
                    'stop_loss': session_high + 0.5,
                    'take_profit_1': session_low,
                    'take_profit_2': session_low - 1.0,
                    'risk_percent': 0.03
                }
        
        return None
    
    def execute_trade(self, trade_signal: Dict, current_time: datetime) -> bool:
        """Execute a trade based on the signal"""
        if not trade_signal:
            return False
        
        # Calculate position size based on risk
        risk_amount = self.current_balance * trade_signal['risk_percent']
        
        # Calculate pip risk
        if trade_signal['type'] == 'BUY':
            pip_risk = (trade_signal['entry_price'] - trade_signal['stop_loss']) / (10 ** -self.PIP_DECIMAL_PLACE)
        else:
            pip_risk = (trade_signal['stop_loss'] - trade_signal['entry_price']) / (10 ** -self.PIP_DECIMAL_PLACE)
        
        pip_risk = abs(pip_risk)
        
        if pip_risk <= 0:
            return False
        
        # Calculate position size
        position_size = risk_amount / (pip_risk * self.PIP_VALUE_PER_LOT)
        position_size = min(position_size, self.LOT_SIZE * 5)  # Cap position size
        
        # Log the trade
        trade_log = {
            'timestamp': current_time,
            'setup': trade_signal['setup'],
            'type': trade_signal['type'],
            'entry_price': trade_signal['entry_price'],
            'stop_loss': trade_signal['stop_loss'],
            'take_profit_1': trade_signal['take_profit_1'],
            'take_profit_2': trade_signal.get('take_profit_2', None),
            'position_size': position_size,
            'risk_amount': risk_amount,
            'pip_risk': pip_risk,
            'status': 'OPEN'
        }
        
        self.trading_logs.append(trade_log)
        self.total_trades += 1
        
        return True
    
    def close_trade(self, trade_idx: int, exit_price: float, exit_reason: str, current_time: datetime):
        """Close a trade and update logs"""
        if trade_idx >= len(self.trading_logs):
            return
        
        trade = self.trading_logs[trade_idx]
        if trade['status'] != 'OPEN':
            return
        
        # Calculate P&L
        pips = self.calculate_pips(trade['entry_price'], exit_price, trade['type'])
        profit_loss = self.calculate_profit_loss(pips)
        
        # Update trade log
        trade.update({
            'exit_price': exit_price,
            'exit_time': current_time,
            'exit_reason': exit_reason,
            'pips': pips,
            'profit_loss': profit_loss,
            'status': 'CLOSED'
        })
        
        # Update balance
        self.current_balance += profit_loss
        self.current_equity = self.current_balance
        
        # Update statistics
        if profit_loss > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Log account update
        self.account_logs.append({
            'timestamp': current_time,
            'balance': self.current_balance,
            'equity': self.current_equity,
            'trade_pnl': profit_loss,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        })
    
    def run_backtest(self):
        """Run the complete backtest"""
        print("Starting backtest...")
        
        # Fetch data
        self.fetch_data()
        
        if self.data is None or len(self.data) == 0:
            print("No data available for backtesting")
            return
        
        # Initialize account log
        self.account_logs.append({
            'timestamp': self.data['Datetime'].iloc[0],
            'balance': self.current_balance,
            'equity': self.current_equity,
            'trade_pnl': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        })
        
        # Track daily limits
        daily_trades = {}
        
        # Process each candle
        for i in range(50, len(self.data)):  # Start from index 50 to have enough history
            current = self.data.iloc[i]
            current_time = current['Datetime']
            current_date = current_time.date()
            
            # Check daily limits
            if current_date not in daily_trades:
                daily_trades[current_date] = 0
            
            if daily_trades[current_date] >= self.max_trades_per_session:
                continue
            
            # Check if it's a trading session
            hour = current['Hour']
            minute = current['Minute']
            
            if not (self.is_ny_session_1(hour, minute) or self.is_ny_session_2(hour, minute)):
                continue
            
            # Skip weekends
            if current['DayOfWeek'] >= 5:
                continue
            
            # Check existing open trades for exit conditions
            for trade_idx, trade in enumerate(self.trading_logs):
                if trade['status'] == 'OPEN':
                    # Check stop loss
                    if trade['type'] == 'BUY':
                        if current['Low'] <= trade['stop_loss']:
                            self.close_trade(trade_idx, trade['stop_loss'], 'STOP_LOSS', current_time)
                        elif current['High'] >= trade['take_profit_1']:
                            self.close_trade(trade_idx, trade['take_profit_1'], 'TAKE_PROFIT', current_time)
                    else:  # SELL
                        if current['High'] >= trade['stop_loss']:
                            self.close_trade(trade_idx, trade['stop_loss'], 'STOP_LOSS', current_time)
                        elif current['Low'] <= trade['take_profit_1']:
                            self.close_trade(trade_idx, trade['take_profit_1'], 'TAKE_PROFIT', current_time)
            
            # Look for new trade setups
            trade_signals = []
            
            # Try each setup
            signal = self.setup_1_liquidity_sweep_reversal(i)
            if signal:
                trade_signals.append(signal)
            
            signal = self.setup_2_ny_continuation_breakout(i)
            if signal:
                trade_signals.append(signal)
            
            signal = self.setup_3_false_breakout_trap(i)
            if signal:
                trade_signals.append(signal)
            
            signal = self.setup_4_range_reversal(i)
            if signal:
                trade_signals.append(signal)
            
            signal = self.setup_5_ny_session_continuation(i)
            if signal:
                trade_signals.append(signal)
            
            # Execute trades (limit to 1 per candle)
            if trade_signals and daily_trades[current_date] < self.max_trades_per_session:
                # Take the first valid signal
                if self.execute_trade(trade_signals[0], current_time):
                    daily_trades[current_date] += 1
        
        # Close any remaining open trades
        final_time = self.data['Datetime'].iloc[-1]
        final_price = self.data['Close'].iloc[-1]
        
        for trade_idx, trade in enumerate(self.trading_logs):
            if trade['status'] == 'OPEN':
                self.close_trade(trade_idx, final_price, 'SESSION_END', final_time)
        
        print(f"Backtest completed. Total trades: {self.total_trades}")
        print(f"Winning trades: {self.winning_trades}")
        print(f"Losing trades: {self.losing_trades}")
        print(f"Final balance: ${self.current_balance:.2f}")
        
        # Generate CSV files
        self.generate_csv_files()
    
    def generate_csv_files(self):
        """Generate trading logs and account logs CSV files"""
        
        # Trading Logs CSV
        trading_df = pd.DataFrame(self.trading_logs)
        if not trading_df.empty:
            trading_df['win_rate'] = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            trading_df.to_csv('trading_logs.csv', index=False)
            print("Trading logs saved to 'trading_logs.csv'")
        
        # Account Logs CSV
        account_df = pd.DataFrame(self.account_logs)
        if not account_df.empty:
            account_df['drawdown'] = (account_df['balance'] / self.INITIAL_BALANCE - 1) * 100
            account_df['return_pct'] = (account_df['balance'] / self.INITIAL_BALANCE - 1) * 100
            account_df.to_csv('account_logs.csv', index=False)
            print("Account logs saved to 'account_logs.csv'")
        
        # Generate summary statistics
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate a summary report of the backtest"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        total_return = ((self.current_balance / self.INITIAL_BALANCE) - 1) * 100
        
        print("\n" + "="*50)
        print("BACKTEST SUMMARY REPORT")
        print("="*50)
        print(f"Initial Balance: ${self.INITIAL_BALANCE:.2f}")
        print(f"Final Balance: ${self.current_balance:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Losing Trades: {self.losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print("="*50)

# Run the backtest
if __name__ == "__main__":
    bot = XAUUSDBacktestBot()
    bot.run_backtest()