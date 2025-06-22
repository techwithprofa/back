import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

class FVGBacktestBot:
    def __init__(self, initial_balance=30):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.lot_size = 0.02
        self.max_trades_per_day = 4
        self.max_sl_pips = 7
        
        # Trade tracking
        self.trades = []
        self.daily_trades = 0
        self.current_trade = None
        self.equity_curve = []

    def calculate_pip_value_for_trade(self, lot_size):
        return lot_size * 10
    
    def identify_fvg(self, df, index):
        if index < 2 or index >= len(df) - 1: return None
        candle1, candle3 = df.iloc[index - 2], df.iloc[index]
        if candle1['High'] < candle3['Low']:
            return {'type': 'bullish', 'top': candle3['Low'], 'bottom': candle1['High'], 'index': index}
        if candle1['Low'] > candle3['High']:
            return {'type': 'bearish', 'top': candle1['Low'], 'bottom': candle3['High'], 'index': index}
        return None
    
    def is_fvg_breakout(self, df, fvg, index):
        """NEW: Checks for initial entry signal (breakout in direction of FVG)"""
        if not fvg or index <= fvg['index']: return False
        current_candle = df.iloc[index]
        if fvg['type'] == 'bullish':
            return current_candle['High'] > fvg['top']
        else: # Bearish
            return current_candle['Low'] < fvg['bottom']

    def is_fvg_failed(self, df, fvg, index):
        """Checks for reversal signal (failure against direction of FVG)"""
        if not fvg or index <= fvg['index']: return False
        current_candle = df.iloc[index]
        if fvg['type'] == 'bullish':
            return current_candle['Close'] < fvg['bottom']
        else: # Bearish
            return current_candle['Close'] > fvg['top']
    
    def calculate_sl_distance(self, df, entry_index, trade_type):
        lookback, start_idx = 10, max(0, entry_index - 10)
        recent_data = df.iloc[start_idx:entry_index + 1]
        entry_price = df.iloc[entry_index]['Close']
        if trade_type == 'buy':
            sl_distance = abs(entry_price - recent_data['Low'].min())
        else:
            sl_distance = abs(recent_data['High'].max() - entry_price)
        sl_pips = sl_distance / 0.1
        return max(1, min(sl_pips, self.max_sl_pips))
    
    def open_trade(self, df, index, trade_type, reason):
        if self.daily_trades >= self.max_trades_per_day: return False
        entry_price = df.iloc[index]['Close']
        sl_pips = self.calculate_sl_distance(df, index, trade_type)
        pip_price_move = 0.1
        
        if trade_type == 'buy':
            sl_price = entry_price - (sl_pips * pip_price_move)
            tp1_price = entry_price + (sl_pips * 2 * pip_price_move)
        else:
            sl_price = entry_price + (sl_pips * pip_price_move)
            tp1_price = entry_price - (sl_pips * 2 * pip_price_move)
            
        tp2_price = entry_price + (25 * pip_price_move) if trade_type == 'buy' else entry_price - (25 * pip_price_move)
        
        self.current_trade = {
            'entry_time': df.iloc[index]['Date'], 'entry_price': entry_price, 'trade_type': trade_type,
            'lot_size': self.lot_size, 'sl_price': sl_price, 'tp1_price': tp1_price, 'tp2_price': tp2_price,
            'sl_pips': sl_pips, 'status': 'open', 'tp1_hit': False, 'reason': reason, 'entry_index': index
        }
        self.daily_trades += 1
        return True
    
    def update_trade(self, df, index):
        if not self.current_trade or self.current_trade['status'] != 'open': return
        high_price, low_price = df.iloc[index]['High'], df.iloc[index]['Low']
        trade = self.current_trade
        
        if trade['trade_type'] == 'buy':
            if low_price <= trade['sl_price']:
                self.close_trade(df, index, trade['sl_price'], 'SL Hit'); return
            if not trade['tp1_hit'] and high_price >= trade['tp1_price']:
                trade['tp1_hit'] = True; trade['sl_price'] = trade['entry_price']
                print(f"  > Trade Alert: TP1 hit for BUY. SL moved to breakeven ({trade['entry_price']:.2f}).")
            if trade['tp1_hit'] and high_price >= trade['tp2_price']:
                self.close_trade(df, index, trade['tp2_price'], 'TP2 Hit')
        else:
            if high_price >= trade['sl_price']:
                self.close_trade(df, index, trade['sl_price'], 'SL Hit'); return
            if not trade['tp1_hit'] and low_price <= trade['tp1_price']:
                trade['tp1_hit'] = True; trade['sl_price'] = trade['entry_price']
                print(f"  > Trade Alert: TP1 hit for SELL. SL moved to breakeven ({trade['entry_price']:.2f}).")
            if trade['tp1_hit'] and low_price <= trade['tp2_price']:
                self.close_trade(df, index, trade['tp2_price'], 'TP2 Hit')
    
    def close_trade(self, df, index, exit_price, reason):
        if not self.current_trade: return
        trade = self.current_trade
        pips = ((exit_price - trade['entry_price']) if trade['trade_type'] == 'buy' else (trade['entry_price'] - exit_price)) / 0.1
        profit = pips * self.calculate_pip_value_for_trade(trade['lot_size'])
        self.balance += profit
        trade.update({
            'exit_time': df.iloc[index]['Date'], 'exit_price': exit_price, 'pips': pips, 'profit': profit,
            'status': 'closed', 'exit_reason': reason, 'balance_after': self.balance
        })
        self.trades.append(trade.copy())
        print(f"  > Trade Closed: {reason}. Pips: {pips:.1f}, P/L: ${profit:.2f}, New Balance: ${self.balance:.2f}")
        self.current_trade = None
    
    def is_trading_session(self, timestamp):
        return timestamp.hour >= 20 or timestamp.hour < 18
    
    def reset_daily_counters(self, current_time, last_time):
        if current_time.date() != last_time.date():
            self.daily_trades = 0; return True
        return False
    
    def run_backtest(self, df):
        print("Starting FVG Backtest (v4 - Breakout/Reversal Logic)...")
        print(f"Initial Balance: ${self.initial_balance}")
        print("-" * 60)
        
        all_fvgs = []
        last_time = df.iloc[0]['Date']
        
        for i in range(10, len(df)):
            current_time = df.iloc[i]['Date']
            
            if self.reset_daily_counters(current_time, last_time):
                print(f"\n--- New Trading Day: {current_time.date()} --- Counters Reset ---")
            last_time = current_time
            
            if not self.is_trading_session(current_time):
                if self.current_trade: self.close_trade(df, i, df.iloc[i]['Close'], 'Session End')
                continue
            
            # --- CORE LOGIC ---
            fvg = self.identify_fvg(df, i)
            if fvg: all_fvgs.append(fvg)
            
            self.update_trade(df, i)

            # A. Check for REVERSAL if a trade is open
            if self.current_trade:
                # The key Two-Mode Logic: Reversals are always active for trades 1&2.
                # For trades 3&4, they are only active BEFORE TP1 is hit.
                can_be_reversed = (self.daily_trades <= 2) or \
                                  (self.daily_trades > 2 and not self.current_trade['tp1_hit'])
                
                if can_be_reversed:
                    for fvg_to_check in all_fvgs[-5:]:
                        if fvg_to_check['index'] > self.current_trade['entry_index'] and self.is_fvg_failed(df, fvg_to_check, i):
                            current_trade_type = self.current_trade['trade_type']
                            signal_type = 'sell' if fvg_to_check['type'] == 'bullish' else 'buy'
                            
                            if signal_type != current_trade_type: # Ensure it's an opposite signal
                                print(f"Reversal Signal on {current_time}: Closing current {current_trade_type} trade.")
                                self.close_trade(df, i, df.iloc[i]['Close'], 'Reversal Signal')
                                if self.daily_trades < self.max_trades_per_day:
                                    reason = f"Reversal on FVG {fvg_to_check['type']} fail"
                                    if self.open_trade(df, i, signal_type, reason):
                                        print(f"Trade {self.daily_trades} (Reversal) opened: {signal_type.upper()} @ {df.iloc[i]['Close']:.2f}")
                                break # Exit loop once reversal is actioned

            # B. Check for a NEW trade (breakout) if no trade is open
            elif not self.current_trade and self.daily_trades < self.max_trades_per_day:
                for fvg_to_check in all_fvgs[-10:]:
                    if not fvg_to_check.get('triggered', False) and self.is_fvg_breakout(df, fvg_to_check, i):
                        trade_type = 'buy' if fvg_to_check['type'] == 'bullish' else 'sell'
                        reason = f"Breakout of {fvg_to_check['type']} FVG"
                        if self.open_trade(df, i, trade_type, reason):
                            print(f"Trade {self.daily_trades} opened: {trade_type.upper()} @ {df.iloc[i]['Close']:.2f}")
                            fvg_to_check['triggered'] = True # Mark FVG as used
                            break # Exit loop once trade is opened
            
            # Track equity curve
            current_equity = self.balance
            if self.current_trade:
                trade, current_price = self.current_trade, df.iloc[i]['Close']
                pips = ((current_price - trade['entry_price']) if trade['trade_type'] == 'buy' else (trade['entry_price'] - current_price)) / 0.1
                current_equity += pips * self.calculate_pip_value_for_trade(trade['lot_size'])
            self.equity_curve.append({'Date': current_time, 'Balance': self.balance, 'Equity': current_equity})

        if self.current_trade: self.close_trade(df, len(df)-1, df.iloc[-1]['Close'], 'End of Data')
        self.print_results()
        return self.trades
    
    def print_results(self):
        print("\n" + "="*60 + "\nBACKTEST RESULTS\n" + "="*60)
        if not self.trades:
            print("No trades executed!")
            print(f"Final Balance: ${self.balance:.2f}")
            return
            
        df_trades = pd.DataFrame(self.trades)
        total_trades, winning_trades = len(df_trades), len(df_trades[df_trades['profit'] > 0])
        losing_trades = len(df_trades[df_trades['profit'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_profit = df_trades['profit'].sum()
        total_gains = df_trades[df_trades['profit'] > 0]['profit'].sum()
        total_losses = abs(df_trades[df_trades['profit'] < 0]['profit'].sum())
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        
        max_drawdown = 0
        peak = self.initial_balance
        for eq in self.equity_curve:
            peak = max(peak, eq['Equity'])
            drawdown = (peak - eq['Equity']) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        print(f"Initial Balance: ${self.initial_balance:.2f} | Final Balance: ${self.balance:.2f}")
        print(f"Total P/L: ${total_profit:.2f} | Return: {(total_profit/self.initial_balance*100):.2f}%")
        print(f"Total Trades: {total_trades} | Wins: {winning_trades} | Losses: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}% | Profit Factor: {profit_factor:.2f} | Max Drawdown: {max_drawdown:.2f}%")
        print("="*60)


def main():
    try:
        df = pd.read_csv('XAU_5m_data_2024_filtered.csv', sep=';')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
        print(f"Loaded {len(df)} rows of data.")
    except FileNotFoundError:
        print("Error: XAU_5m_data_2024_filtered.csv not found!")
        return
    
    bot = FVGBacktestBot(initial_balance=30)
    trades = bot.run_backtest(df)
    
    if trades:
        results_df = pd.DataFrame(trades)
        results_df.to_csv('fvg_backtest_results_v4.csv', index=False)
        print(f"\nResults saved to: fvg_backtest_results_v4.csv")

if __name__ == "__main__":
    main()