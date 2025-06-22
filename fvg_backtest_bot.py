import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

class FVGBacktestBotV14:
    def __init__(self, initial_balance=30):
        # ... constructor is identical ...
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.initial_lot_size = 0.02
        self.max_trades_per_day = 4
        self.max_sl_pips = 7
        self.swing_lookback = 15
        self.trades = []
        self.daily_trades = 0
        self.current_trade = None
        self.equity_curve = []
        self.setup_fvgs = []
        ## NEW: To track reversal signals
        self.consecutive_reversal_signals = 0
        self.last_reversal_signal_type = None

    # ... All helper functions up to run_backtest are identical to v13 ...
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

    def is_fvg_failure_confirmed(self, df, fvg, current_index):
        if not fvg or current_index <= fvg['index'] + 1: return False
        confirmation_candle = df.iloc[current_index - 1]
        if fvg['type'] == 'bullish': return confirmation_candle['Close'] < fvg['bottom']
        else: return confirmation_candle['Close'] > fvg['top']

    def find_swing_point(self, df, check_index, trade_type):
        lookback_data = df.iloc[max(0, check_index - self.swing_lookback) : check_index]
        if trade_type == 'buy':
            lows = lookback_data['Low']
            for i in range(len(lows) - 2, 0, -1):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]: return lows.iloc[i]
            return lookback_data['Low'].min()
        else:
            highs = lookback_data['High']
            for i in range(len(highs) - 2, 0, -1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]: return highs.iloc[i]
            return lookback_data['High'].max()

    def calculate_sl_distance(self, df, entry_index, trade_type):
        entry_price = df.iloc[entry_index]['Open']
        swing_point = self.find_swing_point(df, entry_index, trade_type)
        if swing_point is None: return self.max_sl_pips
        if trade_type == 'buy': sl_distance = abs(entry_price - swing_point)
        else: sl_distance = abs(swing_point - entry_price)
        sl_pips = sl_distance / 0.1
        return max(1, min(sl_pips, self.max_sl_pips))
    
    def open_trade(self, df, index, trade_type, reason):
        if self.daily_trades >= self.max_trades_per_day: return False
        entry_price = df.iloc[index]['Open']
        sl_pips = self.calculate_sl_distance(df, index, trade_type)
        pip_price_move = 0.1
        if trade_type == 'buy':
            sl_price, tp1_price = entry_price - (sl_pips * pip_price_move), entry_price + (sl_pips * 2 * pip_price_move)
        else:
            sl_price, tp1_price = entry_price + (sl_pips * pip_price_move), entry_price - (sl_pips * 2 * pip_price_move)
        tp2_price = entry_price + (25 * pip_price_move) if trade_type == 'buy' else entry_price - (25 * pip_price_move)
        self.daily_trades += 1
        self.current_trade = {'trade_num_daily': self.daily_trades, 'entry_time': df.iloc[index]['Date'], 'entry_price': entry_price, 
                              'trade_type': trade_type, 'lot_size': self.initial_lot_size, 'sl_price': sl_price, 'tp1_price': tp1_price, 
                              'tp2_price': tp2_price, 'sl_pips': sl_pips, 'status': 'open', 'tp1_hit': False, 'reason': reason, 'entry_index': index}
        ## NEW: Reset reversal counter after a new trade is opened
        self.consecutive_reversal_signals = 0
        self.last_reversal_signal_type = None
        return True

    def manage_trade(self, df, index):
        if not self.current_trade: return
        high_price, low_price = df.iloc[index]['High'], df.iloc[index]['Low']
        trade = self.current_trade
        if (trade['trade_type'] == 'buy' and low_price <= trade['sl_price']) or (trade['trade_type'] == 'sell' and high_price >= trade['sl_price']):
            self.close_trade(df, index, trade['sl_price'], 'SL Hit', trade['lot_size'])
            return
        if not trade['tp1_hit']:
            if (trade['trade_type'] == 'buy' and high_price >= trade['tp1_price']) or (trade['trade_type'] == 'sell' and low_price <= trade['tp1_price']):
                self.close_trade(df, index, trade['tp1_price'], 'TP1 Hit (Partial)', self.initial_lot_size / 2)
                trade.update({'tp1_hit': True, 'lot_size': self.initial_lot_size / 2, 'sl_price': trade['entry_price']})
                print(f"  > Alert: TP1 hit. SL for remaining {trade['lot_size']} lots moved to breakeven.")
        if trade['tp1_hit']:
            if (trade['trade_type'] == 'buy' and high_price >= trade['tp2_price']) or (trade['trade_type'] == 'sell' and low_price <= trade['tp2_price']):
                self.close_trade(df, index, trade['tp2_price'], 'TP2 Hit', trade['lot_size'])

    def close_trade(self, df, index, exit_price, reason, lot_size_to_close):
        if not self.current_trade: return
        trade = self.current_trade
        exit_price_actual = df.iloc[index]['Close'] if 'Hit' not in reason else exit_price
        pips = ((exit_price_actual - trade['entry_price']) if trade['trade_type'] == 'buy' else (trade['entry_price'] - exit_price_actual)) / 0.1
        profit = pips * self.calculate_pip_value_for_trade(lot_size_to_close)
        self.balance += profit
        log_entry = trade.copy()
        log_entry.update({'exit_time': df.iloc[index]['Date'], 'exit_price': exit_price_actual, 'pips': pips, 'profit': profit,
                          'status': 'closed_partial' if 'Partial' in reason else 'closed_full',
                          'exit_reason': reason, 'balance_after': self.balance, 'closed_lot_size': lot_size_to_close})
        self.trades.append(log_entry)
        print(f"  > Transaction: {reason} for {lot_size_to_close} lots. Pips: {pips:.1f}, P/L: ${profit:.2f}, New Balance: ${self.balance:.2f}")
        if 'Partial' not in reason:
            self.current_trade = None

    def is_trading_session(self, timestamp):
        if timestamp.hour >= 21 or timestamp.hour < 2: return True
        return False
    
    def reset_daily_counters(self, current_time, last_time):
        if current_time.date() != last_time.date():
            print(f"\n--- New Trading Day: {current_time.date()} ---")
            self.daily_trades = 0; self.setup_fvgs = []
            self.consecutive_reversal_signals = 0 # Reset for new day
            self.last_reversal_signal_type = None
            return True
        return False
    
    def run_backtest(self, df):
        print("Starting FVG Backtest (v14 - Two-Strike Reversal Logic)...")
        all_fvgs = []; last_time = df.iloc[0]['Date']
        
        for i in range(25, len(df)):
            current_time = df.iloc[i]['Date']
            is_new_day = self.reset_daily_counters(current_time, last_time)
            last_time = current_time
            
            if current_time.hour == 20:
                if is_new_day: print(f"Observing setup hour (20:00) for initial FVGs...")
                fvg = self.identify_fvg(df, i)
                if fvg: self.setup_fvgs.append(fvg)
                continue

            if not self.is_trading_session(current_time):
                if self.current_trade:
                    print(f"End of session at {current_time.time()}. Closing open trade.")
                    self.close_trade(df, i, df.iloc[i]['Close'], 'Session End', self.current_trade['lot_size'])
                self.consecutive_reversal_signals = 0 # Reset counter outside session
                continue
            
            fvg = self.identify_fvg(df, i)
            if fvg: all_fvgs.append(fvg)

            self.manage_trade(df, i)

            # --- UPDATED REVERSAL LOGIC ---
            if self.current_trade:
                reversal_signal_found_this_candle = False
                for fvg_to_check in all_fvgs[-5:]:
                    if fvg_to_check.get('reversal_checked', False): continue

                    if fvg_to_check['index'] < self.current_trade['entry_index'] and self.is_fvg_failure_confirmed(df, fvg_to_check, i):
                        signal_type = 'sell' if fvg_to_check['type'] == 'bullish' else 'buy'
                        
                        if signal_type != self.current_trade['trade_type']:
                            reversal_signal_found_this_candle = True
                            fvg_to_check['reversal_checked'] = True 
                            
                            # Check if this signal is consecutive
                            if self.last_reversal_signal_type == signal_type:
                                self.consecutive_reversal_signals += 1
                            else: # New type of signal, reset to 1
                                self.consecutive_reversal_signals = 1
                            
                            self.last_reversal_signal_type = signal_type
                            print(f"  > Reversal warning ({self.consecutive_reversal_signals}/2). Type: {signal_type.upper()}")

                            if self.consecutive_reversal_signals >= 2:
                                print(f"** Confirmed Reversal Signal! Closing {self.current_trade['trade_type']} trade. **")
                                self.close_trade(df, i, df.iloc[i]['Open'], 'Confirmed Reversal', self.current_trade['lot_size'])
                                reason = f"Reversal after 2 FVG fails"
                                self.open_trade(df, i, signal_type, reason)
                                break # Exit loop after reversing
                
                if not reversal_signal_found_this_candle:
                     self.consecutive_reversal_signals = 0 # Reset if a candle has no reversal signals
            
            elif not self.current_trade and self.daily_trades < self.max_trades_per_day:
                # Entry Logic (remains the same as v13)
                fvg_source = self.setup_fvgs if self.daily_trades == 0 else all_fvgs[-20:]
                reason_prefix = "Setup FVG" if self.daily_trades == 0 else "Continuous FVG"

                for fvg_to_check in reversed(fvg_source):
                    if not fvg_to_check.get('triggered', False) and self.is_fvg_failure_confirmed(df, fvg_to_check, i):
                        trade_type = 'sell' if fvg_to_check['type'] == 'bullish' else 'buy'
                        reason = f"{reason_prefix} {fvg_to_check['type']} Fail (Confirmed)"
                        if self.open_trade(df, i, trade_type, reason):
                            print(f"Trade {self.daily_trades} opened: {trade_type.upper()} @ {df.iloc[i]['Open']:.2f} (Reason: {reason})")
                            fvg_to_check['triggered'] = True
                            break
            
            current_equity = self.balance
            if self.current_trade:
                trade, current_price = self.current_trade, df.iloc[i]['Close']
                pips = ((current_price - trade['entry_price']) if trade['trade_type'] == 'buy' else (trade['entry_price'] - current_price)) / 0.1
                current_equity += pips * self.calculate_pip_value_for_trade(trade['lot_size'])
            self.equity_curve.append({'Date': current_time, 'Balance': self.balance, 'Equity': current_equity})

        if self.current_trade: self.close_trade(df, len(df)-1, df.iloc[-1]['Close'], 'End of Data', self.current_trade['lot_size'])
        self.print_results()
        return self.trades
    
    def print_results(self):
        # ... print_results is identical ...
        print("\n" + "="*60 + "\nBACKTEST RESULTS\n" + "="*60)
        if not self.trades:
            print("No trades executed!"); print(f"Final Balance: ${self.balance:.2f}"); return
        df_trades = pd.DataFrame(self.trades)
        total_profit = df_trades['profit'].sum()
        print(f"Initial Balance: ${self.initial_balance:.2f} | Final Balance: ${self.balance:.2f}")
        print(f"Total P/L: ${total_profit:.2f} | Return: {(total_profit/self.initial_balance*100):.2f}%")
        wins = len(df_trades[df_trades['profit'] > 0]); losses = len(df_trades[df_trades['profit'] <= 0])
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        total_gains = df_trades[df_trades['profit'] > 0]['profit'].sum()
        total_losses = abs(df_trades[df_trades['profit'] < 0]['profit'].sum())
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        max_drawdown = 0
        df_equity = pd.DataFrame(self.equity_curve)
        if not df_equity.empty:
            peak = self.initial_balance
            for _, row in df_equity.iterrows():
                peak = max(peak, row['Equity'])
                drawdown = (peak - row['Equity']) / peak * 100 if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        print(f"Total Transactions: {len(df_trades)} | Win Rate (by transaction): {win_rate:.1f}%")
        print(f"Profit Factor: {profit_factor:.2f} | Max Drawdown: {max_drawdown:.2f}%")
        print("="*60)


def main():
    try:
        df = pd.read_csv('XAU_5m_data_2024_filtered.csv', sep=';')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
        print(f"Loaded {len(df)} rows of data.")
    except FileNotFoundError: print("Error: XAU_5m_data_2024_filtered.csv not found!"); return
    
    bot = FVGBacktestBotV14(initial_balance=30)
    trades = bot.run_backtest(df)
    
    if trades:
        results_df = pd.DataFrame(trades)
        results_df.to_csv('fvg_backtest_results_v14_final.csv', index=False)
        print(f"\nResults saved to: fvg_backtest_results_v14_final.csv")

if __name__ == "__main__":
    main()