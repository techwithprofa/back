# chart_backtester.py

import pandas as pd
import plotly.graph_objects as go
from datetime import time
import warnings
warnings.filterwarnings('ignore')

# --- FVG BACKTEST BOT LOGIC (V14 - The most robust version) ---
class FVGBacktestBotV14:
    def __init__(self, initial_balance=30):
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
        self.consecutive_reversal_signals = 0
        self.last_reversal_signal_type = None

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
        if trade_type == 'buy': return lookback_data['Low'].min()
        else: return lookback_data['High'].max()

    def calculate_sl_distance(self, df, entry_index, trade_type):
        entry_price = df.iloc[entry_index]['Open']
        swing_point = self.find_swing_point(df, entry_index, trade_type)
        if swing_point is None: return self.max_sl_pips
        if trade_type == 'buy': sl_distance = abs(entry_price - swing_point)
        else: sl_distance = abs(swing_point - entry_price)
        sl_pips = sl_distance / 0.1
        return max(3, min(sl_pips, self.max_sl_pips))
    
    def open_trade(self, df, index, trade_type, reason):
        if self.daily_trades >= self.max_trades_per_day: return
        entry_price = df.iloc[index]['Open']
        sl_pips = self.calculate_sl_distance(df, index, trade_type)
        pip_price_move = 0.1
        if trade_type == 'buy':
            sl_price, tp1_price = entry_price - (sl_pips * pip_price_move), entry_price + (sl_pips * 2 * pip_price_move)
        else:
            sl_price, tp1_price = entry_price + (sl_pips * pip_price_move), entry_price - (sl_pips * 2 * pip_price_move)
        tp2_price = entry_price + (25 * pip_price_move) if trade_type == 'buy' else entry_price - (25 * pip_price_move)
        self.daily_trades += 1
        self.current_trade = {'trade_num_daily': self.daily_trades, 'entry_time': df.index[index], 'entry_price': entry_price, 
                              'trade_type': trade_type, 'lot_size': self.initial_lot_size, 'sl_price': sl_price, 'tp1_price': tp1_price, 
                              'tp2_price': tp2_price, 'sl_pips': sl_pips, 'status': 'open', 'tp1_hit': False, 'reason': reason, 'entry_index': index}
        self.consecutive_reversal_signals = 0; self.last_reversal_signal_type = None

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
        if trade['tp1_hit']:
            if (trade['trade_type'] == 'buy' and high_price >= trade['tp2_price']) or (trade['trade_type'] == 'sell' and low_price <= trade['tp2_price']):
                self.close_trade(df, index, trade['tp2_price'], 'TP2 Hit', trade['lot_size'])

    def close_trade(self, df, index, exit_price, reason, lot_size_to_close):
        trade = self.current_trade
        exit_price_actual = df.iloc[index]['Close'] if 'Hit' not in reason else exit_price
        pips = ((exit_price_actual - trade['entry_price']) if trade['trade_type'] == 'buy' else (trade['entry_price'] - exit_price_actual)) / 0.1
        profit = pips * self.calculate_pip_value_for_trade(lot_size_to_close)
        self.balance += profit
        log_entry = trade.copy()
        log_entry.update({'exit_time': df.index[index], 'exit_price': exit_price_actual, 'pips': pips, 'profit': profit,
                          'status': 'closed_partial' if 'Partial' in reason else 'closed_full',
                          'exit_reason': reason, 'balance_after': self.balance, 'closed_lot_size': lot_size_to_close})
        self.trades.append(log_entry)
        if 'Partial' not in reason: self.current_trade = None

    def is_trading_session(self, timestamp):
        return timestamp.hour >= 21 or timestamp.hour < 2

    def reset_daily_counters(self, current_time, last_time):
        if last_time and current_time.date() != last_time.date():
            self.daily_trades = 0; self.setup_fvgs = []; self.consecutive_reversal_signals = 0; self.last_reversal_signal_type = None
            return True
        return False
    
    def run_backtest(self, df):
        print("Starting FVG Backtest (v14 - Two-Strike Reversal Logic)...")
        all_fvgs = []; last_time = df.index[0]
        
        for i in range(25, len(df)):
            current_time = df.index[i]
            if self.reset_daily_counters(current_time, last_time):
                print(f"--- New Day: {current_time.date()} ---")
            last_time = current_time
            
            if current_time.hour == 20:
                fvg = self.identify_fvg(df, i)
                if fvg: self.setup_fvgs.append(fvg); all_fvgs.append(fvg)
                continue

            if not self.is_trading_session(current_time):
                if self.current_trade:
                    self.close_trade(df, i, df.iloc[i]['Close'], 'Session End', self.current_trade['lot_size'])
                self.consecutive_reversal_signals = 0
                continue
            
            fvg = self.identify_fvg(df, i)
            if fvg: all_fvgs.append(fvg)

            self.manage_trade(df, i)

            if self.current_trade:
                reversal_signal_found = False
                for fvg_to_check in all_fvgs[-5:]:
                    if fvg_to_check.get('reversal_checked'): continue
                    if fvg_to_check['index'] < self.current_trade['entry_index'] and self.is_fvg_failure_confirmed(df, fvg_to_check, i):
                        signal_type = 'sell' if fvg_to_check['type'] == 'bullish' else 'buy'
                        if signal_type != self.current_trade['trade_type']:
                            reversal_signal_found = True
                            fvg_to_check['reversal_checked'] = True
                            if self.last_reversal_signal_type == signal_type: self.consecutive_reversal_signals += 1
                            else: self.consecutive_reversal_signals = 1
                            self.last_reversal_signal_type = signal_type
                            if self.consecutive_reversal_signals >= 2:
                                self.close_trade(df, i, df.iloc[i]['Open'], 'Confirmed Reversal', self.current_trade['lot_size'])
                                self.open_trade(df, i, signal_type, "Reversal after 2 FVG fails")
                                break
                if not reversal_signal_found: self.consecutive_reversal_signals = 0
            
            elif not self.current_trade and self.daily_trades < self.max_trades_per_day:
                fvg_source = self.setup_fvgs if self.daily_trades == 0 else all_fvgs[-20:]
                for fvg_to_check in reversed(fvg_source):
                    if not fvg_to_check.get('triggered') and self.is_fvg_failure_confirmed(df, fvg_to_check, i):
                        trade_type = 'sell' if fvg_to_check['type'] == 'bullish' else 'buy'
                        self.open_trade(df, i, trade_type, f"FVG Fail Confirmed")
                        fvg_to_check['triggered'] = True
                        break
        
        if self.current_trade: self.close_trade(df, len(df)-1, df.iloc[-1]['Close'], 'End of Data', self.current_trade['lot_size'])
        print("Backtest Complete.")
        return self.trades

def generate_results_chart(df, trades):
    print("Generating results chart...")
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                       open=df['Open'],
                                       high=df['High'],
                                       low=df['Low'],
                                       close=df['Close'],
                                       name='Market')])

    for trade in trades:
        is_buy = trade['trade_type'] == 'buy'
        is_win = trade['profit'] > 0
        
        # Draw line connecting entry and exit
        fig.add_shape(type="line",
                      x0=trade['entry_time'], y0=trade['entry_price'],
                      x1=trade['exit_time'], y1=trade['exit_price'],
                      line=dict(color="gray", width=1, dash="dot"))
        
        # Entry Marker
        fig.add_trace(go.Scatter(x=[trade['entry_time']], y=[trade['entry_price']],
                                 mode='markers',
                                 marker=dict(color='blue', size=10,
                                             symbol='triangle-up' if is_buy else 'triangle-down'),
                                 name='Entry',
                                 hoverinfo='text',
                                 text=f"Entry: {trade['reason']}"))

        # Exit Marker
        fig.add_trace(go.Scatter(x=[trade['exit_time']], y=[trade['exit_price']],
                                 mode='markers',
                                 marker=dict(color='green' if is_win else 'red', size=8, symbol='circle'),
                                 name='Exit',
                                 hoverinfo='text',
                                 text=f"Exit: {trade['exit_reason']}<br>P/L: ${trade['profit']:.2f}"))

    fig.update_layout(title_text="FVG Backtest Results with Trades",
                      xaxis_rangeslider_visible=False,
                      showlegend=False)
    
    fig.write_html("backtest_chart.html")
    print("\nChart saved to 'backtest_chart.html'. Open this file in your browser to see the results.")

def main():
    try:
        df = pd.read_csv('XAU_5m_data_2024_filtered.csv', sep=';')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
        df.set_index('Date', inplace=True)
        print(f"Loaded {len(df)} rows of data.")
    except FileNotFoundError:
        print("Error: XAU_5m_data_2024_filtered.csv not found!")
        return
    
    bot = FVGBacktestBotV14(initial_balance=30)
    trades = bot.run_backtest(df)
    
    if trades:
        results_df = pd.DataFrame(trades)
        results_df.to_csv('final_backtest_trades.csv', index=False)
        print(f"Trade logs saved to: final_backtest_trades.csv")
        generate_results_chart(df, trades)
    else:
        print("No trades were executed.")

if __name__ == "__main__":
    main()