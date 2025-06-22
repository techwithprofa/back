import pandas as pd
import numpy as np
import datetime

# Constants for FVG states (as per specification)
class FVGState:
    PENDING = "pending"
    BAKE = "bake"
    BREAK = "break"
    FAIL = "fail"

POINT_SIZE = 0.01

# --- Data Preprocessing ---
def preprocess_data(df):
    # Updated expected columns to match the CSV file
    expected_columns_original_case = ['Date', 'Open', 'High', 'Low', 'Close']

    # Check if all expected columns are present
    if not all(col in df.columns for col in expected_columns_original_case):
        # Provide a more informative error if essential columns are missing
        missing_cols = [col for col in expected_columns_original_case if col not in df.columns]
        raise ValueError(f"Input CSV must contain columns: {', '.join(expected_columns_original_case)}. Missing: {', '.join(missing_cols)}")

    # Rename columns to lowercase and 'Date' to 'timestamp'
    df.rename(columns={
        'Date': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume' # Ensure volume is also lowercased if present
    }, inplace=True)

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Handle missing data - ffill then bfill to cover edges
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Calculate True Range and ATR(14)
    df['high_low_diff'] = df['high'] - df['low']
    df['high_prev_close_diff'] = abs(df['high'] - df['close'].shift(1))
    df['low_prev_close_diff'] = abs(df['low'] - df['close'].shift(1))

    df['true_range'] = df[['high_low_diff', 'high_prev_close_diff', 'low_prev_close_diff']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()

    df.drop(columns=['high_low_diff', 'high_prev_close_diff', 'low_prev_close_diff', 'true_range'], inplace=True)
    df.dropna(inplace=True) # Drop rows with NaN ATR
    return df

# --- FVG Detection ---
def detect_fvgs(df, config):
    fvgs = []
    for i in range(1, len(df) - 1):
        candle_prev = df.iloc[i-1]
        candle_curr = df.iloc[i]
        candle_next = df.iloc[i+1]

        fvg_details = {
            "index": df.index[i],
            "timestamp": candle_curr['timestamp'],
            "type": None, "top": None, "bottom": None, "mid_price": None, "size_pips": None,
            "state": FVGState.PENDING, "age": 0,
            "formation_candle_idx": df.index[i+1], # FVG confirmed when candle_next (i+1) closes
            "interacted_candle_idx": None, "broken_candle_idx": None, "failed_candle_idx": None
        }

        is_bullish_fvg = candle_prev['high'] < candle_next['low']
        is_bearish_fvg = candle_prev['low'] > candle_next['high']

        if is_bullish_fvg:
            fvg_details["type"] = "bullish"; fvg_details["top"] = candle_next['low']; fvg_details["bottom"] = candle_prev['high']
        elif is_bearish_fvg:
            fvg_details["type"] = "bearish"; fvg_details["top"] = candle_prev['low']; fvg_details["bottom"] = candle_next['high']
        else: continue

        fvg_details["mid_price"] = (fvg_details["top"] + fvg_details["bottom"]) / 2
        fvg_details["size_pips"] = abs(fvg_details["top"] - fvg_details["bottom"]) / POINT_SIZE

        if fvg_details["size_pips"] >= config.get("fvg_min_size", 2):
            fvgs.append(fvg_details)
    return fvgs

# --- Market Structure (Basic) ---
def calculate_market_structure(df, swing_lookback=5):
    df['swing_high'] = np.nan; df['swing_low'] = np.nan
    for i in range(swing_lookback, len(df) - swing_lookback):
        is_swing_high = True; is_swing_low = True
        current_high = df['high'].iloc[i]; current_low = df['low'].iloc[i]
        for j in range(1, swing_lookback + 1):
            if df['high'].iloc[i-j] >= current_high or df['high'].iloc[i+j] >= current_high: is_swing_high = False
            if df['low'].iloc[i-j] <= current_low or df['low'].iloc[i+j] <= current_low: is_swing_low = False
        if is_swing_high: df.loc[df.index[i], 'swing_high'] = current_high
        if is_swing_low: df.loc[df.index[i], 'swing_low'] = current_low
    return df

TRADING_SESSIONS_CONFIG = {
    "london_primary": {"start": "21:00", "end": "23:00", "weight": 0.6},
    "ny_overlap": {"start": "01:00", "end": "03:00", "weight": 0.25},
    "asian_continuation": {"start": "07:00", "end": "09:00", "weight": 0.15}
}

def get_current_session(timestamp_dt, config):
    time_str = timestamp_dt.strftime("%H:%M"); active_sessions = []
    s_config = TRADING_SESSIONS_CONFIG # Use global
    if config.get("trade_london_session", True):
        si = s_config["london_primary"]
        if si["start"] <= si["end"]:
            if si["start"] <= time_str <= si["end"]: active_sessions.append("london_primary")
        else: # Overnight
            if time_str >= si["start"] or time_str <= si["end"]: active_sessions.append("london_primary")
    if config.get("trade_ny_overlap", True):
        si = s_config["ny_overlap"]
        if si["start"] <= si["end"]:
            if si["start"] <= time_str <= si["end"]: active_sessions.append("ny_overlap")
        else:
            if time_str >= si["start"] or time_str <= si["end"]: active_sessions.append("ny_overlap")
    if config.get("trade_asian_session", False):
        si = s_config["asian_continuation"]
        if si["start"] <= si["end"]:
            if si["start"] <= time_str <= si["end"]: active_sessions.append("asian_continuation")
        else:
            if time_str >= si["start"] or time_str <= si["end"]: active_sessions.append("asian_continuation")
    return active_sessions[0] if active_sessions else "out_of_session"

class StopLossManager:
    def __init__(self, config): self.config = config; self.initial_sl_range_pips = (config.get("min_stop_loss",3.0), config.get("max_stop_loss",8.0))
    def calculate_initial_sl(self, entry_price, direction, candle_data, recent_swings):
        atr_val = candle_data.get('atr', self.initial_sl_range_pips[1] * POINT_SIZE) # Default ATR if missing
        vol_adj_pips_component = (atr_val / POINT_SIZE) * 0.3

        struct_sl_pips = self.initial_sl_range_pips[1] # Default to max pips in allowed range
        if direction == 1 and recent_swings.get('low') is not None: # Long
            swing_sl_candidate_pips = (entry_price - (recent_swings['low'] - POINT_SIZE)) / POINT_SIZE
            if swing_sl_candidate_pips > 0 : struct_sl_pips = min(struct_sl_pips, swing_sl_candidate_pips)
        elif direction == -1 and recent_swings.get('high') is not None: # Short
            swing_sl_candidate_pips = ((recent_swings['high'] + POINT_SIZE) - entry_price) / POINT_SIZE
            if swing_sl_candidate_pips > 0: struct_sl_pips = min(struct_sl_pips, swing_sl_candidate_pips)

        # Apply formula: min(max(base_sl + volatility_adjustment, min_sl_cfg), max_sl_cfg)
        # base_sl is struct_sl_pips
        combined_sl_pips = struct_sl_pips + vol_adj_pips_component

        final_sl_pips = max(self.initial_sl_range_pips[0], combined_sl_pips) # At least min_sl_range[0]
        final_sl_pips = min(self.initial_sl_range_pips[1], final_sl_pips)   # At most max_sl_range[1]
        final_sl_pips = max(final_sl_pips, self.config.get("min_stop_loss",3.0)) # Ensure it respects overall min_stop_loss from config again

        sl_price = entry_price - (final_sl_pips * POINT_SIZE * direction)
        return round(sl_price, 5)

    def update_sl(self, trade, current_candle_price_data, df_history, tp_level_hit=None):
        new_sl = trade.stop_loss # Start with current SL
        if tp_level_hit == "tp1_hit" and self.config.get("move_sl_on_tp1", True):
            if 'swing_low' not in df_history.columns or 'swing_high' not in df_history.columns or df_history.empty: new_sl = trade.entry_price # Fallback to BE
            else:
                if trade.direction == 1: # Long trade, find second last swing low
                    swing_lows_available = df_history['swing_low'].dropna()
                    # Ensure there are enough points to get a "second last" or "last before current"
                    relevant_swing_lows = swing_lows_available.iloc[:-1] if len(swing_lows_available) > 1 else swing_lows_available
                    if not relevant_swing_lows.empty:
                        potential_new_sl_price = relevant_swing_lows.iloc[-1] - POINT_SIZE # 1 pip below
                        # Only move SL if it's tighter (higher for long) and above BE
                        if potential_new_sl_price > new_sl and potential_new_sl_price > trade.entry_price: new_sl = potential_new_sl_price
                        else: new_sl = max(new_sl, trade.entry_price) # Ensure at least BE or current SL if better
                    else: new_sl = trade.entry_price # Not enough swing lows, move to BreakEven
                else: # Short trade, find second last swing high
                    swing_highs_available = df_history['swing_high'].dropna()
                    relevant_swing_highs = swing_highs_available.iloc[:-1] if len(swing_highs_available) > 1 else swing_highs_available
                    if not relevant_swing_highs.empty:
                        potential_new_sl_price = relevant_swing_highs.iloc[-1] + POINT_SIZE # 1 pip above
                        if potential_new_sl_price < new_sl and potential_new_sl_price < trade.entry_price: new_sl = potential_new_sl_price
                        else: new_sl = min(new_sl, trade.entry_price) # Ensure at least BE or current SL if better
                    else: new_sl = trade.entry_price # Not enough swing highs, move to BreakEven
        # Ensure SL doesn't move against the trade (e.g. if new_sl calculation results in worse SL than original after TP hit)
        if trade.direction == 1: return round(max(new_sl, trade.stop_loss if not tp_level_hit else trade.entry_price), 5)
        else: return round(min(new_sl, trade.stop_loss if not tp_level_hit else trade.entry_price), 5)

class PositionManager:
    def __init__(self, config): self.config = config
    def calculate_position_size(self, account_balance, stop_loss_pips, risk_per_trade_pct):
        if stop_loss_pips <= 0: return 0.01 # Avoid division by zero
        risk_amount_currency = account_balance * risk_per_trade_pct
        value_per_pip_per_std_lot = 1.00 # For XAUUSD, 1 pip ($0.01 move) on 1 lot = $1 profit
        position_size_std_lots = risk_amount_currency / (stop_loss_pips * value_per_pip_per_std_lot)
        calculated_lot_size = round(position_size_std_lots / 0.01) * 0.01 # Round to 0.01 lot step
        return max(0.01, calculated_lot_size) # Ensure min 0.01

class TakeProfitManager:
    def __init__(self, config): self.config=config; self.tp1_ratio=config.get("tp1_rr_ratio",2.0); self.tp2_fixed_pips=config.get("tp2_fixed_pips",25.0)
    def calculate_targets(self, entry_price, stop_loss_price, direction):
        sl_distance_price = abs(entry_price - stop_loss_price)
        tp1_price = entry_price + (sl_distance_price * self.tp1_ratio * direction)
        tp2_price = entry_price + (self.tp2_fixed_pips * POINT_SIZE * direction)
        return {"tp1": round(tp1_price,5), "tp2": round(tp2_price,5)}

class Trade:
    _id_counter = 0
    def __init__(self, entry_time, direction, entry_price, stop_loss, take_profits, position_size, fvg_info, session, candle_idx_entry):
        Trade._id_counter+=1; self.trade_id=Trade._id_counter; self.entry_time=entry_time; self.entry_candle_idx=candle_idx_entry
        self.direction=direction; self.entry_price=entry_price; self.stop_loss=stop_loss; self.initial_stop_loss=stop_loss
        self.take_profit1=take_profits['tp1']; self.take_profit2=take_profits['tp2']; self.position_size=position_size; self.initial_position_size=position_size
        self.fvg_type_entry=fvg_info.get('type','N/A') if fvg_info else 'N/A'; self.fvg_timestamp_entry=fvg_info.get('timestamp') if fvg_info else None
        self.fvg_index_entry=fvg_info.get('index') if fvg_info else None; self.session=session; self.status="open"
        self.exit_time=None; self.exit_price=None; self.exit_candle_idx=None; self.pnl_pips=0.0; self.pnl_currency=0.0
        self.exit_reason=None; self.tp1_hit_status=False; self.tp1_hit_price=None; self.tp1_hit_time=None
        self.tp1_hit_candle_idx=None; self.partially_closed_pnl=0.0; self.remaining_position_size=position_size

class PerformanceTracker:
    def __init__(self, initial_balance=10000):
        self.trades=[]; self.equity_curve=[(pd.Timestamp('1970-01-01'),initial_balance)]; self.initial_balance=initial_balance
        self.current_balance=initial_balance; self.peak_equity=initial_balance; self.max_drawdown_pct=0.0
        self.metrics = {"total_trades":0, "winning_trades":0, "losing_trades":0, "total_pnl_pips":0.0,
                        "total_pnl_currency":0.0, "max_drawdown_pct":0.0, "profit_factor":0.0, "win_rate_pct":0.0,
                        "avg_win_pips":0.0, "avg_loss_pips":0.0, "avg_rr_ratio":0.0, "fvg_entry_success_rate_pct":0.0}
    def record_trade_closure(self, trade):
        if trade.status != "closed": return
        self.trades.append(trade); self.metrics["total_trades"]+=1
        pip_value_per_lot_per_pip = 1.00
        trade.pnl_pips = (trade.exit_price - trade.entry_price) * trade.direction / POINT_SIZE
        trade.pnl_currency = trade.pnl_pips * trade.initial_position_size * pip_value_per_lot_per_pip # Uses initial size for full trade P&L
        if trade.pnl_currency > 0: self.metrics["winning_trades"]+=1
        else: self.metrics["losing_trades"]+=1
        self.metrics["total_pnl_pips"]+=trade.pnl_pips; self.metrics["total_pnl_currency"]+=trade.pnl_currency
        self.current_balance+=trade.pnl_currency
        if trade.exit_time: self.equity_curve.append((trade.exit_time, self.current_balance))
        else: self.equity_curve.append((self.equity_curve[-1][0] + pd.Timedelta(seconds=1) if self.equity_curve else pd.Timestamp.now(), self.current_balance)) # Fallback
        if self.current_balance > self.peak_equity: self.peak_equity = self.current_balance
        current_drawdown_pct = (self.peak_equity - self.current_balance)/self.peak_equity if self.peak_equity >0 else 0
        if current_drawdown_pct > self.max_drawdown_pct: self.max_drawdown_pct = current_drawdown_pct
        self.metrics["max_drawdown_pct"] = self.max_drawdown_pct * 100 # Store as percentage
    def calculate_final_metrics(self):
        if not self.trades: return self.metrics
        self.metrics["win_rate_pct"] = (self.metrics["winning_trades"]/self.metrics["total_trades"])*100 if self.metrics["total_trades"]>0 else 0
        winning_pips_list = [t.pnl_pips for t in self.trades if t.pnl_pips > 0]
        losing_pips_list = [abs(t.pnl_pips) for t in self.trades if t.pnl_pips < 0]
        total_won_pips = sum(winning_pips_list); total_lost_pips = sum(losing_pips_list)
        self.metrics["profit_factor"] = total_won_pips/total_lost_pips if total_lost_pips > 0 else float('inf')
        self.metrics["avg_win_pips"] = np.mean(winning_pips_list) if winning_pips_list else 0
        self.metrics["avg_loss_pips"] = np.mean(losing_pips_list) if losing_pips_list else 0
        if self.metrics["avg_loss_pips"]>0: self.metrics["avg_rr_ratio"] = self.metrics["avg_win_pips"]/self.metrics["avg_loss_pips"]
        else: self.metrics["avg_rr_ratio"] = float('inf') if self.metrics["avg_win_pips"] > 0 else 0 # Or 0 if no wins
        fvg_entry_trades = [t for t in self.trades if t.fvg_type_entry != 'N/A']
        if fvg_entry_trades:
            winning_fvg_trades = sum(1 for t in fvg_entry_trades if t.pnl_pips > 0)
            self.metrics["fvg_entry_success_rate_pct"] = (winning_fvg_trades/len(fvg_entry_trades))*100 if len(fvg_entry_trades)>0 else 0
        self.metrics["max_drawdown_pct"] = self.max_drawdown_pct # Already set
        return self.metrics
    def generate_report_outputs(self):
        final_metrics = self.calculate_final_metrics()
        report_dict = {"summary_stats": {"total_trades":final_metrics["total_trades"], "win_rate_pct":final_metrics["win_rate_pct"],
                                    "total_pnl_pips":final_metrics["total_pnl_pips"], "total_pnl_currency":final_metrics["total_pnl_currency"],
                                    "profit_factor":final_metrics["profit_factor"], "max_drawdown_pct":final_metrics["max_drawdown_pct"],
                                    "initial_balance":self.initial_balance, "final_balance":self.current_balance, "sharpe_ratio":"N/A"},
                  "fvg_analysis": {"total_fvg_detected_in_trades":len([t for t in self.trades if t.fvg_type_entry != 'N/A']),
                                   "fvg_entry_success_rate_pct":final_metrics["fvg_entry_success_rate_pct"]},
                  "risk_metrics": {"avg_win_pips":final_metrics["avg_win_pips"], "avg_loss_pips":final_metrics["avg_loss_pips"],
                                   "avg_rr_ratio":final_metrics["avg_rr_ratio"]}}
        trade_log_columns = ['trade_id','entry_time','exit_time','direction','entry_price','exit_price','initial_sl','final_sl_at_exit',
                             'tp1','tp2','tp1_hit_status','position_size','pnl_pips','pnl_currency','fvg_type_entry','fvg_timestamp_entry',
                             'exit_reason','session','entry_candle_idx','exit_candle_idx','duration_candles']
        trade_log_data = []
        for t in self.trades:
            duration = (t.exit_candle_idx - t.entry_candle_idx) if t.exit_candle_idx is not None and t.entry_candle_idx is not None else None
            trade_log_data.append({'trade_id':t.trade_id, 'entry_time':t.entry_time, 'exit_time':t.exit_time,
                                   'direction':"LONG" if t.direction==1 else "SHORT", 'entry_price':t.entry_price, 'exit_price':t.exit_price,
                                   'initial_sl':t.initial_stop_loss, 'final_sl_at_exit':t.stop_loss, 'tp1':t.take_profit1, 'tp2':t.take_profit2,
                                   'tp1_hit_status':t.tp1_hit_status, 'position_size':t.initial_position_size, 'pnl_pips':round(t.pnl_pips,2),
                                   'pnl_currency':round(t.pnl_currency,2), 'fvg_type_entry':t.fvg_type_entry,
                                   'fvg_timestamp_entry':t.fvg_timestamp_entry, 'exit_reason':t.exit_reason, 'session':t.session,
                                   'entry_candle_idx':t.entry_candle_idx, 'exit_candle_idx':t.exit_candle_idx, 'duration_candles':duration})
        log_df = pd.DataFrame(trade_log_data, columns=trade_log_columns)
        print("\n--- Performance Report ---")
        for category, stats in report_dict.items():
            print(f"\n{category.replace('_',' ').title()}:")
            for k,v_val in stats.items(): print(f"  {k.replace('_',' ').title()}: {v_val:.2f}" if isinstance(v_val,float) else f"  {k.replace('_',' ').title()}: {v_val}")
        print("\n--- Trade Log ---"); print(log_df.to_string(max_rows=20)) # Print a segment
        return report_dict, log_df

class FVGBacktestEngine:
    def __init__(self, data_df, config, initial_balance=10000):
        self.df=data_df; self.config=config; self.sl_manager=StopLossManager(config); self.pos_manager=PositionManager(config)
        self.tp_manager=TakeProfitManager(config); self.tracker=PerformanceTracker(initial_balance)
        print("Detecting all FVGs upfront..."); self.all_fvgs_master_list=detect_fvgs(self.df,config)
        print(f"Total FVGs detected in dataset: {len(self.all_fvgs_master_list)}")
        self.active_fvgs_being_tracked=[]; self.open_trades=[]
        print("Calculating market structure..."); self.df=calculate_market_structure(self.df.copy()) # Use copy
    def get_recent_swing_points_before_idx(self, current_df_candle_idx, lookback_candles=20):
        start_idx = max(0, current_df_candle_idx - lookback_candles)
        # Ensure we only look at candles *before* current_df_candle_idx by slicing up to current_df_candle_idx-1
        relevant_df_slice = self.df.loc[:current_df_candle_idx-1].iloc[start_idx:] if current_df_candle_idx > 0 else pd.DataFrame()
        if relevant_df_slice.empty: return {'low':None, 'high':None}
        recent_low_val, recent_high_val = None, None
        if 'swing_low' in relevant_df_slice.columns:
            valid_lows=relevant_df_slice['swing_low'].dropna()
            if not valid_lows.empty: recent_low_val=valid_lows.iloc[-1]
        if 'swing_high' in relevant_df_slice.columns:
            valid_highs=relevant_df_slice['swing_high'].dropna()
            if not valid_highs.empty: recent_high_val=valid_highs.iloc[-1]
        return {'low':recent_low_val, 'high':recent_high_val}
    def update_fvg_list_and_states(self, current_candle_series, current_candle_df_idx):
        for fvg_candidate in self.all_fvgs_master_list: # Check all pre-detected FVGs
            if fvg_candidate['formation_candle_idx'] == current_candle_df_idx: # FVG is confirmed now
                # Check if already tracking this FVG (by its middle candle index and type)
                is_new_to_tracking = True
                for existing_fvg in self.active_fvgs_being_tracked:
                    if existing_fvg['index'] == fvg_candidate['index'] and existing_fvg['type'] == fvg_candidate['type']:
                        is_new_to_tracking = False; break
                if is_new_to_tracking:
                    new_fvg_instance=fvg_candidate.copy(); new_fvg_instance['age']=0; self.active_fvgs_being_tracked.append(new_fvg_instance)
        updated_active_list = []
        for fvg in self.active_fvgs_being_tracked:
            fvg['age']+=1
            if fvg['state'] in ["used_for_trade", FVGState.FAIL]:
                if fvg['age'] < self.config.get("fvg_max_age",20)+10: updated_active_list.append(fvg) # Keep for a bit
                continue # Already resolved or too old
            if fvg['age'] > self.config.get("fvg_max_age",20) and fvg['state']==FVGState.PENDING:
                fvg['state']=FVGState.FAIL; fvg['failed_candle_idx']=current_candle_df_idx; fvg['exit_reason']="expired"
                updated_active_list.append(fvg); continue
            price_high,price_low,price_close = current_candle_series['high'],current_candle_series['low'],current_candle_series['close']; interacted_this_candle=False
            if fvg['type']=='bullish':
                if price_low <= fvg['top'] and price_high >= fvg['bottom']: interacted_this_candle=True
                if price_close > fvg['top']:
                    if fvg['state'] != FVGState.BREAK: fvg['state']=FVGState.BREAK; fvg['broken_candle_idx']=current_candle_df_idx
                elif price_close < fvg['bottom']: fvg['state']=FVGState.FAIL; fvg['failed_candle_idx']=current_candle_df_idx; fvg['exit_reason']="closed_below_fvg"
            elif fvg['type']=='bearish':
                if price_high >= fvg['bottom'] and price_low <= fvg['top']: interacted_this_candle=True
                if price_close < fvg['bottom']:
                    if fvg['state'] != FVGState.BREAK: fvg['state']=FVGState.BREAK; fvg['broken_candle_idx']=current_candle_df_idx
                elif price_close > fvg['top']: fvg['state']=FVGState.FAIL; fvg['failed_candle_idx']=current_candle_df_idx; fvg['exit_reason']="closed_above_fvg"
            if interacted_this_candle and fvg['state']==FVGState.PENDING: fvg['state']=FVGState.BAKE; fvg['interacted_candle_idx']=current_candle_df_idx
            updated_active_list.append(fvg)
        def sort_key_fvg(fvg_item): prio={FVGState.PENDING:0,FVGState.BAKE:1,FVGState.BREAK:2,FVGState.FAIL:3,"used_for_trade":4}; return (fvg_item['formation_candle_idx'], prio.get(fvg_item['state'],99))
        updated_active_list.sort(key=sort_key_fvg, reverse=True) # Most recent formation, then by state relevance
        self.active_fvgs_being_tracked = updated_active_list[:self.config.get("fvg_max_tracked",3)]
    def check_for_entry_signals(self, current_candle_series, current_candle_df_idx):
        if len(self.open_trades) >= self.config.get("max_concurrent_trades",1): return
        current_session = get_current_session(current_candle_series['timestamp'], self.config)
        if current_session == "out_of_session": return
        for fvg_oco in self.active_fvgs_being_tracked: # oco = on_current_object
            if fvg_oco['state']==FVGState.BREAK and fvg_oco['broken_candle_idx']==current_candle_df_idx:
                market_bias_aligned=True # Simplified: self.config.get("structure_bias_filter", False)
                if not market_bias_aligned: continue
                direction = 1 if fvg_oco['type']=='bullish' else -1; entry_price = current_candle_series['close']
                recent_swings = self.get_recent_swing_points_before_idx(current_candle_df_idx)
                stop_loss_price = self.sl_manager.calculate_initial_sl(entry_price, direction, current_candle_series, recent_swings)
                sl_pips = abs(entry_price - stop_loss_price)/POINT_SIZE
                min_sl_cfg, max_sl_cfg = self.config.get("min_stop_loss",3.0), self.config.get("max_stop_loss",8.0)
                if not (min_sl_cfg <= sl_pips <= max_sl_cfg): continue
                take_profits = self.tp_manager.calculate_targets(entry_price, stop_loss_price, direction)
                if (direction==1 and (take_profits['tp1']<=entry_price or take_profits['tp2']<=entry_price)) or \
                   (direction==-1 and (take_profits['tp1']>=entry_price or take_profits['tp2']>=entry_price)): continue # TP logic check
                position_size = self.pos_manager.calculate_position_size(self.tracker.current_balance, sl_pips, self.config.get("max_risk_per_trade",0.02))
                if position_size < 0.01: continue # Position size too small
                new_trade = Trade(current_candle_series['timestamp'],direction,entry_price,stop_loss_price,take_profits,position_size,fvg_oco.copy(),current_session,current_candle_df_idx)
                self.open_trades.append(new_trade); fvg_oco['state']="used_for_trade"; break # One trade per candle
    def manage_open_positions(self, current_candle_series, current_candle_df_idx):
        for trade_obj in self.open_trades: # Iterate on a copy if modifying list during iteration, but here we modify trade objects
            if trade_obj.status=="closed": continue
            exit_price_this_candle, exit_reason_this_candle = None, None
            # FVG failure for trade's FVG
            fvg_that_initiated_trade = next((fvg for fvg in self.active_fvgs_being_tracked if fvg['index']==trade_obj.fvg_index_entry and fvg['timestamp']==trade_obj.fvg_timestamp_entry), None)
            if fvg_that_initiated_trade and fvg_that_initiated_trade['state']==FVGState.FAIL and self.config.get("exit_on_trade_fvg_failure",True):
                exit_price_this_candle = current_candle_series['close']; exit_reason_this_candle = f"fvg_failed_{fvg_that_initiated_trade.get('exit_reason','unknown')}"
            # SL check
            if exit_reason_this_candle is None:
                if (trade_obj.direction==1 and current_candle_series['low']<=trade_obj.stop_loss) or \
                   (trade_obj.direction==-1 and current_candle_series['high']>=trade_obj.stop_loss):
                    exit_price_this_candle = trade_obj.stop_loss; exit_reason_this_candle = "sl_hit"
            # TP checks
            if exit_reason_this_candle is None:
                if trade_obj.direction==1: # Long
                    if not trade_obj.tp1_hit_status and current_candle_series['high'] >= trade_obj.take_profit1:
                        trade_obj.tp1_hit_status=True; trade_obj.tp1_hit_price=trade_obj.take_profit1; trade_obj.tp1_hit_time=current_candle_series['timestamp']; trade_obj.tp1_hit_candle_idx=current_candle_df_idx
                        if self.config.get("move_sl_on_tp1",True):
                            df_history = self.df.loc[:current_candle_df_idx-1] if current_candle_df_idx>0 else pd.DataFrame() # History before current candle
                            new_sl = self.sl_manager.update_sl(trade_obj, current_candle_series, df_history, "tp1_hit")
                            if new_sl != trade_obj.stop_loss: trade_obj.stop_loss = new_sl
                    if current_candle_series['high'] >= trade_obj.take_profit2: exit_price_this_candle = trade_obj.take_profit2; exit_reason_this_candle = "tp2_hit"
                else: # Short
                    if not trade_obj.tp1_hit_status and current_candle_series['low'] <= trade_obj.take_profit1:
                        trade_obj.tp1_hit_status=True; trade_obj.tp1_hit_price=trade_obj.take_profit1; trade_obj.tp1_hit_time=current_candle_series['timestamp']; trade_obj.tp1_hit_candle_idx=current_candle_df_idx
                        if self.config.get("move_sl_on_tp1",True):
                            df_history = self.df.loc[:current_candle_df_idx-1] if current_candle_df_idx>0 else pd.DataFrame()
                            new_sl = self.sl_manager.update_sl(trade_obj, current_candle_series, df_history, "tp1_hit")
                            if new_sl != trade_obj.stop_loss: trade_obj.stop_loss = new_sl
                    if current_candle_series['low'] <= trade_obj.take_profit2: exit_price_this_candle = trade_obj.take_profit2; exit_reason_this_candle = "tp2_hit"
            if exit_reason_this_candle:
                trade_obj.status="closed"; trade_obj.exit_price=exit_price_this_candle; trade_obj.exit_reason=exit_reason_this_candle
                trade_obj.exit_time=current_candle_series['timestamp']; trade_obj.exit_candle_idx=current_candle_df_idx
                self.tracker.record_trade_closure(trade_obj)
        self.open_trades = [t for t in self.open_trades if t.status=="open"] # Rebuild list of open trades
    def run_backtest(self):
        if self.df.empty: print("DataFrame is empty, cannot run backtest."); return None,None
        print(f"Starting backtest loop for {len(self.df)} candles...")
        # Iterate through each candle in the DataFrame by its preserved index
        for iteration_count, df_idx in enumerate(self.df.index): # df_idx is the actual index label
            current_candle_series = self.df.loc[df_idx]
            if iteration_count % 5000 == 0: print(f"Processing candle {iteration_count}/{len(self.df)}, Time: {current_candle_series['timestamp']}")
            self.update_fvg_list_and_states(current_candle_series, df_idx) # Pass df_idx
            self.manage_open_positions(current_candle_series, df_idx)     # Pass df_idx
            self.check_for_entry_signals(current_candle_series, df_idx)   # Pass df_idx
        if self.open_trades: # Close any trades still open at the end
            print(f"Closing {len(self.open_trades)} open trades at end of data...")
            last_candle_series = self.df.iloc[-1]; last_df_idx = self.df.index[-1]
            for trade in self.open_trades:
                if trade.status=="open": # Should be all of them
                    trade.exit_price=last_candle_series['close']; trade.exit_time=last_candle_series['timestamp']
                    trade.exit_candle_idx=last_df_idx; trade.exit_reason="end_of_data"; trade.status="closed"
                    self.tracker.record_trade_closure(trade)
        print("Backtest loop finished.")
        return self.tracker.generate_report_outputs()

if __name__ == "__main__":
    Trade._id_counter = 0 # Reset trade ID for multiple script runs (if any)
    STRATEGY_CONFIG = {
        "fvg_min_size":2.0, "fvg_max_age":20, "fvg_priority_distance":0, "fvg_max_tracked":3, "exit_on_trade_fvg_failure":True,
        "max_risk_per_trade":0.02, "min_stop_loss":3.0, "max_stop_loss":8.0,
        "partial_close_at_tp1":True, # This flag enables SL move on TP1; actual partial profit booking is not modeled.
        "move_sl_on_tp1":True, "max_concurrent_trades":1,
        "tp1_rr_ratio":2.0, "tp2_fixed_pips":25.0,
        "trade_london_session":True, "trade_ny_overlap":True, "trade_asian_session":False,
        "structure_bias_filter":False, # Requires market structure indicators (e.g. MAs) not yet added.
    }
    print("--- Backtest Starting ---"); print(f"Strategy Config: {STRATEGY_CONFIG}")
    data_filepath = "XAU_5m_data_2024_filtered.csv"; initial_account_balance = 10000.0
    try:
        print(f"Attempting to load data from: {data_filepath}...");
        # Updated read_csv to handle semicolon delimiter and specific column names
        raw_df = pd.read_csv(data_filepath, delimiter=';', header=0, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        print(f"Successfully loaded data. Shape: {raw_df.shape}")
    except FileNotFoundError:
        print(f"Warning: {data_filepath} not found. Generating dummy data for testing."); num_days=2 # Use fewer days for quicker dummy test
        rng = pd.date_range(start=f'2024-01-01', periods=num_days*24*12, freq='5min') # 12 5-min candles per hour
        data_size = len(rng); base_price = 2000.00
        price_trend = np.linspace(0, 2, data_size); price_noise = np.random.normal(0, 0.5, data_size).cumsum() * 0.2
        volatility = np.random.uniform(0.1, 0.5, data_size)
        open_prices = base_price + price_trend + price_noise; close_offset = np.random.normal(0, volatility, data_size)
        close_prices = open_prices + close_offset
        high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, volatility*0.5, data_size)
        low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, volatility*0.5, data_size)
        final_high = np.maximum.reduce([open_prices, high_prices, low_prices, close_prices]) # Ensure HLOC consistency
        final_low = np.minimum.reduce([open_prices, high_prices, low_prices, close_prices])
        dummy_data = {'timestamp':rng, 'open':np.round(open_prices,2), 'high':np.round(final_high,2),
                      'low':np.round(final_low,2), 'close':np.round(close_prices,2),
                      'volume':np.random.randint(100, 2000, size=data_size)}
        raw_df = pd.DataFrame(dummy_data); raw_df.to_csv("dummy_xau_5m_data_generated.csv", index=False)
        print("Saved dummy_xau_5m_data_generated.csv")
    if raw_df.empty: print("Error: Data loading failed or resulted in an empty DataFrame.")
    else:
        print("Preprocessing data..."); processed_df = preprocess_data(raw_df.copy()) # Use a copy
        print(f"Data preprocessed. Shape: {processed_df.shape}")
        if processed_df.empty or len(processed_df) < 50: # Min data length check
            print("Error: Data is empty or too short after preprocessing. Min length required (e.g. for ATR and swings).")
        else:
            print("Initializing backtest engine..."); engine = FVGBacktestEngine(processed_df, STRATEGY_CONFIG, initial_balance=initial_account_balance)
            performance_report, trade_log = engine.run_backtest()
            print("\n--- Backtest Complete ---")
            if performance_report and trade_log is not None and not trade_log.empty:
                print("Performance report generated.")
                try:
                    trade_log.to_csv("trade_log_output.csv", index=False)
                    print("Trade log saved to trade_log_output.csv")
                except Exception as e:
                    print(f"Error saving trade log: {e}")
            elif performance_report: # Report exists but trade_log might be None or empty
                 print("Performance report generated, but no trades to log or trade log is empty.")
            else:
                print("Backtest finished, but no report was generated (likely no trades or error).")
    print("--- End of Script ---")