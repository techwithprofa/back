import pandas as pd

def manage_trade_bot1(entry_signal, historical_data_df, trade_id):
    """
    Manages a single trade for Bot 1 from entry to completion (SL, TP1, TP2).

    Args:
        entry_signal (dict): Contains entry details like 'entry_timestamp',
                             'direction', 'entry_price', 'sl_price', 'type'.
        historical_data_df (pd.DataFrame): Full market data, NY-time indexed.
        trade_id (str): Unique ID for this trade.

    Returns:
        list: A list of trade event dictionaries.
    """
    events = []
    pip_value = 0.1  # For XAUUSD, 1 pip = $0.10 price movement

    # --- SL Calculation ---
    raw_sl_points = abs(entry_signal['entry_price'] - entry_signal['sl_price'])
    raw_sl_pips = raw_sl_points / pip_value

    if raw_sl_pips > 10.0:
        actual_sl_pips = 10.0
        if entry_signal['direction'] == 'Buy':
            actual_sl_price = entry_signal['entry_price'] - (actual_sl_pips * pip_value)
        else: # Sell
            actual_sl_price = entry_signal['entry_price'] + (actual_sl_pips * pip_value)
    else:
        actual_sl_pips = raw_sl_pips
        actual_sl_price = entry_signal['sl_price']

    # --- TP Calculation ---
    if entry_signal['direction'] == 'Buy':
        tp1_price = entry_signal['entry_price'] + (actual_sl_pips * pip_value)
        tp2_price = entry_signal['entry_price'] + (2 * actual_sl_pips * pip_value)
    else: # Sell
        tp1_price = entry_signal['entry_price'] - (actual_sl_pips * pip_value)
        tp2_price = entry_signal['entry_price'] - (2 * actual_sl_pips * pip_value)

    # --- Initial Open Event ---
    open_event = {
        'trade_id': trade_id,
        'timestamp': entry_signal['entry_timestamp'],
        'action': 'OPEN',
        'price': entry_signal['entry_price'],
        'size_opened_percent': 100.0,
        'size_closed_percent': 0.0,
        'position_still_open_percent': 100.0,
        'direction': entry_signal['direction'],
        'type': entry_signal.get('type', 'N/A'),
        'raw_sl_price': entry_signal['sl_price'],
        'actual_sl_price': actual_sl_price,
        'tp1_price': tp1_price, # Specific to Bot 1
        'tp2_price': tp2_price, # Specific to Bot 1
        'raw_sl_pips': round(raw_sl_pips, 2),
        'actual_sl_pips': round(actual_sl_pips, 2)
    }
    events.append(open_event)

    position_open_percent = 100.0
    tp1_hit = False
    trade_active = True

    try:
        entry_candle_index = historical_data_df.index.get_loc(entry_signal['entry_timestamp'])
    except KeyError:
        fail_event = open_event.copy()
        fail_event['action'] = 'OPEN_FAILED_NO_DATA'
        fail_event['position_still_open_percent'] = 0.0
        fail_event.pop('tp1_price', None)
        fail_event.pop('tp2_price', None)
        return [fail_event]

    for current_candle_idx in range(entry_candle_index + 1, len(historical_data_df)):
        if not trade_active:
            break

        current_candle = historical_data_df.iloc[current_candle_idx]
        current_high = current_candle['High']
        current_low = current_candle['Low']
        current_ts = current_candle.name

        event_details = {
            'trade_id': trade_id,
            'timestamp': current_ts,
            'direction': entry_signal['direction'],
            'type': entry_signal.get('type', 'N/A'),
        }

        if entry_signal['direction'] == 'Buy':
            if current_low <= actual_sl_price: # Use actual_sl_price for Bot1 as SL doesn't move to BE
                event_details.update({
                    'action': 'CLOSE_SL', 'price': actual_sl_price,
                    'size_closed_percent': position_open_percent,
                })
                position_open_percent = 0.0
                trade_active = False
                events.append(event_details)
                break
            if not tp1_hit and current_high >= tp1_price:
                tp1_event = event_details.copy()
                tp1_event.update({
                    'action': 'PARTIAL_TP1', 'price': tp1_price,
                    'size_closed_percent': 50.0,
                })
                events.append(tp1_event)
                position_open_percent -= 50.0
                tp1_hit = True
            if tp1_hit and current_high >= tp2_price:
                tp2_event = event_details.copy()
                tp2_event.update({
                    'action': 'CLOSE_TP2', 'price': tp2_price,
                    'size_closed_percent': position_open_percent,
                })
                events.append(tp2_event)
                position_open_percent = 0.0
                trade_active = False
                break
        else:  # Sell
            if current_high >= actual_sl_price: # Use actual_sl_price for Bot1
                event_details.update({
                    'action': 'CLOSE_SL', 'price': actual_sl_price,
                    'size_closed_percent': position_open_percent,
                })
                position_open_percent = 0.0
                trade_active = False
                events.append(event_details)
                break
            if not tp1_hit and current_low <= tp1_price:
                tp1_event = event_details.copy()
                tp1_event.update({
                    'action': 'PARTIAL_TP1', 'price': tp1_price,
                    'size_closed_percent': 50.0,
                })
                events.append(tp1_event)
                position_open_percent -= 50.0
                tp1_hit = True
            if tp1_hit and current_low <= tp2_price:
                tp2_event = event_details.copy()
                tp2_event.update({
                    'action': 'CLOSE_TP2', 'price': tp2_price,
                    'size_closed_percent': position_open_percent,
                })
                events.append(tp2_event)
                position_open_percent = 0.0
                trade_active = False
                break

    current_pos_open_tracker = 100.0
    for event in events:
        if event['action'] == 'OPEN':
            event['position_still_open_percent'] = 100.0
            current_pos_open_tracker = 100.0
        elif event['action'] == 'PARTIAL_TP1':
            # For Bot 1, size_closed_percent is 50% of original.
            current_pos_open_tracker = 50.0
            event['position_still_open_percent'] = current_pos_open_tracker
        elif event['action'] in ['CLOSE_SL', 'CLOSE_TP2']:
            current_pos_open_tracker = 0.0
            event['position_still_open_percent'] = current_pos_open_tracker
    return events

def manage_trade_bot2(entry_signal, historical_data_df, trade_id):
    events = []
    pip_value = 0.1

    raw_sl_points = abs(entry_signal['entry_price'] - entry_signal['sl_price'])
    raw_sl_pips = raw_sl_points / pip_value

    if raw_sl_pips > 10.0:
        actual_sl_pips = 10.0
        if entry_signal['direction'] == 'Buy':
            actual_sl_price = entry_signal['entry_price'] - (actual_sl_pips * pip_value)
        else:
            actual_sl_price = entry_signal['entry_price'] + (actual_sl_pips * pip_value)
    else:
        actual_sl_pips = raw_sl_pips
        actual_sl_price = entry_signal['sl_price']

    if entry_signal['direction'] == 'Buy':
        tp_price = entry_signal['entry_price'] + (1.5 * actual_sl_pips * pip_value)
    else:
        tp_price = entry_signal['entry_price'] - (1.5 * actual_sl_pips * pip_value)

    open_event = {
        'trade_id': trade_id, 'timestamp': entry_signal['entry_timestamp'], 'action': 'OPEN',
        'price': entry_signal['entry_price'], 'size_opened_percent': 100.0, 'size_closed_percent': 0.0,
        'position_still_open_percent': 100.0, 'direction': entry_signal['direction'],
        'type': entry_signal.get('type', 'N/A'), 'raw_sl_price': entry_signal['sl_price'],
        'actual_sl_price': actual_sl_price, 'tp_price': tp_price,
        'raw_sl_pips': round(raw_sl_pips, 2), 'actual_sl_pips': round(actual_sl_pips, 2)
    }
    events.append(open_event)

    position_open_percent = 100.0
    trade_active = True

    try:
        entry_candle_index = historical_data_df.index.get_loc(entry_signal['entry_timestamp'])
    except KeyError:
        fail_event = open_event.copy()
        fail_event['action'] = 'OPEN_FAILED_NO_DATA'
        fail_event['position_still_open_percent'] = 0.0
        fail_event.pop('tp_price', None)
        return [fail_event]

    for current_candle_idx in range(entry_candle_index + 1, len(historical_data_df)):
        if not trade_active: break
        current_candle = historical_data_df.iloc[current_candle_idx]
        current_high = current_candle['High']
        current_low = current_candle['Low']
        current_ts = current_candle.name
        event_details = {'trade_id': trade_id, 'timestamp': current_ts,
                         'direction': entry_signal['direction'], 'type': entry_signal.get('type', 'N/A')}

        if entry_signal['direction'] == 'Buy':
            if current_low <= actual_sl_price:
                event_details.update({'action': 'CLOSE_SL', 'price': actual_sl_price, 'size_closed_percent': 100.0})
                position_open_percent = 0.0; trade_active = False; events.append(event_details); break
            if current_high >= tp_price:
                event_details.update({'action': 'CLOSE_TP', 'price': tp_price, 'size_closed_percent': 100.0})
                position_open_percent = 0.0; trade_active = False; events.append(event_details); break
        else:  # Sell
            if current_high >= actual_sl_price:
                event_details.update({'action': 'CLOSE_SL', 'price': actual_sl_price, 'size_closed_percent': 100.0})
                position_open_percent = 0.0; trade_active = False; events.append(event_details); break
            if current_low <= tp_price:
                event_details.update({'action': 'CLOSE_TP', 'price': tp_price, 'size_closed_percent': 100.0})
                position_open_percent = 0.0; trade_active = False; events.append(event_details); break

    current_pos_open_tracker = 100.0
    for event in events: # Assumes events are in order
        if event['action'] == 'OPEN': event['position_still_open_percent'] = 100.0
        elif event['action'] in ['CLOSE_SL', 'CLOSE_TP']: event['position_still_open_percent'] = 0.0
    return events

# --- Bot 3 Management Logic ---
def manage_trade_bot3(entry_signal, historical_data_df, trade_id):
    events = []
    pip_value = 0.1
    trailing_stop_pips = 15.0
    trailing_stop_amount = trailing_stop_pips * pip_value

    raw_sl_points = abs(entry_signal['entry_price'] - entry_signal['sl_price'])
    raw_sl_pips = raw_sl_points / pip_value

    if raw_sl_pips > 10.0:
        actual_sl_pips = 10.0
        if entry_signal['direction'] == 'Buy':
            actual_sl_price = entry_signal['entry_price'] - (actual_sl_pips * pip_value)
        else: # Sell
            actual_sl_price = entry_signal['entry_price'] + (actual_sl_pips * pip_value)
    else:
        actual_sl_pips = raw_sl_pips
        actual_sl_price = entry_signal['sl_price']

    if entry_signal['direction'] == 'Buy':
        breakeven_trigger_price = entry_signal['entry_price'] + (actual_sl_pips * pip_value) # 1R
    else: # Sell
        breakeven_trigger_price = entry_signal['entry_price'] - (actual_sl_pips * pip_value) # 1R

    open_event = {
        'trade_id': trade_id, 'timestamp': entry_signal['entry_timestamp'], 'action': 'OPEN',
        'price': entry_signal['entry_price'], 'size_opened_percent': 100.0, 'size_closed_percent': 0.0,
        'position_still_open_percent': 100.0, 'direction': entry_signal['direction'],
        'type': entry_signal.get('type', 'N/A'), 'raw_sl_price': entry_signal['sl_price'],
        'actual_sl_price': actual_sl_price, 'breakeven_trigger_price': breakeven_trigger_price,
        'trailing_stop_pips': trailing_stop_pips,
        'raw_sl_pips': round(raw_sl_pips, 2), 'actual_sl_pips': round(actual_sl_pips, 2)
    }
    events.append(open_event)

    current_stop_price = actual_sl_price
    sl_moved_to_breakeven = False
    position_open_percent = 100.0 # For Bot 3, it's always 100% until closed.
    trade_active = True

    try:
        entry_candle_index = historical_data_df.index.get_loc(entry_signal['entry_timestamp'])
    except KeyError:
        fail_event = open_event.copy()
        fail_event['action'] = 'OPEN_FAILED_NO_DATA'
        fail_event['position_still_open_percent'] = 0.0
        fail_event.pop('breakeven_trigger_price', None)
        fail_event.pop('trailing_stop_pips', None)
        return [fail_event]

    for current_candle_idx in range(entry_candle_index + 1, len(historical_data_df)):
        if not trade_active: break
        current_candle = historical_data_df.iloc[current_candle_idx]
        candle_high = current_candle['High']
        candle_low = current_candle['Low']
        current_ts = current_candle.name

        base_event_details = {'trade_id': trade_id, 'timestamp': current_ts,
                              'direction': entry_signal['direction'], 'type': entry_signal.get('type', 'N/A')}

        # 1. Stop Loss Check
        sl_hit_this_candle = False
        if entry_signal['direction'] == 'Buy':
            if candle_low <= current_stop_price:
                sl_hit_this_candle = True
        else: # Sell
            if candle_high >= current_stop_price:
                sl_hit_this_candle = True

        if sl_hit_this_candle:
            action_type = 'CLOSE_INITIAL_SL'
            if sl_moved_to_breakeven: # If BE was hit, this implies it's a BE or Trailed SL
                action_type = 'CLOSE_MODIFIED_SL' # Covers BE or Trailed SL

            sl_event = {**base_event_details, 'action': action_type, 'price': current_stop_price,
                        'size_closed_percent': 100.0, 'position_still_open_percent': 0.0}
            events.append(sl_event)
            trade_active = False
            break # Trade closed

        # 2. Breakeven Check (if not already at BE)
        if not sl_moved_to_breakeven:
            if entry_signal['direction'] == 'Buy' and candle_high >= breakeven_trigger_price:
                current_stop_price = entry_signal['entry_price']
                sl_moved_to_breakeven = True
                be_event = {**base_event_details, 'action': 'MOVE_SL_TO_BE', 'price': current_stop_price,
                              'new_sl_price': current_stop_price, 'position_still_open_percent': 100.0}
                events.append(be_event)
            elif entry_signal['direction'] == 'Sell' and candle_low <= breakeven_trigger_price:
                current_stop_price = entry_signal['entry_price']
                sl_moved_to_breakeven = True
                be_event = {**base_event_details, 'action': 'MOVE_SL_TO_BE', 'price': current_stop_price,
                              'new_sl_price': current_stop_price, 'position_still_open_percent': 100.0}
                events.append(be_event)

        # 3. Trailing Stop Logic (always active, potentially updating from initial SL or BE SL)
        if entry_signal['direction'] == 'Buy':
            potential_new_sl = candle_high - trailing_stop_amount
            if potential_new_sl > current_stop_price:
                current_stop_price = potential_new_sl
                sl_moved_to_breakeven = True # Once SL is trailed, it's considered modified past initial or BE.
                trail_event = {**base_event_details, 'action': 'TRAIL_SL', 'price': current_stop_price,
                                 'new_sl_price': current_stop_price, 'position_still_open_percent': 100.0}
                events.append(trail_event)
        else: # Sell
            potential_new_sl = candle_low + trailing_stop_amount
            if potential_new_sl < current_stop_price:
                current_stop_price = potential_new_sl
                sl_moved_to_breakeven = True # Once SL is trailed, it's considered modified.
                trail_event = {**base_event_details, 'action': 'TRAIL_SL', 'price': current_stop_price,
                                 'new_sl_price': current_stop_price, 'position_still_open_percent': 100.0}
                events.append(trail_event)

    # Final update of position_still_open_percent for events that don't close position
    for event in events:
        if 'position_still_open_percent' not in event: # OPEN event has it
             event['position_still_open_percent'] = 100.0 if trade_active else 0.0
        if event['action'] == 'OPEN': event['position_still_open_percent'] = 100.0
        elif event['action'] in ['MOVE_SL_TO_BE', 'TRAIL_SL']: event['position_still_open_percent'] = 100.0
        elif event['action'].startswith('CLOSE'): event['position_still_open_percent'] = 0.0


    return events


if __name__ == '__main__':
    print("--- Bot Management Test ---")
    pip_value = 0.1

    # --- Bot 1 Test Cases ---
    print("\n--- Bot 1 Tests ---")
    # ... (Bot 1 tests remain unchanged) ...
    entry_ts_buy1_bot1 = pd.Timestamp('2023-01-01 10:00:00', tz='America/New_York')
    buy_signal1_bot1 = {
        'entry_timestamp': entry_ts_buy1_bot1, 'direction': 'Buy',
        'entry_price': 1900.0, 'sl_price': 1899.5, 'type': 'Test FVG Buy Bot1'
    }
    data_buy1_bot1 = [
        {'Timestamp': entry_ts_buy1_bot1, 'Open': 1900, 'High': 1900.2, 'Low': 1899.8, 'Close': 1900.0},
        {'Timestamp': entry_ts_buy1_bot1 + pd.Timedelta(minutes=5), 'Open': 1900, 'High': 1900.5, 'Low': 1899.9, 'Close': 1900.4},
        {'Timestamp': entry_ts_buy1_bot1 + pd.Timedelta(minutes=10), 'Open': 1900.5, 'High': 1901.1, 'Low': 1900.4, 'Close': 1901.0},
    ]
    df_buy1_bot1 = pd.DataFrame(data_buy1_bot1).set_index('Timestamp')
    print("\nTest Case 1.1: Buy hits TP1, TP2")
    events_buy1_bot1 = manage_trade_bot1(buy_signal1_bot1, df_buy1_bot1, "Trade_Bot1_Buy_TP1_TP2")
    for e in events_buy1_bot1: print(e)

    entry_ts_sell1_bot1 = pd.Timestamp('2023-01-01 11:00:00', tz='America/New_York')
    sell_signal1_bot1 = {
        'entry_timestamp': entry_ts_sell1_bot1, 'direction': 'Sell',
        'entry_price': 1910.0, 'sl_price': 1910.8, 'type': 'Test LS Sell Bot1'
    }
    data_sell1_bot1 = [
        {'Timestamp': entry_ts_sell1_bot1, 'Open': 1910.0, 'High': 1910.2, 'Low': 1909.8, 'Close': 1910.0},
        {'Timestamp': entry_ts_sell1_bot1 + pd.Timedelta(minutes=5), 'Open': 1910.1, 'High': 1910.9, 'Low': 1909.5, 'Close': 1910.5},
    ]
    df_sell1_bot1 = pd.DataFrame(data_sell1_bot1).set_index('Timestamp')
    print("\nTest Case 1.2: Sell hits SL")
    events_sell1_bot1 = manage_trade_bot1(sell_signal1_bot1, df_sell1_bot1, "Trade_Bot1_Sell_SL")
    for e in events_sell1_bot1: print(e)

    entry_ts_buy2_bot1 = pd.Timestamp('2023-01-01 12:00:00', tz='America/New_York')
    buy_signal2_bot1 = {
        'entry_timestamp': entry_ts_buy2_bot1, 'direction': 'Buy',
        'entry_price': 1900.0, 'sl_price': 1898.5, 'type': 'Test BoS Buy Bot1'
    }
    data_buy2_bot1 = [
        {'Timestamp': entry_ts_buy2_bot1, 'Open': 1900, 'High': 1900.2, 'Low': 1899.8, 'Close': 1900.0},
        {'Timestamp': entry_ts_buy2_bot1 + pd.Timedelta(minutes=5), 'Open': 1899.5, 'High': 1900.0, 'Low': 1898.9, 'Close': 1899.0},
    ]
    df_buy2_bot1 = pd.DataFrame(data_buy2_bot1).set_index('Timestamp')
    print("\nTest Case 1.3: Buy with SL capping")
    events_buy2_bot1 = manage_trade_bot1(buy_signal2_bot1, df_buy2_bot1, "Trade_Bot1_Buy_SL_Capped")
    for e in events_buy2_bot1: print(e)

    entry_ts_buy3_bot1 = pd.Timestamp('2023-01-01 13:00:00', tz='America/New_York')
    buy_signal3_bot1 = {
        'entry_timestamp': entry_ts_buy3_bot1, 'direction': 'Buy',
        'entry_price': 1900.0, 'sl_price': 1899.5, 'type': 'Test FVG Buy SL Prio Bot1'
    }
    data_buy3_bot1 = [
        {'Timestamp': entry_ts_buy3_bot1, 'Open': 1900, 'High': 1900.2, 'Low': 1899.8, 'Close': 1900.0},
        {'Timestamp': entry_ts_buy3_bot1 + pd.Timedelta(minutes=5), 'Open': 1900, 'High': 1900.6, 'Low': 1899.4, 'Close': 1899.5},
    ]
    df_buy3_bot1 = pd.DataFrame(data_buy3_bot1).set_index('Timestamp')
    print("\nTest Case 1.4: Buy hits SL and TP1 on same candle")
    events_buy3_bot1 = manage_trade_bot1(buy_signal3_bot1, df_buy3_bot1, "Trade_Bot1_Buy_SL_TP1_SameCandle")
    for e in events_buy3_bot1: print(e)

    # --- Bot 2 Test Cases ---
    print("\n--- Bot 2 Tests ---")
    # ... (Bot 2 tests remain unchanged) ...
    entry_ts_buy1_bot2 = pd.Timestamp('2023-01-02 10:00:00', tz='America/New_York')
    buy_signal1_bot2 = {
        'entry_timestamp': entry_ts_buy1_bot2, 'direction': 'Buy',
        'entry_price': 2000.0, 'sl_price': 1999.0,
        'type': 'Test FVG Buy Bot2 TP'
    }
    data_buy1_bot2 = [
        {'Timestamp': entry_ts_buy1_bot2, 'Open': 2000.0, 'High': 2000.5, 'Low': 1999.5, 'Close': 2000.0},
        {'Timestamp': entry_ts_buy1_bot2 + pd.Timedelta(minutes=5), 'Open': 2000.1, 'High': 2001.6, 'Low': 2000.0, 'Close': 2001.5},
    ]
    df_buy1_bot2 = pd.DataFrame(data_buy1_bot2).set_index('Timestamp')
    print("\nTest Case 2.1: Bot 2 Buy hits TP")
    events_buy1_bot2 = manage_trade_bot2(buy_signal1_bot2, df_buy1_bot2, "Trade_Bot2_Buy_TP")
    for e in events_buy1_bot2: print(e)

    entry_ts_sell1_bot2 = pd.Timestamp('2023-01-02 11:00:00', tz='America/New_York')
    sell_signal1_bot2 = {
        'entry_timestamp': entry_ts_sell1_bot2, 'direction': 'Sell',
        'entry_price': 2010.0, 'sl_price': 2010.7,
        'type': 'Test LS Sell Bot2 SL'
    }
    data_sell1_bot2 = [
        {'Timestamp': entry_ts_sell1_bot2, 'Open': 2010.0, 'High': 2010.2, 'Low': 2009.8, 'Close': 2010.0},
        {'Timestamp': entry_ts_sell1_bot2 + pd.Timedelta(minutes=5), 'Open': 2010.1, 'High': 2010.8, 'Low': 2009.5, 'Close': 2010.7},
    ]
    df_sell1_bot2 = pd.DataFrame(data_sell1_bot2).set_index('Timestamp')
    print("\nTest Case 2.2: Bot 2 Sell hits SL")
    events_sell1_bot2 = manage_trade_bot2(sell_signal1_bot2, df_sell1_bot2, "Trade_Bot2_Sell_SL")
    for e in events_sell1_bot2: print(e)

    entry_ts_buy2_bot2 = pd.Timestamp('2023-01-02 12:00:00', tz='America/New_York')
    buy_signal2_bot2 = {
        'entry_timestamp': entry_ts_buy2_bot2, 'direction': 'Buy',
        'entry_price': 2000.0, 'sl_price': 1998.2,
        'type': 'Test BoS Buy Bot2 SL Cap'
    }
    data_buy2_bot2 = [
        {'Timestamp': entry_ts_buy2_bot2, 'Open': 2000.0, 'High': 2000.2, 'Low': 1999.8, 'Close': 2000.0},
        {'Timestamp': entry_ts_buy2_bot2 + pd.Timedelta(minutes=5), 'Open': 2001.0, 'High': 2001.5, 'Low': 2000.5, 'Close': 2001.2},
    ]
    df_buy2_bot2 = pd.DataFrame(data_buy2_bot2).set_index('Timestamp')
    print("\nTest Case 2.3: Bot 2 Buy with SL capping hits TP")
    events_buy2_bot2 = manage_trade_bot2(buy_signal2_bot2, df_buy2_bot2, "Trade_Bot2_Buy_SLCap_TP")
    for e in events_buy2_bot2: print(e)

    # --- Bot 3 Test Cases ---
    print("\n--- Bot 3 Tests ---")
    # Test Case 3.1: Buy trade hits BE, then trails, then stopped by trailing SL
    entry_ts_buy1_bot3 = pd.Timestamp('2023-01-03 10:00:00', tz='America/New_York')
    buy_signal1_bot3 = {
        'entry_timestamp': entry_ts_buy1_bot3, 'direction': 'Buy',
        'entry_price': 2000.0, 'sl_price': 1999.2, # Raw SL = 0.8 points = 8 pips
        'type': 'Test Bot3 Buy Trail'
    }
    # Actual SL = 8 pips (0.8 points) -> actual_sl_price = 1999.2
    # BE Trigger = 2000.0 + 0.8 = 2000.8
    # Trailing Stop Amount = 15 pips * 0.1 = 1.5 points
    data_buy1_bot3 = [
        # Entry Candle
        {'Timestamp': entry_ts_buy1_bot3, 'Open': 2000.0, 'High': 2000.5, 'Low': 1999.5, 'Close': 2000.0},
        # Candle 1: Hits BE
        {'Timestamp': entry_ts_buy1_bot3 + pd.Timedelta(minutes=5), 'Open': 2000.1, 'High': 2000.9, 'Low': 2000.0, 'Close': 2000.8}, # High=2000.9 >= BE_Trigger=2000.8. SL moves to 2000.0
        # Candle 2: Trails SL. New SL = High(2001.5) - 1.5 = 2000.0. current_stop_price is already 2000.0, so no TRAIL_SL event. Let's make High higher.
        #           High=2002.0. New SL = 2002.0 - 1.5 = 2000.5. This is > 2000.0.
        {'Timestamp': entry_ts_buy1_bot3 + pd.Timedelta(minutes=10), 'Open': 2000.8, 'High': 2002.0, 'Low': 2000.7, 'Close': 2001.8}, # SL trails to 2000.5
        # Candle 3: Trails SL again. New SL = High(2002.8) - 1.5 = 2001.3. This is > 2000.5
        {'Timestamp': entry_ts_buy1_bot3 + pd.Timedelta(minutes=15), 'Open': 2001.8, 'High': 2002.8, 'Low': 2001.7, 'Close': 2002.5}, # SL trails to 2001.3
        # Candle 4: Hits trailed SL. Current SL = 2001.3
        {'Timestamp': entry_ts_buy1_bot3 + pd.Timedelta(minutes=20), 'Open': 2002.5, 'High': 2002.6, 'Low': 2001.2, 'Close': 2001.5}, # Low=2001.2 <= SL=2001.3
    ]
    df_buy1_bot3 = pd.DataFrame(data_buy1_bot3).set_index('Timestamp')
    print("\nTest Case 3.1: Bot 3 Buy BE, Trails, then Stopped by Trailing SL")
    events_buy1_bot3 = manage_trade_bot3(buy_signal1_bot3, df_buy1_bot3, "Trade_Bot3_Buy_Trail")
    for e in events_buy1_bot3: print(e)

    # Test Case 3.2: Sell trade hits initial SL before BE
    entry_ts_sell1_bot3 = pd.Timestamp('2023-01-03 11:00:00', tz='America/New_York')
    sell_signal1_bot3 = {
        'entry_timestamp': entry_ts_sell1_bot3, 'direction': 'Sell',
        'entry_price': 2010.0, 'sl_price': 2010.5, # Raw SL = 0.5 points = 5 pips
        'type': 'Test Bot3 Sell InitialSL'
    }
    # Actual SL = 5 pips (0.5 points) -> actual_sl_price = 2010.5
    # BE Trigger = 2010.0 - 0.5 = 2009.5
    data_sell1_bot3 = [
        {'Timestamp': entry_ts_sell1_bot3, 'Open': 2010.0, 'High': 2010.2, 'Low': 2009.8, 'Close': 2010.0}, # Entry
        {'Timestamp': entry_ts_sell1_bot3 + pd.Timedelta(minutes=5), 'Open': 2010.1, 'High': 2010.6, 'Low': 2009.9, 'Close': 2010.5}, # High=2010.6 >= SL=2010.5. Low=2009.9 > BE_trigger=2009.5 (BE not hit)
    ]
    df_sell1_bot3 = pd.DataFrame(data_sell1_bot3).set_index('Timestamp')
    print("\nTest Case 3.2: Bot 3 Sell hits Initial SL")
    events_sell1_bot3 = manage_trade_bot3(sell_signal1_bot3, df_sell1_bot3, "Trade_Bot3_Sell_InitialSL")
    for e in events_sell1_bot3: print(e)

    # Test Case 3.3: Buy trade with SL capping, then BE, then stopped.
    entry_ts_buy2_bot3 = pd.Timestamp('2023-01-03 12:00:00', tz='America/New_York')
    buy_signal2_bot3 = {
        'entry_timestamp': entry_ts_buy2_bot3, 'direction': 'Buy',
        'entry_price': 2000.0, 'sl_price': 1998.5, # Raw SL = 1.5 points = 15 pips
        'type': 'Test Bot3 Buy CapBE'
    }
    # Capped SL = 10 pips (1.0 points) -> actual_sl_price = 1999.0
    # BE Trigger = 2000.0 + 1.0 = 2001.0
    data_buy2_bot3 = [
        {'Timestamp': entry_ts_buy2_bot3, 'Open': 2000.0, 'High': 2000.5, 'Low': 1999.5, 'Close': 2000.0}, # Entry
        {'Timestamp': entry_ts_buy2_bot3 + pd.Timedelta(minutes=5), 'Open': 2000.1, 'High': 2001.1, 'Low': 2000.0, 'Close': 2001.0}, # High=2001.1 >= BE_Trigger=2001.0. SL moves to 2000.0
        {'Timestamp': entry_ts_buy2_bot3 + pd.Timedelta(minutes=10), 'Open': 2000.5, 'High': 2000.6, 'Low': 1999.9, 'Close': 2000.0}, # Low=1999.9 <= SL=2000.0 (hits BE SL)
    ]
    df_buy2_bot3 = pd.DataFrame(data_buy2_bot3).set_index('Timestamp')
    print("\nTest Case 3.3: Bot 3 Buy SL Cap, then BE, then Stopped at BE")
    events_buy2_bot3 = manage_trade_bot3(buy_signal2_bot3, df_buy2_bot3, "Trade_Bot3_Buy_CapBE_Stop")
    for e in events_buy2_bot3: print(e)


    print("\n--- Bot Management Test Finished ---")
