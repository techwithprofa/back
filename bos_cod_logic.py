import pandas as pd
from liquidity_sweep_logic import is_clear_swing_high, is_clear_swing_low

# --- Configuration ---
PIP_OFFSET_FOR_SL_BOS = 0.0
TOTAL_LOOKBACK_PERIOD = 200
MAX_CANDLES_BETWEEN_SWINGS = 50
DEBUG_BOS = False # Control global debug prints

# --- Helper function to find previous swings ---
def find_previous_swing(df, end_index, is_searching_for_high, max_lookback, context_current_candle_idx=None):
    """
    Finds the most recent clear swing high or low before end_index.
    """
    # Detailed print only if the global DEBUG_BOS is on AND this specific candle is targeted by a test
    verbose_print = DEBUG_BOS and df.attrs.get('is_test_data', False) and \
                    df.attrs.get('debug_specific_candle', -1) == context_current_candle_idx

    if verbose_print:
        print(f"  [find_previous_swing called for current_idx={context_current_candle_idx}] Searching for {'High' if is_searching_for_high else 'Low'} from {end_index} back by {max_lookback}")

    for i in range(end_index, max(1, end_index - max_lookback) -1, -1):
        if i < 2 or i > len(df) - 3:
            continue

        swing_found = False
        if is_searching_for_high:
            if is_clear_swing_high(df, i):
                swing_found = True
        else:
            if is_clear_swing_low(df, i):
                swing_found = True

        if swing_found:
            price = df['High'].iloc[i] if is_searching_for_high else df['Low'].iloc[i]
            if verbose_print:
                print(f"    Found swing {'High' if is_searching_for_high else 'Low'} at index {i} (Timestamp: {df.index[i].strftime('%H:%M')}), price {price:.2f}")
            return {'index': i, 'price': price, 'timestamp': df.index[i]}

    if verbose_print:
        print(f"    No swing found in range for {'High' if is_searching_for_high else 'Low'}")
    return None

# --- Main Logic Function ---
def find_bos_entries(df):
    entries = []
    if len(df) < (3 * MAX_CANDLES_BETWEEN_SWINGS + 10 + 5) : # Min length for a full pattern + buffer
        if DEBUG_BOS and df.attrs.get('verbose_debug', False) : print(f"DataFrame too short: {len(df)}")
        return entries

    trade_window_start = pd.to_datetime('21:00:00').time()
    trade_window_end = pd.to_datetime('23:30:00').time()

    required_min_lookback = 3 * MAX_CANDLES_BETWEEN_SWINGS + 10
    start_iteration_idx = max(5, required_min_lookback)

    if df.attrs.get('is_test_data', False) and df.attrs.get('test_start_iteration_idx', -1) != -1:
        start_iteration_idx = df.attrs.get('test_start_iteration_idx')
        if DEBUG_BOS and df.attrs.get('verbose_debug', False):
            print(f"[DEBUG_BOS] Using start_iteration_idx for test: {start_iteration_idx}")


    for current_candle_idx in range(start_iteration_idx, len(df)):
        current_candle_timestamp = df.index[current_candle_idx]
        current_candle_time = current_candle_timestamp.time()

        # Setup for targeted verbose printing
        is_target_test_candle = False
        if df.attrs.get('is_test_data', False):
            if (current_candle_idx == df.attrs.get('test_break_candle_idx_bullish', -2) or \
                current_candle_idx == df.attrs.get('test_break_candle_idx_bearish', -2)):
                df.attrs['debug_specific_candle'] = current_candle_idx
                is_target_test_candle = True
            elif df.attrs.get('debug_specific_candle', -1) != -1 : # Reset if not the target
                 df.attrs['debug_specific_candle'] = -1

        verbose_debug_current_candle = DEBUG_BOS and df.attrs.get('verbose_debug', False) and is_target_test_candle

        if verbose_debug_current_candle:
             print(f"\n[DEBUG_BOS] Processing Candle at index {current_candle_idx} ({current_candle_timestamp.strftime('%Y-%m-%d %H:%M')})")

        if not (trade_window_start <= current_candle_time <= trade_window_end):
            if verbose_debug_current_candle: print(f"    Skipping due to trade window: {current_candle_time}")
            continue

        current_high = df['High'].iloc[current_candle_idx]
        current_low = df['Low'].iloc[current_candle_idx]
        current_close = df['Close'].iloc[current_candle_idx]

        # --- Bullish BoS (Buy Signal) ---
        csl2_bull = find_previous_swing(df, current_candle_idx - 1, False, MAX_CANDLES_BETWEEN_SWINGS, current_candle_idx)
        if csl2_bull:
            csh2_bull = find_previous_swing(df, csl2_bull['index'] - 1, True, MAX_CANDLES_BETWEEN_SWINGS, current_candle_idx)
            if csh2_bull:
                csl1_bull = find_previous_swing(df, csh2_bull['index'] - 1, False, MAX_CANDLES_BETWEEN_SWINGS, current_candle_idx)
                if csl1_bull:
                    csh1_bull = find_previous_swing(df, csl1_bull['index'] - 1, True, MAX_CANDLES_BETWEEN_SWINGS, current_candle_idx)
                    if csh1_bull:
                        if (current_candle_idx - csh1_bull['index']) <= TOTAL_LOOKBACK_PERIOD:
                            is_downtrend = csh2_bull['price'] < csh1_bull['price'] and csl2_bull['price'] < csl1_bull['price']
                            if verbose_debug_current_candle: print(f"    Bullish Check: CSH1({csh1_bull['price']:.2f}@{csh1_bull['index']}), CSL1({csl1_bull['price']:.2f}@{csl1_bull['index']}), CSH2(SSH)({csh2_bull['price']:.2f}@{csh2_bull['index']}), CSL2({csl2_bull['price']:.2f}@{csl2_bull['index']}). Downtrend={is_downtrend}")

                            if is_downtrend:
                                ssh_price = csh2_bull['price']
                                ssh_timestamp = csh2_bull['timestamp']
                                if current_close > ssh_price and current_high > ssh_price:
                                    if verbose_debug_current_candle: print(f"    >>> BULLISH BoS CONFIRMED: Close({current_close:.2f}) > SSH({ssh_price:.2f}) AND High({current_high:.2f}) > SSH({ssh_price:.2f})")
                                    entry_signal = {
                                        'entry_timestamp': current_candle_timestamp,
                                        'direction': 'Buy',
                                        'entry_price': current_close,
                                        'sl_price': current_low,
                                        'type': 'Bullish BoS',
                                        'broken_swing_price': ssh_price,
                                        'broken_swing_timestamp': ssh_timestamp,
                                        # 'debug_swings': {'csl2':csl2_bull, 'csh2':csh2_bull, 'csl1':csl1_bull, 'csh1':csh1_bull}
                                    }
                                    entries.append(entry_signal)
                                    continue
                                elif verbose_debug_current_candle: print(f"    Bullish BoS FAILED break condition: Close({current_close:.2f}) vs SSH({ssh_price:.2f}), High({current_high:.2f}) vs SSH({ssh_price:.2f})")
                        elif verbose_debug_current_candle: print(f"    Bullish pattern too long: {current_candle_idx - csh1_bull['index']} > {TOTAL_LOOKBACK_PERIOD}")

        # --- Bearish BoS (Sell Signal) ---
        csh2_bear = find_previous_swing(df, current_candle_idx - 1, True, MAX_CANDLES_BETWEEN_SWINGS, current_candle_idx)
        if csh2_bear:
            csl2_bear = find_previous_swing(df, csh2_bear['index'] - 1, False, MAX_CANDLES_BETWEEN_SWINGS, current_candle_idx)
            if csl2_bear:
                csh1_bear = find_previous_swing(df, csl2_bear['index'] - 1, True, MAX_CANDLES_BETWEEN_SWINGS, current_candle_idx)
                if csh1_bear:
                    csl1_bear = find_previous_swing(df, csh1_bear['index'] - 1, False, MAX_CANDLES_BETWEEN_SWINGS, current_candle_idx)
                    if csl1_bear:
                        if (current_candle_idx - csl1_bear['index']) <= TOTAL_LOOKBACK_PERIOD:
                            is_uptrend = csl2_bear['price'] > csl1_bear['price'] and csh2_bear['price'] > csh1_bear['price']
                            if verbose_debug_current_candle: print(f"    Bearish Check: CSL1({csl1_bear['price']:.2f}@{csl1_bear['index']}), CSH1({csh1_bear['price']:.2f}@{csh1_bear['index']}), CSL2(SSL)({csl2_bear['price']:.2f}@{csl2_bear['index']}), CSH2({csh2_bear['price']:.2f}@{csh2_bear['index']}). Uptrend={is_uptrend}")

                            if is_uptrend:
                                ssl_price = csl2_bear['price']
                                ssl_timestamp = csl2_bear['timestamp']
                                if current_close < ssl_price and current_low < ssl_price:
                                    if verbose_debug_current_candle: print(f"    >>> BEARISH BoS CONFIRMED: Close({current_close:.2f}) < SSL({ssl_price:.2f}) AND Low({current_low:.2f}) < SSL({ssl_price:.2f})")
                                    entry_signal = {
                                        'entry_timestamp': current_candle_timestamp,
                                        'direction': 'Sell',
                                        'entry_price': current_close,
                                        'sl_price': current_high,
                                        'type': 'Bearish BoS',
                                        'broken_swing_price': ssl_price,
                                        'broken_swing_timestamp': ssl_timestamp,
                                        # 'debug_swings': {'csh2':csh2_bear, 'csl2':csl2_bear, 'csh1':csh1_bear, 'csl1':csl1_bear}
                                    }
                                    entries.append(entry_signal)
                                elif verbose_debug_current_candle: print(f"    Bearish BoS FAILED break condition: Close({current_close:.2f}) vs SSL({ssl_price:.2f}), Low({current_low:.2f}) vs SSL({ssl_price:.2f})")
                        elif verbose_debug_current_candle: print(f"    Bearish pattern too long: {current_candle_idx - csl1_bear['index']} > {TOTAL_LOOKBACK_PERIOD}")

    # Clean up df.attrs if they were set for testing
    if df.attrs.get('is_test_data', False):
        if 'debug_specific_candle' in df.attrs: del df.attrs['debug_specific_candle']
        if 'verbose_debug' in df.attrs: del df.attrs['verbose_debug']
        # Keep other test_attrs like test_start_iteration_idx for record if needed, or remove.

    return entries

# --- Example Usage ---
if __name__ == '__main__':
    # DEBUG_BOS = True # Keep False for submission, True for local testing if needed
    # VERBOSE_DEBUG_FOR_TESTS = True # For local testing

    # Store original global config
    _orig_total_lookback = TOTAL_LOOKBACK_PERIOD
    _orig_max_candles_between = MAX_CANDLES_BETWEEN_SWINGS
    _orig_debug_bos = DEBUG_BOS

    # --- Test Bullish BoS ---
    # Setup test-specific configurations
    DEBUG_BOS = True # Enable for this test block
    VERBOSE_DEBUG_FOR_TESTS = True
    TOTAL_LOOKBACK_PERIOD = 10
    MAX_CANDLES_BETWEEN_SWINGS = 4

    bullish_bos_data = {
        'Open':  [100,110,101,100,107, 99, 97,100,109],
        'High':  [102,112,103,102,109,101, 99,102,110],
        'Low':   [101,108,100,101,106, 98, 99, 99,107],
        'Close': [101,109,101,101,107, 99, 99,100,109.5],
    }
    idx_bull = []
    base_time_bull = pd.Timestamp('2023-02-01 10:00:00', tz='America/New_York')
    for i in range(len(bullish_bos_data['Open'])):
        if i == 8:
            idx_bull.append(pd.Timestamp('2023-02-01 21:30:00', tz='America/New_York'))
        else:
            idx_bull.append(base_time_bull + pd.Timedelta(minutes=30*i))
    sample_df_bullish = pd.DataFrame(bullish_bos_data, index=pd.DatetimeIndex(idx_bull))
    padding_data_list = [{'Open': 100, 'High': 101, 'Low': 102, 'Close': 100}] * 20
    padding_idx = [base_time_bull - pd.Timedelta(minutes=30*(i+1)) for i in range(20)]
    padding_df = pd.DataFrame(padding_data_list, index=pd.DatetimeIndex(padding_idx[::-1]))
    combined_df_bullish = pd.concat([padding_df, sample_df_bullish])

    combined_df_bullish.attrs['is_test_data'] = True
    combined_df_bullish.attrs['verbose_debug'] = VERBOSE_DEBUG_FOR_TESTS
    combined_df_bullish.attrs['test_start_iteration_idx'] = 28
    combined_df_bullish.attrs['test_break_candle_idx_bullish'] = 28

    print("--- Testing Bullish BoS ---")
    bos_entries_bullish = find_bos_entries(combined_df_bullish)
    print(f"\nFound {len(bos_entries_bullish)} Bullish BoS Entries:")
    for entry in bos_entries_bullish: print(f"- {entry}")
    DEBUG_BOS = _orig_debug_bos # Restore global DEBUG_BOS

    # --- Test Bearish BoS ---
    DEBUG_BOS = True # Enable for this test block
    VERBOSE_DEBUG_FOR_TESTS = True
    TOTAL_LOOKBACK_PERIOD = 10
    MAX_CANDLES_BETWEEN_SWINGS = 4
    bearish_bos_data = {
        'Open':  [100, 90, 99,100, 93,101,103, 95, 91],
        'High':  [100, 92,101,100, 95,103,102, 97, 91.5],
        'Low':   [ 90, 88, 98, 99, 92,100,101, 93, 89],
        'Close': [ 90, 89,100,100, 93,102,102, 94, 90],
    }
    idx_bear = []
    base_time_bear = pd.Timestamp('2023-02-02 10:00:00', tz='America/New_York')
    for i in range(len(bearish_bos_data['Open'])):
        if i == 8:
            idx_bear.append(pd.Timestamp('2023-02-02 22:00:00', tz='America/New_York'))
        else:
            idx_bear.append(base_time_bear + pd.Timedelta(minutes=30*i))
    sample_df_bearish_short = pd.DataFrame(bearish_bos_data, index=pd.DatetimeIndex(idx_bear))
    padding_data_list_bear = [{'Open': 100, 'High': 100, 'Low': 90, 'Close': 100}] * 20
    padding_idx_bear = [base_time_bear - pd.Timedelta(minutes=30*(i+1)) for i in range(20)]
    padding_df_bear = pd.DataFrame(padding_data_list_bear, index=pd.DatetimeIndex(padding_idx_bear[::-1]))
    combined_df_bearish = pd.concat([padding_df_bear, sample_df_bearish_short])

    combined_df_bearish.attrs['is_test_data'] = True
    combined_df_bearish.attrs['verbose_debug'] = VERBOSE_DEBUG_FOR_TESTS
    combined_df_bearish.attrs['test_start_iteration_idx'] = 28
    combined_df_bearish.attrs['test_break_candle_idx_bearish'] = 28

    print("\n--- Testing Bearish BoS ---")
    bos_entries_bearish = find_bos_entries(combined_df_bearish)
    print(f"\nFound {len(bos_entries_bearish)} Bearish BoS Entries:")
    for entry in bos_entries_bearish: print(f"- {entry}")

    # Restore global settings
    TOTAL_LOOKBACK_PERIOD = _orig_total_lookback
    MAX_CANDLES_BETWEEN_SWINGS = _orig_max_candles_between
    DEBUG_BOS = _orig_debug_bos

    print("\n--- End of Example Usage ---")
