import pandas as pd

# --- Configuration ---
SWING_LOOKBACK_CANDLES = 50  # Max candles to look back for a swing high/low
PIP_OFFSET_FOR_SL = 1.0      # 10 pips for XAUUSD (1 pip = 0.10 price movement)

# --- Helper Functions ---
def is_clear_swing_high(df, index):
    """
    Checks if the candle at 'index' is a clear swing high.
    A clear swing high is strictly greater than the highs of 2 prior and 2 post candles.
    """
    if index < 2 or index > len(df) - 3:
        return False  # Not enough data for 2 prior and 2 post candles

    current_high = df['High'].iloc[index]

    # Check 2 prior candles
    if current_high <= df['High'].iloc[index-1] or \
       current_high <= df['High'].iloc[index-2]:
        return False

    # Check 2 post candles
    if current_high <= df['High'].iloc[index+1] or \
       current_high <= df['High'].iloc[index+2]:
        return False

    return True

def is_clear_swing_low(df, index):
    """
    Checks if the candle at 'index' is a clear swing low.
    A clear swing low is strictly less than the lows of 2 prior and 2 post candles.
    """
    if index < 2 or index > len(df) - 3:
        return False  # Not enough data for 2 prior and 2 post candles

    current_low = df['Low'].iloc[index]

    # Check 2 prior candles
    if current_low >= df['Low'].iloc[index-1] or \
       current_low >= df['Low'].iloc[index-2]:
        return False

    # Check 2 post candles
    if current_low >= df['Low'].iloc[index+1] or \
       current_low >= df['Low'].iloc[index+2]:
        return False

    return True

# --- Main Logic Function ---
def find_liquidity_sweep_entries(df):
    """
    Identifies Liquidity Sweep (Failure Swing) entry signals.

    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns,
                           indexed by NY-localized timestamps.

    Returns:
        list: A list of liquidity sweep entry signal dictionaries.
    """
    entries = []
    if len(df) < 5: # Need at least 5 candles for swing checks + current candle
        return entries

    trade_window_start = pd.to_datetime('21:00:00').time()
    trade_window_end = pd.to_datetime('23:30:00').time()

    # Iterate through each candle as a potential sweep candle
    # Start from an index that allows looking back for swings (e.g. SWING_LOOKBACK_CANDLES or at least 2 for swing check context)
    # The check for swing high/low itself needs 2 candles before and 2 after.
    # The sweep candle `i` must be after the swing point.
    for i in range(2, len(df)): # Start from index 2, so i-1 can be 1, i-2 can be 0.
        current_candle_timestamp = df.index[i]
        current_candle_time = current_candle_timestamp.time()

        # Check if current (sweep) candle is within the trading window
        if not (trade_window_start <= current_candle_time <= trade_window_end):
            continue

        current_open = df['Open'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        current_close = df['Close'].iloc[i]

        # --- Check for Buy Signal (Sweep of Low) ---
        # Iterate backwards from i-1 to find the most recent Clear Swing Low (CSL)
        # The CSL must be at least 2 candles before current candle `i` to give it space to form.
        # Search window for CSL: from max(0, i - SWING_LOOKBACK_CANDLES) up to i-3 (inclusive for csl_idx)
        # is_clear_swing_low needs index+2 to be valid, so csl_idx_max = i-3 ensures (i-3)+2 = i-1 < i

        # Search for CSL from i-3 down to max(2, i - SWING_LOOKBACK_CANDLES -1)
        # The csl_idx must be at least 2 itself for is_clear_swing_low to work.
        # The loop for csl_idx should go from i-3 down to a minimum of 2.
        # The lookback limit is also applied.
        start_search_csl = i - 3 # Latest possible CSL (to allow CSL+1, CSL+2 to exist before current candle i)
        end_search_csl = max(2, i - SWING_LOOKBACK_CANDLES) # Oldest possible CSL (must be at least index 2)

        found_csl = None
        csl_timestamp = None
        swing_low_level = None

        if start_search_csl >= 2: # Ensure there's a valid range to search
            for csl_idx in range(start_search_csl, end_search_csl - 1, -1):
                if is_clear_swing_low(df, csl_idx):
                    found_csl = True
                    swing_low_level = df['Low'].iloc[csl_idx]
                    csl_timestamp = df.index[csl_idx]
                    break

        if found_csl:
            # Check if current candle i swept this low and closed back above
            if current_low < swing_low_level and current_close > swing_low_level:
                entry_signal = {
                    'entry_timestamp': current_candle_timestamp,
                    'direction': 'Buy',
                    'entry_price': current_close,
                    'sl_price': current_low - PIP_OFFSET_FOR_SL,
                    'triggering_swing_timestamp': csl_timestamp,
                    'triggering_swing_level': swing_low_level,
                    'type': 'Liquidity Sweep (Buy)'
                }
                entries.append(entry_signal)
                # Optional: once a buy sweep is found for current candle i, can skip sell check for this i
                # continue

        # --- Check for Sell Signal (Sweep of High) ---
        # Iterate backwards from i-1 to find the most recent Clear Swing High (CSH)
        # Similar logic as CSL search window
        start_search_csh = i - 3 # Latest possible CSH
        end_search_csh = max(2, i - SWING_LOOKBACK_CANDLES) # Oldest possible CSH

        found_csh = None
        csh_timestamp = None
        swing_high_level = None

        if start_search_csh >= 2: # Ensure there's a valid range to search
            for csh_idx in range(start_search_csh, end_search_csh - 1, -1):
                if is_clear_swing_high(df, csh_idx):
                    found_csh = True
                    swing_high_level = df['High'].iloc[csh_idx]
                    csh_timestamp = df.index[csh_idx]
                    break

        if found_csh:
            # Check if current candle i swept this high and closed back below
            if current_high > swing_high_level and current_close < swing_high_level:
                entry_signal = {
                    'entry_timestamp': current_candle_timestamp,
                    'direction': 'Sell',
                    'entry_price': current_close,
                    'sl_price': current_high + PIP_OFFSET_FOR_SL,
                    'triggering_swing_timestamp': csh_timestamp,
                    'triggering_swing_level': swing_high_level,
                    'type': 'Liquidity Sweep (Sell)'
                }
                entries.append(entry_signal)

    return entries

# --- Example Usage ---
if __name__ == '__main__':
    # Create a sample DataFrame (ensure it's NY-localized for real use)
    data = {
        # Index: 0   1   2 (CSL) 3   4   5 (Sweep Buy)  6   7   8 (CSH)  9  10  11 (Sweep Sell)
        'Open':  [100,102,98, 100,101,95,             102,103,108,106,105,110],
        'High':  [101,103,99, 101,102,97,             103,104,110,107,106,112], # CSH at index 8 (110)
        'Low':   [99, 101,95, 97, 98, 94,             101,102,105,104,103,107], # CSL at index 2 (95)
        'Close': [100,102,97, 99, 100,96,             102,103,107,105,104,106], # Sweep Buy at 5: L(94)<95, C(96)>95. SL 94-1=93
                                                                             # Sweep Sell at 11: H(112)>110, C(106)<110. SL 112+1=113
    }
    # Make sure timestamps allow for trading window checks
    # Example: one sweep in window, one outside
    idx = []
    base_time = pd.Timestamp('2023-01-01 08:00:00', tz='America/New_York') # Outside window
    for i in range(len(data['Open'])):
        if i == 5: # Sweep Buy candle, put in trade window
            idx.append(pd.Timestamp(f'2023-01-01 21:0{i*5}:00', tz='America/New_York'))
        elif i == 11: # Sweep Sell candle, put in trade window
             idx.append(pd.Timestamp(f'2023-01-01 22:0{(i-6)*5}:00', tz='America/New_York'))
        else: # Other candles outside window
            idx.append(base_time + pd.Timedelta(minutes=5*i))

    sample_df = pd.DataFrame(data, index=pd.DatetimeIndex(idx))

    print("Sample DataFrame:")
    print(sample_df)
    print("\n")

    # Test is_clear_swing_low
    # Candle 2 (index 2) Low is 95. Prev: 101, 99. Post: 97, 98.  95 < 101, 95 < 99, 95 < 97, 95 < 98. Should be CSL.
    print(f"Is candle at index 2 a CSL? {is_clear_swing_low(sample_df, 2)}") # Expected True
    # Candle 5 (index 5) Low is 94. Prev: 98, 97. Post: 101, 102 (using dummy next if needed for test)
    # To test is_clear_swing_low for index 5, we need data up to index 7.
    # Let's test for index 0 (boundary)
    print(f"Is candle at index 0 a CSL? {is_clear_swing_low(sample_df, 0)}") # Expected False (boundary)

    # Test is_clear_swing_high
    # Candle 8 (index 8) High is 110. Prev: 104, 103. Post: 107, 106. 110 > 104, 110 > 103, 110 > 107, 110 > 106. Should be CSH.
    print(f"Is candle at index 8 a CSH? {is_clear_swing_high(sample_df, 8)}") # Expected True
    print(f"Is candle at index 1 a CSH? {is_clear_swing_high(sample_df, 1)}") # Expected False
    print("\n")

    sweep_entries = find_liquidity_sweep_entries(sample_df)
    print(f"Found {len(sweep_entries)} Liquidity Sweep Entries:")
    for entry in sweep_entries:
        print(f"- Timestamp: {entry['entry_timestamp']}, Type: {entry['type']}, Direction: {entry['direction']}, "
              f"Entry: {entry['entry_price']}, SL: {entry['sl_price']}, "
              f"Trigger Swing Level: {entry['triggering_swing_level']} at {entry['triggering_swing_timestamp']}")

    # Test with a DataFrame that's too short
    short_df_data = {'Open': [1,2], 'High': [1,2], 'Low': [1,2], 'Close': [1,2]}
    short_df_idx = pd.to_datetime(['2023-01-01 21:00:00', '2023-01-01 21:05:00'], utc=True).tz_convert('America/New_York')
    short_df = pd.DataFrame(short_df_data, index=short_df_idx)
    print(f"\nTesting with short DataFrame (length {len(short_df)}):")
    short_entries = find_liquidity_sweep_entries(short_df)
    print(f"Found {len(short_entries)} entries in short DataFrame.")


    # Test case: No valid swing in lookback
    no_swing_data = { # No clear swings in the first few candles
        'Open':  [100,100,100,100,100, 95],
        'High':  [101,101,101,101,101, 97],
        'Low':   [99, 99, 99, 99, 99, 94],
        'Close': [100,100,100,100,100, 96],
    }
    no_swing_idx = pd.to_datetime([
        '2023-01-02 21:00:00', '2023-01-02 21:05:00', '2023-01-02 21:10:00',
        '2023-01-02 21:15:00', '2023-01-02 21:20:00', '2023-01-02 21:25:00' # Sweep candle
    ], utc=True).tz_convert('America/New_York')
    no_swing_df = pd.DataFrame(no_swing_data, index=no_swing_idx)
    print("\nTesting with no clear swing in lookback:")
    print(no_swing_df)
    no_swing_entries = find_liquidity_sweep_entries(no_swing_df)
    print(f"Found {len(no_swing_entries)} entries when no swing expected.")


    # Test case: Sweep candle outside trading window
    outside_window_data = { # Identical to first test case data
        'Open':  [100,102,98, 100,101,95,  102,103,108,106,105,110],
        'High':  [101,103,99, 101,102,97,  103,104,110,107,106,112],
        'Low':   [99, 101,95, 97, 98, 94,  101,102,105,104,103,107],
        'Close': [100,102,97, 99, 100,96,  102,103,107,105,104,106],
    }
    outside_window_idx = []
    base_ow_time = pd.Timestamp('2023-01-03 10:00:00', tz='America/New_York') # All outside window
    for i in range(len(outside_window_data['Open'])):
        outside_window_idx.append(base_ow_time + pd.Timedelta(minutes=5*i))

    outside_window_df = pd.DataFrame(outside_window_data, index=pd.DatetimeIndex(outside_window_idx))
    print("\nTesting with sweep candles outside trading window:")
    print(outside_window_df.head(6)) # Show part relevant to the first potential sweep
    ow_entries = find_liquidity_sweep_entries(outside_window_df)
    print(f"Found {len(ow_entries)} entries when sweep candles are outside window.")

    print("\n--- End of Example Usage ---")
