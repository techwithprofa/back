import pandas as pd

def find_fvgs(df):
    """
    Identifies Fair Value Gaps (FVGs) in the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns,
                           indexed by NY-localized timestamps.

    Returns:
        list: A list of FVG signal dictionaries.
    """
    fvgs = []
    # Ensure the DataFrame has enough data
    if len(df) < 3:
        return fvgs

    # Create shifted columns for easier comparison
    # high_minus_1 = df['High'].shift(1) # Candle 2 High
    # low_minus_1 = df['Low'].shift(1)   # Candle 2 Low
    high_minus_2 = df['High'].shift(2) # Candle 1 High
    low_minus_2 = df['Low'].shift(2)   # Candle 1 Low

    for i in range(2, len(df)):
        current_timestamp = df.index[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]

        # Candle 2 (middle candle of the FVG pattern)
        candle2_high = df['High'].iloc[i-1]
        candle2_low = df['Low'].iloc[i-1]

        # Candle 1 (first candle of the FVG pattern)
        candle1_high = high_minus_2.iloc[i]
        candle1_low = low_minus_2.iloc[i]

        fvg_signal = None

        # Bullish FVG: Current (Candle 3) Low is above Candle 1 High
        if current_low > candle1_high:
            fvg_signal = {
                'timestamp': current_timestamp,
                'type': 'Bullish FVG',
                'fvg_high': current_low,
                'fvg_low': candle1_high,
                'candle1_high': candle1_high,
                'candle1_low': candle1_low,
                'candle2_high': candle2_high,
                'candle2_low': candle2_low,
                'candle3_high': current_high,
                'candle3_low': current_low,
            }

        # Bearish FVG: Current (Candle 3) High is below Candle 1 Low
        elif current_high < candle1_low:
            fvg_signal = {
                'timestamp': current_timestamp,
                'type': 'Bearish FVG',
                'fvg_high': candle1_low,
                'fvg_low': current_high,
                'candle1_high': candle1_high,
                'candle1_low': candle1_low,
                'candle2_high': candle2_high,
                'candle2_low': candle2_low,
                'candle3_high': current_high,
                'candle3_low': current_low,
            }

        if fvg_signal:
            fvgs.append(fvg_signal)

    return fvgs

def check_fvg_entry(df, fvg_signal, current_candle_index):
    """
    Checks if the current candle's price re-enters the FVG gap for a trade entry.

    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns,
                           indexed by NY-localized timestamps.
        fvg_signal (dict): A single FVG signal dictionary from find_fvgs.
        current_candle_index (int): Index of the current candle being evaluated for entry.

    Returns:
        dict or None: Entry details if conditions met, otherwise None.
    """
    if current_candle_index < 0 or current_candle_index >= len(df):
        return None

    current_candle = df.iloc[current_candle_index]
    current_candle_timestamp = df.index[current_candle_index]
    current_candle_time = current_candle_timestamp.time()

    trade_window_start = pd.to_datetime('21:00:00').time()
    trade_window_end = pd.to_datetime('23:30:00').time()

    if not (trade_window_start <= current_candle_time <= trade_window_end):
        return None

    entry_details = None
    fvg_type = fvg_signal['type']
    fvg_high_price = fvg_signal['fvg_high']
    fvg_low_price = fvg_signal['fvg_low']

    candle2_low_for_sl = fvg_signal['candle2_low']
    candle2_high_for_sl = fvg_signal['candle2_high']

    current_low = current_candle['Low']
    current_high = current_candle['High']
    current_close = current_candle['Close']

    if fvg_type == 'Bullish FVG':
        if current_low <= fvg_low_price and current_close > fvg_low_price:
            entry_details = {
                'entry_timestamp': current_candle_timestamp,
                'entry_price': fvg_low_price,
                'direction': 'Buy',
                'sl_price': candle2_low_for_sl,
                'fvg_details': fvg_signal,
            }
    elif fvg_type == 'Bearish FVG':
        if current_high >= fvg_high_price and current_close < fvg_high_price:
            entry_details = {
                'entry_timestamp': current_candle_timestamp,
                'entry_price': fvg_high_price,
                'direction': 'Sell',
                'sl_price': candle2_high_for_sl,
                'fvg_details': fvg_signal,
            }

    return entry_details

if __name__ == '__main__':
    # Example Usage (requires a DataFrame named 'data_df')
    data = {
        'Open': [10, 12, 11, 15, 14, 13, 10, 9, 11, 12, 15, 16],
        'High': [11, 13, 15, 16, 15, 14, 13, 10, 12, 15, 17, 18],
        'Low':  [9,  11, 10, 14, 13, 12, 9,  8,  10, 11, 12, 15],
        'Close':[10, 12, 14, 15, 13, 12, 12, 9,  11, 14, 16, 17]
    }
    start_time = pd.Timestamp('2023-03-01 09:00:00', tz='America/New_York')
    index = [start_time + pd.Timedelta(minutes=5*i) for i in range(len(data['Open']))]

    sample_df = pd.DataFrame(data, index=index)

    print("Sample DataFrame:")
    print(sample_df)

    fvg_signals = find_fvgs(sample_df)
    print(f"\nFound {len(fvg_signals)} FVGs:")
    for fvg in fvg_signals:
        print(f"- Type: {fvg['type']}, Timestamp: {fvg['timestamp']}, Gap: [{fvg['fvg_low']}, {fvg['fvg_high']}]")
        print(f"  C1 H/L: {fvg['candle1_high']}/{fvg['candle1_low']}, C2 H/L: {fvg['candle2_high']}/{fvg['candle2_low']}, C3 H/L: {fvg['candle3_high']}/{fvg['candle3_low']}")

    if fvg_signals:
        print("\nChecking for entries (example):")
        if fvg_signals[0]['timestamp'] in sample_df.index:
            fvg_candle_index = sample_df.index.get_loc(fvg_signals[0]['timestamp'])

            for i in range(1, 6):
                entry_candle_idx = fvg_candle_index + i
                if entry_candle_idx < len(sample_df):
                    print(f"  Evaluating candle at {sample_df.index[entry_candle_idx]} for entry into FVG: {fvg_signals[0]['type']} [{fvg_signals[0]['fvg_low']}, {fvg_signals[0]['fvg_high']}]")

                    if i == 2:
                        original_timestamp = sample_df.index[entry_candle_idx]
                        new_timestamp_in_window = original_timestamp.replace(hour=21, minute=30)

                        temp_df_for_check = sample_df.copy()
                        new_index_list = temp_df_for_check.index.tolist()
                        new_index_list[entry_candle_idx] = new_timestamp_in_window
                        temp_df_for_check.index = pd.DatetimeIndex(new_index_list)

                        print(f"    (Modified timestamp for testing trade window: {new_timestamp_in_window})")
                        entry_info = check_fvg_entry(temp_df_for_check, fvg_signals[0], entry_candle_idx)
                    else:
                         entry_info = check_fvg_entry(sample_df, fvg_signals[0], entry_candle_idx)

                    if entry_info:
                        print(f"    Entry found: {entry_info['direction']} at {entry_info['entry_price']} on {entry_info['entry_timestamp']}, SL: {entry_info['sl_price']}")
                        break
                    else:
                        print(f"    No entry on this candle.")
                else:
                    print("    Reached end of dataframe for entry check.")
                    break
    else:
        print("\nNo FVGs found to check for entries.")

    bullish_fvg_data = {
        'Open':  [19, 18, 22, 20],
        'High':  [20, 22, 25, 21],
        'Low':   [18, 17, 21, 19.5],
        'Close': [19.5, 21.5, 24, 20.5]
    }
    bullish_start_time = pd.Timestamp('2023-03-02 21:00:00', tz='America/New_York')
    bullish_idx = [bullish_start_time + pd.Timedelta(minutes=5*i) for i in range(len(bullish_fvg_data['Open']))]
    bullish_df = pd.DataFrame(bullish_fvg_data, index=bullish_idx)

    print("\nTesting specific Bullish FVG entry:")
    print(bullish_df)
    bullish_fvgs = find_fvgs(bullish_df)
    if bullish_fvgs:
        print(f"Found Bullish FVG: {bullish_fvgs[0]}")
        entry_info = check_fvg_entry(bullish_df, bullish_fvgs[0], 3)
        if entry_info:
            print(f"  Bullish Entry Confirmed: {entry_info}")
        else:
            print("  No Bullish Entry.")
    else:
        print("  No Bullish FVG found.")

    bearish_fvg_data = {
        'Open':  [31, 32, 28, 29.8],
        'High':  [32, 33, 29, 30.5],
        'Low':   [30, 28, 27, 29.2],
        'Close': [30.5, 29, 27.5, 29.5]
    }
    bearish_start_time = pd.Timestamp('2023-03-03 22:00:00', tz='America/New_York')
    bearish_idx = [bearish_start_time + pd.Timedelta(minutes=5*i) for i in range(len(bearish_fvg_data['Open']))]
    bearish_df = pd.DataFrame(bearish_fvg_data, index=bearish_idx)

    print("\nTesting specific Bearish FVG entry:")
    print(bearish_df)
    bearish_fvgs = find_fvgs(bearish_df)
    if bearish_fvgs:
        print(f"Found Bearish FVG: {bearish_fvgs[0]}")
        entry_info = check_fvg_entry(bearish_df, bearish_fvgs[0], 3)
        if entry_info:
            print(f"  Bearish Entry Confirmed: {entry_info}")
        else:
            print("  No Bearish Entry.")
    else:
        print("  No Bearish FVG found.")

    non_trade_window_time = pd.Timestamp('2023-03-03 10:00:00', tz='America/New_York')
    bearish_idx_outside_window = [
        non_trade_window_time + pd.Timedelta(minutes=5*i) for i in range(len(bearish_fvg_data['Open'])-1)
    ]
    bearish_idx_outside_window.append(pd.Timestamp('2023-03-03 10:15:00', tz='America/New_York'))
    bearish_df_outside_window = pd.DataFrame(bearish_fvg_data, index=bearish_idx_outside_window)

    print("\nTesting specific Bearish FVG entry (outside trading window):")
    print(bearish_df_outside_window)
    bearish_fvgs_otw = find_fvgs(bearish_df_outside_window)
    if bearish_fvgs_otw:
        print(f"Found Bearish FVG: {bearish_fvgs_otw[0]}")
        entry_info_otw = check_fvg_entry(bearish_df_outside_window, bearish_fvgs_otw[0], 3)
        if entry_info_otw:
            print(f"  Bearish Entry Confirmed (ERROR - SHOULD BE NONE): {entry_info_otw}")
        else:
            print("  No Bearish Entry (Correctly outside window).")
    else:
        print("  No Bearish FVG found.")

    print("\nNote: Timestamps in sample data are illustrative.")
    print("Actual data should be NY-localized as per previous subtask.")

    short_df = sample_df.head(2)
    print(f"\nTesting with short DataFrame (length {len(short_df)}):")
    short_fvgs = find_fvgs(short_df)
    print(f"Found {len(short_fvgs)} FVGs in short DataFrame.")

    print("\nTesting entry check with invalid index:")
    if fvg_signals: # Re-use fvg_signals from the first sample_df
        invalid_entry = check_fvg_entry(sample_df, fvg_signals[0], len(sample_df) + 5)
        if invalid_entry is None:
            print("Correctly returned None for invalid index.")
        else:
            print(f"ERROR: Should have returned None, got {invalid_entry}")
    else:
        print("Skipping invalid index test as no FVGs were found in initial sample_df.")

# Removed the erroneous ``` at the end of the file.
