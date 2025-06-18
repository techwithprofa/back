import pandas as pd
from fvg_logic import find_fvgs, check_fvg_entry
from liquidity_sweep_logic import find_liquidity_sweep_entries
import bos_cod_logic # Import the module to change its constants

def get_all_entry_signals(df):
    """
    Combines signals from FVG, Liquidity Sweep, and BoS/COD logic.
    """
    all_signals = []
    identified_fvgs = find_fvgs(df)
    if identified_fvgs:
        for current_idx in range(2, len(df)):
            current_candle_timestamp = df.index[current_idx]
            for fvg_signal in identified_fvgs:
                if fvg_signal['timestamp'] < current_candle_timestamp:
                    fvg_entry = check_fvg_entry(df, fvg_signal, current_idx)
                    if fvg_entry:
                        all_signals.append(fvg_entry)

    ls_entries = find_liquidity_sweep_entries(df)
    all_signals.extend(ls_entries)

    bos_entries = bos_cod_logic.find_bos_entries(df)
    all_signals.extend(bos_entries)

    if all_signals:
        all_signals.sort(key=lambda x: x['entry_timestamp'])
    return all_signals

if __name__ == '__main__':
    print("--- Core Entry Engine Test ---")

    _orig_bos_total_lookback = bos_cod_logic.TOTAL_LOOKBACK_PERIOD
    _orig_bos_max_candles_between = bos_cod_logic.MAX_CANDLES_BETWEEN_SWINGS
    _orig_bos_debug = bos_cod_logic.DEBUG_BOS

    bos_cod_logic.TOTAL_LOOKBACK_PERIOD = 20
    bos_cod_logic.MAX_CANDLES_BETWEEN_SWINGS = 5
    bos_cod_logic.DEBUG_BOS = True

    data_points = []
    base_time = pd.Timestamp('2023-03-01 18:00:00', tz='America/New_York')

    def add_candle(dt, o, h, l, c):
        data_points.append({'Timestamp': dt, 'Open': o, 'High': h, 'Low': l, 'Close': c})

    for i in range(10):
         add_candle(base_time - pd.Timedelta(minutes=(10-i)*5), 100, 101, 99, 100)

    # Section 1: Bullish FVG (now idx 10-14)
    add_candle(base_time,                             9, 10, 8,  9.5)
    add_candle(base_time + pd.Timedelta(minutes=5),   9, 12, 7,  8)
    add_candle(base_time + pd.Timedelta(minutes=10), 10, 14, 11, 13.5)
    add_candle(base_time + pd.Timedelta(minutes=15),11, 11.5, 10,11.2)
    add_candle(base_time.replace(hour=21, minute=0) + pd.Timedelta(minutes=20), 11, 11.8, 10, 11.5)

    # Section 2: Liquidity Sweep (Buy) (now idx 15-20)
    current_offset_sec2 = 30
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec2),     25, 26, 24, 25)
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec2+5),   23, 24, 22, 23)
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec2+10),  22, 23, 20, 21)
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec2+15),  21, 23, 21, 22)
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec2+20),  22, 24, 22, 23)
    add_candle(base_time.replace(hour=21, minute=30) + pd.Timedelta(minutes=current_offset_sec2+25), 20, 22, 19, 21)

    # Section 3: Bullish BoS (now idx 21-29)
    # Expected: CSH1@23(H=55), CSL1@24(L=48), CSH2@26(H=52,SSH), CSL2@27(L=45). Break@29.
    current_offset_sec3 = current_offset_sec2 + 35

    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec3),     50, 50, 48, 49) # idx 21
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec3+5),   49, 51, 49, 50) # idx 22
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec3+10),  50, 55, 50, 51) # idx 23 (CSH1 H=55)
    # Adjusted H[24] to make CSH2@26 clear. Old H[24]=52
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec3+15),  51, 51, 48, 49) # idx 24 (CSL1 L=48). H changed from 52 to 51.
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec3+20),  49, 50, 47, 48) # idx 25
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec3+25),  48, 52, 47, 47.5)# idx 26 (CSH2 H=52, SSH)
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec3+30),  47, 48, 45, 46)  # idx 27 (CSL2 L=45)
    add_candle(base_time + pd.Timedelta(minutes=current_offset_sec3+35),  46, 47, 46, 46.5) # idx 28 (L[28]=46 > L[27]=45)

    add_candle(base_time.replace(hour=22, minute=0) + pd.Timedelta(minutes=current_offset_sec3+25), 52, 56, 51.5, 53) # idx 29 (Break CSH2@52). Time: 23:30.

    sample_df = pd.DataFrame(data_points)
    sample_df.set_index('Timestamp', inplace=True)

    sample_df.attrs['is_test_data'] = True
    sample_df.attrs['verbose_debug'] = True
    sample_df.attrs['test_start_iteration_idx'] = 29
    sample_df.attrs['test_break_candle_idx_bullish'] = 29
    sample_df.attrs['test_break_candle_idx_bearish'] = -1

    print(f"\nSample DataFrame created with {len(sample_df)} candles.")
    all_signals = get_all_entry_signals(sample_df.copy())

    print(f"\n--- Found {len(all_signals)} signals in total ---")
    for i, signal in enumerate(all_signals):
        print(f"\nSignal {i+1}:")
        print(f"  Timestamp: {signal['entry_timestamp']}")
        signal_type = signal.get('type', 'N/A')
        if 'fvg_details' in signal:
             signal_type = signal['fvg_details']['type']
        print(f"  Type: {signal_type}")
        print(f"  Direction: {signal['direction']}")
        print(f"  Entry Price: {signal['entry_price']:.2f}")
        print(f"  SL Price: {signal['sl_price']:.2f}")
        if 'broken_swing_price' in signal:
            print(f"  Broken Swing Price: {signal['broken_swing_price']:.2f}")
        if 'triggering_swing_level' in signal:
            print(f"  Triggering Swing Level: {signal['triggering_swing_level']:.2f}")

    bos_cod_logic.TOTAL_LOOKBACK_PERIOD = _orig_bos_total_lookback
    bos_cod_logic.MAX_CANDLES_BETWEEN_SWINGS = _orig_bos_max_candles_between
    bos_cod_logic.DEBUG_BOS = _orig_bos_debug

    print("\n--- Core Entry Engine Test Finished ---")
