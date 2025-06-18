import pandas as pd

# Load the CSV file into a pandas DataFrame
try:
    df = pd.read_csv('XAU_5m_data.csv', delimiter=';')
    print("File loaded successfully with semicolon delimiter.")
except FileNotFoundError:
    print("Error: XAU_5m_data.csv not found.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# Print column names
print("\nColumn names:")
print(list(df.columns))

# --- Task 2: Inspect columns and identify timestamp, OHLC columns. Print their names. ---
timestamp_col_name = 'Date'
open_col_name = 'Open'
high_col_name = 'High'
low_col_name = 'Low'
close_col_name = 'Close'

required_cols = [timestamp_col_name, open_col_name, high_col_name, low_col_name, close_col_name]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing expected columns: {missing_cols}")
    exit()

print("\nIdentified columns (actual):")
print(f"Timestamp: {timestamp_col_name}")
print(f"Open: {open_col_name}")
print(f"High: {high_col_name}")
print(f"Low: {low_col_name}")
print(f"Close: {close_col_name}")

# --- Task 3: Convert timestamp column to datetime objects. ---
try:
    df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name], errors='coerce')
    print(f"\n'{timestamp_col_name}' column converted to datetime objects (naive).")
except Exception as e:
    print(f"Error converting '{timestamp_col_name}' to datetime: {e}")
    exit()

if df[timestamp_col_name].isnull().any():
    print(f"Warning: Some values in '{timestamp_col_name}' could not be parsed into dates and are now NaT.")
    # df = df.dropna(subset=[timestamp_col_name]) # Optionally drop rows with NaT

try:
    df[timestamp_col_name] = df[timestamp_col_name].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    print(f"'{timestamp_col_name}' column localized to UTC and converted to 'America/New_York'.")
except Exception as e:
    print(f"Error during timezone conversion: {e}")
    try:
        df[timestamp_col_name] = df[timestamp_col_name].dt.tz_convert('America/New_York')
        print(f"'{timestamp_col_name}' column converted to 'America/New_York' (was already timezone-aware).")
    except Exception as e_inner:
        print(f"Nested error during timezone conversion: {e_inner}")
        exit()

# --- Task 4: Set timestamp as DataFrame index. ---
try:
    df.set_index(timestamp_col_name, inplace=True)
    print(f"\n'{timestamp_col_name}' column set as DataFrame index.")
except Exception as e:
    print(f"Error setting '{timestamp_col_name}' as index: {e}")
    exit()

# --- Task 5: Filter data to include only March and June of any available year(s). ---
try:
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: Index is not a DatetimeIndex. Cannot filter by month.")
        exit()

    df_filtered = df[df.index.month.isin([3, 6])] # 3 for March, 6 for June
    print("\nDataFrame filtered for months March (3) and June (6).")
except Exception as e:
    print(f"Error during filtering by month: {e}")
    exit()

# --- Task 6: Print the shape of the filtered DataFrame and the first 5 rows. ---
print("\nShape of the filtered DataFrame:")
print(df_filtered.shape)

print("\nFirst 5 rows of the filtered DataFrame:")
print(df_filtered.head())

# Task 7 (Saving to CSV) is removed due to file size limitations in the environment.
print("\nTask 7 (Save filtered data to CSV) was skipped due to environment file size limitations.")
print("\nScript finished successfully.")
