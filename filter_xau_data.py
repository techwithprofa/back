import pandas as pd
from datetime import datetime

def filter_xau_data(input_file, output_file, start_date, end_date):
    """
    Filter XAU 5-minute data by date range and save to new CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    """
    try:
        # Read the CSV file with semicolon separator
        df = pd.read_csv(input_file, sep=';')
        
        # Convert Date column to datetime
        # Handle the format: 2004.06.11 07:15
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
        
        # Convert start and end dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Filter data between start and end dates
        filtered_df = df[(df['Date'] >= start_dt) & (df['Date'] < end_dt)]
        
        # Convert Date back to original format for saving
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y.%m.%d %H:%M')
        
        # Save filtered data to new CSV file
        filtered_df.to_csv(output_file, sep=';', index=False)
        
        print(f"Original data: {len(df)} rows")
        print(f"Filtered data: {len(filtered_df)} rows")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Filtered data saved to: {output_file}")
        
        if len(filtered_df) > 0:
            print(f"First date in filtered data: {filtered_df['Date'].iloc[0]}")
            print(f"Last date in filtered data: {filtered_df['Date'].iloc[-1]}")
        else:
            print("No data found in the specified date range!")
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
    except Exception as e:
        print(f"Error processing data: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Configuration
    input_file = "XAU_5m_data.csv"
    output_file = "XAU_5m_data_2024_filtered.csv"
    start_date = "2024-05-01"
    end_date = "2024-07-01"
    
    # Filter the data
    filter_xau_data(input_file, output_file, start_date, end_date)