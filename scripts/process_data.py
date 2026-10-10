import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

def fill_missing_dates_interpolation(stock_df, date_col='date'):
    """
    Compatibility name: normalize observed rows without creating trading bars.

    Interpolation of financial observations was removed because it used future
    prices and fabricated weekend/holiday volume. Use stock_forecasting for
    strict OHLCV validation, return features and versioned research datasets.
    """
    try:
        df = stock_df.copy()

        # Check if date column exists
        if date_col not in df.columns:
            raise ValueError(f"Required column '{date_col}' not found in DataFrame")

        if df.empty:
            raise ValueError("Empty stock history")
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            values = df[date_col].astype(str)
            df[date_col] = pd.to_datetime(values, dayfirst=values.str.contains('/').any())
        df = df.set_index(date_col).sort_index()
        if df.index.duplicated().any():
            raise ValueError("Duplicate dates require reconciliation")

        # For OHLC columns, ensure numeric type before interpolation
        ohlc_columns = ['ouverture', 'haut', 'bas', 'cloture', 'volume']
        for col in ohlc_columns:
            if col in df.columns:
                # Convert to numeric, handling European decimal commas if needed
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

        return df.reset_index().rename(columns={'index': date_col})
    
    except Exception as e:
        raise ValueError(f"Error processing DataFrame: {str(e)}") from e

def process_all_raw_files():
    """
    Process all CSV files in the raw data directory, applying the fill_missing_dates_interpolation function,
    and save processed files to the processed data directory. Only appends new data if needed.
    """
    # Define paths relative to project root
    project_root = PROJECT_ROOT
    input_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'data' / 'processed'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of all CSV files in input directory
    raw_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not raw_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return
    
    logger.info(f"Found {len(raw_files)} CSV files to process")
    
    success_count = 0
    for file_name in raw_files:
        try:
            # Process each file
            input_path = input_dir / file_name
            output_path = output_dir / file_name
            
            logger.info(f"Processing {file_name}...")
            
            # Read raw data with semicolon delimiter
            raw_df = pd.read_csv(input_path, sep=';')
            
            # Drop the 'symbole' column if it exists
            if 'symbole' in raw_df.columns:
                raw_df = raw_df.drop(columns=['symbole'])
                logger.debug(f"Dropped 'symbole' column from {file_name}")
            
            # Convert date column to datetime for comparison
            raw_df['date'] = pd.to_datetime(raw_df['date'], format='%d/%m/%Y', dayfirst=True)
            
            # Rebuild from observed raw rows to remove legacy synthetic dates and
            # incorporate provider corrections; never append interpolated history.
            processed_df = fill_missing_dates_interpolation(raw_df)

            # Save processed data
            processed_df.to_csv(output_path, index=False, sep=';')
            logger.info(f"Successfully processed and saved to {output_path}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
    
    logger.info(f"Processing complete! Successfully processed {success_count}/{len(raw_files)} files")

if __name__ == "__main__":
    # Run the processing pipeline
    process_all_raw_files()
