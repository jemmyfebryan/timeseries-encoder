import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def export_alpha_prices_to_csv():
    """Export alpha_prices data to CSV files, one per netuid (0-128)"""

    # Create SQLAlchemy engine for better pandas compatibility
    db_url = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)

    # Create output directory
    output_dir = 'alpha_prices_csv'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Check table structure
        query = "SELECT * FROM alpha_prices LIMIT 1"
        df_sample = pd.read_sql(query, engine)
        print(f"Table columns: {list(df_sample.columns)}")

        # Export data for each netuid (0-128)
        for netuid in range(129):
            query = f"""
                SELECT id, date, block, netuid, price, volume
                FROM alpha_prices
                WHERE netuid = {netuid} AND ochl = 'hour'
                ORDER BY block ASC
            """

            df = pd.read_sql(query, engine)

            if not df.empty:
                # Calculate actual volume by taking difference of accumulated volume
                df['volume'] = df['volume'].diff()

                # Remove first row since we can't calculate actual volume (accumulated from unknown start)
                df = df.iloc[1:].reset_index(drop=True)

                csv_file = os.path.join(output_dir, f'alpha_prices_netuid_{netuid}.csv')
                df.to_csv(csv_file, index=False)
                print(f"✓ Exported {len(df)} rows for netuid {netuid} -> {csv_file}")
            else:
                print(f"○ No data found for netuid {netuid}")

        print(f"\nExport complete! CSV files saved to '{output_dir}/' directory")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    export_alpha_prices_to_csv()
