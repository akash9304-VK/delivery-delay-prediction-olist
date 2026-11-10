
from pathlib import Path
import getpass

import pandas as pd
from sqlalchemy import create_engine, URL, text

# Locate the project folders
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"

# MySQL connection settings

DB_USER = "root"
DB_PASSWORD = getpass.getpass("Enter MySQL password: ")
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "delivery_delay"

connection_url = URL.create(
    "mysql+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)

engine = create_engine(connection_url)


try:
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DATA_DIR}"
        )

    for csv_path in csv_files:
        table_name = csv_path.stem

        print(f"\nLoading {csv_path.name}...")
        df = pd.read_csv(csv_path)

        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=1000,
        )

        print(f"Rows loaded: {len(df):,}")

    with engine.connect() as connection:
        database = connection.execute(
            text("SELECT DATABASE()")
        ).scalar()

        print(f"\nConnected database: {database}")
        print("Import completed successfully!")

finally:
    engine.dispose()
