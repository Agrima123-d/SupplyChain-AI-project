from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "supplychain.db"

TABLES = ["products", "suppliers", "warehouses", "inventory", "orders", "shipments"]

def run_etl():
    """Load CSV files, apply basic quality checks, and write SQLite tables."""
    if not all((RAW / f"{table}.csv").exists() for table in TABLES):
        raise FileNotFoundError("CSV files missing. Run: python src/generate_data.py")

    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as conn:
        for table in TABLES:
            df = pd.read_csv(RAW / f"{table}.csv")
            before = len(df)
            df = df.drop_duplicates()
            df = df.dropna()
            print(f"{table}: {before} rows read, {len(df)} rows after basic cleaning")
            df.to_sql(table, conn, if_exists="replace", index=False)

        # Add useful indexes for common joins and filters.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(product_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_supplier ON orders(supplier_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shipments_order ON shipments(order_id)")

    print(f"ETL complete. SQLite database created at: {DB}")

if __name__ == "__main__":
    run_etl()
