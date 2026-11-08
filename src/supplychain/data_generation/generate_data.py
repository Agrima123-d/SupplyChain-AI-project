"""Synthetic supply-chain data generator.

Generates six relational CSV tables with realistic relationships:
  - products, suppliers, warehouses, inventory, orders, shipments

Key design decisions
--------------------
1. **Fixed seed** (`numpy.random.default_rng(42)`) ensures every run produces
   identical data, making analyses reproducible.
2. **Dirty-data injection** adds nulls, duplicates, typos, outliers, and
   inconsistent casing so the ETL pipeline (Phase 2) has meaningful work.
3. **Configurable scale** via command-line flags (`--orders`, `--seed`, etc.)
   supports benchmarking at 500 000+ orders.
4. **Realistic columns** match the project specification (lead_time_days,
   reorder_level, reliability_score, capacity, reserved_quantity,
   last_updated, order_status, proper shipment dates).

Usage
-----
    # Default (2 500 orders):
    python -m src.supplychain.data_generation.generate_data

    # Large-scale benchmark (500 000 orders):
    python -m src.supplychain.data_generation.generate_data --orders 500000

    # Custom seed:
    python -m src.supplychain.data_generation.generate_data --seed 123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project paths (works when run as module or script)
# ---------------------------------------------------------------------------
try:
    from src.supplychain.config import RAW_DIR, DEFAULT_SEED
except ImportError:
    # Fallback: compute paths relative to this file
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    RAW_DIR = _PROJECT_ROOT / "data" / "raw"
    DEFAULT_SEED = 42


# ===================================================================
# Helper utilities
# ===================================================================

def _inject_nulls(
    df: pd.DataFrame,
    rng: np.random.Generator,
    columns: list[str],
    fraction: float = 0.02,
) -> pd.DataFrame:
    """Set ~`fraction` of values in `columns` to NaN (dirty-data injection).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to modify *in place*.
    rng : numpy random Generator
        For reproducibility.
    columns : list[str]
        Columns eligible for null injection.
    fraction : float
        Approximate fraction of rows per column to nullify (default 2 %).

    Returns
    -------
    pd.DataFrame
        The same dataframe, modified in place.
    """
    n_rows = len(df)
    n_nulls = max(1, int(n_rows * fraction))
    for col in columns:
        if col in df.columns:
            idx = rng.choice(n_rows, size=n_nulls, replace=False)
            df.loc[df.index[idx], col] = np.nan
    return df


def _inject_duplicates(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_dupes: int = 5,
) -> pd.DataFrame:
    """Append a few exact-duplicate rows (dirty-data injection).

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    rng : numpy random Generator
        For reproducibility.
    n_dupes : int
        Number of duplicate rows to add.

    Returns
    -------
    pd.DataFrame
        Dataframe with duplicates appended (index reset).
    """
    if n_dupes <= 0 or len(df) == 0:
        return df
    idx = rng.choice(len(df), size=min(n_dupes, len(df)), replace=False)
    dupes = df.iloc[idx].copy()
    return pd.concat([df, dupes], ignore_index=True)


# ===================================================================
# Table generators
# ===================================================================

def generate_products(
    rng: np.random.Generator,
    n: int = 80,
) -> pd.DataFrame:
    """Create the *products* table.

    Columns: product_id, product_name, category, unit_cost,
             lead_time_days, reorder_level.
    """
    categories = ["Laptop", "Monitor", "Keyboard", "Mouse",
                  "Accessory", "Server", "Networking", "Storage"]

    # Map categories to realistic cost ranges (min, max)
    cost_ranges: dict[str, tuple[float, float]] = {
        "Laptop":      (35_000, 1_50_000),
        "Monitor":     (8_000,  45_000),
        "Keyboard":    (500,    5_000),
        "Mouse":       (300,    3_500),
        "Accessory":   (200,    8_000),
        "Server":      (80_000, 3_00_000),
        "Networking":  (2_000,  25_000),
        "Storage":     (3_000,  40_000),
    }

    # More descriptive product names per category
    name_prefixes: dict[str, list[str]] = {
        "Laptop":      ["ProBook", "EliteBook", "ZenBook", "ThinkPad",
                        "Latitude", "IdeaPad", "Vostro", "Pavilion"],
        "Monitor":     ["UltraSharp", "ProDisplay", "ClearView", "VividScreen"],
        "Keyboard":    ["MechForce", "SlimKey", "TypeMaster", "ProType"],
        "Mouse":       ["ErgoClick", "SwiftPoint", "PrecisePro", "OptiFlex"],
        "Accessory":   ["FlexDock", "PowerHub", "CableKit", "CoolPad"],
        "Server":      ["RackPro", "CloudEdge", "DataNode", "PowerEdge"],
        "Networking":  ["NetSwitch", "RouterPro", "WifiMax", "LinkMaster"],
        "Storage":     ["DataVault", "SpeedDrive", "CloudStore", "FlashArray"],
    }

    cat_choices = rng.choice(categories, n)
    product_names: list[str] = []
    costs: list[float] = []
    for i, cat in enumerate(cat_choices):
        prefixes = name_prefixes[cat]
        prefix = rng.choice(prefixes)
        product_names.append(f"{prefix} {cat} {i + 1:03d}")
        lo, hi = cost_ranges[cat]
        costs.append(round(float(rng.uniform(lo, hi)), 2))

    df = pd.DataFrame({
        "product_id":     [f"P{i:03d}" for i in range(1, n + 1)],
        "product_name":   product_names,
        "category":       cat_choices,
        "unit_cost":      costs,
        "lead_time_days": rng.integers(3, 22, n).astype(int),
        "reorder_level":  rng.integers(50, 300, n).astype(int),
    })

    # --- Dirty data: inject a few category typos ---
    typo_map = {"Laptop": "Lapto", "Monitor": "Moniter", "Keyboard": "Keybord"}
    n_typos = max(1, int(n * 0.04))  # ~4 % of rows
    typo_idx = rng.choice(n, size=n_typos, replace=False)
    for idx in typo_idx:
        original = df.at[idx, "category"]
        if original in typo_map:
            df.at[idx, "category"] = typo_map[original]

    # --- Dirty data: a couple of outlier costs ---
    outlier_idx = rng.choice(n, size=2, replace=False)
    df.loc[df.index[outlier_idx[0]], "unit_cost"] = 9_99_999.99  # absurdly high
    df.loc[df.index[outlier_idx[1]], "unit_cost"] = -50.0        # negative (invalid)

    # --- Dirty data: nulls in non-key columns ---
    _inject_nulls(df, rng, ["unit_cost", "lead_time_days", "reorder_level"])

    return df


def generate_suppliers(
    rng: np.random.Generator,
    n: int = 15,
) -> pd.DataFrame:
    """Create the *suppliers* table.

    Columns: supplier_id, supplier_name, supplier_region, reliability_score.
    """
    regions = ["Asia-Pacific", "North America", "Europe", "South America"]

    # Give some suppliers deliberately inconsistent casing (dirty data)
    region_dirty = list(rng.choice(regions, n))
    n_dirty = max(1, int(n * 0.13))
    for idx in rng.choice(n, size=n_dirty, replace=False):
        region_dirty[idx] = region_dirty[idx].lower()  # e.g., "asia-pacific"

    df = pd.DataFrame({
        "supplier_id":     [f"S{i:02d}" for i in range(1, n + 1)],
        "supplier_name":   [f"Supplier {i:02d}" for i in range(1, n + 1)],
        "supplier_region": region_dirty,
        "reliability_score": np.round(rng.uniform(0.45, 1.0, n), 2),
    })

    # --- Dirty data: one null reliability_score ---
    _inject_nulls(df, rng, ["reliability_score"], fraction=0.07)

    return df


def generate_warehouses(
    rng: np.random.Generator,
    n: int = 5,
) -> pd.DataFrame:
    """Create the *warehouses* table.

    Columns: warehouse_id, warehouse_name, location, capacity.
    """
    cities = ["Bengaluru", "Chennai", "Pune", "Delhi", "Hyderabad",
              "Mumbai", "Kolkata", "Ahmedabad"]

    locations = list(rng.choice(cities, n, replace=False) if n <= len(cities)
                     else rng.choice(cities, n))

    df = pd.DataFrame({
        "warehouse_id":   [f"W{i:02d}" for i in range(1, n + 1)],
        "warehouse_name": [f"{loc} Distribution Centre" for loc in locations],
        "location":       locations,
        "capacity":       rng.integers(5_000, 50_001, n).astype(int),
    })
    return df


def generate_inventory(
    rng: np.random.Generator,
    products: pd.DataFrame,
    warehouses: pd.DataFrame,
) -> pd.DataFrame:
    """Create the *inventory* table (one row per product × warehouse).

    Columns: inventory_id, product_id, warehouse_id, stock_quantity,
             reserved_quantity, last_updated.
    """
    rows: list[dict] = []
    inv_id = 1
    today = pd.Timestamp.today().normalize()

    for p in products.itertuples(index=False):
        for w in warehouses.itertuples(index=False):
            stock = int(rng.integers(5, 900))
            reserved = int(rng.integers(0, max(1, stock // 3)))
            days_ago = int(rng.integers(0, 31))
            rows.append({
                "inventory_id":     f"INV-{inv_id:04d}",
                "product_id":       p.product_id,
                "warehouse_id":     w.warehouse_id,
                "stock_quantity":    stock,
                "reserved_quantity": reserved,
                "last_updated":     (today - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            })
            inv_id += 1

    df = pd.DataFrame(rows)

    # --- Dirty data: a few negative stock values (invalid) ---
    bad_idx = rng.choice(len(df), size=3, replace=False)
    for idx in bad_idx:
        df.loc[df.index[idx], "stock_quantity"] = int(rng.integers(-50, -1))

    # --- Dirty data: nulls ---
    _inject_nulls(df, rng, ["stock_quantity", "reserved_quantity", "last_updated"])

    # --- Dirty data: duplicates ---
    df = _inject_duplicates(df, rng, n_dupes=4)

    return df


def generate_orders(
    rng: np.random.Generator,
    products: pd.DataFrame,
    suppliers: pd.DataFrame,
    warehouses: pd.DataFrame,
    n: int = 2_500,
) -> pd.DataFrame:
    """Create the *orders* table.

    Columns: order_id, product_id, supplier_id, warehouse_id,
             order_date, quantity, order_status.
    """
    today = pd.Timestamp.today().normalize()

    # Spread orders over the past 365 days for richer time-series data
    days_back = rng.integers(0, 365, n)
    order_dates = today - pd.to_timedelta(days_back, unit="D")

    statuses = ["Pending", "Processing", "Shipped", "Delivered", "Cancelled"]
    status_probs = [0.08, 0.07, 0.12, 0.65, 0.08]

    df = pd.DataFrame({
        "order_id":     [f"O{i:05d}" for i in range(1, n + 1)],
        "product_id":   rng.choice(products["product_id"].values, n),
        "supplier_id":  rng.choice(suppliers["supplier_id"].values, n),
        "warehouse_id": rng.choice(warehouses["warehouse_id"].values, n),
        "order_date":   order_dates.strftime("%Y-%m-%d"),
        "quantity":     rng.integers(1, 100, n).astype(int),
        "order_status": rng.choice(statuses, n, p=status_probs),
    })

    # --- Dirty data: a few negative / zero quantities ---
    bad_idx = rng.choice(n, size=max(1, int(n * 0.005)), replace=False)
    for idx in bad_idx:
        df.loc[df.index[idx], "quantity"] = int(rng.integers(-10, 0))

    # --- Dirty data: nulls ---
    _inject_nulls(df, rng, ["quantity", "order_status", "order_date"])

    # --- Dirty data: duplicate rows ---
    n_dupes = max(3, int(n * 0.004))  # ~0.4 %
    df = _inject_duplicates(df, rng, n_dupes=n_dupes)

    return df


def generate_shipments(
    rng: np.random.Generator,
    orders: pd.DataFrame,
    suppliers: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Create the *shipments* table (one shipment per non-cancelled order).

    Columns: shipment_id, order_id, shipment_date, expected_delivery_date,
             actual_delivery_date, shipment_status.

    - `shipment_date` is 0-2 days after the order date.
    - `expected_delivery_date` = shipment_date + product lead time.
    - `actual_delivery_date` = expected + random delay (or NaT if in transit).
    - `shipment_status` is derived from the actual delivery date.
    """
    # Build lookup dicts for fast access
    supplier_lead: dict[str, int] = {}
    for s in suppliers.itertuples(index=False):
        supplier_lead[s.supplier_id] = 7  # default; we'll use product lead time

    product_lead: dict[str, int] = {}
    for p in products.itertuples(index=False):
        lt = p.lead_time_days
        product_lead[p.product_id] = int(lt) if not pd.isna(lt) else 7

    # Only create shipments for non-cancelled orders with valid dates
    eligible = orders[
        (orders["order_status"] != "Cancelled")
        & orders["order_date"].notna()
    ].copy()

    rows: list[dict] = []
    for row in eligible.itertuples(index=False):
        try:
            order_dt = pd.Timestamp(row.order_date)
        except Exception:
            continue

        # Shipment dispatched 0-2 days after order
        dispatch_delay = int(rng.integers(0, 3))
        ship_date = order_dt + pd.Timedelta(days=dispatch_delay)

        # Expected delivery based on product lead time
        lead = product_lead.get(row.product_id, 7)
        expected_date = ship_date + pd.Timedelta(days=lead)

        # Actual delivery: 48% on-time, rest delayed by 1-10 days
        delay_days = int(rng.choice(
            [0, 0, 1, 2, 3, 5, 7, 10],
            p=[0.36, 0.12, 0.15, 0.12, 0.10, 0.07, 0.05, 0.03],
        ))
        actual_date = expected_date + pd.Timedelta(days=delay_days)

        # ~10% of shipments are still "In Transit" (no actual delivery date)
        in_transit = rng.random() < 0.10
        if in_transit:
            actual_str = None
            status = "In Transit"
        else:
            actual_str = actual_date.strftime("%Y-%m-%d")
            status = "Delayed" if delay_days > 0 else "Delivered"

        rows.append({
            "shipment_id":             f"SH-{row.order_id}",
            "order_id":                row.order_id,
            "shipment_date":           ship_date.strftime("%Y-%m-%d"),
            "expected_delivery_date":  expected_date.strftime("%Y-%m-%d"),
            "actual_delivery_date":    actual_str,
            "shipment_status":         status,
        })

    df = pd.DataFrame(rows)

    # --- Dirty data: nulls in shipment_date, expected_delivery_date ---
    _inject_nulls(df, rng, ["shipment_date", "expected_delivery_date"], fraction=0.01)

    # --- Dirty data: a few duplicate shipment rows ---
    df = _inject_duplicates(df, rng, n_dupes=3)

    return df


# ===================================================================
# Main generation pipeline
# ===================================================================

def generate_all(
    n_products: int = 80,
    n_suppliers: int = 15,
    n_warehouses: int = 5,
    n_orders: int = 2_500,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate all six synthetic tables and save as CSV.

    Parameters
    ----------
    n_products : int
        Number of products to generate.
    n_suppliers : int
        Number of suppliers.
    n_warehouses : int
        Number of warehouses.
    n_orders : int
        Number of purchase orders (scales shipments proportionally).
    seed : int
        Random seed for full reproducibility.
    output_dir : Path, optional
        Directory to write CSV files into. Defaults to ``data/raw/``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of table name → DataFrame.
    """
    rng = np.random.default_rng(seed)
    out = output_dir or RAW_DIR
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic data (seed={seed}) ...")
    print(f"  Products:   {n_products}")
    print(f"  Suppliers:  {n_suppliers}")
    print(f"  Warehouses: {n_warehouses}")
    print(f"  Orders:     {n_orders:,}")

    # 1. Dimension tables (no FK dependencies)
    products   = generate_products(rng, n_products)
    suppliers  = generate_suppliers(rng, n_suppliers)
    warehouses = generate_warehouses(rng, n_warehouses)

    # 2. Fact / bridge tables (depend on dimensions)
    inventory  = generate_inventory(rng, products, warehouses)
    orders_df  = generate_orders(rng, products, suppliers, warehouses, n_orders)
    shipments  = generate_shipments(rng, orders_df, suppliers, products)

    tables = {
        "products":   products,
        "suppliers":  suppliers,
        "warehouses": warehouses,
        "inventory":  inventory,
        "orders":     orders_df,
        "shipments":  shipments,
    }

    # Write CSVs
    for name, frame in tables.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)

    # Summary
    print(f"\n{'Table':<14} {'Rows':>8}  {'Columns':>8}")
    print("-" * 34)
    for name, frame in tables.items():
        print(f"  {name:<12} {len(frame):>8,}  {len(frame.columns):>8}")

    total_nulls = sum(frame.isna().sum().sum() for frame in tables.values())
    total_dupes = sum(frame.duplicated().sum() for frame in tables.values())
    print(f"\nDirty data injected: ~{total_nulls:.0f} null values, "
          f"~{total_dupes} duplicate rows")
    print(f"CSV files saved to: {out.resolve()}")

    return tables


# ===================================================================
# CLI entry point
# ===================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic supply-chain data.",
    )
    parser.add_argument(
        "--products", type=int, default=80,
        help="Number of products (default: 80)",
    )
    parser.add_argument(
        "--suppliers", type=int, default=15,
        help="Number of suppliers (default: 15)",
    )
    parser.add_argument(
        "--warehouses", type=int, default=5,
        help="Number of warehouses (default: 5)",
    )
    parser.add_argument(
        "--orders", type=int, default=2_500,
        help="Number of orders (default: 2500; use 500000 for benchmarks)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for CSV files (default: data/raw/)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    out_dir = Path(args.output_dir) if args.output_dir else None
    generate_all(
        n_products=args.products,
        n_suppliers=args.suppliers,
        n_warehouses=args.warehouses,
        n_orders=args.orders,
        seed=args.seed,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
