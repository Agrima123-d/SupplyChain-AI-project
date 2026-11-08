from pathlib import Path
import numpy as np
import pandas as pd

# Reproducible synthetic data: the same seed creates the same dataset.
rng = np.random.default_rng(42)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

n_products, n_suppliers, n_warehouses, n_orders = 80, 15, 5, 2500

products = pd.DataFrame({
    "product_id": [f"P{i:03d}" for i in range(1, n_products + 1)],
    "product_name": [f"Product {i:03d}" for i in range(1, n_products + 1)],
    "category": rng.choice(["Laptop", "Monitor", "Keyboard", "Mouse", "Accessory"], n_products),
    "unit_cost": np.round(rng.uniform(8, 900, n_products), 2),
    "avg_daily_demand": rng.integers(2, 35, n_products)
})

suppliers = pd.DataFrame({
    "supplier_id": [f"S{i:03d}" for i in range(1, n_suppliers + 1)],
    "supplier_name": [f"Supplier {i:02d}" for i in range(1, n_suppliers + 1)],
    "region": rng.choice(["North", "South", "East", "West"], n_suppliers),
    "expected_lead_days": rng.integers(3, 16, n_suppliers)
})

warehouses = pd.DataFrame({
    "warehouse_id": [f"W{i:02d}" for i in range(1, n_warehouses + 1)],
    "warehouse_name": [f"Warehouse {i}" for i in range(1, n_warehouses + 1)],
    "city": rng.choice(["Bengaluru", "Chennai", "Pune", "Delhi", "Hyderabad"], n_warehouses)
})

inventory_rows = []
for p in products.itertuples(index=False):
    for w in warehouses.itertuples(index=False):
        stock = int(rng.integers(10, 900))
        inventory_rows.append({
            "product_id": p.product_id,
            "warehouse_id": w.warehouse_id,
            "current_stock": stock,
            "reorder_point": int(p.avg_daily_demand * rng.integers(4, 12)),
            "avg_daily_demand": int(p.avg_daily_demand)
        })
inventory = pd.DataFrame(inventory_rows)

order_dates = pd.Timestamp.today().normalize() - pd.to_timedelta(
    rng.integers(0, 180, n_orders), unit="D"
)
orders = pd.DataFrame({
    "order_id": [f"O{i:05d}" for i in range(1, n_orders + 1)],
    "product_id": rng.choice(products["product_id"], n_orders),
    "supplier_id": rng.choice(suppliers["supplier_id"], n_orders),
    "warehouse_id": rng.choice(warehouses["warehouse_id"], n_orders),
    "order_date": pd.to_datetime(order_dates).strftime("%Y-%m-%d"),
    "quantity": rng.integers(1, 80, n_orders)
})

shipment_rows = []
for order in orders.itertuples(index=False):
    supplier = suppliers.loc[suppliers.supplier_id == order.supplier_id].iloc[0]
    delay = int(rng.choice([0, 1, 2, 4, 7, 10], p=[.48, .18, .13, .10, .07, .04]))
    shipment_rows.append({
        "shipment_id": f"SH-{order.order_id}",
        "order_id": order.order_id,
        "supplier_id": order.supplier_id,
        "expected_delivery_days": int(supplier.expected_lead_days),
        "actual_delivery_days": int(supplier.expected_lead_days + delay),
        "shipment_status": "Delivered" if rng.random() > .08 else "In Transit"
    })
shipments = pd.DataFrame(shipment_rows)

for name, frame in {
    "products.csv": products,
    "suppliers.csv": suppliers,
    "warehouses.csv": warehouses,
    "inventory.csv": inventory,
    "orders.csv": orders,
    "shipments.csv": shipments
}.items():
    frame.to_csv(RAW / name, index=False)

print(f"Generated 6 synthetic CSV files in: {RAW}")
print(f"Products: {len(products):,} | Orders: {len(orders):,} | Inventory rows: {len(inventory):,}")
