from pathlib import Path
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "supplychain.db"

st.set_page_config(page_title="SupplyChain AI", page_icon="📦", layout="wide")

@st.cache_data
def load_data():
    if not DB.exists():
        return None
    with sqlite3.connect(DB) as conn:
        inventory = pd.read_sql_query("""
            SELECT i.*, p.product_name, p.category, p.unit_cost,
                   w.warehouse_name, w.city
            FROM inventory i
            JOIN products p ON i.product_id = p.product_id
            JOIN warehouses w ON i.warehouse_id = w.warehouse_id
        """, conn)
        supplier = pd.read_sql_query("""
            SELECT s.supplier_id, s.supplier_name, s.region,
                   COUNT(sh.shipment_id) AS shipments,
                   SUM(CASE WHEN sh.actual_delivery_days > sh.expected_delivery_days
                       THEN 1 ELSE 0 END) AS delayed_shipments,
                   ROUND(100.0 * SUM(CASE WHEN sh.actual_delivery_days <= sh.expected_delivery_days
                       THEN 1 ELSE 0 END) / COUNT(sh.shipment_id), 1) AS on_time_pct
            FROM suppliers s
            LEFT JOIN shipments sh ON s.supplier_id = sh.supplier_id
            GROUP BY s.supplier_id, s.supplier_name, s.region
        """, conn)
        orders = pd.read_sql_query("""
            SELECT o.*, p.category, s.supplier_name, w.warehouse_name
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            JOIN suppliers s ON o.supplier_id = s.supplier_id
            JOIN warehouses w ON o.warehouse_id = w.warehouse_id
        """, conn)
        shipments = pd.read_sql_query("SELECT * FROM shipments", conn)
    return inventory, supplier, orders, shipments

st.title("📦 SupplyChain AI")
st.caption("Intelligent Operations & Analytics Copilot | Demonstration using synthetic data")

data = load_data()
if data is None:
    st.error("Database not found. First run: python src/generate_data.py and then python src/etl.py")
    st.stop()

inventory, suppliers, orders, shipments = data

page = st.sidebar.radio("Navigate", ["Executive Overview", "Inventory Analytics", "Supplier Performance", "Data Explorer"])
st.sidebar.info("All records are synthetic and created for this portfolio project.")

if page == "Executive Overview":
    total_units = int(inventory.current_stock.sum())
    inventory_value = float((inventory.current_stock * inventory.unit_cost).sum())
    low_stock = int((inventory.current_stock <= inventory.reorder_point).sum())
    on_time = float((shipments.actual_delivery_days <= shipments.expected_delivery_days).mean() * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Units in inventory", f"{total_units:,}")
    c2.metric("Inventory value", f"₹{inventory_value:,.0f}")
    c3.metric("Low-stock locations", f"{low_stock:,}")
    c4.metric("On-time delivery", f"{on_time:.1f}%")

    left, right = st.columns(2)
    with left:
        by_category = inventory.groupby("category", as_index=False)["current_stock"].sum()
        st.plotly_chart(px.bar(by_category, x="category", y="current_stock",
                               title="Inventory units by product category"), use_container_width=True)
    with right:
        by_city = inventory.groupby("city", as_index=False)["current_stock"].sum()
        st.plotly_chart(px.pie(by_city, names="city", values="current_stock",
                               title="Inventory distribution by city"), use_container_width=True)

    st.subheader("Products requiring attention")
    risk = inventory[inventory.current_stock <= inventory.reorder_point].copy()
    risk["shortage"] = (risk.reorder_point - risk.current_stock).clip(lower=0)
    st.dataframe(risk[["product_name", "warehouse_name", "current_stock",
                       "reorder_point", "shortage"]].sort_values("shortage", ascending=False).head(12),
                 use_container_width=True, hide_index=True)

elif page == "Inventory Analytics":
    st.subheader("Inventory Analytics")
    categories = ["All"] + sorted(inventory.category.unique().tolist())
    chosen = st.selectbox("Filter by category", categories)
    view = inventory if chosen == "All" else inventory[inventory.category == chosen]
    view = view.copy()
    view["risk_status"] = view.apply(
        lambda r: "Low stock" if r.current_stock <= r.reorder_point else "Healthy", axis=1
    )
    st.plotly_chart(px.scatter(view, x="avg_daily_demand", y="current_stock",
                               color="risk_status", hover_name="product_name",
                               size="unit_cost", title="Stock versus average daily demand"),
                    use_container_width=True)
    st.dataframe(view[["product_name", "category", "warehouse_name", "current_stock",
                       "reorder_point", "avg_daily_demand", "risk_status"]],
                 use_container_width=True, hide_index=True)

elif page == "Supplier Performance":
    st.subheader("Supplier Delivery Performance")
    st.plotly_chart(px.bar(suppliers.sort_values("on_time_pct"),
                           x="supplier_name", y="on_time_pct", color="region",
                           title="Supplier on-time delivery percentage"),
                    use_container_width=True)
    st.dataframe(suppliers.sort_values("on_time_pct"), use_container_width=True, hide_index=True)

else:
    st.subheader("Data Explorer")
    table_name = st.selectbox("Choose a dataset", ["orders", "inventory", "suppliers", "shipments"])
    mapping = {"orders": orders, "inventory": inventory,
               "suppliers": suppliers, "shipments": shipments}
    st.write(f"Showing {len(mapping[table_name]):,} rows")
    st.dataframe(mapping[table_name], use_container_width=True, hide_index=True)

st.divider()
st.caption("Built with Python, Pandas, SQLite, SQL, Streamlit and Plotly.")
