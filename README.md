# SupplyChain AI – Intelligent Operations & Analytics Copilot

A portfolio MVP demonstrating a supply-chain data pipeline, SQLite analytics database, and interactive Streamlit dashboard.

> **Data note:** This project uses synthetic demonstration data. It is not connected to Dell systems or real company operations.

## Features
- Synthetic products, suppliers, warehouses, inventory, orders and shipment data
- Python/Pandas ETL into SQLite
- SQL joins and aggregations for supplier performance
- Streamlit dashboard with inventory and delivery KPIs
- Inventory threshold-based risk flags

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/generate_data.py
python src/etl.py
streamlit run app.py
```

If PowerShell blocks activation, run this once in the terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Project structure
- `src/generate_data.py`: creates reproducible synthetic CSV datasets
- `src/etl.py`: performs basic cleaning and loads SQLite tables
- `app.py`: interactive dashboard
- `data/raw/`: generated CSV files
- `data/supplychain.db`: generated SQLite database (not committed)

## Next improvements
- Add automated tests and data-quality checks
- Add demand forecasting with time-aware validation
- Add a natural-language SQL assistant with read-only query validation
- Add RAG over supply-chain policy documents
- Add FastAPI, Docker and CI/CD
