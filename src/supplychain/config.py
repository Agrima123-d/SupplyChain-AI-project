"""Central configuration for SupplyChain AI.

Loads settings from environment variables with sensible defaults.
Use a .env file locally (never commit it to version control).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # two levels up from src/supplychain/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = DATA_DIR / "docs"
DB_PATH = DATA_DIR / "supplychain.db"
CONFIG_DIR = PROJECT_ROOT / "configs"

# ---------------------------------------------------------------------------
# Data generation defaults
# ---------------------------------------------------------------------------
DEFAULT_N_PRODUCTS = 80
DEFAULT_N_SUPPLIERS = 15
DEFAULT_N_WAREHOUSES = 5
DEFAULT_N_ORDERS = 2_500
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# LLM settings (used in later phases)
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Embedding settings (used in later phases)
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
