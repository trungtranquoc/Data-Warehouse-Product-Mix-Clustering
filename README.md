# Product Clustering Analysis

A multipage Streamlit application for analyzing product clusters with detailed visualizations and numerical analysis. Includes ETL and clustering pipelines for data warehouse management.

## Features

### 📊 Main Page - Clustering Analysis
- ETL pipeline execution status with last update timestamp
- Overview of product clusters with boxplots
- Profit, quantity, and customer distribution visualizations
- Cluster statistics and summary metrics
- One-click clustering pipeline execution
- Quick navigation to cluster details

### 📦 Page 1 - Cluster Details
- Filter products by cluster
- View detailed quarterly performance for each product
- Interactive charts showing sales, costs, and profit trends
- Quantity analysis across quarters

### 🔢 Page 2 - Product Details
- Comprehensive numerical analysis of selected products
- Quarter-over-quarter growth metrics
- ROI and per-unit calculations
- Revenue and cost breakdown visualizations
- Export functionality for data and metrics

### 🏪 Page 3 - Product Categories
- Category overview with profit and product count analysis (always visible at top)
- Browse products by category and subcategory
- Interactive button-based navigation
- Search functionality to find products by name or ID
- Sort products by profit (high/low), quantity (high/low), name, or ID
- View all products in selected category/subcategory
- Click on any product to view detailed information in Product Details page

## Installation

Install dependencies using uv:

```bash
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

## Running the Application

### Using main.py (Recommended)

The project includes a unified entry point that supports multiple operations:

```bash
# Run the Streamlit Decision Support System
uv run main.py -t dss

# Run the ETL pipeline
uv run main.py -t etl

# Run the clustering pipeline
uv run main.py -t clustering
```

### Running Streamlit Directly

```bash
streamlit run Clustering_Analysis.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using Docker

Start all services (SQL Server, PostgreSQL, and Streamlit app):

```bash
docker compose up -d --build
```

The Streamlit app will be available at `http://localhost:8501`. The container includes a cron job that automatically runs the ETL pipeline on the 1st of each month.

## Pipeline Management

### ETL Pipeline
Extracts data from SQL Server, transforms it, and loads it into the PostgreSQL data warehouse:
```bash
uv run etl.py
# or
uv run main.py -t etl
```

### Clustering Pipeline
Performs product clustering analysis based on warehouse data:
```bash
uv run main.py -t clustering
```

## Data Files

- PostgreSQL data warehouse tables in `dwh` schema
- SQL Server source database (CompanyX)
- Pipeline execution logs in `PipelineLog` table

## Project Structure

```
dwh/
├── main.py                               # Unified entry point for all operations
├── etl.py                                # ETL pipeline standalone script
├── Clustering_Analysis.py                # Main page - Clustering overview
├── Dockerfile                            # Docker configuration for Streamlit app
├── compose.yaml                          # Docker Compose configuration
├── pages/
│   ├── 1_Product_Categories.py          # Category & subcategory navigation
│   └── 2_Product_Details.py             # Detailed numerical analysis
├── notebooks/                            # Jupyter notebooks for exploration
│   ├── data_clustering.ipynb            # Clustering exploration
│   ├── etl_transform.ipynb              # ETL transformation steps
│   └── LSTM_features_learning.ipynb     # LSTM feature learning
├── src/
│   ├── config/
│   │   ├── database_connection.py       # SQL Server connection
│   │   └── warehouse_connection.py      # PostgreSQL connection
│   ├── models/                          # ML models (GAT, LSTM Autoencoder)
│   ├── pipelines/
│   │   ├── etl.py                       # ETL pipeline implementation
│   │   └── clustering.py                # Clustering pipeline implementation
│   └── utils/                           # Data processing utilities
├── data/                                # Data files and backups
├── pyproject.toml                       # Project dependencies
└── README.md                            # This file
```

## Dependencies

- streamlit >= 1.51.0
- pandas >= 2.2.0
- plotly >= 5.24.0
- sqlalchemy >= 2.0.44
- pyodbc >= 5.3.0
- psycopg2-binary >= 2.9.11
- scikit-learn >= 1.7.2

## Architecture

The application uses a data warehouse architecture:
- **Source**: SQL Server database (CompanyX)
- **Warehouse**: PostgreSQL database with `dwh` schema
- **ETL**: Automated monthly pipeline (via cron in Docker)
- **Analytics**: Streamlit-based decision support system
