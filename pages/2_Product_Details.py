import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sqlalchemy import create_engine

st.set_page_config(page_title="Product Details", page_icon="🔢", layout="wide")

st.title('🔢 Product Details')

@st.cache_data
def load_cluster_data():
    """Load the product clusters data from PostgreSQL database"""
    try:
        engine = create_engine('postgresql+psycopg2://postgres:Trungtq@localhost:5432/postgres')
        query = """
        SELECT 
            pc."ProductID",
            pc.cluster as cluster_id,
            pc.profit,
            pc.nunique_customer as "#accumulated_customer",
            pc.profit_margin,
            COALESCE(SUM(fps."OrderQty"), 0) as quantity,
            0 as new_customer_ratio,
            0 as quantity_growth
        FROM dwh.product_clustering pc
        LEFT JOIN dwh."FactProductSales" fps ON pc."ProductID" = fps."ProductID"
        GROUP BY pc."ProductID", pc.cluster, pc.profit, pc.nunique_customer, pc.profit_margin
        """
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None

@st.cache_data
def load_categories():
    """Load product categories from PostgreSQL database"""
    try:
        engine = create_engine('postgresql+psycopg2://postgres:Trungtq@localhost:5432/postgres')
        query = """
        SELECT 
            dp."ProductID",
            dp."Name" as "ProductName",
            dpc."Name" as "Category",
            dps."Name" as "Subcategory"
        FROM dwh."DimProduct" dp
        JOIN dwh."DimProductSubcategory" dps ON dp."ProductSubcategoryID" = dps."ProductSubcategoryID"
        JOIN dwh."DimProductCategory" dpc ON dps."ProductCategoryID" = dpc."ProductCategoryID"
        """
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None

@st.cache_data
def load_product_detail(product_id):
    """Load detailed quarterly data for a specific product"""
    file_path = f'product_{product_id}.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

@st.cache_data
def get_product_name(product_id):
    """Get product name from database"""
    try:
        engine = create_engine('postgresql+psycopg2://postgres:Trungtq@localhost:5432/postgres')
        query = f"""
        SELECT "Name" as "ProductName"
        FROM dwh."DimProduct"
        WHERE "ProductID" = {product_id}
        """
        df = pd.read_sql(query, engine)
        engine.dispose()
        if len(df) > 0:
            return df['ProductName'].iloc[0]
        return f"Product {product_id}"
    except:
        return f"Product {product_id}"

# Load cluster data
df = load_cluster_data()
df_categories = load_categories()

if df is None or df_categories is None:
    st.error("Failed to load cluster data from database.")
    st.stop()

# Merge data to get complete product information
df_products = df_categories.merge(df, on='ProductID', how='left')

# Filters
col1, col2, col3 = st.columns(3)

with col1:
    # Category filter
    categories = ['All'] + sorted(df_categories['Category'].unique().tolist())
    selected_category = st.selectbox("Category", categories, key="filter_category")

with col2:
    # Subcategory filter
    if selected_category == 'All':
        subcategories = ['All'] + sorted(df_categories['Subcategory'].unique().tolist())
    else:
        subcategories = ['All'] + sorted(
            df_categories[df_categories['Category'] == selected_category]['Subcategory'].unique().tolist()
        )
    selected_subcategory = st.selectbox("Subcategory", subcategories, key="filter_subcategory")

with col3:
    # Cluster filter
    clusters = ['All'] + sorted([int(c) for c in df['cluster_id'].dropna().unique()])
    selected_cluster = st.selectbox("Cluster", clusters, key="filter_cluster")

# Apply filters
filtered_products = df_products.copy()

if selected_category != 'All':
    filtered_products = filtered_products[filtered_products['Category'] == selected_category]

if selected_subcategory != 'All':
    filtered_products = filtered_products[filtered_products['Subcategory'] == selected_subcategory]

if selected_cluster != 'All':
    filtered_products = filtered_products[filtered_products['cluster_id'] == selected_cluster]

# Display summary metrics
st.markdown("### 📈 Summary Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Products", len(filtered_products))
with col2:
    avg_profit = filtered_products['profit'].mean()
    st.metric("Avg Profit", f"${avg_profit:,.2f}")
with col3:
    total_quantity = filtered_products['quantity'].sum()
    st.metric("Total Quantity", f"{total_quantity:,.0f}")
with col4:
    avg_margin = filtered_products['profit_margin'].mean()
    st.metric("Avg Margin", f"{avg_margin:.2%}")

st.markdown("---")

# Search and sort options
col1, col2 = st.columns([2, 1])
with col1:
    search_query = st.text_input("🔍 Search products by name or ID", key="product_search", placeholder="Enter product name or ID...")
with col2:
    sort_by = st.selectbox(
        "Sort by",
        ["Profit (High to Low)", "Profit (Low to High)", 
            "Product ID", "Product Name", "Quantity (High to Low)", "Quantity (Low to High)"],
        key="sort_option"
    )

# Apply search filter
if search_query:
    filtered_products = filtered_products[
        filtered_products['ProductName'].str.contains(search_query, case=False, na=False) |
        filtered_products['ProductID'].astype(str).str.contains(search_query, case=False, na=False)
    ]

# Apply sorting
if sort_by == "Profit (High to Low)":
    filtered_products = filtered_products.sort_values('profit', ascending=False)
elif sort_by == "Profit (Low to High)":
    filtered_products = filtered_products.sort_values('profit', ascending=True)
elif sort_by == "Product ID":
    filtered_products = filtered_products.sort_values('ProductID')
elif sort_by == "Product Name":
    filtered_products = filtered_products.sort_values('ProductName')
elif sort_by == "Quantity (High to Low)":
    filtered_products = filtered_products.sort_values('quantity', ascending=False)
elif sort_by == "Quantity (Low to High)":
    filtered_products = filtered_products.sort_values('quantity', ascending=True)

st.markdown(f"### 📋 Product List ({len(filtered_products)} products)")

# Pagination
if len(filtered_products) > 0:
    # Initialize session state for pagination
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 1
    
    # Products per page
    products_per_page = 20
    total_products = len(filtered_products)
    total_pages = (total_products + products_per_page - 1) // products_per_page  # Ceiling division
    
    # Reset to page 1 if filters changed
    if 'last_filter_state' not in st.session_state:
        st.session_state['last_filter_state'] = (selected_category, selected_subcategory, selected_cluster, search_query, sort_by)
    
    current_filter_state = (selected_category, selected_subcategory, selected_cluster, search_query, sort_by)
    if st.session_state['last_filter_state'] != current_filter_state:
        st.session_state['current_page'] = 1
        st.session_state['last_filter_state'] = current_filter_state
    
    # Ensure current page is within bounds
    if st.session_state['current_page'] > total_pages:
        st.session_state['current_page'] = total_pages if total_pages > 0 else 1
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page_cols = st.columns([1, 3, 1])
        with page_cols[0]:
            if st.button("⬅️ Previous", disabled=(st.session_state['current_page'] <= 1), width='stretch'):
                st.session_state['current_page'] -= 1
                st.rerun()
        
        with page_cols[1]:
            st.markdown(f"<div style='text-align: center; padding: 8px;'><b>Page {st.session_state['current_page']} of {total_pages}</b></div>", unsafe_allow_html=True)
        
        with page_cols[2]:
            if st.button("Next ➡️", disabled=(st.session_state['current_page'] >= total_pages), width='stretch'):
                st.session_state['current_page'] += 1
                st.rerun()
    
    # Calculate start and end indices for current page
    start_idx = (st.session_state['current_page'] - 1) * products_per_page
    end_idx = min(start_idx + products_per_page, total_products)
    
    # Get products for current page
    page_products = filtered_products.iloc[start_idx:end_idx]
    
    st.markdown(f"Showing {start_idx + 1} to {end_idx} of {total_products} products")
    st.markdown("---")
    
    # Display products as cards
    for idx, row in page_products.iterrows():
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 1.5, 1.5, 1.5, 1.5])
            
            with col1:
                st.markdown(f"**{row['ProductName']}**")
                cluster_display = int(row['cluster_id']) if pd.notna(row['cluster_id']) else 'N/A'
                st.caption(f"ID: {row['ProductID']} | Cluster: {cluster_display}")
                st.caption(f"📁 {row['Category']} > {row['Subcategory']}")
            
            with col2:
                profit_color = "🟢" if row['profit'] > 0 else "🔴"
                st.markdown(f"{profit_color} **Profit**")
                st.markdown(f"${row['profit']:,.2f}")
            
            with col3:
                st.markdown(f"📦 **Quantity**")
                st.markdown(f"{row['quantity']:,.0f}")
            
            with col4:
                st.markdown(f"📊 **Margin**")
                st.markdown(f"{row['profit_margin']:.2%}")
            
            with col5:
                st.markdown("")  # Spacing
                if st.button("View Analysis", key=f"view_{row['ProductID']}"):
                    # Store selected product in session state and switch to analysis tab
                    st.session_state['selected_product'] = row['ProductID']
                    if pd.notna(row['cluster_id']):
                        st.session_state['selected_cluster'] = int(row['cluster_id'])
                    st.rerun()
            
            st.divider()
    
    # Bottom pagination controls
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page_cols = st.columns([1, 3, 1])
        with page_cols[0]:
            if st.button("⬅️ Prev", disabled=(st.session_state['current_page'] <= 1), width='stretch', key="prev_bottom"):
                st.session_state['current_page'] -= 1
                st.rerun()
        
        with page_cols[1]:
            st.markdown(f"<div style='text-align: center; padding: 8px;'><b>Page {st.session_state['current_page']} of {total_pages}</b></div>", unsafe_allow_html=True)
        
        with page_cols[2]:
            if st.button("Next ➡️", disabled=(st.session_state['current_page'] >= total_pages), width='stretch', key="next_bottom"):
                st.session_state['current_page'] += 1
                st.rerun()
else:
    st.warning("No products found matching your search criteria.")