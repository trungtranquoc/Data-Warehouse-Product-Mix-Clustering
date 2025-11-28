import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sqlalchemy import create_engine

st.set_page_config(page_title="Product Categories", page_icon="🏪", layout="wide")

st.title('🏪 Product Categories')
st.markdown("### Browse products by category and subcategory")

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

# Load data
df_clusters = load_cluster_data()
df_categories = load_categories()

if df_clusters is None or df_categories is None:
    st.error("Failed to load data from database.")
    st.stop()

# Merge data to get complete product information
df_products = df_categories.merge(df_clusters, on='ProductID', how='left')

# Initialize session state
if 'selected_category' not in st.session_state:
    st.session_state['selected_category'] = None
if 'selected_subcategory' not in st.session_state:
    st.session_state['selected_subcategory'] = None
if 'selected_product_id' not in st.session_state:
    st.session_state['selected_product_id'] = None

# Category Overview - Always show at the top
st.markdown("## 📊 Category Overview")

# Category statistics
category_stats = df_categories.merge(df_clusters, on='ProductID', how='left')

# For charts - use simple aggregation
category_summary_charts = category_stats.groupby('Category').agg({
    'ProductID': 'count',
    'profit': 'sum',
    'quantity': 'sum'
}).reset_index()
category_summary_charts.columns = ['Category', 'Product Count', 'Total Profit', 'Total Quantity']
category_summary_charts = category_summary_charts.sort_values('Total Profit', ascending=False)

# For table - include subcategories with separate rows
category_summary = category_stats.groupby(['Category', 'Subcategory']).agg({
    'ProductID': 'count',
    'profit': 'sum',
    'quantity': 'sum'
}).reset_index()
category_summary.columns = ['Category', 'Subcategory', 'Product Count', 'Total Profit', 'Total Quantity']
category_summary = category_summary.sort_values(['Category', 'Subcategory'])

# Display as chart
col1, col2 = st.columns(2)

with col1:
    fig_profit = px.bar(
        category_summary_charts,
        x='Category',
        y='Total Profit',
        title='Total Profit by Category',
        color_discrete_sequence=['#00CC96']
    )
    fig_profit.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_profit, width='stretch')

with col2:
    fig_count = px.bar(
        category_summary_charts,
        x='Category',
        y='Product Count',
        title='Number of Products by Category',
        color_discrete_sequence=['#636EFA']
    )
    fig_count.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_count, width='stretch')

st.dataframe(category_summary, width='stretch', hide_index=True)

# Add cluster distribution pie charts for each category
st.markdown("---")
st.markdown("### 🎯 Cluster Distribution by Category")

# Define consistent colors for clusters
color_map = {
    0: '#636EFA',
    1: '#EF553B',
    2: '#00CC96',
    3: '#AB63FA',
    4: '#FFA15A',
    5: '#19D3F3',
    6: '#FF6692',
    7: '#B6E880'
}

# Get cluster distribution for each category
categories = sorted(df_products['Category'].unique())
num_categories = len(categories)

# Determine optimal grid layout
if num_categories <= 3:
    cols_per_row = num_categories
elif num_categories == 4:
    cols_per_row = 2
else:
    cols_per_row = 3

rows = [categories[i:i + cols_per_row] for i in range(0, len(categories), cols_per_row)]

for row in rows:
    cols = st.columns(len(row))  # Create exactly as many columns as items in this row
    for idx, category in enumerate(row):
        with cols[idx]:
            # Filter products by category and get cluster distribution
            cat_products = df_products[df_products['Category'] == category]
            cluster_dist = cat_products.groupby('cluster_id').size().reset_index(name='count')
            cluster_dist = cluster_dist[cluster_dist['cluster_id'].notna()]  # Remove NaN clusters
            
            if len(cluster_dist) > 0:
                cluster_dist['cluster_id'] = cluster_dist['cluster_id'].astype(int)
                
                fig_pie = px.pie(
                    cluster_dist,
                    values='count',
                    names='cluster_id',
                    title=f'{category}<br>({len(cat_products)} products)',
                    hole=0.3,
                    color='cluster_id',
                    color_discrete_map=color_map
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    showlegend=True,
                    legend_title="Cluster ID",
                    height=350,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig_pie, width='stretch')
            else:
                st.info(f"No history purchasing data for {category} in recent 1 year")

st.markdown("---")

# Category selection
st.markdown("## 📁 Select a Category")

categories = sorted(df_categories['Category'].unique())

# Create buttons for categories in columns
cols_per_row = 4
rows = [categories[i:i + cols_per_row] for i in range(0, len(categories), cols_per_row)]

for row in rows:
    cols = st.columns(cols_per_row)
    for idx, category in enumerate(row):
        # Count products in category
        count = len(df_categories[df_categories['Category'] == category])
        with cols[idx]:
            if st.button(f"📦 {category}\n({count} products)", 
                        key=f"cat_{category}",
                        width='stretch'):
                st.session_state['selected_category'] = category
                st.session_state['selected_subcategory'] = None
                st.session_state['selected_product_id'] = None
                st.rerun()

# Show selected category
if st.session_state['selected_category']:
    st.markdown("---")
    st.markdown(f"## 🎯 Category: {st.session_state['selected_category']}")
    
    # Add subcategory cluster distribution pie charts
    st.markdown("### 🎯 Cluster Distribution by Subcategory")
    
    # Get subcategories for selected category
    selected_subcategories = sorted(df_products[
        df_products['Category'] == st.session_state['selected_category']
    ]['Subcategory'].unique())
    
    num_subcats = len(selected_subcategories)
    
    # Determine optimal grid layout
    if num_subcats <= 3:
        cols_per_row = num_subcats
    elif num_subcats == 4:
        cols_per_row = 2
    else:
        cols_per_row = 3
    
    rows = [selected_subcategories[i:i + cols_per_row] for i in range(0, len(selected_subcategories), cols_per_row)]
    
    for row in rows:
        cols = st.columns(len(row))
        for idx, subcategory in enumerate(row):
            with cols[idx]:
                # Filter products by subcategory and get cluster distribution
                subcat_products = df_products[
                    (df_products['Category'] == st.session_state['selected_category']) &
                    (df_products['Subcategory'] == subcategory)
                ]
                cluster_dist = subcat_products.groupby('cluster_id').size().reset_index(name='count')
                cluster_dist = cluster_dist[cluster_dist['cluster_id'].notna()]  # Remove NaN clusters
                
                if len(cluster_dist) > 0:
                    cluster_dist['cluster_id'] = cluster_dist['cluster_id'].astype(int)
                    
                    fig_pie = px.pie(
                        cluster_dist,
                        values='count',
                        names='cluster_id',
                        title=f'{subcategory}<br>({len(subcat_products)} products)',
                        hole=0.3,
                        color='cluster_id',
                        color_discrete_map=color_map
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(
                        showlegend=True,
                        legend_title="Cluster ID",
                        height=350,
                        margin=dict(l=20, r=20, t=60, b=20)
                    )
                    st.plotly_chart(fig_pie, width='stretch')
                else:
                    st.info(f"No purchasing data for {subcategory} in recent 1 year")
    
    st.markdown("---")
    st.info("💡 Go to **Product Details** page to browse all products with advanced filters.")
