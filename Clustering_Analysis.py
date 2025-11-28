import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

import os
import dotenv

dotenv.load_dotenv()

server_name = os.getenv('POSTGRESQL_SERVER_NAME')
database_name = os.getenv('POSTGRESQL_DATABASE_NAME')
username = os.getenv('POSTGRESQL_USER_NAME')
password = os.getenv('POSTGRESQL_PASSWORD')
driver_name = os.getenv('POSTGRESQL_DRIVER_NAME')
port_number = os.getenv('POSTGRESQL_PORT_NUMBER')

st.set_page_config(page_title="Product Clustering Analysis", page_icon="📊", layout="wide")

st.title('📊 Product Clustering Analysis')
st.markdown("### Overview of Product Clusters")

# Display ETL Pipeline Last Update
@st.cache_data(ttl=60)
def get_last_etl_update():
    """Get the last ETL pipeline update time from the database"""
    try:
        engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{server_name}:{port_number}/{database_name}')
        query = """
        SELECT lastupdate 
        FROM dwh."PipelineLog"
        ORDER BY lastupdate DESC 
        LIMIT 1
        """
        result = pd.read_sql(query, engine)
        engine.dispose()
        if not result.empty:
            return result['lastupdate'].iloc[0]
        return None
    except Exception as e:
        return None

last_update = get_last_etl_update()

# Create columns for ETL update info and clustering button
col_info, col_button = st.columns([3, 1])

with col_info:
    if last_update:
        st.info(f"📅 **Data Last Updated:** {last_update}")
    else:
        st.warning("⚠️ No ETL pipeline execution recorded yet.")

with col_button:
    if st.button("🔄 Run Clustering Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running clustering pipeline... This may take a few minutes."):
            try:
                from src.pipelines.clustering import ClusteringPipeline
                
                clustering_pipeline = ClusteringPipeline(warehouse_schema_name="dwh")
                clustering_pipeline.run()
                
                st.success("✅ Clustering pipeline completed successfully!")
                st.info("🔄 Refreshing data... Please wait.")
                
                # Clear cache to reload new data
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error running clustering pipeline: {str(e)}")
                st.exception(e)

# Add Run Clustering Pipeline button
st.markdown("---")

@st.cache_data
def load_cluster_data():
    """Load the product clusters data from PostgreSQL database"""
    try:
        # Create database connection
        engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{server_name}:{port_number}/{database_name}')

        # Query to get all product cluster data with quantity from fact table
        query = """
        SELECT 
            pc."ProductID",
            pc.cluster as cluster_id,
            pc.profit,
            pc.nunique_customer as "#accumulated_customer",
            pc.profit_margin,
            pc.selling_duration,
            pc.revenue_growth,
            pc.average_unit_price,
            COALESCE(COUNT(DISTINCT fps."SalesOrderID"), 0) as order_frequency,
            COALESCE(SUM(fps."OrderQty"), 0) as quantity
        FROM dwh.product_clustering pc
        LEFT JOIN dwh."FactProductSales" fps ON pc."ProductID" = fps."ProductID"
        GROUP BY pc."ProductID", pc.cluster, pc.profit, pc.nunique_customer, pc.profit_margin, 
                 pc.selling_duration, pc.revenue_growth, pc.average_unit_price
        """
        
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None

# Load data
df = load_cluster_data()

if df is not None:
    # Display summary statistics
    st.markdown("#### Cluster Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Products", len(df))
    with col2:
        st.metric("Number of Clusters", df['cluster_id'].nunique())
    with col3:
        st.metric("Avg Profit", f"${df['profit'].mean():,.2f}")
    with col4:
        st.metric("Total Quantity", f"{df['quantity'].sum():,.0f}")

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Profit Margin Analysis", "Selling Duration Analysis", "Revenue Growth Analysis"])

    with tab1:
        st.markdown("#### Profit Margin Distribution by Cluster")
        fig_margin = px.box(df, 
                            x='cluster_id', 
                            y='profit_margin',
                            color='cluster_id',
                            title='Profit Margin Distribution Across Clusters',
                            labels={'cluster_id': 'Cluster ID', 'profit_margin': 'Profit Margin'},
                            color_discrete_sequence=px.colors.qualitative.Set3)
        fig_margin.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_margin, width='stretch')

    with tab2:
        st.markdown("#### Selling Duration Distribution by Cluster")
        fig_duration = px.box(df, 
                              x='cluster_id', 
                              y='selling_duration',
                              color='cluster_id',
                              title='Selling Duration Distribution Across Clusters',
                              labels={'cluster_id': 'Cluster ID', 'selling_duration': 'Selling Duration (days)'},
                              color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_duration.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_duration, width='stretch')

    with tab3:
        st.markdown("#### Revenue Growth Distribution by Cluster")
        fig_growth = px.box(df, 
                           x='cluster_id', 
                           y='revenue_growth',
                           color='cluster_id',
                           title='Revenue Growth Distribution Across Clusters',
                           labels={'cluster_id': 'Cluster ID', 'revenue_growth': 'Revenue Growth'},
                           color_discrete_sequence=px.colors.qualitative.Safe)
        fig_growth.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_growth, width='stretch')

    # Display cluster statistics table and database summary side by side
    st.markdown("---")
    
    # Add clustering scatter plot
    st.markdown("#### Clustering Visualization")
    st.markdown("##### Average Unit Price vs Number of Orders by Cluster")
    
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
    
    # Convert cluster_id to string for discrete colors
    df_scatter = df.copy()
    df_scatter['cluster_id'] = df_scatter['cluster_id'].astype(str)
    
    fig_scatter = px.scatter(
        df_scatter,
        x='average_unit_price',
        y='order_frequency',
        color='cluster_id',
        title='Product Clustering: Average Unit Price vs Number of Orders',
        labels={
            'average_unit_price': 'Average Unit Price ($)',
            'order_frequency': 'Number of Orders',
            'cluster_id': 'Cluster ID'
        },
        color_discrete_map={str(k): v for k, v in color_map.items()},
        hover_data=['ProductID', 'profit', 'quantity'],
        height=600
    )
    
    fig_scatter.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    fig_scatter.update_layout(
        xaxis_title="Average Unit Price ($)",
        yaxis_title="Number of Orders",
        legend_title="Cluster ID",
        hovermode='closest'
    )
    
    st.plotly_chart(fig_scatter, width='stretch')
    
    st.markdown("---")

    # Create two columns for the tables
    col_stats, col_db = st.columns([1, 1])

    with col_stats:
        st.markdown("#### Cluster Statistics")
        cluster_stats = df.groupby('cluster_id').agg({
            'ProductID': 'count',
            'profit': ['mean', 'median', 'sum'],
            'quantity': ['mean', 'sum'],
            '#accumulated_customer': 'mean',
            'profit_margin': 'mean'
        }).round(2)

        cluster_stats.columns = ['Product Count', 'Avg Profit', 'Median Profit', 'Total Profit', 
                                 'Avg Quantity', 'Total Quantity', 'Avg Customers', 'Avg Profit Margin']
        st.dataframe(cluster_stats, width='stretch')

    with col_db:
        st.markdown("#### Cluster Summary from Database")
        # Calculate summary from loaded data
        db_summary = df.groupby('cluster_id').agg({
            'ProductID': 'count',
            'profit': 'sum'
        }).reset_index()
        
        db_summary.columns = ['Cluster ID', 'Number of Products', 'Profit']
        
        # Calculate profit percentage
        total_profit = db_summary['Profit'].sum()
        db_summary['Profit %'] = (db_summary['Profit'] / total_profit * 100).round(2)
        
        # Format columns
        db_summary['Profit'] = db_summary['Profit'].apply(lambda x: f"${x:,.2f}")
        db_summary['Profit %'] = db_summary['Profit %'].apply(lambda x: f"{x}%")
        
        st.dataframe(db_summary, width='stretch', hide_index=True)

    # Display pie charts below the tables
    st.markdown("---")
    st.markdown("#### Distribution Analysis")

    # Calculate summary for pie charts
    db_summary = df.groupby('cluster_id').agg({
        'ProductID': 'count',
        'profit': 'sum'
    }).reset_index()
    
    db_summary.columns = ['Cluster ID', 'Number of Products', 'Profit']
    
    # Create two columns for the pie charts
    col_products, col_profit = st.columns(2)
    
    # Extract numeric values for pie charts
    product_counts = db_summary['Number of Products'].values
    cluster_ids = db_summary['Cluster ID'].values
    profit_values = db_summary['Profit'].values
    
    colors = [color_map.get(cid, '#999999') for cid in cluster_ids]
    
    with col_products:
        st.markdown("##### Product Distribution")
        fig_products = px.pie(
            values=product_counts,
            names=cluster_ids,
            title='Percentage of Products per Cluster',
            labels={'names': 'Cluster ID', 'values': 'Number of Products'},
            hole=0.3,
            category_orders={'names': sorted(cluster_ids)},
            color_discrete_sequence=colors
        )
        fig_products.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_products, width='stretch')
    
    with col_profit:
        st.markdown("##### Profit Distribution")
        fig_profit = px.pie(
            values=profit_values,
            names=cluster_ids,
            title='Percentage of Profit per Cluster',
            labels={'names': 'Cluster ID', 'values': 'Profit'},
            hole=0.3,
            category_orders={'names': sorted(cluster_ids)},
            color_discrete_sequence=colors
        )
        fig_profit.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_profit, width='stretch')

    # Show raw data option
    if st.checkbox('Show raw cluster data'):
        st.subheader('Raw Data')
        st.dataframe(df, width='stretch')
else:
    st.error("Failed to load cluster data from database.")