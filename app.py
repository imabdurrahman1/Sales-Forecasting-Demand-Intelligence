import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

.block-container{
    padding-top:2rem;
}

h1,h2,h3{
    color:#1F3B73;
}

section[data-testid="stSidebar"]{
    background-color:#0E1117;
}

section[data-testid="stSidebar"] *{
    color:white;
}

</style>
""", unsafe_allow_html=True)


# Page Configuration

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)


# Title

st.title("📈 End-to-End Sales Forecasting & Demand Intelligence System")

st.markdown("""
This dashboard provides business managers with a consolidated view of historical sales, forecasts, anomalies, and demand segmentation to support inventory planning and strategic decision-making.
""")

st.divider()


# Load Dataset

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    return df

sales_df = load_data()


# Load Analysis Outputs


forecast_df = pd.read_csv("outputs/forecast_summary.csv")

model_df = pd.read_csv("outputs/model_comparison.csv")

cluster_df = pd.read_csv("outputs/cluster_summary.csv")

anomaly_df = pd.read_csv("outputs/anomaly_report.csv")


# Sidebar

with st.sidebar:

    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=70)

    st.title("Sales Forecasting")

    st.caption("Demand Intelligence Dashboard")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Sales Overview",
            "Forecast Explorer",
            "Anomaly Report",
            "Demand Segmentation"
        ]
    )

    st.divider()

    st.markdown("### Project")

    st.write("Xylofy AI Internship")

    st.write("Version 1.7")

    st.write("Developed by")

    st.success("Abdur Rahman M")

    
if page == "Sales Overview":

    st.header("📊 Sales Overview")

    total_sales = sales_df["Sales"].sum()
    total_orders = sales_df["Order ID"].nunique()
    total_customers = sales_df["Customer ID"].nunique()
    total_products = sales_df["Product ID"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
    col2.metric("🧾 Orders", total_orders)
    col3.metric("👥 Customers", total_customers)
    col4.metric("📦 Products", total_products)

    st.info("""
    ### Executive Summary

    - 💰 Total Sales exceeded **$2.26 Million**
    - 📈 XGBoost delivered the most accurate forecasts
    - 🚨 11 major sales anomalies were detected
    - 📦 Products were grouped into three demand segments
    - 🎯 The dashboard supports data-driven inventory planning
    """)

    monthly_sales = (
        sales_df
        .groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        title="Monthly Sales Trend"
    )

    st.plotly_chart(fig, use_container_width=True)
    

    st.subheader("Regional & Category Analysis")

    col1, col2 = st.columns(2)

    region = col1.selectbox(
        "Select Region",
        sorted(sales_df["Region"].unique())
    )

    category = col2.selectbox(
        "Select Category",
        sorted(sales_df["Category"].unique())
    )

    filtered = sales_df[
        (sales_df["Region"] == region) &
        (sales_df["Category"] == category)
    ]

    sales_summary = (
        filtered
        .groupby("Sub-Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        sales_summary,
        x="Sub-Category",
        y="Sales",
        title=f"{category} Sales in {region}"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.caption(
        "Abdur Rahman M | Xylofy AI Internship Project | Sales Forecasting & Demand Intelligence System"
    )
    
elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    st.write(
        "Explore the forecast generated using the best-performing XGBoost model."
    )

    # Best Model Metrics
    best_model = model_df.loc[model_df["MAPE"].idxmin()]

    c1, c2, c3 = st.columns(3)

    c1.metric("Best Model", best_model["Model"])
    c2.metric("MAPE", f"{best_model['MAPE']:.2f}%")
    c3.metric("RMSE", f"{best_model['RMSE']:.2f}")

    st.divider()

    # Forecast Type Selection
    forecast_type = st.radio(
        "Forecast Level",
        ["Category", "Region"],
        horizontal=True
    )

    if forecast_type == "Category":
        options = ["Furniture", "Technology", "Office Supplies"]
    else:
        options = ["East", "West"]

    selected = st.selectbox(
        f"Select {forecast_type}",
        options
    )

    row = forecast_df[
        forecast_df["Segment"] == selected
    ]

    chart_df = pd.DataFrame({
        "Month": ["Month 1", "Month 2", "Month 3"],
        "Forecast": [
            row["Month 1"].values[0],
            row["Month 2"].values[0],
            row["Month 3"].values[0]
        ]
    })

    st.dataframe(row, use_container_width=True)

    fig = px.line(
        chart_df,
        x="Month",
        y="Forecast",
        markers=True,
        title=f"{selected} Sales Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"The {selected} segment is forecasted to achieve "
        f"{chart_df['Forecast'].sum():,.0f} total sales over the next three months."
    )

    st.info("""
    **Business Insight**

    - Forecasts help optimize inventory and procurement planning.
    - Prepare additional stock before expected demand peaks.
    - Review forecasts monthly as new sales data becomes available.
    """)
    st.divider()

    st.caption(
        "Abdur Rahman M | Xylofy AI Internship Project | Sales Forecasting & Demand Intelligence System"
    )

elif page == "Anomaly Report":

    st.header("🚨 Sales Anomaly Report")

    st.write("""
    This page highlights unusual sales periods detected using machine learning
    and statistical techniques. Identifying these anomalies helps businesses
    investigate unexpected spikes or drops in sales and improve future planning.
    """)

    st.divider()

    total_if = len(anomaly_df)

    total_sales = len(sales_df)

    anomaly_percent = (total_if / total_sales) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        total_sales
    )

    col2.metric(
        "Detected Anomalies",
        total_if
    )

    col3.metric(
        "Anomaly Rate",
        f"{anomaly_percent:.2f}%"
    )

    st.divider()

    st.subheader("Sales Anomaly Detection")

    st.image(
        "charts/anomaly.png",
        use_container_width=True
    )

    st.caption(
        "Detected anomaly weeks highlighted using Isolation Forest and Z-Score analysis."
    )

    st.divider()
    

    # Table

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomaly_df,
        use_container_width=True
    )

    st.divider()

    # Business Interpretation


    st.subheader("Business Interpretation")

    st.success("""
• Sudden positive spikes may indicate festive seasons, promotional campaigns,
  bulk customer purchases or successful marketing events.

• Sudden drops may suggest supply shortages, operational disruptions,
  product availability issues or lower customer demand.

• Monitoring these events allows supply chain teams to respond quickly
  and improve forecasting accuracy.
""")

    st.info("""
Recommendation

✔ Investigate every detected anomaly with business teams.

✔ Record the business reason (promotion, holiday, stock issue, etc.).

✔ Use this information to continuously improve forecasting models.
""")

    st.warning("""
Business Note

Not every anomaly represents a problem.

Some anomalies are positive opportunities (unexpected high demand),
while others may indicate operational issues requiring immediate attention.
""")
    st.divider()

    st.caption(
        "Abdur Rahman M | Xylofy AI Internship Project | Sales Forecasting & Demand Intelligence System"
    )

elif page == "Demand Segmentation":

    st.header("📦 Product Demand Segmentation")

    st.write("""
Products have been grouped using Machine Learning (K-Means Clustering)
based on sales behaviour, growth trend, volatility and average order value.

These clusters help businesses apply different stocking strategies instead
of treating every product equally.
    """)

    st.divider()


    # Cluster Summary


    st.subheader("Demand Segment Summary")

    st.dataframe(
        cluster_df,
        use_container_width=True
    )

    st.divider()


    # Cluster Visualization


    st.subheader("Cluster Visualization")

    st.image(
        "charts/product_clusters.png",
        use_container_width=True
    )

    st.caption(
        "Product demand clusters obtained using PCA visualization."
    )

    st.divider()


    # Recommended Stocking Strategy


    st.subheader("Recommended Stocking Strategy")

    strategy = pd.DataFrame({

        "Cluster":[0,1,2],

        "Business Interpretation":[
            "Growing Demand Products",
            "Low Volume Stable Products",
            "High Volume Core Products"
        ],

        "Recommended Action":[
            "Increase inventory and monitor future growth",
            "Maintain conservative stock levels",
            "Prioritize inventory and avoid stock-outs"
        ]

    })

    st.dataframe(
        strategy,
        use_container_width=True
    )

    st.divider()


    # Business Recommendations


    st.success("""
Business Recommendations

• Prioritize inventory investment for High Volume Core Products.

• Increase monitoring of Growing Demand Products to capture
future sales opportunities.

• Keep Low Volume products under controlled inventory
to reduce warehousing costs.

• Review cluster assignments periodically as customer demand changes.
""")

    st.info("""
Business Value

Demand segmentation allows the supply chain team to
allocate inventory budgets more effectively,
reducing stock shortages while minimizing excess inventory.
""")
    st.divider()

    st.caption(
        "Abdur Rahman M | Xylofy AI Internship Project | Sales Forecasting & Demand Intelligence System"
    )