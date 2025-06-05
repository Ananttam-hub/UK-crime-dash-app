import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="UK Crime Dashboard", layout="wide")

@st.cache_data(show_spinner=True)
def load_data():
    file_path =  'Master_file_cleaned.xlsb'
    df = pd.read_excel(file_path)
    df.dropna(subset=["Longitude", "Latitude"], inplace=True)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    return df

df = load_data()

st.title("📊 UK Crime Dashboard")
st.markdown("Explore, analyze, and forecast crime patterns in the UK.")

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
months = df["Month"].dt.to_period("M").astype(str).unique()
selected_month = st.sidebar.selectbox("Select Month", sorted(months))

df_month = df[df["Month"].dt.to_period("M").astype(str) == selected_month]

crime_types = df_month["Crime type"].unique()
selected_crimes = st.sidebar.multiselect("Crime Types", crime_types, default=list(crime_types))

forces = df_month["Reported by"].unique()
selected_forces = st.sidebar.multiselect("Police Forces", forces, default=list(forces))

df_filtered = df_month[
    (df_month["Crime type"].isin(selected_crimes)) &
    (df_month["Reported by"].isin(selected_forces))
]

st.write(f"🔎 Showing **{len(df_filtered)}** records for **{selected_month}**.")

# KPIs
col1, col2 = st.columns(2)
with col1:
    st.metric("📌 Total Crimes", len(df_filtered))
with col2:
    prev_month = pd.Period(selected_month) - 1
    prev_df = df[df["Month"].dt.to_period("M").astype(str) == str(prev_month)]
    if not prev_df.empty:
        prev_count = len(prev_df[
            (prev_df["Crime type"].isin(selected_crimes)) &
            (prev_df["Reported by"].isin(selected_forces))
        ])
        change = len(df_filtered) - prev_count
        pct = (change / prev_count) * 100 if prev_count > 0 else 0
        st.metric("📈 Change from Previous Month", f"{change:+}", f"{pct:.1f}%")
    else:
        st.metric("📈 Change from Previous Month", "N/A")

if df_filtered.empty:
    st.warning("No data for selected filters.")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Crime Types", "📍 Hotspots & Distribution", "📈 Trends", "🔮 Forecast", "⬇️ Downloads"
])

# Tab 1: Crime Types
with tab1:
    st.subheader("📂 Crimes by Type")
    crime_count = df_filtered["Crime type"].value_counts().reset_index()
    crime_count.columns = ["Crime Type", "Count"]
    fig1 = px.bar(crime_count, x="Crime Type", y="Count", color="Count", text="Count",
                  title="Crime Distribution by Type")
    fig1.update_traces(textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

# Tab 2: Hotspots & Area Analysis
with tab2:
    st.subheader("📍 Top Crime Hotspots")
    if "Location" in df_filtered.columns:
        top_locations = df_filtered["Location"].value_counts().head(10).reset_index()
        top_locations.columns = ["Location", "Count"]
        st.dataframe(top_locations, use_container_width=True)

    st.subheader("📊 Area-wise Crime Distribution")
    area_chart = df_filtered["LSOA name"].value_counts().nlargest(10).reset_index()
    area_chart.columns = ["LSOA Area", "Count"]
    fig2 = px.bar(area_chart, x="LSOA Area", y="Count", title="Top 10 Areas with Crimes",
                  color="Count", text="Count")
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔥 Crime Location Heatmap")
    heatmap_data = df_filtered[["Latitude", "Longitude"]]
    if not heatmap_data.empty:
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=heatmap_data["Latitude"].mean(),
                longitude=heatmap_data["Longitude"].mean(),
                zoom=10,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "HeatmapLayer",
                    data=heatmap_data,
                    get_position="[Longitude, Latitude]",
                    radius=200,
                    threshold=0.3
                )
            ],
        ))

# Tab 3: Trends
with tab3:
    st.subheader("📆 Monthly Crime Trend by Selected Filters")
    df_time = df[
        (df["Crime type"].isin(selected_crimes)) &
        (df["Reported by"].isin(selected_forces))
    ]
    trend = df_time.groupby([df_time["Month"].dt.to_period("M"), "Crime type"]).size().reset_index(name="Crimes")
    trend["Month"] = trend["Month"].dt.to_timestamp()
    fig3 = px.line(trend, x="Month", y="Crimes", color="Crime type", markers=True,
                   title="Monthly Crime Trends")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("⚖️ Crime Outcomes by Type (Stacked Bar Chart)")
    outcome_df = df_filtered.groupby(["Crime type", "Last outcome category"]).size().reset_index(name="Count")
    outcome_pivot = outcome_df.pivot(index="Crime type", columns="Last outcome category", values="Count").fillna(0)
    outcome_pivot = outcome_pivot.sort_index()
    outcome_stacked = outcome_pivot.reset_index().melt(id_vars="Crime type", var_name="Outcome", value_name="Count")
    fig4 = px.bar(outcome_stacked, x="Crime type", y="Count", color="Outcome",
                 title="Crime Outcomes by Type (Stacked)", barmode="stack", text="Count")
    fig4.update_traces(textposition="inside")
    st.plotly_chart(fig4, use_container_width=True)

# Tab 4: Forecast
with tab4:
    st.subheader("🔮 Forecast Future Crimes (Linear Regression)")

    forecast_crime = st.selectbox("Select Crime Type to Forecast", sorted(df["Crime type"].unique()))
    forecast_months = st.slider("Forecast Months Ahead", min_value=3, max_value=24, value=6)

    forecast_data = df[df["Crime type"] == forecast_crime]
    monthly_data = forecast_data.groupby(forecast_data["Month"].dt.to_period("M")).size().reset_index(name="Crimes")
    monthly_data["Month"] = monthly_data["Month"].dt.to_timestamp()
    monthly_data = monthly_data.sort_values("Month")

    if len(monthly_data) >= 12:
        monthly_data["MonthNum"] = np.arange(len(monthly_data)).reshape(-1, 1)
        X = monthly_data["MonthNum"].values.reshape(-1, 1)
        y = monthly_data["Crimes"].values
        model = LinearRegression().fit(X, y)

        future_months_idx = np.arange(len(monthly_data), len(monthly_data) + forecast_months).reshape(-1, 1)
        predictions = model.predict(future_months_idx)

        last_date = monthly_data["Month"].max()
        future_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(forecast_months)]

        future_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Crimes": predictions.round().astype(int)
        })

        std_dev = np.std(y - model.predict(X))
        future_df["Lower Bound"] = (predictions - std_dev).round().astype(int)
        future_df["Upper Bound"] = (predictions + std_dev).round().astype(int)

        all_df = pd.concat([
            monthly_data[["Month", "Crimes"]].rename(columns={"Month": "Date", "Crimes": "Actual Crimes"}),
            future_df[["Date", "Predicted Crimes"]]
        ], ignore_index=True)

        fig5 = px.line(all_df, x="Date", y=["Actual Crimes", "Predicted Crimes"], markers=True,
                       title=f"{forecast_crime} Forecast using Linear Regression")
        st.plotly_chart(fig5, use_container_width=True)

        forecast_csv = future_df.to_csv(index=False)
        st.download_button("📅 Download Forecast CSV", data=forecast_csv,
                           file_name=f"{forecast_crime}_forecast.csv", mime="text/csv")
    else:
        st.warning("Not enough historical data for forecasting this crime type (minimum 12 months required).")

# Tab 5: Download Filtered Data
with tab5:
    st.subheader("⬇️ Download Filtered Data")
    csv_data = df_filtered.to_csv(index=False)
    st.download_button("Download Filtered CSV", data=csv_data, file_name="filtered_crime_data.csv", mime="text/csv")
