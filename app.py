import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Monthly Item-wise Demand Forecasting", layout="centered")
st.title("📦 Monthly Demand Forecasting (Holt-Winters)")

uploaded_file = st.file_uploader("Upload Excel with: Date, Item Code, Demand", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [col.strip().title() for col in df.columns]

        required_cols = {"Date", "Item Code", "Demand"}
        if not required_cols.issubset(df.columns):
            st.error("❌ Required columns: Date, Item Code, Demand")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
            monthly_df = df.groupby(['Month', 'Item Code'])['Demand'].sum().reset_index()

            items = monthly_df['Item Code'].unique()
            selected_item = st.selectbox("Select Item Code", items)

            item_df = monthly_df[monthly_df['Item Code'] == selected_item].copy()
            item_df.set_index('Month', inplace=True)

            st.subheader(f"📊 Monthly Demand: {selected_item}")
            st.line_chart(item_df['Demand'])

            if len(item_df) < 12:
                st.warning("⚠️ Less than 12 months of data. Forecast may be unreliable.")
            else:
                model = ExponentialSmoothing(
                    item_df['Demand'],
                    trend='add',
                    seasonal='add',
                    seasonal_periods=12
                ).fit()

                forecast = model.forecast(12)
                forecast_df = pd.DataFrame({'Forecast': forecast})
                
                st.subheader("📈 12-Month Forecast")
                st.line_chart(pd.concat([item_df['Demand'], forecast_df['Forecast']]))

                forecast_df.index.name = 'Month'
                forecast_df.reset_index(inplace=True)
                forecast_df.insert(0, 'Item Code', selected_item)

                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=forecast_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{selected_item}_monthly_forecast.csv"
                )
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Upload an Excel file to get started.")
