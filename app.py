import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from io import BytesIO

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

            forecast_horizon = 5  # months
            result = {}

            for item in monthly_df['Item Code'].unique():
                item_df = monthly_df[monthly_df['Item Code'] == item].copy()
                item_df.set_index('Month', inplace=True)

                # Choose model based on data availability
                if len(item_df) >= 24:
                    model = ExponentialSmoothing(
                        item_df['Demand'],
                        trend='add',
                        seasonal='add',
                        seasonal_periods=12
                    ).fit()
                else:
                    model = ExponentialSmoothing(
                        item_df['Demand'],
                        trend='add'
                    ).fit()

                forecast = model.forecast(forecast_horizon)
                forecast.index = forecast.index.to_period('M').to_timestamp()
                result[item] = forecast

            # Prepare output DataFrame
            forecast_df = pd.DataFrame(result).T
            forecast_df.columns = [col.strftime('%B %Y') for col in forecast_df.columns]
            forecast_df.index.name = 'Item Name'

            # Convert to Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                forecast_df.to_excel(writer, sheet_name='Forecast')
            output.seek(0)

            st.download_button(
                label="📥 Download Forecast Excel",
                data=output,
                file_name="item_wise_monthly_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success("✅ Forecast ready! Download using the button above.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Upload an Excel file to get started.")
