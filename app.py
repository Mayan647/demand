import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Demand Forecasting", layout="centered")

st.title("📈 Demand Forecasting (Holt-Winters Method)")

uploaded_file = st.file_uploader("Upload Excel file with 'Date' and 'Demand' columns", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [col.strip().capitalize() for col in df.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()

        st.subheader("📊 Uploaded Data")
        st.line_chart(df['Demand'])

        model = ExponentialSmoothing(df['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
        forecast = model.forecast(12)
        
        st.subheader("📈 Forecast")
        forecast_df = pd.DataFrame({'Forecast': forecast})
        st.line_chart(pd.concat([df['Demand'], forecast_df['Forecast']]))

        st.download_button("Download Forecast CSV", forecast_df.to_csv().encode('utf-8'), file_name='forecast.csv')
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Awaiting Excel file upload...")

