import streamlit as st
import pandas as pd
from lightgbm import LGBMRegressor
from io import BytesIO

st.set_page_config(page_title="ML-Based Monthly Forecast", layout="centered")
st.title("🤖 LightGBM Forecasting (Monthly Data Input)")

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

            forecast_results = []
            skipped_items = []

            for item in monthly_df['Item Code'].unique():
                item_df = monthly_df[monthly_df['Item Code'] == item].copy()
                item_df = item_df.sort_values('Month').reset_index(drop=True)

                if len(item_df) < 6:
                    skipped_items.append(item)
                    continue

                # Create lag features
                item_df['Lag1'] = item_df['Demand'].shift(1)
                item_df['Lag2'] = item_df['Demand'].shift(2)
                item_df['Lag3'] = item_df['Demand'].shift(3)
                item_df = item_df.dropna().reset_index(drop=True)

                if len(item_df) < 3:
                    skipped_items.append(item)
                    continue

                X = item_df[['Lag1', 'Lag2', 'Lag3']]
                y = item_df['Demand']
                model = LGBMRegressor()
                model.fit(X, y)

                # Rolling forecast
                lags = item_df.iloc[-1][['Lag1', 'Lag2', 'Lag3']].tolist()
                forecast = []
                months = []

                last_month = item_df['Month'].max()

                for i in range(5):
                    pred = model.predict([lags])[0]
                    forecast.append(pred)
                    last_month += pd.offsets.MonthBegin(1)
                    months.append(last_month)
                    lags = [pred] + lags[:2]  # update lags

                forecast_results.append(pd.DataFrame({
                    'Item Name': [item],
                    **{month.strftime('%B %Y'): [val] for month, val in zip(months, forecast)}
                }))

            if not forecast_results:
                st.error("❌ No forecastable items found.")
            else:
                result_df = pd.concat(forecast_results, ignore_index=True)

                # Downloadable Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    result_df.to_excel(writer, index=False, sheet_name='Forecast')
                output.seek(0)

                st.download_button(
                    label="📥 Download Forecast Excel",
                    data=output,
                    file_name="ml_item_forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.dataframe(result_df)

            if skipped_items:
                st.warning(f"⚠️ Skipped items due to insufficient data: {', '.join(skipped_items)}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Upload a monthly Excel file to begin forecasting.")
