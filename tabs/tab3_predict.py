import streamlit as st
import pandas as pd


def render_tab3_predict():
    st.header("Tab 3: Predict")
    st.caption("Prediction tab placeholder. We can connect this to your trained model/API next.")

    st.info(
        "After Tab 1 and Tab 2 data are uploaded, this tab can pull recent rows from Supabase "
        "and create prediction inputs."
    )

    st.subheader("Manual test input")

    sku_no = st.text_input("SKU No", value="")
    qty_before_cutoff = st.number_input("Qty before cutoff", min_value=0.0, value=0.0)
    lag_1_day_qty = st.number_input("Lag 1 day qty", min_value=0.0, value=0.0)
    lag_7_day_qty = st.number_input("Lag 7 day qty", min_value=0.0, value=0.0)
    rolling_7d_avg = st.number_input("Rolling 7d avg", min_value=0.0, value=0.0)
    temp_max = st.number_input("Temp max", value=20.0)
    precipitation = st.number_input("Precipitation", min_value=0.0, value=0.0)
    is_holiday = st.selectbox("Is holiday", options=[0, 1])
    is_promo = st.selectbox("Is promo", options=[0, 1])
    closing_hour = st.number_input("Closing hour", min_value=0, max_value=24, value=21)

    input_df = pd.DataFrame([
        {
            "sku_no": sku_no,
            "qty_before_cutoff": qty_before_cutoff,
            "lag_1_day_qty": lag_1_day_qty,
            "lag_7_day_qty": lag_7_day_qty,
            "rolling_7d_avg": rolling_7d_avg,
            "temp_max": temp_max,
            "precipitation": precipitation,
            "is_holiday": is_holiday,
            "is_promo": is_promo,
            "closing_hour": closing_hour,
        }
    ])

    st.dataframe(input_df, use_container_width=True)

    if st.button("Predict"):
        st.warning("Prediction API/model is not connected in this placeholder yet.")
