import streamlit as st
import pandas as pd
import numpy as np
import joblib
from supabase import create_client
from datetime import datetime

st.set_page_config(page_title="HEYTEA Forecast", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

model = joblib.load("item_after_cutoff_model.pkl")

FEATURES = [
    "sku_no",
    "qty_before_cutoff",
    "lag_1_day_qty",
    "lag_7_day_qty",
    "lag_14_day_qty",
    "rolling_7d_avg",
    "rolling_14d_avg",
    "store_qty_before_cutoff",
    "store_transactions_before_cutoff",
    "sku_qty_last_hour",
    "store_qty_last_hour",
    "store_transactions_last_hour",
    "sku_sales_share",
    "top5_flag",
    "day_of_week",
    "month",
    "is_holiday",
    "closing_hour",
    "temp_max",
    "precipitation",
]


def upload_batches(table, records, batch_size=500):
    for i in range(0, len(records), batch_size):
        supabase.table(table).upsert(
            records[i:i + batch_size]
        ).execute()


def clean_hourly_data(df):
    df["sales_date"] = pd.to_datetime(
        df["report_date统计时间"],
        errors="coerce"
    )

    df = df[df["sales_date"].notna()].copy()

    df["sales_hour"] = pd.to_numeric(df["report_hour时段"], errors="coerce")
    df["sku_no"] = pd.to_numeric(df["sku_no商品规格编码"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty商品数量"], errors="coerce").fillna(0)
    df["ord_qty"] = pd.to_numeric(df["ord_qty订单数"], errors="coerce").fillna(0)

    df = df.dropna(subset=["sales_hour", "sku_no", "sku_name商品名称"])

    df["sku_no"] = df["sku_no"].astype("int64")

    df = df[
        df["sku_no"].astype(str).str.startswith("3300")
    ].copy()

    return pd.DataFrame({
        "sales_date": df["sales_date"].dt.strftime("%Y-%m-%d"),
        "sales_hour": df["sales_hour"].astype(int),
        "sku_no": df["sku_no"].astype(int),
        "sku_name": df["sku_name商品名称"].astype(str),
        "qty": df["qty"].astype(float),
        "ord_qty": df["ord_qty"].astype(float),
    })


def get_daily_history(sku_no, date):
    date = pd.to_datetime(date)

    start_date = (date - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    res = (
        supabase.table("daily_sku_sales")
        .select("*")
        .eq("sku_no", sku_no)
        .gte("sales_date", start_date)
        .lte("sales_date", end_date)
        .execute()
    )

    hist = pd.DataFrame(res.data)

    if hist.empty:
        return {
            "lag_1_day_qty": 0,
            "lag_7_day_qty": 0,
            "lag_14_day_qty": 0,
            "rolling_7d_avg": 0,
            "rolling_14d_avg": 0,
        }

    hist["sales_date"] = pd.to_datetime(hist["sales_date"])
    hist["daily_qty"] = pd.to_numeric(hist["daily_qty"], errors="coerce").fillna(0)

    def get_lag(days):
        target_date = date - pd.Timedelta(days=days)
        row = hist[hist["sales_date"] == target_date]
        return float(row["daily_qty"].iloc[0]) if not row.empty else 0

    last_7 = hist[hist["sales_date"] >= date - pd.Timedelta(days=7)]
    last_14 = hist[hist["sales_date"] >= date - pd.Timedelta(days=14)]

    return {
        "lag_1_day_qty": get_lag(1),
        "lag_7_day_qty": get_lag(7),
        "lag_14_day_qty": get_lag(14),
        "rolling_7d_avg": float(last_7["daily_qty"].mean()) if not last_7.empty else 0,
        "rolling_14d_avg": float(last_14["daily_qty"].mean()) if not last_14.empty else 0,
    }


def predict_one(row):
    sales_date = pd.to_datetime(row["date"])

    history = get_daily_history(
        int(row["sku_no"]),
        sales_date
    )

    store_qty_before_cutoff = float(row["store_qty_before_cutoff"])
    qty_before_cutoff = float(row["qty_before_cutoff"])

    sku_sales_share = (
        qty_before_cutoff / store_qty_before_cutoff
        if store_qty_before_cutoff > 0
        else 0
    )

    feature_row = {
        "sku_no": int(row["sku_no"]),
        "qty_before_cutoff": qty_before_cutoff,

        **history,

        "store_qty_before_cutoff": store_qty_before_cutoff,
        "store_transactions_before_cutoff": float(row["store_transactions_before_cutoff"]),

        "sku_qty_last_hour": float(row["sku_qty_last_hour"]),
        "store_qty_last_hour": float(row["store_qty_last_hour"]),
        "store_transactions_last_hour": float(row["store_transactions_last_hour"]),

        "sku_sales_share": sku_sales_share,
        "top5_flag": int(row.get("top5_flag", 0)),

        "day_of_week": int(sales_date.dayofweek),
        "month": int(sales_date.month),
        "is_holiday": int(row.get("is_holiday", 0)),

        "closing_hour": int(row["closing_hour"]),
        "temp_max": float(row["temp_max"]),
        "precipitation": float(row["precipitation"]),
    }

    X = pd.DataFrame([feature_row])[FEATURES]
    pred = model.predict(X)[0]

    return max(0, round(float(pred)))


tab1, tab2, tab3 = st.tabs([
    "1. Upload Hourly Data",
    "2. Upload Screenshot/Daily Data",
    "3. Predict"
])


with tab1:
    st.header("Upload Hourly Sales Data to Supabase")

    uploaded_file = st.file_uploader(
        "Upload hourly sales Excel",
        type=["xlsx", "xls"],
        key="hourly_upload"
    )

    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file)

        st.subheader("Raw File Preview")
        st.dataframe(df_raw.head(30))

        try:
            df_clean = clean_hourly_data(df_raw)

            st.subheader("Cleaned Data Preview")
            st.dataframe(df_clean.head(100))

            st.write("Cleaned rows:", len(df_clean))
            st.write("Date range:", df_clean["sales_date"].min(), "to", df_clean["sales_date"].max())
            st.write("Unique SKUs:", df_clean["sku_no"].nunique())
            st.write("Total Qty:", df_clean["qty"].sum())
            st.write("Total Orders:", df_clean["ord_qty"].sum())

            duplicate_count = df_clean.duplicated(
                subset=["sales_date", "sales_hour", "sku_no"]
            ).sum()

            st.write("Duplicate rows:", duplicate_count)

            if duplicate_count > 0:
                st.warning("Duplicate sales_date + sales_hour + sku_no rows found. They will be aggregated before upload.")

                df_clean = (
                    df_clean
                    .groupby(
                        ["sales_date", "sales_hour", "sku_no", "sku_name"],
                        as_index=False
                    )
                    .agg({
                        "qty": "sum",
                        "ord_qty": "sum"
                    })
                )

                st.subheader("After Duplicate Aggregation")
                st.dataframe(df_clean.head(100))
                st.write("Final rows:", len(df_clean))

            confirm_upload = st.checkbox(
                "I confirm the cleaned data is correct and ready to upload."
            )

            if confirm_upload:
                if st.button("Upload Cleaned Hourly Data"):
                    records = df_clean.to_dict("records")

                    uploaded_rows = 0
                    failed_batches = []

                    progress = st.progress(0)

                    for i in range(0, len(records), 500):
                        batch = records[i:i + 500]

                        try:
                            supabase.table("sales_hourly").upsert(
                                batch,
                                on_conflict="sales_date,sales_hour,sku_no"
                            ).execute()

                            uploaded_rows += len(batch)

                        except Exception as e:
                            failed_batches.append({
                                "start_row": i,
                                "end_row": i + len(batch),
                                "error": str(e)
                            })

                        progress.progress(
                            min((i + 500) / len(records), 1.0)
                        )

                    if failed_batches:
                        st.error("Some batches failed.")
                        st.dataframe(pd.DataFrame(failed_batches))
                    else:
                        st.success(f"Upload complete: {uploaded_rows} rows uploaded/upserted.")

                    st.subheader("Upload Summary")
                    st.write("Uploaded rows:", uploaded_rows)
                    st.write("Date range:", df_clean["sales_date"].min(), "to", df_clean["sales_date"].max())
                    st.write("Unique SKUs:", df_clean["sku_no"].nunique())
                    st.write("Total Qty:", df_clean["qty"].sum())
                    st.write("Total Orders:", df_clean["ord_qty"].sum())

        except Exception as e:
            st.error("Failed to clean uploaded file.")
            st.exception(e)

with tab2:
    st.header("Upload Screenshot / Daily SKU Sales")

    st.info("For now, paste or upload cleaned daily screenshot data: date, sku_no, sku_name, daily_qty.")

    daily_file = st.file_uploader(
        "Upload daily SKU sales CSV/Excel",
        type=["csv", "xlsx", "xls"],
        key="daily_upload"
    )

    if daily_file:
        if daily_file.name.endswith(".csv"):
            daily_df = pd.read_csv(daily_file)
        else:
            daily_df = pd.read_excel(daily_file)

        daily_df["sales_date"] = pd.to_datetime(daily_df["sales_date"]).dt.strftime("%Y-%m-%d")
        daily_df["sku_no"] = daily_df["sku_no"].astype(int)
        daily_df["sku_name"] = daily_df["sku_name"].astype(str)
        daily_df["daily_qty"] = pd.to_numeric(daily_df["daily_qty"], errors="coerce").fillna(0)

        daily_clean = daily_df[[
            "sales_date",
            "sku_no",
            "sku_name",
            "daily_qty"
        ]].copy()

        daily_clean["source"] = "screenshot"

        st.dataframe(daily_clean)

        if st.button("Upload Daily Screenshot Data"):
            records = daily_clean.to_dict("records")

            for i in range(0, len(records), 500):
                supabase.table("daily_sku_sales").upsert(
                    records[i:i + 500],
                    on_conflict="sales_date,sku_no"
                ).execute()

            st.success(f"Uploaded {len(records)} daily rows.")


with tab3:
    st.header("Predict After-Cutoff Sales")

    st.info("Upload a prediction input table, or enter one item manually.")

    mode = st.radio("Input mode", ["Manual", "Upload Table"])

    if mode == "Manual":
        col1, col2, col3 = st.columns(3)

        with col1:
            date = st.date_input("Date")
            sku_no = st.number_input("SKU No", value=33000029, step=1)
            sku_name = st.text_input("SKU Name", value="椰椰芒芒")
            qty_before_cutoff = st.number_input("Qty Before Cutoff", value=0.0)

        with col2:
            store_qty_before_cutoff = st.number_input("Store Qty Before Cutoff", value=0.0)
            store_transactions_before_cutoff = st.number_input("Store Transactions Before Cutoff", value=0.0)
            sku_qty_last_hour = st.number_input("SKU Qty Last Hour", value=0.0)
            store_qty_last_hour = st.number_input("Store Qty Last Hour", value=0.0)

        with col3:
            store_transactions_last_hour = st.number_input("Store Transactions Last Hour", value=0.0)
            closing_hour = st.number_input("Closing Hour", value=21, step=1)
            temp_max = st.number_input("Temp Max", value=25.0)
            precipitation = st.number_input("Precipitation", value=0.0)
            is_holiday = st.checkbox("Holiday")
            top5_flag = st.checkbox("Top 5 SKU")

        if st.button("Predict"):
            row = {
                "date": str(date),
                "sku_no": sku_no,
                "sku_name": sku_name,
                "qty_before_cutoff": qty_before_cutoff,
                "store_qty_before_cutoff": store_qty_before_cutoff,
                "store_transactions_before_cutoff": store_transactions_before_cutoff,
                "sku_qty_last_hour": sku_qty_last_hour,
                "store_qty_last_hour": store_qty_last_hour,
                "store_transactions_last_hour": store_transactions_last_hour,
                "closing_hour": closing_hour,
                "temp_max": temp_max,
                "precipitation": precipitation,
                "is_holiday": int(is_holiday),
                "top5_flag": int(top5_flag),
            }

            prediction = predict_one(row)

            st.success(f"Predicted after-cutoff sales: {prediction} cups")

    else:
        pred_file = st.file_uploader(
            "Upload prediction input CSV/Excel",
            type=["csv", "xlsx", "xls"],
            key="predict_upload"
        )

        if pred_file:
            if pred_file.name.endswith(".csv"):
                pred_df = pd.read_csv(pred_file)
            else:
                pred_df = pd.read_excel(pred_file)

            predictions = []

            for _, row in pred_df.iterrows():
                pred = predict_one(row)
                predictions.append(pred)

            pred_df["predicted_after_cutoff"] = predictions

            st.dataframe(pred_df)

            csv = pred_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "Download Prediction Result",
                csv,
                "prediction_result.csv",
                "text/csv"
            )
