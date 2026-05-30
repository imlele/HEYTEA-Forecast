import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("item_after_cutoff_model.pkl")

app = FastAPI()

FEATURES = [
    "sku_no商品规格编码",
    "qty_before_cutoff",
    "lag_1_day_qty",
    "lag_7_day_qty",
    "rolling_7d_avg",
    "sku_sales_share",
    "temp_max",
    "precipitation",
    "day_of_week",
    "month",
    "is_holiday",
    "is_promo",
    "closing_hour"
]

class PredictInput(BaseModel):
    sku_no: int
    qty_before_cutoff: float
    lag_1_day_qty: float = 0
    lag_7_day_qty: float = 0
    rolling_7d_avg: float = 0
    sku_sales_share: float = 0
    temp_max: float
    precipitation: float = 0
    date: str
    is_holiday: bool = False
    is_promo: bool = False
    closing_hour: int = 22

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: PredictInput):
    dt = pd.to_datetime(input.date)

    row = pd.DataFrame([{
        "sku_no商品规格编码": input.sku_no,
        "qty_before_cutoff": input.qty_before_cutoff,
        "lag_1_day_qty": input.lag_1_day_qty,
        "lag_7_day_qty": input.lag_7_day_qty,
        "rolling_7d_avg": input.rolling_7d_avg,
        "sku_sales_share": input.sku_sales_share,
        "temp_max": input.temp_max,
        "precipitation": input.precipitation,
        "day_of_week": dt.dayofweek,
        "month": dt.month,
        "is_holiday": int(input.is_holiday),
        "is_promo": int(input.is_promo),
        "closing_hour": input.closing_hour
    }])

    prediction = model.predict(row[FEATURES])[0]
    prediction = max(0, round(float(prediction)))

    return {
        "sku_no": input.sku_no,
        "date": input.date,
        "cutoff_hour": input.closing_hour - 4,
        "predicted_after_cutoff": prediction
    }