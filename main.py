from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np

app = FastAPI(
    title="Solar Power Generation & Deviation API",
    description="Estimates AC Power output and detects performance deviations",
    version="1.0"
)

# Load the trained Random Forest model
try:
    model = joblib.load("solar_model.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load solar_model.pkl: {e}")

DEVIATION_THRESHOLD = 38.19

class SolarInput(BaseModel):
    irradiation: float = Field(..., example=0.78, description="Solar irradiation in kW/m²")
    ambient_temperature: float = Field(..., example=34.5, description="Ambient air temperature in °C")
    module_temperature: float = Field(..., example=52.1, description="Solar panel module temperature in °C")
    hour: int = Field(..., ge=0, le=23, example=12, description="Hour of the day (0-23)")
    month: int = Field(..., ge=1, le=12, example=6, description="Month of the year (1-12)")
    day: int = Field(..., ge=1, le=366, example=163, description="Day of the year (1-366)")
    actual_ac_power: Optional[float] = Field(None, example=430.5, description="Actual measured AC power (kW)")

@app.get("/")
def home():
    return {
        "status": "online",
        "project": "Project 5: Solar Power Generation Prediction",
        "docs_url": "/docs"
    }

@app.post("/predict")
def predict_solar_power(data: SolarInput):
    try:
        features = np.array([[
            data.irradiation,
            data.ambient_temperature,
            data.module_temperature,
            data.hour,
            data.month,
            data.day
        ]])

        if data.irradiation == 0.0 or (data.hour >= 19 or data.hour <= 5):
            predicted_ac = 0.0
        else:
            predicted_ac = float(model.predict(features)[0])
            predicted_ac = max(0.0, round(predicted_ac, 2))

        response = {
            "predicted_ac_power_kw": predicted_ac,
            "unit": "kW"
        }

        if data.actual_ac_power is not None:
            deviation = round(data.actual_ac_power - predicted_ac, 2)
            abs_dev = abs(deviation)
            status = "Needs Investigation" if abs_dev > DEVIATION_THRESHOLD else "Normal"

            response["actual_ac_power_kw"] = data.actual_ac_power
            response["deviation_kw"] = deviation
            response["deviation_threshold_kw"] = DEVIATION_THRESHOLD
            response["status"] = status

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
