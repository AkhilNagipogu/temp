"""
EMS & First Responder Rapid Dispatch FastAPI Backend
Connects the emergency response triage console to the Random Forest model
and live weather telemetry.
"""

from datetime import datetime
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import predictor

app = FastAPI(title="EMS & First Responder Dispatch Console")
STATIC_DIR = Path(__file__).parent / "static"
HTML_FILE = STATIC_DIR / "index.html"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TriageRequest(BaseModel):
    weather: str = "Clear"
    hour: int = Field(default=17, ge=0, le=23)
    weekday: int = Field(default=2, ge=0, le=6)
    month: int = Field(default=6, ge=1, le=12)
    lighting: str = "Day"
    visibility: float = Field(default=10.0, ge=0, le=100)
    temperature: float = Field(default=68.0, ge=-60, le=140)
    wind_speed: float = Field(default=5.0, ge=0, le=200)
    junction: bool = False
    traffic_signal: bool = False
    crossing: bool = False
    incident_call_id: str = "CAD-911-8841"
    reported_vehicles: int = 2


def map_weather_code(code: int) -> str:
    if code in (0, 1):
        return "Clear"
    if code in (2, 3):
        return "Cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (71, 73, 75, 77, 85, 86):
        return "Snow"
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99):
        return "Rain"
    return "Other"


@app.get("/")
def serve_home():
    return FileResponse(HTML_FILE)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "role": "EMS & First Responder Dispatch Commander",
        "model": predictor.MODEL_NAME,
        "stakeholder": "Emergency Medical Services & 911 Communications"
    }


@app.get("/options")
def get_options():
    return {"weather": predictor.WEATHER_OPTIONS}


@app.get("/stats")
def get_stats():
    return predictor.STATS


@app.post("/predict-triage")
def evaluate_triage(req: TriageRequest):
    try:
        res = predictor.predict_triage(
            weather=req.weather,
            lighting=req.lighting,
            hour=req.hour,
            weekday=req.weekday,
            month=req.month,
            visibility=req.visibility,
            temperature=req.temperature,
            wind_speed=req.wind_speed,
            junction=req.junction,
            traffic_signal=req.traffic_signal,
            crossing=req.crossing,
            reported_vehicles=req.reported_vehicles
        )
        res["incident_call_id"] = req.incident_call_id
        res["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/live-weather")
async def live_weather(city: str):
    coords = {
        "atlanta": (33.7490, -84.3880, "Atlanta, Georgia"),
        "chicago": (41.8781, -87.6298, "Chicago, Illinois"),
        "miami": (25.7617, -80.1918, "Miami, Florida"),
        "seattle": (47.6062, -122.3321, "Seattle, Washington"),
        "denver": (39.7392, -104.9903, "Denver, Colorado"),
        "dallas": (32.7767, -96.7970, "Dallas, Texas"),
        "new york": (40.7128, -74.0060, "New York, NY"),
        "los angeles": (34.0522, -118.2437, "Los Angeles, CA"),
    }
    key = city.strip().lower()
    if key not in coords:
        raise HTTPException(status_code=404, detail="City not found in emergency dispatch catalog")

    lat, lon, full_name = coords[key]
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,weather_code,wind_speed_10m,visibility"
           f"&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto")

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(url)
            data = resp.json()
            curr = data.get("current", {})
            return {
                "city": full_name,
                "temperature": round(curr.get("temperature_2m", 65)),
                "weather": map_weather_code(curr.get("weather_code", 0)),
                "wind_speed": round(curr.get("wind_speed_10m", 5)),
                "visibility": round(curr.get("visibility", 16000) / 1609.34, 1),
            }
    except Exception:
        # Fallback simulation
        return {
            "city": full_name + " (Telemetry Fallback)",
            "temperature": 68,
            "weather": "Clear",
            "wind_speed": 6,
            "visibility": 10.0,
        }
