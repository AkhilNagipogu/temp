"""
EMS & First Responder Dispatch Predictor
Wraps the core Random Forest severity model with emergency triage logic:
- Dynamic Code Red / Amber / Green classification
- Unit dispatch allocation recommendations (ALS, BLS, Heavy Extrication)
- On-scene extrication & corridor clearance time estimation
- Regional Trauma Center routing delays
"""

import math
from pathlib import Path
import joblib
import pandas as pd
import sklearn

MODEL_PATH = Path(__file__).parent / "model" / "roadsafety_model.pkl"

_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"]
MODEL_NAME = _bundle["model_name"]
COLUMNS = _bundle["columns"]
WEATHER_OPTIONS = _bundle["weather_options"]
MEDIUM_CUTOFF, HIGH_CUTOFF = _bundle["risk_cutoffs"]
CLASSES = [int(c) for c in MODEL.classes_]
STATS = {name: {str(k): float(v) for k, v in table.items()} for name, table in _bundle["stats"].items()}


def predict_triage(weather, lighting, hour, weekday, month,
                   visibility, temperature, wind_speed,
                   junction, traffic_signal, crossing,
                   units_called=None, reported_vehicles=2):
    """Predict crash severity and generate EMS response manifest."""
    if weather not in WEATHER_OPTIONS:
        raise ValueError("Unknown weather: " + str(weather))
    if lighting not in ("Day", "Night"):
        raise ValueError("Lighting must be Day or Night")

    row = {col: 0 for col in COLUMNS}
    row.update({
        "Visibility": visibility, "Temperature": temperature, "Wind_Speed": wind_speed,
        "Hour": hour, "Weekday": weekday, "Month": month,
        "Is_Night": 1 if lighting == "Night" else 0,
        "Junction": int(junction), "Traffic_Signal": int(traffic_signal), "Crossing": int(crossing),
    })
    weather_col = "Weather_" + weather
    if weather_col in row:
        row[weather_col] = 1

    X = pd.DataFrame([row], columns=COLUMNS)
    probs = MODEL.predict_proba(X)[0]
    scores = {str(c): round(float(p), 4) for c, p in zip(CLASSES, probs)}
    risk_score = float(probs[CLASSES.index(3)] + probs[CLASSES.index(4)])
    p4 = float(probs[CLASSES.index(4)])

    # Emergency Triage Classification
    if p4 >= 0.12 or risk_score >= 0.60:
        triage_code = "CODE RED (Priority 1 - Critical Trauma & Extrication)"
        triage_level = "CRITICAL"
        recommended_units = [
            {"type": "ALS Paramedic Unit (Advanced Life Support)", "count": 2, "icon": "fa-truck-medical", "priority": "Urgent"},
            {"type": "Heavy Rescue Extrication Squad", "count": 1, "icon": "fa-screwdriver-wrench", "priority": "Urgent"},
            {"type": "Engine Company (Fire Suppression / Washdown)", "count": 1, "icon": "fa-fire-extinguisher", "priority": "Standard"},
            {"type": "Highway Patrol Cruiser (Corridor Blockade)", "count": 2, "icon": "fa-shield-halved", "priority": "Immediate"},
        ]
        extrication_minutes = round(35 + (p4 * 60) + (10 if junction else 0))
        clearance_minutes = round(75 + (risk_score * 80) + (20 if weather in ("Snow", "Fog") else 0))
        trauma_center_tier = "Level-1 Comprehensive Adult Trauma Center"
        secondary_crash_risk = "HIGH (Corridor flare-pattern & warning matrix required)"
    elif risk_score >= 0.35:
        triage_code = "CODE AMBER (Priority 2 - Serious Injury / Significant Delay)"
        triage_level = "SERIOUS"
        recommended_units = [
            {"type": "BLS Ambulance Unit (Basic Life Support)", "count": 1, "icon": "fa-truck-medical", "priority": "Urgent"},
            {"type": "Quick Response Paramedic (QRV)", "count": 1, "icon": "fa-car-side", "priority": "Standard"},
            {"type": "Traffic Management Truck (Arrow Board)", "count": 1, "icon": "fa-triangle-exclamation", "priority": "Standard"},
        ]
        extrication_minutes = round(15 + (risk_score * 25))
        clearance_minutes = round(40 + (risk_score * 40))
        trauma_center_tier = "Level-2 Regional Trauma Center"
        secondary_crash_risk = "MEDIUM (Active shoulder buffer recommended)"
    else:
        triage_code = "CODE GREEN (Priority 3 - Non-Critical / Minor Traffic Impairment)"
        triage_level = "MINOR"
        recommended_units = [
            {"type": "Community Paramedic / BLS Standby", "count": 1, "icon": "fa-user-nurse", "priority": "Standard"},
            {"type": "DOT Courtesy Service Patrol / Tow", "count": 1, "icon": "fa-truck-pickup", "priority": "Routine"},
        ]
        extrication_minutes = round(5 + (risk_score * 10))
        clearance_minutes = round(20 + (risk_score * 20))
        trauma_center_tier = "Local Community Urgent Care / Walk-in Facility"
        secondary_crash_risk = "LOW (Normal lane resumption expected)"

    # Weather Delay Penalty for transit
    delay_factor = 1.0
    if weather in ("Rain", "Snow"):
        delay_factor += 0.35
    if visibility < 3:
        delay_factor += 0.25
    if lighting == "Night":
        delay_factor += 0.15

    hospitals = [
        {
            "name": "Metro Central Level-1 Trauma Hospital",
            "tier": "Level-1 Comprehensive",
            "distance_miles": 5.4,
            "base_minutes": 10,
            "adjusted_minutes": round(10 * delay_factor, 1),
            "helipad_available": True,
            "icu_status": "Operating (4 beds open)"
        },
        {
            "name": "Mercy Regional Emergency Trauma Center",
            "tier": "Level-2 Regional",
            "distance_miles": 3.8,
            "base_minutes": 7,
            "adjusted_minutes": round(7 * delay_factor, 1),
            "helipad_available": True,
            "icu_status": "Operating (6 beds open)"
        },
        {
            "name": "Highland Valley Community Hospital",
            "tier": "Level-3 Acute Care",
            "distance_miles": 2.2,
            "base_minutes": 5,
            "adjusted_minutes": round(5 * delay_factor, 1),
            "helipad_available": False,
            "icu_status": "Normal Capacity"
        }
    ]

    return {
        "risk_score": round(risk_score, 3),
        "severe_p3_p4": round(risk_score * 100, 1),
        "p4_critical": round(p4 * 100, 1),
        "class_scores": scores,
        "triage_code": triage_code,
        "triage_level": triage_level,
        "recommended_units": recommended_units,
        "extrication_minutes": extrication_minutes,
        "clearance_minutes": clearance_minutes,
        "trauma_center_tier": trauma_center_tier,
        "secondary_crash_risk": secondary_crash_risk,
        "hospitals": hospitals,
        "model_name": MODEL_NAME,
        "cutoffs": {"medium": round(MEDIUM_CUTOFF, 3), "high": round(HIGH_CUTOFF, 3)}
    }
