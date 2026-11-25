# EMS & First Responder Dispatch Console

**Stakeholder Persona**: Emergency Medical Services (EMS) & First Responder Dispatch Commander  
**Port**: 8001  
**Default URL**: http://127.0.0.1:8001/

## Project Overview
This website provides 911 dispatchers and incident response commanders with predictive triage intelligence powered by a machine learning model trained on the US Accidents dataset.

## Core Features
1. **Incident Triage & Resource Sizing**: Evaluates atmospheric and road geometry factors to classify incidents into **CODE RED** (Critical Trauma / Extrication), **CODE AMBER** (Serious Injury), or **CODE GREEN** (Minor / Property Damage), recommending appropriate emergency units (ALS Paramedics, Heavy Rescue, Engine, Patrol).
2. **Corridor Delay & Extrication Timers**: Predicts on-scene extrication duration, corridor clearance time, and secondary crash prevention perimeters.
3. **Trauma Hospital Routing**: Regional Level-1 and Level-2 trauma center status with weather-adjusted ambulance transit times and ICU bed availability.
4. **Shift Pre-Staging Heatmap**: 24-hour casualty probability curves to pre-position ambulances along major highway interchanges.
5. **Printable 911 Dispatch Brief**: Generates official CAD incident manifest summaries.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
   ```
   Or double-click `start.bat`.
3. Open your browser at `http://127.0.0.1:8001/`.
