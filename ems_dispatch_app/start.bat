@echo off
echo Starting EMS & First Responder Dispatch App on Port 8001...
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
pause
