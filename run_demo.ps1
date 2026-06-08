$ErrorActionPreference = "Stop"

if (!(Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\src\drone_delivery_sim.py

Write-Host ""
Write-Host "Done. See outputs\final_trajectory.png, outputs\drone_delivery_demo.gif, outputs\drone_delivery_demo.mp4, and outputs\summary.json"
