1) Open the folder
2) Start the Backend (FastAPI, port 8100)
Open PowerShell as Administrator, then:
cd C:\AgriBrain\mvp\backend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# Activate your venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
# (matplotlib offscreen safety)
$env:MPLBACKEND="Agg"
# Start the API
python -m uvicorn src.app:API --reload --port 8100   # (works)
python -m uvicorn src.app:API --host 127.0.0.1 --port 8111  (workiig)
# python -m uvicorn src.app:app --reload --port 8100  # (won’t work unless you rename API→app)
✅ Checkpoints
•	Open http://127.0.0.1:8100/health → should return {"ok": true}
•	(Optional) http://127.0.0.1:8100/docs
o	Try POST /case/load then GET /kpis
o	Add-on endpoints now appear too: /governance/*, /audit/logs, /scenarios, /decide
________________________________________
3) Start the Dashboard (React/Tailwind, port 5173)
In another PowerShell window:
cd C:\AgriBrain\mvp\frontend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
npm install
npm run dev
If it doesn’t print a local URL, run with host:
npm run dev -- --host
Open the printed URL (usually http://127.0.0.1:5173).
If you mounted the route, open http://127.0.0.1:5173/admin.
What you’ll see (single site with tabs / route):
•	Operations: KPIs (waste baseline→AGRI, avg SLCA, anomaly count, avg temp)
•	Quality: IoT streams (Temp/RH/Ambient/Shock) + Spoilage curve (shelf life)
•	Decisions: live Decision Memos (route, SLCA, carbon, reasoning) + PDF download
•	Admin (new):
o	Policy (edit thresholds, factors, distances, SLCA weights)
o	Blockchain (RPC/ChainID/private key/addresses)
o	Scenarios (baseline, heatwave, reverse logistics, cyber outage, adaptive pricing)
o	Audit (decision log incl. tx hash when blockchain is on)
o	QuickDecision (hit /decide once and log the memo)
________________________________________
4) (Optional) Turn on Blockchain / DAO (Hardhat)
To get non-zero on-chain tx hashes in Audit:
cd C:\AgriBrain\mvp\contracts\hardhat
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
npm install
npx hardhat node
In a new PowerShell (same folder):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
npx hardhat run scripts/deploy.js --network localhost
# copy the printed contract address
In Admin → Blockchain panel:
•	RPC: http://127.0.0.1:8545
•	Chain ID: 31337
•	Private Key: optional for demo (leave blank for dry-run tx "0x0")
•	Addresses (JSON): paste something like
•	{ "AGRIValidator": "0xYourDeployedAddress" }
Note: the add-on expects AGRIValidator (a simple “recordDecision” contract).
If your deploy script prints multiple addresses, just paste the one that records decisions.
________________________________________
5) Start Autonomous Agents (to generate live decisions)
cd C:\AgriBrain\mvp\agents
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# Activate your venv
.venv\Scripts\Activate.ps1
pip install -U pip requests pandas numpy pyyaml
.venv\Scripts\python.exe -m pip install -U pip requests pandas numpy pyyaml

$env:API_BASE="http://127.0.0.1:8111"
python runner.py
You’ll see a loop like:
[farm] reroute_to_near_dc slca=0.734 tx=0x0 note=Decision=...
[processor] standard_cold_chain ...
...
These memos stream into the Decisions view, and also appear in Admin → Audit.
If blockchain is configured (RPC + address + private key), you’ll see real tx hashes instead of 0x0.
________________________________________
6) Your 5-minute demo script
1.	Operations: “This is the spinach lane (72h). KPI shows waste down from baseline → AGRI.”
2.	Quality: Point at three temperature excursions (loading, transfer, mini failure). Show shelf-life dips on Spoilage curve.
3.	Decisions: Click Take decision to create a memo. Read the memo (route, SLCA, carbon, reason).
4.	Admin → Policy: Change min_shelf_reroute from 0.70 to 0.65, click Save Policy. Take another decision—note behavior shift.
5.	Admin → Audit & Reports: See new memo at top; if blockchain is on, show tx hash.
6.	Admin → Scenarios: Toggle Heatwave or Reverse Logistics and take decisions again to illustrate resilience/circularity effects.
________________________________________
7) How to control the MVP
•	Scenario control (Admin → Scenarios): pick heatwave, reverse_logistics, cyber_outage, or adaptive_pricing. Decisions adapt immediately.
•	Data control: backend ships with data_spinach.csv (72h @ 15 min). Replace it with your real CSV (same columns: timestamp,tempC,RH,shockG,ambientC,inventory_units,demand_units) and hit /case/load (or refresh the dashboard).
•	On-chain control: with Hardhat running, set RPC/Chain ID/addresses in Admin → Blockchain; decisions will log on-chain (non-zero tx hashes).
________________________________________
8) Common pitfalls & quick fixes
•	FastAPI app variable: python -m uvicorn src.app:API (works). :app won’t unless you rename the variable.
•	PowerShell activation: On Windows, use .venv\Scripts\Activate.ps1 (not source).
•	Ports busy: If 8100 is taken, run the backend on 8111:
python -m uvicorn src.app:API --port 8111
Then set the frontend to that API:
localStorage.setItem('API_BASE','http://127.0.0.1:8111') and refresh.
•	Blank dashboard: Ensure backend is running; visit http://127.0.0.1:8100/health; then reload the site.
•	Admin route not found: Make sure you added the Admin route/tab and copied frontend/src/mvp/* correctly.
________________________________________
9) After the meeting (optional)
•	Swap in a partner CSV → regenerate decisions & PDF.
•	Tune SLCA weights to match their ESG targets (Admin → Policy).
•	Turn on blockchain and share tx hashes in follow-ups.
________________________________________
If you want this guide embedded in your repo as RUNBOOK.md (with your org paths and screenshots), I can generate it now.




import io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from .models.spoilage import compute_spoilage, volatility_flags
from .models.forecast import yield_demand_forecast
from .models.slca import slca_score
from .models.policy import Policy
from .chain.client import ChainClient
from src.routers import compat as _compat
from src.routers import case as _case
from src.routers import decide as _decide
from src.routers import debug as _debug
from src.routers import scenarios as _scenarios
from src.routers import governance

from src.routers import governance as _gov, audit as _audit, scenarios as _scn, decide as _decide

API = FastAPI(title="AGRI BRAIN MVP API")

API.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173","http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API.include_router(_case.router, prefix="/case", tags=["case"])
API.include_router(_decide.router, tags=["decide"])
API.include_router(_compat.router, tags=["compat"])
API.include_router(_debug.router, tags=["debug"])
API.include_router(_gov.router,    prefix="/governance", tags=["governance"])
API.include_router(_audit.router,  prefix="/audit",      tags=["audit"])
API.include_router(_scn.router,    prefix="/scenarios",  tags=["scenarios"])
API.include_router(_decide.router, prefix="/decide",     tags=["decide"])
API.include_router(_scenarios.router, prefix="/scenarios", tags=["scenarios"])
API.include_router(governance.router, prefix="/governance")

# If your FastAPI variable is `app` instead of `API`, just replace `API.` with `app.`

API.include_router(_compat.router, tags=["compat"])   # use app.include_router(...) if your instance is named app

API.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DATA = Path(__file__).parent / "data_spinach.csv"
state: Dict[str, Any] = {"df": None, "policy": Policy(), "chain": {"rpc": None, "addresses": {}, "chain_id": 31337, "private_key": None}}

class ChainConfig(BaseModel):
    rpc: Optional[str] = None
    chain_id: int = 31337
    private_key: Optional[str] = None
    addresses: Dict[str, str] = {}

@API.get("/health")
def health(): return {"ok": True}

@API.post("/case/load")
def case_load():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = compute_spoilage(df)
    df["volatility"] = volatility_flags(df)
    state["df"] = df
    return {"ok": True, "records": len(df)}

@API.get("/kpis")
def kpis():
    if state["df"] is None: case_load()
    df = state["df"]
    waste_baseline = float((df["shelf_left"]<0.0).sum()/len(df))
    waste_agri = float((df["shelf_left"]<0.0).rolling(4).max().fillna(0).mean()*0.6)
    return {
        "records": len(df),
        "avg_tempC": float(df["tempC"].mean()),
        "anomaly_points": int((df["volatility"]=="anomaly").sum()),
        "waste_rate_baseline": waste_baseline,
        "waste_rate_agri": waste_agri
    }

@API.get("/telemetry")
def telemetry():
    if state["df"] is None: case_load()
    df = state["df"]
    return {
        "timestamp": df["timestamp"].astype(str).tolist(),
        "tempC": df["tempC"].tolist(),
        "RH": df["RH"].tolist(),
        "ambientC": df["ambientC"].tolist(),
        "shockG": df["shockG"].tolist(),
        "inventory_units": df["inventory_units"].tolist(),
        "demand_units": df["demand_units"].tolist(),
    }

@API.get("/predictions")
def predictions():
    if state["df"] is None: case_load()
    df = state["df"]
    yf = yield_demand_forecast(df, horizon=24)
    return {
        "timestamp": df["timestamp"].astype(str).tolist(),
        "shelf_left": df["shelf_left"].round(4).tolist(),
        "volatility": df["volatility"].tolist(),
        "yield_forecast_24h": yf
    }

@API.get("/policy")
def get_policy():
    return state["policy"].model_dump()

@API.post("/policy")
def set_policy(p: Policy):
    state["policy"] = p
    return {"ok": True, "policy": p.model_dump()}

class DecideIn(BaseModel):
    agent_id: str
    role: str

@API.post("/decide")
def decide(d: DecideIn):
    if state["df"] is None: case_load()
    p = state["policy"]; df = state["df"]
    row = df.iloc[-1]; shelf = float(row["shelf_left"]); vol = str(row["volatility"])
    if shelf < p.min_shelf_expedite:
        action="expedite_to_retail"; km=p.km_expedited; price=p.msrp*0.92
    elif shelf < p.min_shelf_reroute or vol=="anomaly":
        action="reroute_to_near_dc"; km=p.km_farm_to_dc*0.6; price=p.msrp*0.95
    else:
        action="standard_cold_chain"; km=p.km_farm_to_dc+p.km_dc_to_retail; price=p.msrp
    carbon = km * p.carbon_per_km
    slca = slca_score(carbon)
    memo = {
        "time": datetime.utcnow().isoformat(), "agent": d.agent_id, "role": d.role,
        "decision": action, "shelf_left": round(shelf,3), "volatility": vol, "km": km,
        "carbon_kg": round(carbon,2), "unit_price": round(price,2), "slca": round(slca,3),
        "note": f"Decision={action} because shelf_left={shelf:.2f} and volatility={vol}."
    }
    chain = ChainClient(**state["chain"]); tx = chain.log_decision(d.agent_id, action, int(slca*1e6), "")
    memo["tx"] = tx
    state.setdefault("log", []).append(memo)
    return {"ok": True, "memo": memo}

@API.get("/decisions")
def list_decisions():
    return {"decisions": state.get("log", [])[-500:]}

@API.post("/chain/config")
def chain_config(cfg: ChainConfig):
    state["chain"] = cfg.model_dump()
    return {"ok": True, "chain": state["chain"]}

@API.get("/report/pdf")
def report_pdf():
    if state["df"] is None: case_load()
    kp = kpis(); last = (state.get("log") or [{}])[-1] if state.get("log") else {}
    buf = io.BytesIO(); from reportlab.lib.pagesizes import A4; from reportlab.lib.units import mm
    c = canvas.Canvas(buf, pagesize=A4); w,h=A4; y=h-20*mm
    c.setFont("Helvetica-Bold", 16); c.drawString(20*mm, y, "AGRI BRAIN Spinach — Decision Memo"); y-=10*mm
    c.setFont("Helvetica", 10)
    for k in ("records","avg_tempC","anomaly_points","waste_rate_baseline","waste_rate_agri"):
        c.drawString(20*mm, y, f"{k}: {kp.get(k)}"); y-=6*mm
    y-=4*mm; c.setFont("Helvetica-Bold",12); c.drawString(20*mm, y, "Last Decision"); y-=7*mm; c.setFont("Helvetica",10)
    for k in ("time","agent","role","decision","shelf_left","volatility","km","carbon_kg","unit_price","slca","tx","note"):
        c.drawString(20*mm, y, f"{k}: {last.get(k,'')}"); y-=6*mm
        if y<20*mm: c.showPage(); y=h-20*mm
    c.showPage(); c.save()
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="application/pdf")

1) What the system is
•	A tiny digital twin of a spinach cold-chain (farm → processor → distributor → retail).
•	It ingests a demo telemetry stream (temperature, humidity, shocks, inventory/demand).
•	It computes a simple shelf-left signal and flags volatility/anomalies.
•	A lightweight policy converts the latest state into a logistics action (standard / reroute / expedite).
•	Every action is audited (local log + optional blockchain tx), and a one-page PDF memo is generated.
•	Optional agent workers can run in the background and take decisions automatically.
•	A WebSocket pushes live events (new decisions, chain head) to the UI.
2) The core loop (what happens when you click “Take decision”)
1.	Read latest state
The backend looks at the last row of the dataset (already computed shelf_left and volatility).
2.	Compare to policy thresholds
o	If shelf_left < min_shelf_expedite ⇒ expedite_to_retail
o	Else if shelf_left < min_shelf_reroute or volatility == "anomaly" ⇒ reroute_to_near_dc
o	Else ⇒ standard_cold_chain
3.	Compute impact
o	Distance traveled (km) depends on the action.
o	Carbon = km × carbon_per_km.
o	SLCA score is derived from carbon (lower carbon → better score).
o	Price can be nudged (e.g., 0.92× MSRP when expediting).
4.	Build the memo
A JSON memo: time, agent/role, decision, shelf_left, volatility, km, carbon_kg, unit_price, slca, note.
5.	Write the audit log (always)
The memo is appended to in-memory state["log"].
6.	Try to write on-chain (optional)
If chain config is set, the DecisionLogger is called; you get a tx/tx_hash.
7.	Broadcast live
The memo is emitted over WebSocket so the UI can show it instantly.
8.	Return to UI
The UI displays a “Decision Memo” card, the Audit table updates, and you can download the PDF.
9.	PDF
/report/pdf renders top KPIs + the last decision section.
3) What the pages show
•	Operations (Ops)
o	KPI tiles: records, average temp, anomaly count, waste rates (baseline vs Agri-Brain).
o	Live charts: telemetry (temp/ambient/shock) and a preview of shelf_left.
•	Admin → Policy
o	Sliders/inputs for thresholds and carbon factors.
•	Admin → Scenarios
o	Buttons to apply research scenarios (e.g., heatwave), plus a reset.
•	Admin → QuickDecision
o	Pick a role (farm/processor/distributor/retail) and trigger the decision. (All roles use the same policy now; the role tags the memo.)
•	Admin → Audit
o	Log of decisions and a “Download Decision Memo (PDF)” button.
•	Admin → Blockchain
o	Shows RPC reachability, chain id, current block, contract presence, and last tx receipt.
o	With the WebSocket, you also see “Block # (live)” bump as the chain advances.
4) How scenarios change the outcome
Scenarios alter the underlying state (directly or through recomputed KPIs), which changes the decision at step 2:
•	Climate shock (heatwave) → temps rise → shelf_left falls faster → your action slides from standard → reroute → expedite.
•	Cyber outage → may force certain routes offline (in extended versions) → policy biases to alternative routes.
•	Reverse logistics / Adaptive pricing → shift supply/demand or price modifiers (in the MVP, the visible effect is via shelf_left/volatility, but you can easily add price/yield hooks).
In short: the inputs move, the policy re-evaluates, and you see different actions and different carbon/SLCA in the memo and PDF.
5) The “agentic” part (how it shows your strength)
•	You can run a tiny agent process (agents/runner.py) that periodically calls /decide for different roles.
•	Each agent decision appears live in the UI via WebSocket (no page refresh).
•	The Blockchain panel updates last tx status and block # (live) automatically.
•	This demonstrates an autonomous, closed-loop system: telemetry → policy → action → immutable audit → live status.
6) What’s novel / demo talking points
•	Self-auditing supply chain: every automated decision is logged and optionally anchored on-chain → tamper-evident trace.
•	Policy-driven + agentic: a policy can be hand-tuned today, learned tomorrow; agents execute continuously.
•	Real-time transparency: live memos, live chain head, instant PDF memos for compliance and hand-offs.
•	Scenario-aware: stress-test and show how your system adapts (heatwaves, outages, gluts).
7) A clean 5-minute demo script
1.	Start backend and frontend. Open Ops → show base KPIs & charts.
2.	Go to Admin → QuickDecision, pick farm, click Take decision → show memo card, Audit row, and PDF.
3.	Go to Admin → Scenarios, run Climate shock (intensity ~1.3). Return to Ops → avg temp up.
4.	Back to QuickDecision, click Take decision again → action likely switches (reroute/expedite). Show carbon/SLCA differences.
5.	Admin → Blockchain: show chain id, block #; copy the tx hash from the latest audit row; point out “success”.
6.	(Optional) Run python agents/runner.py → watch memos appear live; block # ticks; PDF updates with the last decision.
8) If something looks “off”
•	PDF header shows zeros → call /case/load once (or click “Load spinach demo data” in Scenarios).
•	Two decision popups → you fixed this by preventing a double POST; keep your data-skip-global-take on the button.
•	422 errors from agents → ensure they POST { "agent_id": "...", "role": "..." }.


PowerShell is tripping you up here—on Windows, curl is an alias for Invoke-WebRequest, so the -H/-d flags don’t work. Use one of these Windows-friendly options:
Option A — PowerShell-native (recommended)
# LIST TOOLS (GET)
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8111/mcp/tools

# CALL CALCULATOR (POST)
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8111/mcp/call `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"name":"calculator","args":{"expr":"3*(4+5)"}}'

# UNIT CONVERT (POST)
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8111/mcp/call `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"name":"convert_units","args":{"value":10,"from_unit":"kPa","to_unit":"bar"}}'
RAG endpoints
# INGEST (POST)
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8111/rag/ingest `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '[{"id":"doc1","text":"Spinach shelf life at 4 C is 10 days.","metadata":{"src":"SOP"}}]'

# ASK (POST)
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8111/rag/ask `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"question":"What is spinach shelf life at 4 C? Convert 10 kPa to bar.","k":4}'
Option B — Force real curl on Windows
Use curl.exe (not the PowerShell alias) and keep the classic flags:
curl.exe -X POST http://127.0.0.1:8111/mcp/call ^
  -H "Content-Type: application/json" ^
  -d "{\"name\":\"calculator\",\"args\":{\"expr\":\"3*(4+5)\"}}"
Option C — PowerShell “stop parsing” trick
curl --% -X POST http://127.0.0.1:8111/mcp/call -H "Content-Type: application/json" -d '{"name":"calculator","args":{"expr":"3*(4+5)"}}'
Quick success checks
•	GET /mcp/tools returns a list including calculator, convert_units, simulate, policy_check.
•	POST /mcp/call with calculator returns {"ok":true,"result":19.2} for 3.2*(10-4) (or 27 for 3*(4+5)).
•	POST /rag/ingest returns {"ok":true,"n":1} (or more).
•	POST /rag/ask returns JSON with answer, citations[...sha256], guards_passed, merkle_root.
If you want, I can give you a tiny .ps1 script that runs all four calls and prints friendly pass/fail messages.

