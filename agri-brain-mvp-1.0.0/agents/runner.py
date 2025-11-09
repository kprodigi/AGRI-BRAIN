# agents/runner.py
import os
import time
import random
import requests

API = os.environ.get("API_BASE", "http://127.0.0.1:8111").rstrip("/")
ROLES = ["farm", "processor", "distributor", "retail"]

def take_decision(role: str):
    url = f"{API}/decide"
    payload = {
        "agent_id": f"agent:{role}",   # <-- REQUIRED by your backend
        "role": role                   # <-- REQUIRED by your backend
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        # Show useful info if it fails
        if not r.ok:
            print("error", r.status_code, r.reason, "for", url)
            try:
                print("body:", r.text)
            except Exception:
                pass
            return
        data = r.json()
        memo = data.get("memo", data)
        action = memo.get("decision") or memo.get("action")
        slca   = memo.get("slca") or memo.get("slca_score")
        tx     = memo.get("tx") or memo.get("tx_hash")
        print(f"OK role={role} action={action} SLCA={slca} tx={tx}")
    except requests.RequestException as e:
        print("error", e)

def main():
    print(f"Agents hitting {API}")
    # fire a few decisions, one per second
    for i in range(5):
        role = random.choice(ROLES)
        take_decision(role)
        time.sleep(1.0)

if __name__ == "__main__":
    main()
