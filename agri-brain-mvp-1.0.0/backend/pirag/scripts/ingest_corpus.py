
import os, sys, glob, requests
API = os.environ.get("PIRAG_API","http://127.0.0.1:8000/rag/ingest")
def load_texts(folder):
    for p in glob.glob(os.path.join(folder, "**", "*.*"), recursive=True):
        if p.lower().endswith((".txt",".md",".log",".csv",".json")):
            try:
                with open(p,"r",encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                yield {"id": os.path.relpath(p, folder), "text": txt, "metadata": {"path": p}}
            except Exception:
                pass
def main():
    if len(sys.argv) < 2:
        print("Provide folder of text files."); sys.exit(1)
    folder = sys.argv[1]
    batch = list(load_texts(folder))
    print(f"Ingesting {len(batch)} docs → {API}")
    r = requests.post(API, json=batch)
    print(r.status_code, r.text)
if __name__ == "__main__":
    main()
