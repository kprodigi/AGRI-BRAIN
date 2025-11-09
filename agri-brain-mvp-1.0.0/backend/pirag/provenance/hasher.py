
import hashlib, json
def hash_artifact(obj):
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()
def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
