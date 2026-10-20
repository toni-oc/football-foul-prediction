import hashlib
import json
from pathlib import Path

def sha256_archivo(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()

def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8-sig"))
