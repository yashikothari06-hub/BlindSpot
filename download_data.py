"""Download the three UCI datasets into data/raw/. If a download fails, fetch the
files manually from the UCI pages and place them in data/raw/ (see README)."""
import io, urllib.request, zipfile
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
URLS = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "bank": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    "wine": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
}

def extract(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(RAW)
        for n in z.namelist():           # bank zip contains a nested zip
            if n.endswith(".zip"):
                z.extract(n, RAW)
                with zipfile.ZipFile(RAW / n) as inner:
                    inner.extractall(RAW)

for name, url in URLS.items():
    try:
        print("downloading", name)
        extract(urllib.request.urlopen(url, timeout=60).read())
    except Exception as e:
        print(f"  FAILED ({e}). Download {name} manually from UCI into data/raw/")
# UCI archives sometimes put CSVs inside nested folders. Copy the expected files
# to RAW's root so the loaders can find them consistently.
import shutil
expected = {"adult.data", "bank-full.csv", "winequality-red.csv", "winequality-white.csv"}
for p in list(RAW.rglob("*")):
    if p.is_file() and p.name in expected and p.parent != RAW:
        target = RAW / p.name
        if not target.exists():
            shutil.copy2(p, target)
print("files in data/raw:", sorted(p.name for p in RAW.iterdir()))
