"""Download Kvasir-SEG, CVC-ClinicDB and a subset of HyperKvasir colonoscopy videos into data/raw.

    python scripts/download_data.py            # everything (~750 MB)
    python scripts/download_data.py --skip-videos

All three datasets are licensed for research/education only; read their terms before use.
CVC-ClinicDB ships as .rar: extraction uses `bsdtar` (macOS built-in; `apt install libarchive-tools`).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

RAW = Path("data/raw")
KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
CVC_URL = "https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=1"
OSF = "https://api.osf.io/v2/nodes/mh9sj/files/osfstorage"  # HyperKvasir
OSF_LABELS = "https://osf.io/download/5ebc56b9ec5a9800a023fded/"
OSF_FOLDERS = {"polyps": "pathological-findings/polyps", "normal": "quality-of-mucosal-view/BBPS-2-3"}


def fetch(url: str, dst: Path) -> Path:
    if dst.exists():
        print(f"exists  {dst}")
        return dst
    print(f"get     {url}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    return dst


def osf_list(folder_id: str) -> list[dict]:
    url, out = f"{OSF}/{folder_id}/", []
    while url:
        page = json.load(urllib.request.urlopen(url))
        out += page["data"]
        url = page["links"].get("next")
    return out


def osf_find(path: str) -> str:
    """Resolve a folder path inside HyperKvasir's labeled-videos/lower-gi-tract to an OSF id."""
    node = next(x for x in osf_list("") if x["attributes"]["name"] == "labeled-videos")["id"]
    node = next(x for x in osf_list(node) if x["attributes"]["name"] == "lower-gi-tract")["id"]
    for part in path.split("/"):
        node = next(x for x in osf_list(node) if x["attributes"]["name"] == part)["id"]
    return node


def videos(n_polyp: int) -> None:
    labels_csv = fetch(OSF_LABELS, RAW / "hyperkvasir-video-labels.csv")
    labels = {r["Video file"]: r for r in csv.DictReader(open(labels_csv))}
    for sub, path in OSF_FOLDERS.items():
        files = sorted(osf_list(osf_find(path)), key=lambda x: x["attributes"]["size"])
        picked = []
        for f in files:
            stem = f["attributes"]["name"].rsplit(".", 1)[0]
            second = labels.get(stem, {}).get("Finding 2", "").strip().lower()
            if sub == "normal" and second == "normal mucosa":
                picked.append(f)  # good bowel prep + explicitly normal mucosa: no polyp expected
            elif sub == "polyps" and second == "" and len(picked) < n_polyp:
                picked.append(f)
        for f in picked:
            fetch(f"https://osf.io/download/{f['id']}/",
                  RAW / "hyperkvasir-videos" / sub / f["attributes"]["name"])  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--skip-videos", action="store_true")
    ap.add_argument("--n-polyp-videos", type=int, default=5)
    args = ap.parse_args()

    z = fetch(KVASIR_URL, RAW / "kvasir-seg.zip")
    if not (RAW / "Kvasir-SEG").exists():
        zipfile.ZipFile(z).extractall(RAW)

    rar = fetch(CVC_URL, RAW / "cvc.rar")
    if not (RAW / "cvc" / "CVC-ClinicDB").exists():
        if not shutil.which("bsdtar"):
            raise SystemExit("bsdtar is required to extract CVC-ClinicDB (.rar)")
        (RAW / "cvc").mkdir(exist_ok=True)
        subprocess.run(["bsdtar", "-xf", str(rar), "-C", str(RAW / "cvc")], check=True)

    if not args.skip_videos:
        videos(args.n_polyp_videos)
    print("done")


if __name__ == "__main__":
    main()
