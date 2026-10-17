"""Fetch polyp-free lower-GI images from the HyperKvasir labeled-images archive without
downloading all 3.9 GB: the zip's central directory is read over HTTP range requests and only
the selected members are transferred.

    python scripts/fetch_negatives.py --per-class 300
"""

from __future__ import annotations

import argparse
import io
import random
import re
import subprocess
import zipfile
from pathlib import Path

URL = "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip"
CLASSES = {  # lower-GI classes without polyps
    "quality-of-mucosal-views/bbps-2-3",
    "quality-of-mucosal-views/bbps-0-1",
    "anatomical-landmarks/cecum",
    "anatomical-landmarks/retroflex-rectum",
}


class HttpRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP range requests, enough for ``zipfile``.

    Requests go through curl: the server sends an incomplete TLS chain that curl (via the OS
    trust store) accepts but Python's ssl module rejects, and certificate checks stay on.
    """

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = subprocess.run(["curl", "-sfIL", url], check=True, capture_output=True, text=True)
        self.size = int(re.findall(r"(?im)^content-length:\s*(\d+)", head.stdout)[-1])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        cmd = ["curl", "-sfL", "--retry", "3", "-r", f"{self.pos}-{end - 1}", self.url]
        data = subprocess.run(cmd, check=True, capture_output=True).stdout
        self.pos += len(data)
        return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--out", default="data/raw/hyperkvasir-negatives")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    zf = zipfile.ZipFile(HttpRangeFile(URL))
    by_class: dict[str, list[zipfile.ZipInfo]] = {}
    for info in zf.infolist():
        parts = info.filename.split("/")
        if info.filename.endswith(".jpg") and "lower-gi-tract" in parts:
            cls = "/".join(parts[parts.index("lower-gi-tract") + 1 : -1])
            if cls in CLASSES:
                by_class.setdefault(cls, []).append(info)
    rng = random.Random(args.seed)
    for cls, infos in sorted(by_class.items()):
        infos = sorted(infos, key=lambda i: i.filename)
        picked = rng.sample(infos, min(args.per_class, len(infos)))
        dst_dir = Path(args.out) / cls.split("/")[-1]
        dst_dir.mkdir(parents=True, exist_ok=True)
        for info in picked:
            dst = dst_dir / Path(info.filename).name
            if not dst.exists():
                dst.write_bytes(zf.read(info))
        print(f"{cls}: {len(picked)} of {len(infos)}")


if __name__ == "__main__":
    main()
