"""Download a subset of dermoscopy images from the ISIC public API.

    python scripts/download_isic.py --out data/isic --limit 200

Notes
-----
* The ISIC v2 API serves images + clinical metadata but does **not** expose the
  pixel-level segmentation masks. For supervised training, pair these images
  with the ISIC-2018 Task-1 ground-truth masks
  (https://challenge.isic-archive.com/data/) and arrange them as::

      data/isic/images/ISIC_xxxxxxx.jpg
      data/isic/masks/ISIC_xxxxxxx_segmentation.png

* This downloader is sufficient on its own for inference/demo data.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

API = "https://api.isic-archive.com/api/v2/images/"


def fetch_page(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/isic")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    img_dir = Path(args.out) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = []

    url = f"{API}?limit={min(args.limit, 100)}"
    got = 0
    while url and got < args.limit:
        page = fetch_page(url)
        for item in page.get("results", []):
            if got >= args.limit:
                break
            files = item.get("files", {})
            src = (files.get("full") or files.get("thumbnail_256") or {}).get("url")
            if not src:
                continue
            isic_id = item["isic_id"]
            dst = img_dir / f"{isic_id}.jpg"
            try:
                urllib.request.urlretrieve(src, dst)
            except Exception as exc:  # pragma: no cover
                print(f"skip {isic_id}: {exc}")
                continue
            clin = (item.get("metadata") or {}).get("clinical", {})
            meta_rows.append({"isic_id": isic_id,
                              "diagnosis": clin.get("diagnosis"),
                              "benign_malignant": clin.get("benign_malignant"),
                              "anatom_site": clin.get("anatom_site_general")})
            got += 1
            print(f"[{got}/{args.limit}] {isic_id}")
        url = page.get("next")

    (Path(args.out) / "metadata.json").write_text(json.dumps(meta_rows, indent=2))
    print(f"done: {got} images -> {img_dir}")


if __name__ == "__main__":
    main()
