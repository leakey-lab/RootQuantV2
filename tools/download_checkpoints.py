#!/usr/bin/env python3
"""Download released RootQuant-V2 checkpoints from Google Drive.

The trained weights are hosted on Google Drive rather than in git. This script
reads ``checkpoints.json``, fetches the files you ask for, verifies their
SHA-256, and drops them where the training / inference code expects them:

    RootQuantV2/runs/checkpoints/<name>/best.pt

Usage (from the repository root)
--------------------------------
    python tools/download_checkpoints.py --list
    python tools/download_checkpoints.py                     # headline model only
    python tools/download_checkpoints.py --all
    python tools/download_checkpoints.py rootquant-v2-weights rootquant-v2-soybean-to-maize-fft
    python tools/download_checkpoints.py --verify-only       # re-check local files

If a checkpoint has no ``drive_file_id`` in the manifest, pass the share link
directly:

    python tools/download_checkpoints.py rootquant-v2-weights \\
        --url "https://drive.google.com/file/d/XXXXXXXXXXXX/view?usp=sharing"

Large Google Drive files are served behind a virus-scan interstitial. ``gdown``
handles that reliably, so install it if a download fails:  pip install gdown
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "checkpoints.json")
DEFAULT_NAME = "rootquant-v2-weights"


# ── manifest ────────────────────────────────────────────────────────────────

def load_manifest() -> Dict[str, Any]:
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        return json.load(fh)


def entry_for(manifest: Dict[str, Any], name: str) -> Dict[str, Any]:
    for entry in manifest["checkpoints"]:
        if entry["name"] == name:
            return entry
    known = ", ".join(e["name"] for e in manifest["checkpoints"])
    raise SystemExit(f"unknown checkpoint {name!r}. Known checkpoints:\n  {known}")


def target_path(manifest: Dict[str, Any], entry: Dict[str, Any]) -> str:
    root = manifest.get("install_root", "RootQuantV2/runs/checkpoints")
    return os.path.join(REPO_ROOT, root, entry["name"], "best.pt")


# ── hashing ─────────────────────────────────────────────────────────────────

def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: str, entry: Dict[str, Any]) -> bool:
    if not os.path.isfile(path):
        return False
    size = os.path.getsize(path)
    if entry.get("bytes") and size != entry["bytes"]:
        print(f"  size mismatch: {size} bytes on disk, {entry['bytes']} expected")
        return False
    got = sha256_of(path)
    if got != entry["sha256"]:
        print(f"  sha256 mismatch:\n    got      {got}\n    expected {entry['sha256']}")
        return False
    return True


# ── Google Drive ────────────────────────────────────────────────────────────

def file_id_from_url(url: str) -> Optional[str]:
    for pattern in (r"/file/d/([A-Za-z0-9_-]{10,})", r"[?&]id=([A-Za-z0-9_-]{10,})"):
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_with_gdown(file_id: str, dest: str) -> bool:
    """Preferred path: gdown deals with the large-file confirmation page."""
    exe = shutil.which("gdown")
    # Positional URL: gdown >= 5 removed the --id flag; the URL form works in all versions.
    cmd = ([exe] if exe else [sys.executable, "-m", "gdown"]) + [
        f"https://drive.google.com/uc?id={file_id}", "--output", dest,
    ]
    try:
        subprocess.run(cmd, check=True)
        return os.path.isfile(dest) and os.path.getsize(dest) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_with_urllib(file_id: str, dest: str) -> bool:
    """Stdlib fallback. Works for small files and often for large ones."""
    import urllib.parse
    import urllib.request
    from http.cookiejar import CookieJar

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar())
    )
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    base = "https://drive.usercontent.google.com/download"
    url = f"{base}?{urllib.parse.urlencode({'id': file_id, 'export': 'download'})}"

    try:
        response = opener.open(url, timeout=60)
        head = response.read(8192)
        # An HTML body means Drive returned the virus-scan interstitial; retry
        # with the confirm token it embeds.
        if head.lstrip()[:1] == b"<":
            page = (head + response.read()).decode("utf-8", "replace")
            token = re.search(r'name="confirm"\s+value="([^"]+)"', page)
            uuid = re.search(r'name="uuid"\s+value="([^"]+)"', page)
            params = {"id": file_id, "export": "download",
                      "confirm": token.group(1) if token else "t"}
            if uuid:
                params["uuid"] = uuid.group(1)
            response = opener.open(f"{base}?{urllib.parse.urlencode(params)}", timeout=60)
            head = response.read(8192)
            if head.lstrip()[:1] == b"<":
                return False

        with open(dest, "wb") as fh:
            fh.write(head)
            shutil.copyfileobj(response, fh, length=1 << 20)
        return os.path.getsize(dest) > 0
    except Exception as exc:                                    # noqa: BLE001
        print(f"  urllib download failed: {exc}")
        return False


def fetch(entry: Dict[str, Any], dest: str, url_override: Optional[str]) -> bool:
    file_id = (
        file_id_from_url(url_override) if url_override else entry.get("drive_file_id")
    )
    if not file_id:
        print(
            f"  no drive_file_id for {entry['name']}.\n"
            f"  Fill it into checkpoints.json, or pass --url <share link>."
        )
        return False

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        delete=False, dir=os.path.dirname(dest), suffix=".part"
    )
    tmp.close()
    try:
        ok = download_with_gdown(file_id, tmp.name) or download_with_urllib(file_id, tmp.name)
        if not ok:
            print("  download failed. Try:  pip install gdown")
            return False
        os.replace(tmp.name, dest)
        return True
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ── CLI ─────────────────────────────────────────────────────────────────────

def cmd_list(manifest: Dict[str, Any]) -> None:
    folder = manifest.get("drive_folder_url", "")
    print(f"Drive folder: {folder}\n")
    for entry in manifest["checkpoints"]:
        mb = entry["bytes"] / (1 << 20)
        tags = ",".join(entry.get("tags", []))
        marker = "*" if "headline" in entry.get("tags", []) else " "
        local = target_path(manifest, entry)
        state = "on disk" if os.path.isfile(local) else "-"
        print(f"{marker} {entry['name']:<40} {mb:8.1f} MB  [{tags}]  {state}")
        print(f"    {entry['description']}")
    print("\n* = headline model (the default download)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("names", nargs="*", help="checkpoint name(s) to download (default: headline).")
    ap.add_argument("--all", action="store_true", help="Download every non-optional entry.")
    ap.add_argument("--include-optional", action="store_true",
                    help="With --all, also fetch entries flagged optional (the 1.2 GB baseline).")
    ap.add_argument("--list", action="store_true", help="List available checkpoints and exit.")
    ap.add_argument("--url", default=None,
                    help="Google Drive share link for a single checkpoint (overrides the manifest id).")
    ap.add_argument("--verify-only", action="store_true",
                    help="Check SHA-256 of already-downloaded files; download nothing.")
    ap.add_argument("--force", action="store_true", help="Re-download even if the file verifies.")
    args = ap.parse_args()

    manifest = load_manifest()

    if args.list:
        cmd_list(manifest)
        return

    if args.all:
        wanted: List[str] = [
            e["name"] for e in manifest["checkpoints"]
            if args.include_optional or not e.get("optional")
        ]
    else:
        wanted = args.names or [DEFAULT_NAME]

    if args.url and len(wanted) != 1:
        raise SystemExit("--url applies to exactly one checkpoint name.")

    failures = 0
    for name in wanted:
        entry = entry_for(manifest, name)
        dest = target_path(manifest, entry)
        print(f"\n{name}  ->  {os.path.relpath(dest, REPO_ROOT)}")

        if os.path.isfile(dest) and not args.force:
            print("  already present, verifying ...")
            if verify(dest, entry):
                print("  OK (sha256 matches)")
                continue
            if args.verify_only:
                failures += 1
                continue
            print("  re-downloading")

        if args.verify_only:
            print("  missing")
            failures += 1
            continue

        if not fetch(entry, dest, args.url):
            failures += 1
            continue

        print("  verifying ...")
        if verify(dest, entry):
            print("  OK (sha256 matches)")
        else:
            print("  CORRUPT — delete it and retry")
            failures += 1

    print()
    if failures:
        raise SystemExit(f"{failures} checkpoint(s) failed.")
    print("done.")


if __name__ == "__main__":
    main()
