"""Publish the verified standalone replay as the GitHub Pages entrypoint."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
offline = root / "PlayVision_Offline.html"
assert offline.exists(), "Run src/export_demo.py before building Pages."
(root / "index.html").write_bytes(offline.read_bytes())
