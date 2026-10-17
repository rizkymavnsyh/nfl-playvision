"""Create a lightweight GitHub Pages entrypoint from the offline dashboard template."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
template = (root / "src" / "demo_template.html").read_text(encoding="utf-8")
template = template.replace(
    "<script>__PLOTLY_JS__</script>",
    '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
)
template = template.replace(
    "__DATA_JSON__",
    (root / "assets" / "replay_data.json").read_text(encoding="utf-8"),
)
assert "__PLOTLY_JS__" not in template and "__DATA_JSON__" not in template
(root / "index.html").write_text(template, encoding="utf-8")
