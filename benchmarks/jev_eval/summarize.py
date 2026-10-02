"""Print a markdown table of SetFit results next to Jev / gpt-5.4-mini / gpt-5.6-luna on the same items.

Usage: python summarize.py results/probe_bge-small results/probe_bge-base ...
"""

import json
import sys
from pathlib import Path

rows = []
for d in sys.argv[1:]:
    for f in sorted(Path(d).glob("*.json")):
        r = json.loads(f.read_text())
        b = r.get("baselines", {})
        lat = {k: v.get("p50_ms") for k, v in r.items() if k.startswith("latency_") and isinstance(v, dict)}
        rows.append(
            (
                r["dataset"],
                Path(r.get("model") or r["env"]["args"]["model"]).name,
                r.get("device", "cpu"),
                r["shots"],
                r["n_test"],
                r["accuracy"],
                b.get("jev-latest", {}).get("accuracy"),
                b.get("gpt-5.4-mini", {}).get("accuracy"),
                b.get("gpt-5.6-luna", {}).get("accuracy"),
                r["train_seconds"],
                lat,
            )
        )

pct = lambda x: "–" if x is None else f"{100 * x:.0f}"
print("| dataset | body | device | shots/class | n | SetFit | Jev | gpt-5.4-mini | gpt-5.6-luna | train s | batch-1 p50 ms |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for ds, body, dev, shots, n, acc, jev, mini, luna, ts, lat in sorted(rows):
    lat_s = ", ".join(f"{k.removeprefix('latency_')} {v:.0f}" for k, v in lat.items() if v is not None) or "–"
    print(f"| {ds} | {body} | {dev} | {shots} | {n} | {pct(acc)} | {pct(jev)} | {pct(mini)} | {pct(luna)} | {ts:.0f} | {lat_s} |")
