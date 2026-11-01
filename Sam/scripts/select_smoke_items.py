"""Write the Stage 3B smoke-test item list (IDs and lengths only, no text).

Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/select_smoke_items.py
Writes Sam/data/manifests/stage3b_madlad_smoke5.csv. The rule is in sot_mt/smoke.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import paths, session, smoke  # noqa: E402
from sot_mt.files import load_yaml  # noqa: E402

OUT = paths.SAM_ROOT / "data" / "manifests" / "stage3b_madlad_smoke5.csv"


def main() -> None:
    manifest, base = session.load_inputs()
    exp = load_yaml("experiment.yaml")["experiment_id"]
    sel = smoke.select_smoke_items(base, exp, "madlad-3b", "mt-default")
    s1 = base[(base.translator == "madlad-3b") & (base.output_stage == "S1")].set_index("item_id")
    legacy = s1.legacy_fields.map(json.loads)
    sel["t0_words"] = sel.item_id.map(manifest.set_index("item_id").t0_words).astype(int)
    sel["round1_segments_en_es"] = sel.item_id.map(legacy.map(lambda d: d["chunks_en"]))
    sel["round1_segments_es_en"] = sel.item_id.map(legacy.map(lambda d: d["chunks_es"]))
    smoke.write_ids(sel, OUT)
    print(sel.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
