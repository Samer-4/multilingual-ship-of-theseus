"""Finished MADLAD XSum pilot: the remainder export's integrity, the one invalid chain, and the final folder.
CPU only; skipped when the downloads are not present."""
import importlib.util
import json
import sys

import pandas as pd
import pytest

from sot_mt import checks, paths, runner, schema, session
from sot_mt.files import load_yaml, sha256_file

SAM = paths.SAM_ROOT
REM = SAM / "data" / "generated" / "madlad_pilot" / "remainder" / "xsum" / "madlad-3b"
FINAL = SAM / "data" / "generated" / "madlad_final"
ITEM = "xsum|Eleuther-AI|xsum-4729"
CONTAINS = ["data/generated/madlad_pilot/b01/xsum/madlad-3b", "data/generated/smoke5_v2/xsum/madlad-3b"]


def _script(name):
    sys.path.insert(0, str(SAM / "scripts"))
    spec = importlib.util.spec_from_file_location(name, SAM / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def merged():
    if not (REM / runner.MERGED).exists() or not (FINAL / "xsum" / "madlad-3b" / "FINAL.json").exists():
        pytest.skip("remainder download or final folder unavailable")
    return pd.read_parquet(REM / runner.MERGED)


def test_remainder_integrity_and_counts(merged):
    r = _script("verify_madlad_pilot").verify(REM, CONTAINS)
    assert (r["records"], r["stage2_records"], r["new_records"]) == (1186, 400, 786)
    assert (r["eligible_items"], r["complete_items"]) == (197, 196)
    assert all(v == {"records": n, "missing": 0, "changed": 0}
               for v, n in zip(r["earlier_exports_kept"].values(), (516, 420)))
    assert r["invalid_new"] == [{"item_id": ITEM, "output_stage": "T2", "validation_reason": "engine:generation_cap"}]
    assert merged.record_id.is_unique and set(merged.source_label) and merged.source_label.nunique() == 7
    assert schema.chain_violations(merged, session.load_inputs()[0]) == []


def test_the_invalid_t2_is_a_decoding_loop_and_blocks_its_successors(merged):
    ch = merged[merged.item_id == ITEM].set_index("output_stage")
    assert set(ch.index) == {"S1", "T1", "S2", "T2"}                  # S3, T3 never generated from it
    t2 = ch.loc["T2"]
    meta = json.loads(t2.generation_meta)
    assert meta["finish"] == "length" and meta["segment_hit_eos"] == [False, True, True, True]
    assert meta["segment_output_tokens"][0] == 513 > 2 * meta["segmentation"]["segment_input_tokens"][0]
    assert t2.output_text.count("the fourth segment North Britain Wind Park – 100 MW of wind power") == 19   # + a truncated 20th
    # whole-text repetition share is diluted by the 3 clean segments, so only the cap marks it: still blocking
    rep = load_yaml("experiment.yaml")["validation"]["repetition"]
    assert not checks.repetition(t2.output_text, rep["trigram_count_min"], rep["trigram_share_min"])
    assert "generation_cap" in checks.BLOCKING


def test_final_folder_is_the_verified_export(merged):
    gen = FINAL / "xsum" / "madlad-3b"
    info = json.loads((gen / "FINAL.json").read_text())
    assert sha256_file(gen / runner.MERGED) == sha256_file(REM / runner.MERGED) == info["generations_merged_sha256"]
    chains = pd.read_csv(gen / "chains.csv")
    assert len(chains) == 200 and chains.complete.sum() == 196 == info["complete_items"]
    assert chains.groupby("source_label").size().to_dict() == {k: v["items"] for k, v in info["source_labels"].items()}
    m = session.load_inputs()[0].set_index("item_id")
    assert (chains.group_id.values == m.loc[chains.item_id, "group_id"].values).all()
    audit = json.loads((FINAL / "audit" / "invalid_T2_xsum_Eleuther-AI_xsum-4729.json").read_text())
    assert audit["output_text"] == merged.set_index(["item_id", "output_stage"]).loc[(ITEM, "T2"), "output_text"]


def test_finalize_never_overwrites(merged):
    with pytest.raises(SystemExit, match="never overwritten"):
        _script("finalize_madlad").finalize()
