import pandas as pd
import pytest

from sot_mt import manifest
from sot_mt.files import ImmutableFileError, write_immutable_parquet


def synthetic_raw():
    rows = []
    for key in ["k2", "k1", "k10"]:
        for src in ["B", "A", "C"]:
            rows.append({"source": src, "key": key, "text": f"{src} {key}", "version_name": "original"})
            rows.append({"source": src, "key": key, "text": f"para {src} {key}", "version_name": "chatgpt"})
    rows.append({"source": "A", "key": "k1", "text": "dup", "version_name": "original"})   # duplicate (key, source)
    rows.append({"source": "D", "key": "k1", "text": None, "version_name": "original"})    # missing text
    return pd.DataFrame(rows).rename_axis("source_row_index").reset_index()


def test_round_robin_matches_worker_order():
    items = manifest.madlad_round_robin(synthetic_raw(), "xsum", None)
    # string sort of item_id: k1 < k10 < k2; round-robin over sources A, B, C at each position
    assert items.item_id.tolist() == [
        "xsum|A|k1", "xsum|B|k1", "xsum|C|k1", "xsum|A|k10", "xsum|B|k10", "xsum|C|k10",
        "xsum|A|k2", "xsum|B|k2", "xsum|C|k2"]
    assert items.set_index("item_id").text["xsum|A|k1"] == "A k1"      # first occurrence kept, not "dup"
    assert "D" not in set(items.source)                                  # missing text dropped


def test_round_robin_prefix_and_count_guard():
    raw = synthetic_raw()
    assert manifest.madlad_round_robin(raw, "xsum", 4).item_id.tolist()[-1] == "xsum|A|k10"
    with pytest.raises(RuntimeError, match="expected 99 originals"):
        manifest.madlad_round_robin(raw, "xsum", 4, expected_originals=99)


def test_shared_prefix_words():
    assert manifest.shared_prefix_words("a b c d", "a b x d") == 2
    assert manifest.shared_prefix_words("a b", "a b c") == 2
    assert manifest.shared_prefix_words("x", "y") == 0


def test_immutable_write(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    p = tmp_path / "m.parquet"
    assert write_immutable_parquet(df, p) == "written"
    assert write_immutable_parquet(df.copy(), p) == "unchanged"
    with pytest.raises(ImmutableFileError):
        write_immutable_parquet(pd.DataFrame({"a": [1, 3]}), p)


# -- real data -------------------------------------------------------------------------------------

def test_real_manifest_reproduces_legacy_exactly(real):
    m, legacy = real["manifest"], real["legacy"]
    manifest.check_against_legacy(m, legacy)                 # same ids, order, T0 text, labels
    assert len(m) == 200 and m.item_id.is_unique
    assert m.sample_index.tolist() == list(range(200))
    assert m.t0_text.tolist() == legacy.S.tolist()


def test_real_counts_and_sources(real):
    m = real["manifest"]
    assert m.article_id.nunique() == 31
    assert m.source_label.value_counts().to_dict() == {
        "BigScience": 29, "Eleuther-AI": 29, "Human": 29, "LLAMA": 29, "OpenAI": 28, "PaLM": 28, "Tsinghua": 28}
    assert not m.t0_text.duplicated().any()


def test_real_provenance(real, cfg):
    m, raw = real["manifest"], real["raw"]
    assert (m.corpus_commit == cfg["corpus"]["commit"]).all()
    # source_row_index points at the exact original row in the corpus file
    src = raw.set_index("source_row_index").loc[m.source_row_index]
    assert (src.version_name == "original").all()
    assert src.text.tolist() == m.t0_text.tolist()
    assert (src.source.values == m.source_label.values).all() and (src.key.values == m.article_id.values).all()


def test_real_staged_manifest_matches_rebuild(real):
    from sot_mt import paths
    p = paths.staged_dir() / "t0_manifest.parquet"
    if not p.exists():
        pytest.skip("staged manifest not built yet")
    assert pd.read_parquet(p).equals(real["manifest"])
