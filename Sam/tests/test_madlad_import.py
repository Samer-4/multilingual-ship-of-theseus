import json

import pandas as pd

from sot_mt import schema
from sot_mt.files import sha256_file

FLAGGED = {"xsum|BigScience|xsum-4713": "generation_cap", "xsum|LLAMA|xsum-4714": "generation_cap;severe_expansion",
           "xsum|LLAMA|xsum-4733": "generation_cap"}


def test_record_counts(real):
    gen = real["gen"]
    assert len(gen) == 400 and gen.item_id.nunique() == 200
    assert gen.output_stage.value_counts().to_dict() == {"S1": 200, "T1": 200}
    assert real["stats"] == {"inserted": 400, "unchanged": 0}
    assert schema.chain_violations(gen, real["manifest"]) == []


def test_exact_text_alignment(real):
    gen, legacy, m = real["gen"], real["legacy"].set_index("item_id"), real["manifest"].set_index("item_id")
    s1 = gen[gen.output_stage == "S1"].set_index("item_id")
    t1 = gen[gen.output_stage == "T1"].set_index("item_id")
    ids = legacy.index
    assert (s1.loc[ids, "input_text"] == m.loc[ids, "t0_text"]).all()       # T0
    assert (s1.loc[ids, "input_text"] == legacy.S).all()
    assert (s1.loc[ids, "output_text"] == legacy.F).all()                   # S1
    assert (t1.loc[ids, "input_text"] == legacy.F).all()
    assert (t1.loc[ids, "output_text"] == legacy.B_F).all()                 # T1
    assert (s1.direction == "en->es").all() and (t1.direction == "es->en").all()


def test_source_labels_and_articles_preserved(real):
    gen, legacy = real["gen"], real["legacy"].set_index("item_id")
    g = gen.set_index("item_id")
    assert (g.source_label == legacy.loc[g.index, "source"]).all()
    assert (g.article_id == legacy.loc[g.index, "key"]).all()


def test_legacy_flags_preserved(real):
    gen = real["gen"]
    inv = gen[gen.validation_status == "invalid"]
    assert set(inv.item_id) == set(FLAGGED) and len(inv) == 6               # S1 and T1 of each flagged item
    for r in inv.itertuples():
        for code in FLAGGED[r.item_id].split(";"):
            assert f"legacy:{code}" in r.validation_reason.split(";")
        assert "legacy:direction_unattributed" in r.validation_reason
        assert json.loads(r.legacy_fields)["bad_reason"] == FLAGGED[r.item_id]
    legacy_ok = gen[~gen.item_id.isin(FLAGGED)]
    assert (legacy_ok.validation_status == "valid").all()
    assert legacy_ok.legacy_fields.map(lambda s: json.loads(s)["bad"]).eq(False).all()


def test_provenance_and_source_file_untouched(real):
    gen = real["gen"]
    assert gen.provenance.str.startswith("legacy:results/madlad_translation/chain_es_madlad4003b.csv#row=").all()
    assert (gen.run_id == "legacy-madlad-explorer-v100-r1").all()
    assert gen.generation_config_id.nunique() == 1
    assert gen.legacy_fields.map(lambda s: json.loads(s)["source_file_sha256"]).eq(real["legacy_sha"]).all()
    assert sha256_file(real["legacy_path"]) == real["legacy_sha"]


def test_reimport_is_idempotent(real):
    again, stats = schema.upsert(real["gen"], real["rows"], real["manifest"], allow_invalid_predecessor=True)
    assert stats == {"inserted": 0, "unchanged": 400}
    assert again.equals(real["gen"])


def test_round_two_only_from_valid_t1(real):
    e = schema.eligible_inputs(real["gen"], real["manifest"], "msot_xsum_madlad200", "madlad-3b", "mt-default", "S2")
    assert len(e) == 197 and not set(e.item_id) & set(FLAGGED)
    t1 = real["gen"][real["gen"].output_stage == "T1"].set_index("item_id").output_text
    assert (e.set_index("item_id").input_text == t1.loc[e.item_id].values).all()


def test_staged_generations_match_rebuild(real):
    from sot_mt import paths
    p = paths.staged_dir() / "generations.parquet"
    if not p.exists():
        import pytest
        pytest.skip("staged generations not built yet")
    staged = schema.conform(pd.read_parquet(p))        # staged before generation_meta existed
    assert staged.equals(real["gen"])
