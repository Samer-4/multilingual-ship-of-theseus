import pandas as pd
import pytest

from sot_mt import schema


def upsert(rows, t0, existing=None, **kw):
    existing = schema.empty_table() if existing is None else existing
    return schema.upsert(existing, pd.DataFrame(rows), t0, **kw)


def test_full_chain_accepted_in_order(t0, full_chain):
    gen, stats = upsert(full_chain, t0)
    assert stats == {"inserted": 6, "unchanged": 0}
    assert gen.output_stage.tolist() == ["S1", "T1", "S2", "T2", "S3", "T3"]
    assert schema.chain_violations(gen, t0) == []


def test_chain_can_be_built_one_stage_at_a_time(t0, full_chain):
    gen = schema.empty_table()
    for row in full_chain:
        gen, _ = upsert([row], t0, existing=gen)
    assert len(gen) == 6


def test_s1_must_start_from_t0(t0, make_row):
    with pytest.raises(schema.ChainError, match="not the pathway's T0"):
        upsert([make_row("xsum|Human|a1", "S1", "some other English text", "x")], t0)


def test_later_round_must_not_restart_from_t0(t0, full_chain, make_row):
    gen, _ = upsert(full_chain[:2], t0)
    t0_text = t0.set_index("item_id").t0_text["xsum|Human|a1"]
    with pytest.raises(schema.ChainError, match="not the pathway's T1"):
        upsert([make_row("xsum|Human|a1", "S2", t0_text, "x")], t0, existing=gen)


def test_stage_cannot_be_skipped(t0, full_chain):
    with pytest.raises(schema.ChainError, match="predecessor T1 missing"):
        upsert([full_chain[0], full_chain[2]], t0)          # S1 then S2, no T1


def test_invalid_generation_never_fed_forward(t0, make_row):
    item = "xsum|Human|a1"
    text = t0.set_index("item_id").t0_text[item]
    s1 = make_row(item, "S1", text, "loop loop loop", status="invalid", reason="check:repetition")
    gen, _ = upsert([s1], t0)
    t1 = make_row(item, "T1", "loop loop loop", "back")
    with pytest.raises(schema.ChainError, match="non-valid S1|derived from invalid S1"):
        upsert([t1], t0, existing=gen)
    # even an invalid new record may not be generated from an invalid input
    with pytest.raises(schema.ChainError, match="non-valid S1"):
        upsert([dict(t1, validation_status="invalid", validation_reason="x")], t0, existing=gen)
    assert schema.eligible_inputs(gen, t0, "exp", "madlad-3b", "mt-default", "T1").empty


def test_historical_import_flag_still_forbids_valid_from_invalid(t0, make_row):
    item = "xsum|Human|a1"
    text = t0.set_index("item_id").t0_text[item]
    s1 = make_row(item, "S1", text, "es", status="invalid", reason="legacy:generation_cap")
    t1_valid = make_row(item, "T1", "es", "en")
    with pytest.raises(schema.ChainError, match="valid record derived from invalid"):
        upsert([s1, t1_valid], t0, allow_invalid_predecessor=True)
    t1_invalid = make_row(item, "T1", "es", "en", status="invalid", reason="legacy:generation_cap")
    gen, _ = upsert([s1, t1_invalid], t0, allow_invalid_predecessor=True)
    assert len(gen) == 2


def test_eligible_inputs(t0, make_row):
    texts = t0.set_index("item_id").t0_text
    rows = [make_row("xsum|Human|a1", "S1", texts["xsum|Human|a1"], "es1"),
            make_row("xsum|OpenAI|a1", "S1", texts["xsum|OpenAI|a1"], "", status="invalid", reason="check:blank")]
    gen, _ = upsert(rows, t0)
    s1 = schema.eligible_inputs(gen, t0, "exp", "madlad-3b", "mt-default", "S1")
    assert s1.item_id.tolist() == ["xsum|Human|a2"]                       # only the ungenerated one
    t1 = schema.eligible_inputs(gen, t0, "exp", "madlad-3b", "mt-default", "T1")
    assert t1.to_dict("records") == [{"item_id": "xsum|Human|a1", "input_text": "es1", "input_stage": "S1"}]


def test_pathways_are_independent(t0, full_chain, make_row):
    gen, _ = upsert(full_chain[:2], t0)                                     # madlad S1, T1
    item = "xsum|Human|a1"
    madlad_t1 = full_chain[1]["output_text"]
    # an NLLB S2 cannot chain onto MADLAD's T1: NLLB has no T1 of its own
    with pytest.raises(schema.ChainError, match="predecessor T1 missing"):
        upsert([make_row(item, "S2", madlad_t1, "x", translator="nllb", model="nllb-m")], t0, existing=gen)
    # nor can a Qwen 'literal' S2 chain onto Qwen 'free' T1
    texts = t0.set_index("item_id").t0_text
    q = [make_row(item, "S1", texts[item], "qs1", translator="qwen3-4b", prompt_variant="free", model="qwen"),
         make_row(item, "T1", "qs1", "qt1", translator="qwen3-4b", prompt_variant="free", model="qwen")]
    gen, _ = upsert(q, t0, existing=gen)
    with pytest.raises(schema.ChainError, match="predecessor T1 missing"):
        upsert([make_row(item, "S2", "qt1", "x", translator="qwen3-4b", prompt_variant="literal", model="qwen")],
               t0, existing=gen)


def test_one_checkpoint_per_translator(t0, make_row):
    texts = t0.set_index("item_id").t0_text
    rows = [make_row("xsum|Human|a1", "S1", texts["xsum|Human|a1"], "a"),
            make_row("xsum|Human|a2", "S1", texts["xsum|Human|a2"], "b", model="another/model")]
    with pytest.raises(schema.ChainError, match="more than one model_checkpoint"):
        upsert(rows, t0)


def test_attrition_and_complete_cases(t0, make_row):
    texts = t0.set_index("item_id").t0_text
    rows = [make_row("xsum|Human|a1", "S1", texts["xsum|Human|a1"], "es1"),
            make_row("xsum|Human|a1", "T1", "es1", "en1"),
            make_row("xsum|OpenAI|a1", "S1", texts["xsum|OpenAI|a1"], "", status="invalid", reason="check:blank")]
    gen, _ = upsert(rows, t0)
    att = schema.attrition(gen, t0).set_index("output_stage")
    assert att.loc["S1", ["valid", "invalid", "not_attempted"]].tolist() == [1, 1, 1]
    assert att.loc["T1", ["valid", "blocked", "pending"]].tolist() == [1, 1, 1]
    assert att.loc["S2", ["not_attempted", "blocked", "pending"]].tolist() == [1, 1, 1]
    assert (att[["valid", "invalid", "blocked", "not_attempted", "pending"]].sum(axis=1) == 3).all()
    assert schema.complete_cases(gen, t0, "T1") == ["xsum|Human|a1"]
    assert schema.complete_cases(gen, t0, "T2") == []
