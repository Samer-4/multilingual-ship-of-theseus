import pandas as pd
import pytest

from sot_mt import schema


def test_import_twice_is_idempotent(t0, full_chain):
    first, s1 = schema.upsert(schema.empty_table(), pd.DataFrame(full_chain), t0)
    second, s2 = schema.upsert(first, pd.DataFrame(full_chain), t0)
    assert s1 == {"inserted": 6, "unchanged": 0}
    assert s2 == {"inserted": 0, "unchanged": 6}
    assert second.equals(first)


def test_idempotent_after_parquet_round_trip(t0, full_chain, tmp_path):
    first, _ = schema.upsert(schema.empty_table(), pd.DataFrame(full_chain), t0)
    p = tmp_path / "g.parquet"
    first.to_parquet(p, index=False)
    again, s = schema.upsert(pd.read_parquet(p), pd.DataFrame(full_chain), t0)
    assert s["inserted"] == 0 and len(again) == 6


def test_changed_content_for_existing_key_conflicts(t0, full_chain, make_row):
    gen, _ = schema.upsert(schema.empty_table(), pd.DataFrame(full_chain[:1]), t0)
    changed = make_row("xsum|Human|a1", "S1", full_chain[0]["input_text"], "a different translation")
    with pytest.raises(schema.ConflictError, match="output_text"):
        schema.upsert(gen, pd.DataFrame([changed]), t0)


def test_duplicate_key_within_batch_rejected(t0, full_chain):
    with pytest.raises(schema.DuplicateRecordError):
        schema.upsert(schema.empty_table(), pd.DataFrame([full_chain[0], full_chain[0]]), t0)


def test_missing_records_show_up_as_attrition(t0, full_chain):
    gen, _ = schema.upsert(schema.empty_table(), pd.DataFrame(full_chain[:2]), t0)
    att = schema.attrition(gen, t0).set_index("output_stage")
    assert att.loc["S1", "valid"] == 1 and att.loc["S1", "not_attempted"] == 2   # two items never imported
