import pandas as pd
import pytest

from sot_mt import schema


def test_stage_sequence():
    assert schema.STAGES == ["T0", "S1", "T1", "S2", "T2", "S3", "T3"]
    expected = {
        "S1": (1, "en->es", "T0"), "T1": (1, "es->en", "S1"),
        "S2": (2, "en->es", "T1"), "T2": (2, "es->en", "S2"),
        "S3": (3, "en->es", "T2"), "T3": (3, "es->en", "S3"),
    }
    for stage, (rt, direction, inp) in expected.items():
        i = schema.stage_info(stage)
        assert (i["round_trip"], i["direction"], i["input_stage"]) == (rt, direction, inp)
        assert i["input_language"] == direction[:2] and i["output_language"] == direction[-2:]
    for bad in ["T0", "S4", "X1", ""]:
        with pytest.raises(schema.SchemaError):
            schema.stage_info(bad)


def test_record_id_is_stable_and_key_only(make_row):
    a = make_row("xsum|Human|a1", "S1", "The cat sat on the mat.", "El gato.")
    b = make_row("xsum|Human|a1", "S1", "The cat sat on the mat.", "Otro texto.")
    c = make_row("xsum|Human|a1", "S1", "The cat sat on the mat.", "El gato.", translator="nllb", model="m")
    assert a["record_id"] == b["record_id"]               # content does not change the key
    assert a["record_id"] != c["record_id"]               # pathway does
    assert len(a["record_id"]) == 16
    assert a["record_id"] == schema.record_id(dict(a))    # deterministic


def test_valid_rows_pass(t0, full_chain):
    schema.check_rows(pd.DataFrame(full_chain), t0)


@pytest.mark.parametrize("mutate, match", [
    (lambda r: r.pop("run_id"), "missing columns"),
    (lambda r: r.update(generation_config_id=None), "null values"),
    (lambda r: r.update(direction="es->en"), "direction"),
    (lambda r: r.update(input_stage="T1"), "input_stage"),
    (lambda r: r.update(validation_status="ok"), "validation_status"),
    (lambda r: r.update(validation_status="invalid", validation_reason=None), "without validation_reason"),
    (lambda r: r.update(input_sha256="0" * 64), "input_sha256"),
    (lambda r: r.update(output_sha256="0" * 64), "output_sha256"),
    (lambda r: r.update(record_id="deadbeefdeadbeef"), "record_id"),
    (lambda r: r.update(source_label="PaLM"), "source_label"),     # labels must match the manifest
    (lambda r: r.update(article_id="a2"), "source_label"),
])
def test_invalid_rows_rejected(t0, full_chain, mutate, match):
    row = dict(full_chain[0])
    mutate(row)
    with pytest.raises(schema.SchemaError, match=match):
        schema.check_rows(pd.DataFrame([row]), t0)


def test_unknown_item_rejected(t0, full_chain):
    row = dict(full_chain[0], item_id="xsum|Human|zzz")
    row["record_id"] = schema.record_id(row)
    with pytest.raises(schema.SchemaError, match="not in the T0 manifest"):
        schema.check_rows(pd.DataFrame([row]), t0)
