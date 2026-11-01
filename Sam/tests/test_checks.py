import pytest

from sot_mt.checks import check_step, repetition
from sot_mt.files import load_yaml

CFG = load_yaml("experiment.yaml")["validation"]
EN = "The minister said the plan was approved on Monday and that it would start next year."
ES = "El ministro dijo que el plan fue aprobado el lunes y que comenzaría el próximo año."


def test_clean_steps():
    assert check_step(EN, ES, "es", CFG) == {"blocking": [], "warnings": []}
    assert check_step(ES, EN, "en", CFG) == {"blocking": [], "warnings": []}


@pytest.mark.parametrize("out, code", [(None, "check:missing_output"), (float("nan"), "check:missing_output"),
                                       ("", "check:blank"), ("   \n", "check:blank")])
def test_missing_and_blank(out, code):
    assert check_step(EN, out, "es", CFG)["blocking"] == [code]


def test_unchanged_is_untranslated():
    r = check_step(EN, "  " + EN.replace(" ", "  "), "es", CFG)
    assert "check:unchanged" in r["blocking"] and "check:wrong_language" in r["blocking"]


def test_wrong_language_both_directions():
    assert "check:wrong_language" in check_step(ES, ES + " ", "en", CFG)["blocking"]
    assert "check:wrong_language" in check_step(EN, "Translation: " + EN, "es", CFG)["blocking"]


def test_repetition_loop():
    loop = "Tags: FSA, " + ", ".join(["pigs"] * 60)
    assert repetition(loop, 6, 0.10)
    assert not repetition(EN, 6, 0.10)
    assert "check:repetition" in check_step(EN, loop, "en", CFG)["blocking"]


def test_identical_round_trip_is_valid_but_recorded():
    from sot_mt.checks import roundtrip_warnings
    assert check_step(ES, EN, "en", CFG)["blocking"] == []                 # T_r == T_{r-1} is not a failure
    assert roundtrip_warnings(EN, EN + " ") == ["check:roundtrip_unchanged"]
    assert roundtrip_warnings(EN, "Something else.") == [] and roundtrip_warnings(EN, None) == []


def test_meta_text_is_flagged_not_removed():
    r = check_step(EN, "Aquí está la traducción: " + ES, "es", CFG)
    assert r["blocking"] == [] and "check:meta_text" in r["warnings"]


def test_warnings_do_not_block():
    r = check_step(EN, "El ministro dijo que el plan", "es", CFG)
    assert r["blocking"] == []
    assert "check:possible_truncation" in r["warnings"]
    assert any(w.startswith("check:length_ratio=") for w in r["warnings"])
