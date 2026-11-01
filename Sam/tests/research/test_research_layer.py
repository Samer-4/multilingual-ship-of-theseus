"""Research layer (RQ1-RQ3): canonical corpus, pair metrics, classifiers. CPU only, synthetic data, no models."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sacrebleu")
pytest.importorskip("yaml")

from sot_research import classify, corpus, features, metrics  # noqa: E402

STAGES = corpus.STAGES


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def _manifest(n_articles=4, sources=("Human", "OpenAI", "PaLM")):
    rows = []
    for a in range(n_articles):
        opening = f"Opening words of article {a} are shared by every source here ."
        for s in sources:
            t0 = f"{opening} Then {s} writes its own continuation number {a} with 12 apples in London ."
            rows.append({"dataset": "xsum", "item_id": f"xsum|{s}|art-{a}", "article_id": f"art-{a}",
                         "group_id": f"xsum|art-{a}", "source_label": s, "is_human": s == "Human", "t0_text": t0,
                         "t0_sha256": sha(t0), "shared_prefix_words_with_human": None if s == "Human" else 11})
    return pd.DataFrame(rows)


def _translator_records(man, name, fail=None):
    """A full valid chain per item; `fail` = {item_id: stage} makes that stage invalid and stops the chain."""
    fail = fail or {}
    rows = []
    for m in man.itertuples():
        prev = m.t0_text
        for st in STAGES:
            out = (f"[{name} {st}] " + prev) if st[0] == "S" else prev.split("] ", 1)[-1] + f" ({name})"
            status = "invalid" if fail.get(m.item_id) == st else "valid"
            rows.append({"dataset": "xsum", "item_id": m.item_id, "translator": name, "output_stage": st,
                         "output_text": out, "output_sha256": sha(out), "validation_status": status,
                         "validation_reason": "check:repetition" if status == "invalid" else None,
                         "validation_warnings": None, "input_sha256": sha(prev), "model_checkpoint": f"org/{name}",
                         "generation_config_id": f"cfg-{name}", "run_id": "r1", "record_id": f"{m.item_id}{st}",
                         "output_language": "es" if st[0] == "S" else "en"})
            if status == "invalid":
                break
            prev = out
    return pd.DataFrame(rows)


def _write_final(root, name, recs):
    d = root / f"{name}_final" / "xsum" / name
    d.mkdir(parents=True)
    recs.to_parquet(d / "generations_merged.parquet", index=False)
    (d / "FINAL.json").write_text(json.dumps({"generations_merged_sha256": corpus.sha256_file(
        d / "generations_merged.parquet"), "complete_items": None, "source_export": None}))


@pytest.fixture
def built(tmp_path):
    man = _manifest()
    mp = tmp_path / "t0_manifest.parquet"
    man.to_parquet(mp, index=False)
    gen = tmp_path / "generated"
    _write_final(gen, "mt-a", _translator_records(man, "mt-a", {"xsum|OpenAI|art-1": "T2"}))
    _write_final(gen, "llm-b", _translator_records(man, "llm-b"))
    _write_final(gen, "llm-c", _translator_records(man, "llm-c", {"xsum|PaLM|art-2": "S1"}))
    return man, mp, gen


def test_corpus_grid_statuses_and_cohort(built):
    man, mp, gen = built
    texts, chains, info = corpus.build(mp, gen)
    assert sorted(info["translators"]) == ["llm-b", "llm-c", "mt-a"]          # discovered, not hardcoded
    assert (texts.stage == "T0").sum() == len(man)                             # T0 once per item
    assert len(texts) == len(man) * (1 + 3 * len(STAGES))
    t = texts.set_index(["item_id", "translator", "stage"]).status
    assert t[("xsum|OpenAI|art-1", "mt-a", "T2")] == "invalid"
    assert t[("xsum|OpenAI|art-1", "mt-a", "S3")] == "blocked"
    assert t[("xsum|PaLM|art-2", "llm-c", "T1")] == "blocked"
    cohort = corpus.cohort_table(chains)
    assert cohort.matched.sum() == len(man) - 2
    assert info["lineage"]["valid_records_checked"] == int(((texts.translator != "original") & texts.valid).sum())


def test_corpus_rejects_broken_lineage(tmp_path):
    man = _manifest(2)
    mp = tmp_path / "m.parquet"
    man.to_parquet(mp, index=False)
    recs = _translator_records(man, "x")
    recs.loc[recs.output_stage == "T1", "input_sha256"] = "0" * 64
    _write_final(tmp_path / "g", "x", recs)
    with pytest.raises(ValueError, match="lineage"):
        corpus.build(mp, tmp_path / "g")


def test_corpus_rejects_hash_mismatch(tmp_path):
    man = _manifest(2)
    mp = tmp_path / "m.parquet"
    man.to_parquet(mp, index=False)
    _write_final(tmp_path / "g", "x", _translator_records(man, "x"))
    fin = tmp_path / "g" / "x_final" / "xsum" / "x" / "FINAL.json"
    fin.write_text(json.dumps({"generations_merged_sha256": "bad"}))
    with pytest.raises(ValueError, match="sha256"):
        corpus.build(mp, tmp_path / "g")


def test_pairs_cumulative_step_and_baselines(built):
    man, mp, gen = built
    texts, _, _ = corpus.build(mp, gen)
    p = metrics.build_pairs(corpus.english(texts), seed=1)
    full = p[(p.translator == "llm-b") & (p.item_id == "xsum|Human|art-0")]
    assert sorted(zip(full.kind, full.ref_stage, full.hyp_stage)) == sorted(
        [("cumulative", "T0", "T1"), ("cumulative", "T0", "T2"), ("cumulative", "T0", "T3"),
         ("step", "T0", "T1"), ("step", "T1", "T2"), ("step", "T2", "T3")])
    cut = p[(p.translator == "mt-a") & (p.item_id == "xsum|OpenAI|art-1")]
    assert set(cut.hyp_stage) == {"T1"}                                        # invalid T2 ends the chain
    un = p[p.kind == "unrelated"]
    art = texts[texts.stage == "T0"].set_index("text_id").article_id
    assert len(un) == len(man) and (un.ref_text_id.map(art) != un.hyp_text_id.map(art)).all()
    cs = p[p.kind == "cross_source"]
    assert len(cs) == (~man.is_human).sum() and cs.ref_text_id.str.contains("Human").all()


def test_lexical_definitions():
    r = "The cat sat on the mat. It was 12 degrees."
    m = metrics.lexical(r, r)
    assert m["bleu"] == pytest.approx(1) and m["exact_equal"] == 1 and m["word_edit_rate"] == 0
    m = metrics.lexical("the the cat", "the cat")
    assert m["word_retention"] == pytest.approx(2 / 3)                          # clipped multiset recall
    assert metrics.levenshtein("a b c d".split(), "a x c".split()) == 2


def test_number_normalization_and_quantities():
    nums, qty = features.numbers("It cost £11,200 or 3.50 dollars, up 12% on 2019.")
    assert nums == ["11200", "3.5", "12", "2019"]
    assert "gbp11200" in qty and "12 %" in qty and "3.5 dollars" in qty


def test_information_metrics_flag_losses():
    rf = pd.Series({"entities_json": json.dumps([["london", "GPE"], ["bbc", "ORG"]]),
                    "numbers_json": json.dumps(["12", "2019"]), "quantities_json": json.dumps([])})
    hf = pd.Series({"numbers_json": json.dumps(["12", "2020"]), "quantities_json": json.dumps([])})
    m = metrics.information(rf, "The BBC said 12 people in 2020.", hf)
    assert m["entities_lost"] == 1 and m["entity_recall"] == 0.5
    assert m["numbers_lost"] == 1 and m["numbers_added"] == 1 and m["number_recall"] == 0.5


def test_article_folds_never_split_an_article():
    df = pd.DataFrame({"article_id": [f"a{i % 7}" for i in range(70)]})
    f = classify.article_folds(df.article_id, 5, seed=3)
    assert set(f) == set(df.article_id) and set(f.values()) <= set(range(5))


def test_grouped_cv_has_no_article_leakage():
    rng = np.random.default_rng(0)
    rows = []
    for a in range(10):
        for lab in ("x", "y"):
            for st in ("T0", "T1"):
                rows.append({"text_id": f"{a}{lab}{st}", "item_id": f"{a}{lab}", "article_id": f"art{a}",
                             "source_label": lab, "translator": "t", "stage": st, "f1": rng.normal(lab == "x"),
                             "f2": rng.normal()})
    df = pd.DataFrame(rows)
    seen = []

    def spy(tr):
        seen.append(set(tr.article_id))
        return tr
    preds = classify.grouped_cv(df[df.stage == "T0"], {"T1": df[df.stage == "T1"]}, "source_label", "stylometric",
                                cols=["f1", "f2"], n_splits=5, repeats=2, seed=1, train_filter=spy)
    for (r, k), part in preds.groupby(["repeat", "fold"]):
        assert not (set(part.article_id) & seen[r * 5 + k])
    assert preds.groupby("repeat").text_id.nunique().eq(20).all()               # every eval text predicted once
    s = classify.scores(preds, ["eval_set"], ["x", "y"], n_boot=50)
    assert s.chance.iloc[0] == 0.5 and 0 <= s.macro_f1_ci_low.iloc[0] <= s.macro_f1_ci_high.iloc[0] <= 1


def test_shared_opening_is_measured_per_group():
    df = pd.DataFrame({"article_id": ["a"] * 3 + ["b"] * 2, "translator": "t", "stage": "T1",
                       "text": ["one two three four x", "one two three five y", "one two zz", "p q r", "s t u"]})
    n = classify.shared_opening_lengths(df)
    assert n.tolist() == [3, 3, 2, 0, 0]
    assert classify.strip_opening(df, n).text.tolist()[:2] == ["four x", "five y"]
