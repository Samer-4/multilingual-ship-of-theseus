import pandas as pd

from sot_mt import legacy_madlad, paths, regression


def _pairs(real):
    gen, m = real["gen"], real["manifest"]
    ok = real["legacy"].item_id[~real["legacy"].bad]
    t1 = gen[gen.output_stage == "T1"].set_index("item_id").output_text
    sec = real["legacy"].set_index("item_id").seconds
    mm = m.set_index("item_id").loc[ok]
    return pd.DataFrame({"item_id": ok.values, "source": mm.source_label.values, "words": mm.t0_words.values,
                         "seconds": sec.loc[ok].values, "S": mm.t0_text.values, "B_F": t1.loc[ok].values})


def test_worker_metric_code_is_extracted_from_notebook(cfg):
    w = legacy_madlad.worker_metrics(paths.repo_file(cfg["legacy"]["madlad_notebook"]))
    assert len(w["func"]) == 52
    assert w["word_retention"]("a b b", "b") == 1 / 3
    assert w["punctuation_f1"]("a, b.", "a, b.") == 1.0


def test_imported_outputs_reproduce_recorded_metrics(real, cfg):
    recomputed = legacy_madlad.recompute_legacy_metrics(_pairs(real), paths.repo_file(cfg["legacy"]["madlad_notebook"]))
    recorded = pd.read_csv(paths.repo_file(cfg["legacy"]["madlad_metrics_csv"]), float_precision="round_trip")
    per_item = regression.per_item_comparison(recomputed, recorded)
    assert (per_item.presence == "both").all() and len(per_item) == 197
    s = regression.summary(per_item, cfg["readme_madlad_medians"]).set_index("metric_id")
    for m in regression.METRICS:
        row = s.loc[f"madlad_v1.{m}"]
        assert row.status == "pass", (m, row.to_dict())
        assert row.max_abs_diff <= regression.TOLERANCE
    for m in ["bleu", "rougeL", "chrf", "word_retention", "function_word_cosine", "punctuation_f1"]:
        assert round(s.loc[f"madlad_v1.{m}", "recomputed_median"], 3) == cfg["readme_madlad_medians"][m]
    assert (per_item.words_abs_diff == 0).all()                            # T0 word counts identical


def test_unavailable_metrics_are_pending_not_fabricated(cfg):
    s = regression.summary(pd.DataFrame({"item_id": [], "presence": []}).assign(
        **{f"{m}_{k}": [] for m in regression.METRICS for k in ["recorded", "recomputed", "abs_diff"]}),
        cfg["readme_madlad_medians"]).set_index("metric_id")
    for m in ["madlad_readme_only.e5_cosine_multilingual", "madlad_readme_only.bertscore_f1"]:
        assert s.loc[m, "status"].startswith("pending")
        assert pd.isna(s.loc[m, "recomputed_median"])
