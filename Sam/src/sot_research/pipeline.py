"""Build the canonical corpus, then compute and cache every shared feature and pairwise metric.

Each step resumes from its cache (keyed by text/pair sha256 and a configuration fingerprint), so adding a
translator only computes the new texts and the pairs that involve them.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import corpus, features, metrics, neural, paths


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_corpus(log=_log) -> dict:
    texts, chains, info = corpus.build()
    summary = corpus.write(texts, chains, info)
    log(f"corpus: {summary['items']} items, {summary['articles']} articles, translators {summary['translators']}, "
        f"matched items {summary['matched_items']}, complete chains {summary['complete_chains']}")
    return summary


def run(skip_neural: bool = False, workers: int = 4, log=_log) -> dict:
    t_start = time.time()
    cfg = corpus.research_config()
    summary = build_corpus(log)
    texts = corpus.load_texts()
    en = corpus.english(texts)
    t0 = en[en.stage == "T0"]
    refl = features.reference_lists(t0.text, cfg["function_words"], cfg["burrows_mfw"])
    out_sf = paths.results("shared_features")
    (out_sf / "reference_lists.json").write_text(json.dumps(refl, indent=1))

    feats = features.compute(en, refl, n_process=workers, log=log)
    feature_fp = features.fingerprint(refl)
    tf = en.drop(columns=["text"]).merge(feats.drop(columns=features.TEXT_COLUMNS), on="text_sha256", how="left")
    tf.to_parquet(out_sf / "text_features.parquet", index=False)

    pairs = metrics.build_pairs(en, cfg["seed"])
    text_of = dict(zip(en.text_sha256, en.text))
    cpu, cpu_fp = metrics.compute_cpu(pairs, text_of, feats, refl, feature_fp, workers=workers, log=log)
    table = pairs.merge(cpu, on=metrics.PAIR_KEY, how="left")
    manifest = {"started": datetime.fromtimestamp(t_start, timezone.utc).isoformat(), "corpus": summary,
                "fingerprints": {"text_features": feature_fp, "pair_cpu": cpu_fp},
                "python": platform.python_version(), "platform": platform.platform()}
    for lib in ["numpy", "pandas", "sacrebleu", "rouge_score", "sklearn", "spacy", "torch", "transformers",
                "bert_score"]:
        try:
            manifest.setdefault("libraries", {})[lib] = __import__(lib).__version__
        except Exception:  # noqa: BLE001
            manifest.setdefault("libraries", {})[lib] = None

    if not skip_neural:
        nc = cfg["neural"]
        doc_items = dict(zip(en.text_sha256, en.text))
        sents = {}
        F = feats.set_index("text_sha256")
        sent_lists = {sha: json.loads(F.loc[sha, "sentences_json"]) for sha in doc_items}
        for lst in sent_lists.values():
            for s in lst:
                sents[corpus.sha256_text(s)] = s
        e5 = neural.ChunkEmbedder(nc["e5"]["model"], nc["e5"]["max_tokens"], nc["e5"]["prefix"])
        v_e5 = neural.embed_cached(e5, doc_items, "e5", log)
        info_e5 = {"repo": e5.repo, "revision": e5.revision, "fingerprint": e5.fingerprint, "device": e5.device,
                   "texts_chunked_this_run": e5.chunked_texts}
        del e5
        mp = neural.ChunkEmbedder(nc["mpnet"]["model"], nc["mpnet"]["max_tokens"])
        v_mp = neural.embed_cached(mp, doc_items, "mpnet", log)
        v_sent = neural.embed_cached(mp, sents, "mpnet_sentences", log, batch=256)
        info_mp = {"repo": mp.repo, "revision": mp.revision, "fingerprint": mp.fingerprint, "device": mp.device,
                   "texts_chunked_this_run": mp.chunked_texts}
        del mp
        bs = neural.LongBertScore(nc["bertscore"]["model"], nc["bertscore"]["layer"], nc["bertscore"]["window"],
                                  nc["bertscore"]["step"])
        bsc = neural.bertscore_pairs(pairs, text_of, bs, log)
        info_bs = {"repo": bs.repo, "revision": bs.revision, "fingerprint": bs.fingerprint, "device": bs.device,
                   "layer": bs.layer, "baseline": bs.baseline, "bert_score": bs.bert_score_version,
                   "windowed_texts_this_run": bs.windowed_texts}
        del bs
        thr = cfg["sentence_coverage_threshold"]
        u = pairs.drop_duplicates(metrics.PAIR_KEY)
        sem = []
        for r, h in u[metrics.PAIR_KEY].to_numpy():
            sem.append({"ref_sha256": r, "hyp_sha256": h, "e5_cosine": float(v_e5[r] @ v_e5[h]),
                        "mpnet_cosine": float(v_mp[r] @ v_mp[h]),
                        **neural.coverage(sent_lists[r], sent_lists[h], v_sent, thr)})
        table = table.merge(pd.DataFrame(sem), on=metrics.PAIR_KEY, how="left").merge(bsc, on=metrics.PAIR_KEY,
                                                                                      how="left")
        # token lengths under each encoder policy (to document how many texts needed chunking/windowing)
        manifest["neural"] = {"e5": info_e5, "mpnet": info_mp, "bertscore": info_bs,
                              "sentence_coverage_threshold": thr, "unique_sentences": len(sents)}

    # heuristic content warnings (never "errors")
    table["flag_entity_loss"] = table.entities_lost.fillna(0) > 0
    table["flag_number_change"] = (table.numbers_lost.fillna(0) + table.numbers_added.fillna(0)) > 0
    if "sentences_uncovered" in table:
        table["flag_sentence_omission"] = table.sentences_uncovered.fillna(0) > 0
    flags = [c for c in table.columns if c.startswith("flag_")]
    table["flag_any_content_warning"] = table[flags].any(axis=1)

    meta = texts.set_index("text_id")
    for side in ("ref", "hyp"):
        table[f"{side}_words"] = table[f"{side}_text_id"].map(meta.words)
    ids = texts[texts.stage == "T0"].set_index("item_id")[["article_id", "group_id", "source_label", "is_human"]]
    table = table.join(ids, on="item_id")
    cohort = pd.read_csv(paths.results("data_coverage") / "cohort.csv").set_index("item_id").matched
    table["matched"] = table.item_id.map(cohort).fillna(False).astype(bool)
    table.to_parquet(paths.local("pair_metrics_full.parquet"), index=False)
    public = table.drop(columns=metrics.LOCAL_ONLY)
    out_m = paths.results("metrics")
    public.to_parquet(out_m / "pair_metrics.parquet", index=False)
    public.round(6).to_csv(out_m / "pair_metrics.csv.gz", index=False)
    manifest.update(pairs=int(len(table)), unique_pairs=int(len(table.drop_duplicates(metrics.PAIR_KEY))),
                    pairs_by_kind=table.groupby(["translator", "kind"]).size().rename("n").reset_index()
                    .to_dict("records"),
                    seconds=round(time.time() - t_start, 1), finished=datetime.now(timezone.utc).isoformat())
    (paths.results("run_manifests") / "feature_run.json").write_text(json.dumps(manifest, indent=1, default=str))
    log(f"done: {len(table)} pair rows, {manifest['seconds']} s")
    return manifest


def token_lengths(en: pd.DataFrame) -> pd.DataFrame:
    """Per-text token counts under the three encoders' tokenizers (no model weights loaded)."""
    from transformers import AutoTokenizer

    cfg = corpus.research_config()["neural"]
    out = en[["text_id", "text_sha256"]].copy()
    for k in ("e5", "mpnet", "bertscore"):
        tok = AutoTokenizer.from_pretrained(neural.snapshot(cfg[k]["model"])[0])
        out[f"tokens_{k}"] = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in en.text]
    return out
