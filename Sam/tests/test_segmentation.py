import ast
import random
import re
from types import SimpleNamespace

import pytest

from sot_mt import legacy_madlad, paths, segmentation
from sot_mt.files import load_yaml


def words_plus_prefix(s):          # stand-in tokenizer: one token per word + 1 for the "<2xx>" prefix
    return len(s.split()) + 1


def worker_split_chunks(count, budget):
    """The MADLAD worker's own split_chunks(), extracted from the notebook and bound to a fake tokenizer."""
    nb = paths.repo_file(load_yaml("experiment.yaml")["legacy"]["madlad_notebook"])
    if not nb.exists():
        pytest.skip("MADLAD notebook not available")
    tree = ast.parse(legacy_madlad.worker_source(nb))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "split_chunks")
    ns = {"re": re, "token_count": lambda s, lang: count(s), "a": SimpleNamespace(budget=budget)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "worker", "exec"), ns)
    return lambda text: ns["split_chunks"](text, "es")


def random_doc(rng):
    sents = []
    for _ in range(rng.randint(1, 12)):
        n = rng.choice([1, 3, 8, 15, 40])
        sents.append(" ".join(rng.choice(["alpha", "beta", "gamma", "delta"]) for _ in range(n))
                     + rng.choice([".", "!", "?", ""]))
    return rng.choice([" ", "  ", "\n"]).join(sents)


@pytest.mark.parametrize("budget", [5, 12, 30, 300])
def test_madlad_pack_matches_worker_exactly(budget):
    worker = worker_split_chunks(words_plus_prefix, budget)
    rng = random.Random(budget)
    for _ in range(300):
        doc = random_doc(rng)
        try:
            expected = worker(doc)
        except RuntimeError as e:
            with pytest.raises(segmentation.SegmentationError, match=str(e)):
                segmentation.madlad_pack(doc, words_plus_prefix, budget)
            continue
        assert segmentation.madlad_pack(doc, words_plus_prefix, budget) == expected


def test_madlad_pack_on_real_legacy_text_matches_worker(real):
    worker = worker_split_chunks(words_plus_prefix, 60)
    for text in real["manifest"].t0_text.head(40):
        assert segmentation.madlad_pack(text, words_plus_prefix, 60) == worker(text)


def test_segments_respect_budget_and_keep_every_word():
    text = "One two three. Four five six seven eight nine ten eleven twelve! Thirteen?"
    for method in ["madlad_pack", "sentence"]:
        segs = segmentation.segment(text, method, words_plus_prefix, 6)
        assert all(words_plus_prefix(s) <= 6 for s in segs)
        assert " ".join(segs).split() == text.split()          # nothing lost, nothing reordered


def test_sentence_mode_is_one_sentence_per_segment():
    text = "A b. C d. E f."
    assert segmentation.sentences(text, words_plus_prefix, 100) == ["A b.", "C d.", "E f."]
    assert segmentation.madlad_pack(text, words_plus_prefix, 100) == ["A b. C d. E f."]


def test_errors_are_deterministic():
    with pytest.raises(segmentation.SegmentationError, match="one_word_exceeds_budget"):
        segmentation.madlad_pack("tiny enormousword", lambda s: len(s), 5)
    for method in ["madlad_pack", "sentence", "none"]:
        with pytest.raises(segmentation.SegmentationError, match="no_input_chunks"):
            segmentation.segment("   ", method, words_plus_prefix, 10)
    with pytest.raises(ValueError):
        segmentation.segment("x", "pysbd", words_plus_prefix, 10)


def test_reconstruction_and_metadata():
    assert segmentation.reconstruct(["Hola.", "Adiós."]) == "Hola. Adiós."
    segs = ["A b.", "C d e."]
    meta = segmentation.describe(segs, "sentence", words_plus_prefix, 10)
    assert meta == {"method": "sentence", "budget_tokens": 10, "n_segments": 2, "joiner": " ",
                    "segment_input_tokens": [3, 4], "segment_input_words": [2, 3]}
    assert segmentation.segment("whole doc. two", "none", words_plus_prefix, None) == ["whole doc. two"]
