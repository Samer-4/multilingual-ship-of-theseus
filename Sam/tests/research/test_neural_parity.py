"""Neural metric parity with the reference libraries (needs the cached encoder snapshots; skipped otherwise).

  - LongBertScore == bert_score.score (roberta-large, layer 17, no idf) on texts that fit one window
  - ChunkEmbedder(MPNet) == sentence-transformers encode on a text that fits one window
  - long texts are chunked/windowed, never truncated
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bert_score")
pytest.importorskip("sentence_transformers")

from sot_research import neural  # noqa: E402


def _cached(repo):
    try:
        neural.snapshot(repo)
        return True
    except FileNotFoundError:
        return False


needs = pytest.mark.skipif(not (_cached("FacebookAI/roberta-large") and
                                _cached("sentence-transformers/all-mpnet-base-v2")),
                           reason="encoder snapshots not in the local Hugging Face cache")

REF = "The council said the new bridge would open in March, two years later than planned."
HYP = "According to the council, the new bridge will open in March, two years behind schedule."


@needs
def test_bertscore_matches_reference_package():
    import bert_score

    bs = neural.LongBertScore(device="cpu")
    ours = bs.score(*[e.float() for e in bs.embed([HYP, REF])])
    P, R, F = bert_score.score([HYP], [REF], model_type=str(bs.path), num_layers=17, idf=False, device="cpu",
                               lang="en")
    assert ours["bertscore_p"] == pytest.approx(P.item(), abs=1e-4)
    assert ours["bertscore_r"] == pytest.approx(R.item(), abs=1e-4)
    assert ours["bertscore_f1"] == pytest.approx(F.item(), abs=1e-4)


@needs
def test_long_text_is_windowed_not_truncated():
    bs = neural.LongBertScore(device="cpu")
    long = " ".join([REF] * 60)                                  # ~1100 tokens
    n = len(bs.tok(long, add_special_tokens=False, add_prefix_space=True)["input_ids"])
    e = bs.embed([long])[0]
    assert n > 510 and e.shape[0] == n + 2                        # every token embedded once (+ <s>, </s>)
    assert bs.windowed_texts == 1


@needs
def test_mpnet_chunk_embedder_matches_sentence_transformers():
    from sentence_transformers import SentenceTransformer

    emb = neural.ChunkEmbedder("sentence-transformers/all-mpnet-base-v2", 384, device="cpu")
    st = SentenceTransformer(str(emb.path), device="cpu")
    ours = emb.embed([REF])[0]
    ref = st.encode([REF], normalize_embeddings=True)[0]
    assert float(ours @ ref) == pytest.approx(1.0, abs=1e-5)
    long = " ".join([REF] * 40)
    chunks = emb.chunks(long)
    assert len(chunks) > 1 and all(emb._len(c) <= 384 for c in chunks)
    assert " ".join(chunks).split() == long.split()                # nothing dropped
    v = emb.embed([long])[0]
    assert np.isclose(np.linalg.norm(v), 1, atol=1e-5)
