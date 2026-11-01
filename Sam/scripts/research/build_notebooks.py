"""Generate the three research notebooks (edit this file, not the .ipynb files).

  notebooks/01_RQ1_Translation_Decay.ipynb
  notebooks/02_RQ2_Source_Attribution.ipynb
  notebooks/03_RQ3_Translator_Fingerprint.ipynb

Every notebook reads the shared research layer (src/sot_research) and the cached tables written by
scripts/research/compute_features.py; translators are discovered at run time, so adding one needs no edits here.
Usage: Sam/.venv-research/bin/python Sam/scripts/research/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[2] / "notebooks"

SETUP = r'''import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd()
while not (ROOT / "src" / "sot_research").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sot_research import analysis, classify, corpus, features, paths, plots

%matplotlib inline
plots.setup()
pd.set_option("display.width", 200, "display.max_columns", 40, "display.precision", 3)
CFG = corpus.research_config()
SEED, NB, NR = CFG["seed"], CFG["bootstrap_samples"], CFG["cv_repeats"]
print("python", sys.version.split()[0], "|", Path(sys.executable).parent.parent.name, "| pandas", pd.__version__)
'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "sot-research", "display_name": "Python (sot-research)", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    p = NB / name
    nbf.write(nb, p)
    print("wrote", p)


# ======================================================================================================== RQ1
RQ1 = [md(r'''
# RQ1 - How do English texts decay under repeated English→Spanish→English round trips?

**Question.** When a text goes through round trips T0 → S1 → T1 → S2 → T2 → S3 → T3, how much of its wording (lexical),
meaning (semantic), style and information survives, and do translators differ in that decay?

**Design.**
- **Inputs:** the same 200 XSum T0 documents (31 original articles × 7 source labels: Human + 6 LLM generators).
- **Chains:** every translator runs every document independently through three round trips. No translator ever reads
  another's output.
- **Comparisons:** each English stage is compared with the original (**T0 → Tk**, cumulative) and with the previous
  English stage (**Tk-1 → Tk**, consecutive).
- **Primary cohort:** the *matched* cohort, i.e. documents whose chain is complete (all six stages valid) for **every**
  translator. All translators are then compared on exactly the same documents. The full available cohort is
  reported as a supplement.
- **Uncertainty:** 95% confidence intervals come from an article-cluster bootstrap. The statistical unit is the
  **original article**, because the seven texts of one article share content.

**Where the numbers come from.** Shared, cached metrics (`Sam/results/research/metrics/pair_metrics.parquet`), built by
`Sam/scripts/research/compute_features.py`. Definitions are in the docstrings of `src/sot_research/metrics.py`
and `neural.py`. Translators are discovered from the finalized datasets, so a newly added translator appears here
automatically.
'''), code(SETUP + r'''
D = analysis.load(require_text=True)
P, TR, NAMES, COL, T = D["pairs"], D["translators"], D["names"], D["colors"], D["texts"]
OUT = paths.results("rq1")
TR_PAIRS = P[P.translator.isin(TR)]
MATCHED = TR_PAIRS[TR_PAIRS.matched]
BASE = P[P.translator == "baseline"]
print("translators (discovered):", [NAMES[t] for t in TR])
print("matched cohort:", len(D["matched_items"]), "documents from",
      D["cohort"][D["cohort"].matched].article_id.nunique(), "articles")
'''),
md('''## 1. Data coverage and attrition'''),
code(r'''
chains = D["chains"]
cov = (chains.groupby("translator").agg(documents=("item_id", "nunique"), complete_chains=("complete", "sum"))
       .reindex(TR))
cov["incomplete"] = cov.documents - cov.complete_chains
cov["matched_cohort"] = len(D["matched_items"])
cov.index = [NAMES[t] for t in cov.index]
display(cov)

stage_cov = pd.read_csv(paths.results("data_coverage") / "coverage_by_stage.csv")
stage_cov["translator"] = stage_cov.translator.map(NAMES)
display(stage_cov.set_index(["translator", "stage"]).unstack("stage")["valid"][corpus.STAGES])

inv = T[T.status == "invalid"][["translator", "stage", "item_id", "source_label", "validation_reason"]]
inv = inv.assign(translator=inv.translator.map(NAMES)).sort_values(["translator", "stage"])
print(f"{len(inv)} invalid records (kept as attrition, never repaired); every later stage of those chains is blocked:")
display(inv.reset_index(drop=True))
'''),
code(r'''
fig, ax = plt.subplots(figsize=(7, 3.4))
sc = pd.read_csv(paths.results("data_coverage") / "coverage_by_stage.csv")
for t in TR:
    s = sc[sc.translator == t].set_index("stage").reindex(corpus.STAGES)
    ax.plot(range(6), s["valid"], marker="o", color=COL[t], label=NAMES[t])
    ax.annotate(f"{int(s['valid'].iloc[-1])}", (5, s["valid"].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=plots.INK2)
ax.set_xticks(range(6), corpus.STAGES)
ax.set_ylabel("valid documents")
ax.set_title("Surviving valid chains by stage (200 documents per translator)")
ax.legend(loc="lower left")
plots.save(fig, "rq1_attrition_by_stage")
plt.show()
by_src = chains.groupby(["source_label", "translator"]).complete.agg(["sum", "size"])
display((by_src["size"] - by_src["sum"]).unstack("translator").rename(columns=NAMES).rename_axis(
    index="incomplete chains by source"))
'''),
md(r'''
**Missingness.** Attrition is small, at most 4 of 200 documents per translator, and it is reported, not imputed.
- **Cause:** every invalid record failed a blocking check: degenerate repetition, or a generation that hit its token
  cap.
- **Consequence:** the later stages of those chains were never generated, by design, because an invalid output is
  never fed into another round.
- **Missing data:** missing chains are *not* missing at random. Repetition loops occur in particular documents. This
  is why translator comparisons use the matched cohort.
'''),
md('''## 2. Summary statistics (T0 → Tk, matched cohort)'''),
code(r'''
CUM_METRICS = [m for m in analysis.METRICS if m in P.columns]
cum = TR_PAIRS[TR_PAIRS.kind == "cumulative"]
summ = pd.concat([analysis.cluster_ci(cum[cum.matched], ["translator", "hyp_stage"], m, n_boot=NB, seed=SEED)
                  .assign(cohort="matched") for m in CUM_METRICS] +
                 [analysis.cluster_ci(cum, ["translator", "hyp_stage"], m, n_boot=NB, seed=SEED)
                  .assign(cohort="all_available") for m in CUM_METRICS], ignore_index=True)
summ.to_csv(OUT / "summary_cumulative.csv", index=False)
base = pd.concat([analysis.cluster_ci(BASE, ["kind"], m, n_boot=NB, seed=SEED) for m in CUM_METRICS])
base.to_csv(OUT / "baselines.csv", index=False)

show = ["bleu", "chrf", "word_retention", "e5_cosine", "mpnet_cosine", "bertscore_f1", "burrows_delta",
        "entity_recall", "number_recall", "sent_coverage"]
tab = summ[(summ.cohort == "matched") & summ.metric.isin(show)].copy()
tab["value"] = tab.apply(analysis.fmt_ci, axis=1)
tab["translator"] = tab.translator.map(NAMES)
wide = tab.pivot_table(index=["metric"], columns=["translator", "hyp_stage"], values="value", aggfunc="first")
display(wide.reindex(show))
print("n pairs per cell (matched):", tab.groupby(["translator", "hyp_stage"]).n_pairs.max().to_dict())
bt = base[base.metric.isin(show)].pivot_table(index="metric", columns="kind", values="mean").reindex(show)
display(bt.rename(columns={"unrelated": "unrelated T0 pair (floor)",
                           "cross_source": "same article, other author (Human vs LLM T0)"}))
'''),
code(r'''
F = D["features"]
L = F[F.item_id.isin(D["matched_items"])]
lens = (L.groupby(["translator", "stage"]).agg(words=("n_words", "mean"), sentences=("n_sentences", "mean"),
                                               sentence_len=("sent_len_mean", "mean"), mattr=("mattr", "mean"))
        .reset_index())
lens["translator"] = lens.translator.map(lambda t: NAMES.get(t, "original T0"))
display(lens.set_index(["translator", "stage"]).round(2))
'''),
md(r'''## 3. Decay relative to the original (T0 → T1, T2, T3)

Each panel shows one metric. Lines are translator means with 95% article-cluster bootstrap bands (matched cohort).
Grey reference lines help read the scale:
- **dotted:** two *unrelated* originals, a floor for that metric;
- **dashed:** the same article written by a different author (Human vs LLM T0), roughly the similarity of an
  independent paraphrase.
'''),
code(r'''
PANELS = [("bleu", "lexical"), ("chrf", "lexical"), ("rougeL", "lexical"), ("word_retention", "lexical"),
          ("e5_cosine", "semantic"), ("mpnet_cosine", "semantic"), ("bertscore_f1", "semantic"),
          ("bertscore_f1_rescaled", "semantic"), ("fw_cosine", "style"), ("pos_jsd", "style"),
          ("burrows_delta", "style"), ("char3_cosine", "style"), ("entity_recall", "information"),
          ("number_recall", "information"), ("sent_coverage", "information"), ("length_ratio", "lexical")]
PANELS = [p for p in PANELS if p[0] in P.columns]
m = summ[summ.cohort == "matched"]
nc = 4
fig, axes = plt.subplots(int(np.ceil(len(PANELS) / nc)), nc, figsize=(13, 2.7 * np.ceil(len(PANELS) / nc)))
for ax, (met, fam) in zip(axes.flat, PANELS):
    for t in TR:
        s = m[(m.metric == met) & (m.translator == t)].sort_values("hyp_stage")
        x = s.hyp_stage.str[1].astype(int)
        ax.plot(x, s["mean"], marker="o", color=COL[t], label=NAMES[t])
        ax.fill_between(x, s.ci_low, s.ci_high, color=COL[t], alpha=0.15, linewidth=0)
    for kind, ls in (("unrelated", ":"), ("cross_source", "--")):
        b = base[(base.metric == met) & (base.kind == kind)]
        if len(b):
            ax.axhline(b["mean"].iloc[0], color=plots.INK2, ls=ls, lw=1)
    ax.set_title(f"{analysis.METRICS[met][0]}  ({fam})", fontsize=9)
    ax.set_xticks([1, 2, 3], ["T1", "T2", "T3"])
for ax in axes.flat[len(PANELS):]:
    ax.axis("off")
fig.suptitle("T0 → Tk: decay across rounds (matched cohort; dotted = unrelated floor, dashed = other-author paraphrase)",
             y=1.03, fontsize=11)
fig.tight_layout()
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=len(TR), bbox_to_anchor=(0.5, 1.01), fontsize=8)
plots.save(fig, "rq1_decay_curves")
plt.show()
'''),
md('''## 4. Consecutive-round change (Tk-1 → Tk)'''),
code(r'''
step = TR_PAIRS[(TR_PAIRS.kind == "step") & TR_PAIRS.matched].copy()
step["step"] = step.ref_stage + "→" + step.hyp_stage
STEP_METRICS = ["chrf", "bleu", "word_edit_rate", "bertscore_f1", "mpnet_cosine", "burrows_delta", "exact_equal"]
ssum = pd.concat([analysis.cluster_ci(step, ["translator", "step"], mm, n_boot=NB, seed=SEED) for mm in STEP_METRICS],
                 ignore_index=True)
ssum.to_csv(OUT / "summary_step.csv", index=False)
t = ssum.copy()
t["value"] = t.apply(analysis.fmt_ci, axis=1)
t["translator"] = t.translator.map(NAMES)
display(t.pivot_table(index="metric", columns=["translator", "step"], values="value", aggfunc="first").reindex(STEP_METRICS))

fig, axes = plt.subplots(1, 4, figsize=(13, 3))
order = ["T0→T1", "T1→T2", "T2→T3"]
for ax, met in zip(axes, ["word_edit_rate", "chrf", "bertscore_f1", "exact_equal"]):
    for tt in TR:
        s = ssum[(ssum.metric == met) & (ssum.translator == tt)].set_index("step").reindex(order)
        ax.plot(range(3), s["mean"], marker="o", color=COL[tt], label=NAMES[tt])
        ax.fill_between(range(3), s.ci_low, s.ci_high, color=COL[tt], alpha=0.15, linewidth=0)
    ax.set_xticks(range(3), order)
    ax.set_title(analysis.METRICS[met][0] + (" (share of chains)" if met == "exact_equal" else ""), fontsize=9)
axes[0].legend(fontsize=7)
fig.suptitle("Consecutive-round change (matched cohort)", y=1.03, fontsize=11)
fig.tight_layout()
plots.save(fig, "rq1_consecutive_change")
plt.show()
'''),
md('''## 5. Convergence: do chains stabilise?'''),
code(r'''
rows = []
for tt in TR:
    s = step[step.translator == tt]
    c = cum[(cum.translator == tt) & cum.matched]
    r = {"translator": NAMES[tt], "chains": s.item_id.nunique(),
         "T2 == T1 (exact)": s[s.hyp_stage == "T2"].exact_equal.mean(),
         "T3 == T2 (exact)": s[s.hyp_stage == "T3"].exact_equal.mean(),
         "edit rate T0→T1": s[s.hyp_stage == "T1"].word_edit_rate.mean(),
         "edit rate T1→T2": s[s.hyp_stage == "T2"].word_edit_rate.mean(),
         "edit rate T2→T3": s[s.hyp_stage == "T3"].word_edit_rate.mean()}
    for met in ["chrf", "bertscore_f1"]:
        mm = c.groupby("hyp_stage")[met].mean()
        r[f"round-1 share of T3 loss ({met})"] = (1 - mm["T1"]) / (1 - mm["T3"])
    floor = base[(base.metric == "bertscore_f1") & (base.kind == "unrelated")]["mean"].iloc[0]
    r["BERTScore T3, normalised to floor"] = (c[c.hyp_stage == "T3"].bertscore_f1.mean() - floor) / (1 - floor)
    rows.append(r)
conv = pd.DataFrame(rows).set_index("translator")
conv.to_csv(OUT / "convergence.csv")
display(conv.round(3))
'''),
md(r'''
Reading the convergence table:
- **Fixed points:** "T3 == T2" is the share of chains in which the third round trip reproduced the second round's text
  *exactly*, i.e. a fixed point of the translator.
- **Round-1 share of T3 loss:** the fraction of the total T0→T3 loss that already happened in the first round trip.
'''),
md('''## 6. Translator comparison on the same documents (paired, T3)'''),
code(r'''
CMP = ["chrf", "bleu", "bertscore_f1", "e5_cosine", "mpnet_cosine", "burrows_delta", "pos_jsd", "entity_recall",
       "number_recall", "sent_coverage"]
CMP = [c for c in CMP if c in P.columns]
pdiff = pd.concat([analysis.paired_differences(TR_PAIRS, mm, TR, stage=s, n_boot=NB, seed=SEED)
                   for mm in CMP for s in ("T1", "T3")], ignore_index=True)
pdiff.to_csv(OUT / "paired_differences.csv", index=False)
x = pdiff[pdiff.stage == "T3"].copy()
x["pair"] = x.a.map(NAMES) + " − " + x.b.map(NAMES)
x["diff [95% CI]"] = x.apply(lambda r: f"{r.mean_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
display(x.pivot_table(index="metric", columns="pair", values="diff [95% CI]", aggfunc="first").reindex(CMP))
print("items per comparison:", x.n_items.unique(), " articles:", x.n_articles.unique())

fig, axes = plt.subplots(1, len(x.pair.unique()), figsize=(4 * len(x.pair.unique()), 3.6), sharey=True)
axes = np.atleast_1d(axes)
for ax, (pr, g) in zip(axes, x.groupby("pair", sort=False)):
    g = g.set_index("metric").reindex(CMP[::-1])
    ax.errorbar(g.mean_diff, range(len(g)), xerr=[g.mean_diff - g.ci_low, g.ci_high - g.mean_diff], fmt="o",
                color=plots.INK, ecolor=plots.INK2, capsize=2, markersize=5)
    ax.axvline(0, color=plots.INK2, lw=1)
    ax.set_yticks(range(len(g)), [analysis.METRICS[mm][0] for mm in g.index])
    ax.set_title(pr, fontsize=9)
    ax.set_xlabel("mean paired difference at T3")
fig.suptitle("Paired translator differences at T3 (95% article-cluster CI; Δ in each metric's own units)", y=1.03,
             fontsize=11)
fig.tight_layout()
plots.save(fig, "rq1_paired_differences_T3")
plt.show()
'''),
md(r'''## 7. Information preservation: entities, numbers, sentences

These are **heuristic warnings, not validated factual errors**:
- **Entity loss:** a named entity of the original is no longer found verbatim in the output. Paraphrases and
  spelling variants also trigger it.
- **Number change:** the multiset of numbers differs. Unit conversions and reformatting (e.g. "11.2 billion" →
  "11,200 million") also count.
- **Sentence omission:** an original sentence has no output sentence above the MPNet cosine threshold.

None of these has been manually validated. The unrelated-pair row shows how often each flag fires when two texts
share no content.
'''),
code(r'''
FLAGS = [c for c in ["flag_entity_loss", "flag_number_change", "flag_sentence_omission", "flag_any_content_warning"]
         if c in P.columns]
fr = cum[cum.matched].groupby(["translator", "hyp_stage"])[FLAGS].mean()
fr = pd.concat([fr, BASE.groupby(["translator", "kind"])[FLAGS].mean().rename_axis(["translator", "hyp_stage"])])
fr.to_csv(OUT / "information_flags.csv")
disp = fr.copy()
disp.index = [(NAMES.get(a, a), b) for a, b in disp.index]
display(disp.round(3))

fig, axes = plt.subplots(1, len(FLAGS) - 1, figsize=(12, 3), sharey=True)
w = 0.8 / len(TR)
for ax, f in zip(axes, FLAGS[:-1]):
    for i, tt in enumerate(TR):
        v = fr.loc[tt][f].reindex(["T1", "T2", "T3"])
        ax.bar(np.arange(3) + (i - (len(TR) - 1) / 2) * w, v, width=w * 0.92, color=COL[tt], label=NAMES[tt])
    ax.set_xticks(range(3), ["T1", "T2", "T3"])
    ax.set_title(f.replace("flag_", "").replace("_", " ") + " (share of documents)", fontsize=9)
fig.suptitle("Heuristic information-loss warnings vs T0 (matched cohort)", y=1.08, fontsize=11)
fig.tight_layout()
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=len(TR), bbox_to_anchor=(0.5, 1.03), fontsize=8)
plots.save(fig, "rq1_information_flags")
plt.show()
'''),
code(r'''
full = pd.read_parquet(paths.local("pair_metrics_full.parquet"))
lost = full[(full.kind == "cumulative") & (full.hyp_stage == "T3") & full.matched & full.translator.isin(TR)]
top = (lost.assign(e=lost.lost_entities_json.map(json.loads)).explode("e").dropna(subset=["e"])
       .groupby("translator").e.value_counts().groupby(level=0).head(8).rename("documents").reset_index())
top["translator"] = top.translator.map(NAMES)
print("Most frequently 'lost' entity strings at T3 (inspect: many are spelling/translation variants, not losses)")
display(top.pivot_table(index=top.groupby("translator").cumcount(), columns="translator",
                        values=["e", "documents"], aggfunc="first").swaplevel(axis=1).sort_index(axis=1))
by_src = analysis.cluster_ci(cum[cum.matched & (cum.hyp_stage == "T3")], ["translator", "source_label"],
                             "bertscore_f1", n_boot=NB, seed=SEED)
by_src.to_csv(OUT / "bertscore_T3_by_source.csv", index=False)
display(by_src.pivot_table(index="source_label", columns="translator", values="mean").rename(columns=NAMES)
        .rename_axis(index="BERTScore F1, T0→T3, by original source"))
'''),
md('''## 8. Segmentation and length: the translators' design differs'''),
code(r'''
seg = cum[cum.matched].groupby(["translator", "hyp_stage"])[["length_ratio", "sentence_count_ratio",
                                                              "sent_len_ratio"]].mean()
seg.index = [(NAMES[a], b) for a, b in seg.index]
display(seg.round(3))
tt = D["translators_table"].set_index("translator")
print("Unit of translation (from configs/translators.yaml):")
for t in TR:
    print(f"  {NAMES[t]:<16} {tt.loc[t, 'model_checkpoint']}")
'''),
md(r'''
**Segmentation is a design confound.**
- **MADLAD and NLLB** translate sentence-sized segments. MADLAD packs sentences up to 300 tokens; NLLB translates one
  sentence per segment. Sentence boundaries are therefore preserved almost by construction.
- **Qwen (and any LLM translator)** translates the whole document in one prompt and can merge or split sentences.

Sentence-count and punctuation metrics reflect this design choice as well as each model's behaviour.
'''),
md('''## 9. Representative examples'''),
code(r'''
txt = T.set_index("text_id").text
f0 = D["features"][(D["features"].stage == "T0") & D["features"].item_id.isin(D["matched_items"])]
ex_item = f0.assign(d=(f0.n_words - 70).abs()).sort_values(["d", "item_id"]).item_id.iloc[0]
print("Typical short document:", ex_item)
print(f"\nT0: {analysis.excerpt(txt[ex_item + '|original|T0'], 70)}")
for t in TR:
    for k in (1, 3):
        print(f"\n{NAMES[t]} T{k}: {analysis.excerpt(txt[f'{ex_item}|{t}|T{k}'], 70)}")
'''),
code(r'''
cols = ["bertscore_f1", "chrf", "entity_recall", "number_recall"]
ex_rows = []
for t in TR:
    c = cum[(cum.translator == t) & cum.matched & (cum.hyp_stage == "T3")].sort_values("bertscore_f1")
    worst = c.iloc[0]
    ex_rows.append({"translator": t, "kind": "largest semantic drop", "item_id": worst.item_id, **worst[cols].to_dict()})
    print(f"\n=== {NAMES[t]}: largest BERTScore drop at T3 ({worst.item_id}, {worst.ref_words} words) ===")
    print(worst[cols].round(3).to_dict())
    print("T0:", analysis.excerpt(txt[worst.ref_text_id], 45))
    print("T3:", analysis.excerpt(txt[worst.hyp_text_id], 45))
pd.DataFrame(ex_rows).to_csv(OUT / "examples.csv", index=False)
'''),
md('''## 10. Key results'''),
code(r'''
m3 = summ[(summ.cohort == "matched") & (summ.hyp_stage == "T3")].set_index(["metric", "translator"])
m1 = summ[(summ.cohort == "matched") & (summ.hyp_stage == "T1")].set_index(["metric", "translator"])
lines = []
for t in TR:
    lines.append(f"- **{NAMES[t]}**: chrF {m1.loc[('chrf', t), 'mean']:.3f} (T1) → {m3.loc[('chrf', t), 'mean']:.3f} (T3); "
                 f"BERTScore F1 {m1.loc[('bertscore_f1', t), 'mean']:.3f} → {m3.loc[('bertscore_f1', t), 'mean']:.3f}; "
                 f"entity recall {m3.loc[('entity_recall', t), 'mean']:.3f} at T3; "
                 f"T3 identical to T2 in {conv.loc[NAMES[t], 'T3 == T2 (exact)']:.0%} of chains.")
display(Markdown("\n".join(lines) + f"\n\n(n = {len(D['matched_items'])} matched documents per translator and stage, "
                 f"{D['cohort'][D['cohort'].matched].article_id.nunique()} articles)"))
'''),
md('''RQ1_INTERPRETATION'''),
]

# ======================================================================================================== RQ2
RQ2 = [md(r'''
# RQ2 - Can the original source of a text still be identified after translation?

**Question.** Each original T0 text was written either by a human (XSum) or by one of six LLM generators. A classifier
is trained to recognise this **source label** on original T0 texts. How well does it still recognise the source
after one, two or three English→Spanish→English round trips through each translator?

**Protocol (no leakage).**
- **Classes:** the seven actual source labels (Human, BigScience, Eleuther-AI, LLAMA, OpenAI, PaLM, Tsinghua).
  Chance level is 1/7.
- **Splitting:** folds are formed over **original article IDs** (31 articles, 5 folds, 5 shufflings). An article's
  texts (all 7 sources, every translator, every round) are always together on one side.
- **Training:** **original T0 texts of the training articles only**. The model never sees a translation.
- **Evaluation:** held-out articles' T0 texts, and each translator's T1/T2/T3 of the same held-out articles.
- **Models:** reproducible classical baselines with fixed (untuned) hyperparameters:
  - TF-IDF word 1-2-grams + logistic regression;
  - TF-IDF character 2-5-grams + logistic regression;
  - stylometric features + logistic regression (function words, punctuation, POS, dependency labels, sentence
    and word statistics);
  - a length-only control.
- **Primary cohort:** the matched cohort (documents complete for every translator), so every translator is scored on
  the same documents. All available texts are reported as a supplement.
- **Uncertainty:** the spread over the 5 shufflings (SD), and a 95% article-cluster bootstrap CI.
'''), code(SETUP + r'''
D = analysis.load(require_text=True, pairs=False)
TR, NAMES, COL = D["translators"], D["names"], D["colors"]
OUT = paths.results("rq2")
F = D["features"]
STYLE = features.style_columns(F)
en = corpus.english(D["texts"]).merge(F[["text_id", "n_words", "n_sentences"] + STYLE], on="text_id", how="left")
en = classify.add_length_cols(en)
LABELS = [s for s in plots.SOURCE_ORDER if s in set(en.source_label)] + sorted(set(en.source_label) - set(plots.SOURCE_ORDER))
CHANCE = 1 / len(LABELS)
train_pool = en[en.stage == "T0"]
EVAL = {"original|T0": train_pool}
for t in TR:
    for k in (1, 2, 3):
        EVAL[f"{t}|T{k}"] = en[(en.translator == t) & (en.stage == f"T{k}")]
print(f"{len(LABELS)} classes, chance = {CHANCE:.3f}; {len(STYLE)} stylometric features")
print("training pool: T0 of", train_pool.item_id.nunique(), "documents /", train_pool.article_id.nunique(), "articles")
display(pd.Series({k: len(v) for k, v in EVAL.items()}, name="texts available").to_frame().T)
'''),
md('''## 1. Confound audit: the shared opening and text length'''),
code(r'''
en["opening_words"] = classify.shared_opening_lengths(en)
t0 = en[en.stage == "T0"]
chk = t0[~t0.is_human & t0.shared_prefix_words_with_human.notna()]
agree = (chk.opening_words >= chk.shared_prefix_words_with_human).mean()
print(f"T0: measured shared opening >= manifest's shared prefix with the Human text in {agree:.0%} of LLM texts")
op = en.groupby(["translator", "stage", "source_label"]).opening_words.median().unstack("source_label")[LABELS]
op.index = [(NAMES.get(a, "original"), b) for a, b in op.index]
display(op.rename_axis(columns="median shared-opening words"))
en.groupby(["translator", "stage"]).opening_words.describe().to_csv(OUT / "shared_opening_audit.csv")
L0 = t0.groupby("source_label").n_words.describe()[["mean", "50%", "min", "max"]].reindex(LABELS)
display(L0.rename_axis(index="T0 length (words) by source"))
'''),
md(r'''
**What the audit shows.**
- **The shared opening:** the six LLM sources were generated from the first ~30 words of the human article, so at T0
  almost every LLM text starts with the same ~30 words as the Human text of its article. That opening carries
  article content, not source style; a classifier can only use it indirectly, e.g. through how a generator continues
  it. Its measured length per translator and stage is shown above.
- **Translations:** a sentence-level MT system reproduces the shared opening identically across sources. A
  whole-document LLM translator may render it slightly differently in each text, so its measured shared opening is
  shorter.
- **The ablation:** section 4 removes the measured opening from every text (training and test) and repeats the
  experiment.
- **Length:** text length differs by source, so a length-only control model is included.
'''),
md('''## 2. Grouped cross-validation (train on T0, test on held-out articles' T0/T1/T2/T3)'''),
code(r'''
MODELS = ["tfidf_word", "tfidf_char", "stylometric", "length_only"]
COLS = {"stylometric": STYLE, "length_only": classify.LENGTH_COLS}
def run_all(pool, evals, tag):
    out = []
    for mname in MODELS:
        out.append(classify.grouped_cv(pool, evals, "source_label", mname, cols=COLS.get(mname),
                                       n_splits=CFG["cv_folds"], repeats=NR, seed=SEED).assign(ablation=tag))
    p = pd.concat(out, ignore_index=True)
    p[["translator", "stage"]] = p.eval_set.str.split("|", expand=True)
    return p
preds = run_all(train_pool, EVAL, "full_text")
preds["matched"] = preds.item_id.isin(D["matched_items"])
splits = pd.DataFrame([{"repeat": r, "seed": SEED + r, "article_id": a, "fold": f}
                       for r in range(NR) for a, f in classify.article_folds(en.article_id, CFG["cv_folds"], SEED + r).items()])
splits.to_csv(OUT / "splits.csv", index=False)
print(preds.groupby(["model"]).size().rename("predictions"))
'''),
code(r'''
BY = ["model", "translator", "stage"]
sc = classify.scores(preds[preds.matched], BY, LABELS, n_boot=1000, seed=SEED).assign(cohort="matched")
sc_all = classify.scores(preds, BY, LABELS, n_boot=1000, seed=SEED).assign(cohort="all_available")
scores = pd.concat([sc, sc_all], ignore_index=True)
scores.to_csv(OUT / "scores.csv", index=False)
t = sc.copy()
t["macro-F1 [95% CI]"] = t.apply(lambda r: f"{r.macro_f1:.3f} [{r.macro_f1_ci_low:.3f}, {r.macro_f1_ci_high:.3f}]", axis=1)
t["accuracy (±SD)"] = t.apply(lambda r: f"{r.accuracy:.3f} ± {r.accuracy_sd:.3f}", axis=1)
t["translator"] = t.translator.map(lambda x: NAMES.get(x, "original"))
display(t.pivot_table(index=["translator", "stage"], columns="model", values="macro-F1 [95% CI]", aggfunc="first")[MODELS])
display(t.pivot_table(index=["translator", "stage"], columns="model", values="accuracy (±SD)", aggfunc="first")[MODELS])
print("texts per cell (matched):", sorted(sc.n_texts.unique()), "| articles:", sorted(sc.n_articles.unique()),
      "| chance:", round(CHANCE, 3))
'''),
code(r'''
def curve_plot(s, title, fname, metric="macro_f1"):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.4 * len(MODELS), 3.2), sharey=True)
    for ax, mname in zip(axes, MODELS):
        g = s[s.model == mname]
        t0v = g[g.translator == "original"].iloc[0]
        for t in TR:
            h = g[g.translator == t].sort_values("stage")
            x = [0] + h.stage.str[1].astype(int).tolist()
            y = [t0v[metric]] + h[metric].tolist()
            lo = [t0v[metric + "_ci_low"]] + h[metric + "_ci_low"].tolist()
            hi = [t0v[metric + "_ci_high"]] + h[metric + "_ci_high"].tolist()
            ax.plot(x, y, marker="o", color=COL[t], label=NAMES[t])
            ax.fill_between(x, lo, hi, color=COL[t], alpha=0.12, linewidth=0)
        ax.axhline(CHANCE, color=plots.INK2, ls=":", lw=1)
        ax.text(3, CHANCE, " chance", va="bottom", ha="right", fontsize=7, color=plots.INK2)
        ax.set_xticks(range(4), ["T0", "T1", "T2", "T3"])
        ax.set_title(mname, fontsize=9)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel(metric.replace("_", "-"))
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(title, y=1.03, fontsize=11)
    fig.tight_layout()
    plots.save(fig, fname)
    plt.show()
curve_plot(sc, "Source attribution after translation (trained on T0 only; matched cohort, 95% CI)", "rq2_macro_f1_by_round")
'''),
md('''## 3. Per-class results and confusion matrices'''),
code(r'''
best = sc[sc.translator == "original"].sort_values("macro_f1", ascending=False).model.iloc[0]
print("best model on original T0:", best)
pc = classify.per_class(preds[preds.matched], BY, LABELS)
pc.to_csv(OUT / "per_class.csv", index=False)
b = pc[pc.model == best].copy()
b["set"] = b.translator.map(lambda x: NAMES.get(x, "original")) + " " + b.stage
heat = b.pivot_table(index="class", columns="set", values="f1").reindex(LABELS)
heat = heat[["original T0"] + [f"{NAMES[t]} T{k}" for t in TR for k in (1, 2, 3)]]
fig, ax = plt.subplots(figsize=(1.0 + 0.75 * heat.shape[1], 3.6))
im = ax.imshow(heat.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(heat.shape[1]), heat.columns, rotation=45, ha="right")
ax.set_yticks(range(len(LABELS)), LABELS)
ax.grid(False)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.iat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else plots.INK)
fig.colorbar(im, ax=ax, fraction=0.03, label="F1")
ax.set_title(f"Per-class F1, {best} (matched cohort, pooled over {NR} repeats)")
plots.save(fig, "rq2_per_class_f1")
plt.show()
display(b[b.translator == "original"][["class", "precision", "recall", "f1", "support_per_repeat"]].set_index("class"))
'''),
code(r'''
sets = [("original", "T0")] + [(t, "T3") for t in TR]
fig, axes = plt.subplots(1, len(sets), figsize=(3.3 * len(sets), 3.4))
conf_rows = []
for ax, (t, s) in zip(axes, sets):
    part = preds[preds.matched & (preds.model == best) & (preds.translator == t) & (preds.stage == s)]
    cm = classify.confusion(part, LABELS)
    raw = classify.confusion(part, LABELS, normalize=False)
    conf_rows.append(raw.stack().rename("count").reset_index().assign(model=best, translator=t, stage=s))
    ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), [l[:6] for l in LABELS], rotation=60, fontsize=7)
    ax.set_yticks(range(len(LABELS)), LABELS if ax is axes[0] else [""] * len(LABELS), fontsize=7)
    ax.grid(False)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = cm.iat[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else plots.INK)
    ax.set_title(f"{NAMES.get(t, 'original')} {s}", fontsize=9)
    ax.set_xlabel("predicted")
axes[0].set_ylabel("true source")
fig.suptitle(f"Row-normalised confusion matrices, {best} (matched cohort, pooled over repeats)", y=1.02, fontsize=11)
fig.tight_layout()
plots.save(fig, "rq2_confusion_T0_T3")
plt.show()
for t in TR:
    for s in ("T1", "T2"):
        part = preds[preds.matched & (preds.model == best) & (preds.translator == t) & (preds.stage == s)]
        conf_rows.append(classify.confusion(part, LABELS, normalize=False).stack().rename("count").reset_index()
                         .assign(model=best, translator=t, stage=s))
pd.concat(conf_rows).to_csv(OUT / "confusion_counts.csv", index=False)
'''),
md(r'''## 4. Ablation: shared opening removed

The measured shared opening (section 1) is removed from **every** text: training T0 texts and all test texts, for
every source including Human. The full experiment is then repeated. Stylometric features are recomputed on the
stripped texts with the same cached spaCy pipeline. Original texts are not modified.
'''),
code(r'''
strip = classify.strip_opening(en[["text_id", "item_id", "article_id", "source_label", "translator", "stage", "text"]],
                               en.opening_words)
strip["text_sha256"] = strip.text.map(corpus.sha256_text)
refl = json.loads((paths.results("shared_features") / "reference_lists.json").read_text())
sf = features.compute(strip, refl, n_process=1, log=lambda m: None)
strip = strip.merge(sf[["text_sha256", "n_words", "n_sentences"] + STYLE], on="text_sha256", how="left")
strip = classify.add_length_cols(strip)
print("words removed per text: mean", round(en.opening_words.mean(), 1), "| texts left empty:", int((strip.text.str.len() == 0).sum()))
EVAL_S = {k: strip[strip.text_id.isin(v.text_id)] for k, v in EVAL.items()}
preds_s = run_all(strip[strip.stage == "T0"], EVAL_S, "opening_removed")
preds_s["matched"] = preds_s.item_id.isin(D["matched_items"])
sc_s = classify.scores(preds_s[preds_s.matched], BY, LABELS, n_boot=1000, seed=SEED).assign(cohort="matched")
sc_s.to_csv(OUT / "scores_opening_removed.csv", index=False)
cmp_ = sc.merge(sc_s, on=BY, suffixes=("_full", "_ablated"))
cmp_["delta_macro_f1"] = cmp_.macro_f1_ablated - cmp_.macro_f1_full
cmp_["translator"] = cmp_.translator.map(lambda x: NAMES.get(x, "original"))
display(cmp_.pivot_table(index=["translator", "stage"], columns="model",
                         values=["macro_f1_full", "macro_f1_ablated"]).round(3))
curve_plot(sc_s, "ABLATION: shared opening removed from every text (matched cohort, 95% CI)", "rq2_macro_f1_opening_removed")
pd.concat([preds, preds_s]).to_parquet(OUT / "predictions.parquet", index=False)
'''),
md('''## 5. Key results'''),
code(r'''
def get(s, m, t, st):
    r = s[(s.model == m) & (s.translator == t) & (s.stage == st)].iloc[0]
    return f"{r.macro_f1:.3f} [{r.macro_f1_ci_low:.3f}, {r.macro_f1_ci_high:.3f}]"
lines = [f"- Best model on original T0 (**{best}**): macro-F1 {get(sc, best, 'original', 'T0')} (chance {CHANCE:.3f})."]
for t in TR:
    lines.append(f"- {NAMES[t]}: T1 {get(sc, best, t, 'T1')}, T3 {get(sc, best, t, 'T3')}; "
                 f"opening removed T3 {get(sc_s, best, t, 'T3')}.")
lines.append(f"- Length-only control on T0: {get(sc, 'length_only', 'original', 'T0')}.")
display(Markdown("\n".join(lines)))
json.dump({"labels": LABELS, "models": MODELS, "best_model": best, "seed": SEED, "folds": CFG["cv_folds"],
           "repeats": NR, "train": "T0 of training articles", "translators": TR, "chance": CHANCE,
           "hyperparameters": "fixed; see src/sot_research/classify.py make_model"},
          open(OUT / "config.json", "w"), indent=1)
'''),
md('''RQ2_INTERPRETATION'''),
]

# ======================================================================================================== RQ3
RQ3 = [md(r'''
# RQ3 - Do translators leave a fingerprint? Identifying the translation model from the text

**Question.** Given an English text produced by a round trip (T1, T2 or T3), can a classifier tell which translator
produced it?

**Protocol.**
- **Classes:** the translators discovered from the finalized datasets (currently MADLAD-400-3B, NLLB-200-1.3B and
  Qwen3-4B). Chance is computed as 1/K at run time, and a new translator (e.g. Phi-4-mini) is added automatically.
- **Inputs:** only the **English translated text**. Model names, metadata and source labels are never features.
- **Balance:** the matched cohort, i.e. documents complete for every translator, so each translator contributes the
  *same* documents at every round and classes are exactly balanced.
- **Splitting:** grouped folds over **original article IDs** (5 folds × 5 shufflings). One article's texts (every
  source, translator and round) are always on the same side, so the classifier can never match a test text to a
  training text of the same content.
- **Primary experiment:** train on the pooled T1-T3 texts of the training articles, then evaluate each round
  separately. Further experiments train per round and cross-round.
- **Uncertainty:** the 31 articles, not individual texts, are the resampling unit in the bootstrap CIs.
'''), code(SETUP + r'''
D = analysis.load(require_text=True, pairs=False)
TR, NAMES, COL = D["translators"], D["names"], D["colors"]
OUT = paths.results("rq3")
F = D["features"]
STYLE = features.style_columns(F)
en = corpus.english(D["texts"]).merge(F[["text_id", "n_words", "n_sentences"] + STYLE], on="text_id", how="left")
en = classify.add_length_cols(en)
pool = en[en.translator.isin(TR) & en.stage.isin(["T1", "T2", "T3"]) & en.item_id.isin(D["matched_items"])].copy()
K = len(TR)
CHANCE = 1 / K
print(f"K = {K} translators -> chance = {CHANCE:.3f}")
bal = pool.groupby(["stage", "translator"]).size().unstack().rename(columns=NAMES)
display(bal.rename_axis(index="texts per stage"))
assert bal.nunique(axis=1).eq(1).all(), "classes must be balanced (same documents per translator)"
print("articles:", pool.article_id.nunique(), "| documents:", pool.item_id.nunique(), "| texts:", len(pool))
'''),
md('''## 1. Descriptive confound check: length and segmentation'''),
code(r'''
t0w = en[en.stage == "T0"].set_index("item_id")[["n_words", "n_sentences"]]
d = pool.join(t0w, on="item_id", rsuffix="_t0")
d["word_ratio"] = d.n_words / d.n_words_t0
d["sentence_ratio"] = d.n_sentences / d.n_sentences_t0
desc = d.groupby(["translator", "stage"])[["n_words", "word_ratio", "sentence_ratio", "sent_len_mean",
                                          "punct_01", "mattr"]].mean()
desc.index = [(NAMES[a], b) for a, b in desc.index]
display(desc.rename(columns={"punct_01": "comma rate"}).round(3))
desc.to_csv(OUT / "length_segmentation_descriptives.csv")
print("Sources are balanced by construction (same documents per translator):")
display(pool.groupby(["source_label", "translator"]).size().unstack().rename(columns=NAMES).head(8))
'''),
md('''## 2. Main experiment: train on pooled T1-T3, evaluate per round'''),
code(r'''
SENT = ["sent_len_mean", "sent_len_sd", "punct_00", "dep_ROOT", "cap_rate"]
STYLE_NOSENT = [c for c in STYLE if c not in SENT]
MODELS = {"tfidf_word": None, "tfidf_char": None, "stylometric": STYLE, "stylometric_no_sentence": STYLE_NOSENT,
          "length_only": classify.LENGTH_COLS}
def model_kind(m):
    return "stylometric" if m.startswith("stylometric") else m
EVALS = {s: pool[pool.stage == s] for s in ["T1", "T2", "T3"]}
runs = []
for mname, cols in MODELS.items():
    p = classify.grouped_cv(pool, EVALS, "translator", model_kind(mname), cols=cols, n_splits=CFG["cv_folds"],
                            repeats=NR, seed=SEED)
    runs.append(p.assign(model=mname))
preds = pd.concat(runs, ignore_index=True)
preds["stage"] = preds.eval_set
LAB = TR
sc = classify.scores(preds, ["model", "stage"], LAB, n_boot=1000, seed=SEED)
sc_all = classify.scores(preds, ["model"], LAB, n_boot=1000, seed=SEED).assign(stage="T1-T3")
scores = pd.concat([sc, sc_all], ignore_index=True)
scores.to_csv(OUT / "scores_main.csv", index=False)
t = scores.copy()
t["macro-F1 [95% CI]"] = t.apply(lambda r: f"{r.macro_f1:.3f} [{r.macro_f1_ci_low:.3f}, {r.macro_f1_ci_high:.3f}]", axis=1)
t["accuracy (±SD)"] = t.apply(lambda r: f"{r.accuracy:.3f} ± {r.accuracy_sd:.3f}", axis=1)
display(t.pivot_table(index="model", columns="stage", values="macro-F1 [95% CI]", aggfunc="first").reindex(list(MODELS)))
display(t.pivot_table(index="model", columns="stage", values="accuracy (±SD)", aggfunc="first").reindex(list(MODELS)))
print("texts per round:", sorted(sc.n_texts.unique()), "| articles:", sorted(sc.n_articles.unique()), "| chance:", round(CHANCE, 3))
splits = pd.DataFrame([{"repeat": r, "seed": SEED + r, "article_id": a, "fold": f}
                       for r in range(NR) for a, f in classify.article_folds(pool.article_id, CFG["cv_folds"], SEED + r).items()])
splits.to_csv(OUT / "splits.csv", index=False)
'''),
code(r'''
fig, ax = plt.subplots(figsize=(7.5, 3.4))
names = list(MODELS)
w = 0.8 / 3
for i, s in enumerate(["T1", "T2", "T3"]):
    g = sc[sc.stage == s].set_index("model").reindex(names)
    x = np.arange(len(names)) + (i - 1) * w
    ax.bar(x, g.macro_f1, width=w * 0.92, color=["#9ec5f0", "#5a9be4", "#2a78d6"][i], label=s)
    ax.errorbar(x, g.macro_f1, yerr=[g.macro_f1 - g.macro_f1_ci_low, g.macro_f1_ci_high - g.macro_f1], fmt="none",
                ecolor=plots.INK2, capsize=2, lw=1)
ax.axhline(CHANCE, color=plots.INK2, ls=":", lw=1)
ax.text(len(names) - 0.5, CHANCE, f"chance 1/{K}", va="bottom", ha="right", fontsize=7, color=plots.INK2)
ax.set_xticks(range(len(names)), names, rotation=15)
ax.set_ylim(0, 1.05)
ax.set_ylabel("macro-F1")
ax.legend(title="evaluated on", fontsize=7, title_fontsize=7)
ax.set_title("Translator identification by round (trained on pooled T1-T3; 95% article-cluster CI)")
plots.save(fig, "rq3_macro_f1_by_model_round")
plt.show()
'''),
md('''## 3. Confusion matrices and per-class results'''),
code(r'''
best = sc_all.sort_values("macro_f1", ascending=False).model.iloc[0]
print("best model (pooled):", best)
pc = classify.per_class(preds, ["model", "stage"], LAB)
pc.to_csv(OUT / "per_class.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(3.2 * 3 + 0.5, 3.2))
conf = []
for ax, s in zip(axes, ["T1", "T2", "T3"]):
    part = preds[(preds.model == best) & (preds.stage == s)]
    cm = classify.confusion(part, LAB)
    conf.append(classify.confusion(part, LAB, normalize=False).stack().rename("count").reset_index().assign(stage=s, model=best))
    ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(K), [NAMES[x] for x in LAB], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(K), [NAMES[x] for x in LAB] if s == "T1" else [""] * K, fontsize=7)
    for i in range(K):
        for j in range(K):
            v = cm.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else plots.INK)
    ax.set_title(f"evaluated on {s}", fontsize=9)
axes[0].set_ylabel("true translator")
fig.suptitle(f"Row-normalised confusion, {best} (pooled over {NR} repeats)", y=1.03, fontsize=11)
fig.tight_layout()
plots.save(fig, "rq3_confusion_by_round")
plt.show()
pd.concat(conf).to_csv(OUT / "confusion_counts.csv", index=False)
x = pc[pc.model == best].copy()
x["class"] = x["class"].map(NAMES)
display(x.pivot_table(index="class", columns="stage", values=["precision", "recall", "f1"]).round(3))
'''),
md('''## 4. Per-round training and cross-round generalisation'''),
code(r'''
xr = []
for mname in ["tfidf_char", "stylometric"]:
    for s_train in ["T1", "T2", "T3"]:
        p = classify.grouped_cv(pool[pool.stage == s_train], EVALS, "translator", model_kind(mname),
                                cols=MODELS[mname], n_splits=CFG["cv_folds"], repeats=NR, seed=SEED)
        xr.append(p.assign(model=mname, train_stage=s_train, stage=p.eval_set))
xr = pd.concat(xr, ignore_index=True)
xs = classify.scores(xr, ["model", "train_stage", "stage"], LAB, n_boot=500, seed=SEED)
xs.to_csv(OUT / "scores_cross_round.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, mname in zip(axes, ["tfidf_char", "stylometric"]):
    m = xs[xs.model == mname].pivot(index="train_stage", columns="stage", values="macro_f1")
    ax.imshow(m.to_numpy(), cmap="Blues", vmin=CHANCE, vmax=1)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m.iat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if m.iat[i, j] > 0.8 else plots.INK)
    ax.set_xticks(range(3), m.columns)
    ax.set_yticks(range(3), m.index)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("trained on")
    ax.set_title(mname, fontsize=9)
fig.suptitle("Cross-round macro-F1 (grouped by article)", y=1.03, fontsize=11)
fig.tight_layout()
plots.save(fig, "rq3_cross_round")
plt.show()
'''),
md('''## 5. Most predictive features (models fit on all matched texts; interpretation only)'''),
code(r'''
imp = []
for mname, kind in [("tfidf_word", "text"), ("stylometric", "features")]:
    k, est = classify.make_model(mname)
    X = pool["text"] if k == "text" else pool[STYLE].to_numpy(float)
    est.fit(X, pool["translator"])
    imp.append(classify.top_features(est, k, STYLE if k == "features" else None, k=12).assign(model=mname))
imp = pd.concat(imp)
imp.to_csv(OUT / "top_features.csv", index=False)
tw = imp[imp.model == "tfidf_word"].copy()
tw["rank"] = tw.groupby("class").cumcount() + 1
display(tw.assign(**{"class": tw["class"].map(NAMES)}).pivot(index="rank", columns="class", values="feature"))
st = imp[imp.model == "stylometric"]
fig, axes = plt.subplots(1, K, figsize=(3.6 * K, 3.4))
for ax, c in zip(np.atleast_1d(axes), LAB):
    g = st[st["class"] == c].iloc[::-1]
    ax.barh(range(len(g)), g.coef, color=COL[c], height=0.7)
    ax.set_yticks(range(len(g)), [classify.clean_feature_name(f) for f in g.feature], fontsize=7)
    ax.set_title(NAMES[c], fontsize=9)
    ax.set_xlabel("LR coefficient (standardised)")
fig.suptitle("Stylometric features most indicative of each translator", y=1.03, fontsize=11)
fig.tight_layout()
plots.save(fig, "rq3_top_stylometric_features")
plt.show()
'''),
md('''## 6. Confound checks: source, length and segmentation'''),
code(r'''
bysrc = classify.scores(preds[preds.model == best], ["source_label"], LAB, n_boot=500, seed=SEED)
display(bysrc[["source_label", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "n_texts"]]
        .set_index("source_label").round(3).rename_axis(index=f"{best}, by original source"))
loso = []
srcs = sorted(pool.source_label.unique())
for mname in ["tfidf_char", "stylometric"]:
    for s in srcs:
        p = classify.grouped_cv(pool, {s: pool[pool.source_label == s]}, "translator", model_kind(mname),
                                cols=MODELS[mname], n_splits=CFG["cv_folds"], repeats=2, seed=SEED,
                                train_filter=lambda tr, s=s: tr[tr.source_label != s])
        loso.append(p.assign(model=mname, held_out_source=s))
loso = pd.concat(loso, ignore_index=True)
ls = classify.scores(loso, ["model", "held_out_source"], LAB, n_boot=300, seed=SEED)
ls.to_csv(OUT / "scores_leave_one_source_out.csv", index=False)
display(ls.pivot_table(index="held_out_source", columns="model", values="macro_f1").round(3)
        .rename_axis(index="leave-one-source-out macro-F1"))
bysrc.to_csv(OUT / "scores_by_source.csv", index=False)

# Typography: curly quotes/apostrophes and long dashes are a surface signal any char model can pick up.
TYPO_CHARS = "\u2018\u2019\u201c\u201d\u2013\u2014"
TYPO = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})
typo_share = pool.groupby(["translator", "stage"]).text.apply(lambda x: x.str.contains(f"[{TYPO_CHARS}]").mean())
display(typo_share.unstack().rename(index=NAMES).round(3).rename_axis(index="share of texts with typographic quotes/dashes"))
t0_typo = en[en.stage == "T0"].text.str.contains(f"[{TYPO_CHARS}]").mean()
print(f"original T0 texts with typographic quotes/dashes: {t0_typo:.3f}")
pool_n = pool.assign(text=pool.text.str.translate(TYPO))
NO_TYPO = [c for c in STYLE if c not in ("punct_11", "punct_12", "punct_13", "punct_14", "punct_15", "punct_16")]
typo_runs = [classify.grouped_cv(pool_n, {x: pool_n[pool_n.stage == x] for x in ["T1", "T2", "T3"]}, "translator",
                                 "tfidf_char", n_splits=CFG["cv_folds"], repeats=NR, seed=SEED).assign(model="tfidf_char_typography_normalised"),
             classify.grouped_cv(pool, EVALS, "translator", "stylometric", cols=NO_TYPO, n_splits=CFG["cv_folds"],
                                 repeats=NR, seed=SEED).assign(model="stylometric_no_typography")]
tp = pd.concat(typo_runs, ignore_index=True)
tp["stage"] = tp.eval_set
ts = pd.concat([classify.scores(tp, ["model", "stage"], LAB, n_boot=500, seed=SEED),
                classify.scores(tp, ["model"], LAB, n_boot=500, seed=SEED).assign(stage="T1-T3")])
ts.to_csv(OUT / "scores_typography_controls.csv", index=False)
scores = pd.concat([scores, ts], ignore_index=True)
scores.to_csv(OUT / "scores_main.csv", index=False)
conf_tab = scores[scores.stage == "T1-T3"].set_index("model")[["macro_f1", "macro_f1_ci_low", "macro_f1_ci_high"]]
display(conf_tab.round(3).rename_axis(index="pooled macro-F1: full vs controls"))
'''),
md(r'''
The confound checks above ask:
- **Length only:** can text length and sentence/word statistics alone identify the translator?
- **Segmentation:** does the stylometric model still work without sentence-boundary features
  (`stylometric_no_sentence` drops sentence-length statistics, the full-stop rate, the ROOT-dependency rate and the
  capitalisation rate)?
- **Source:** does a fingerprint learned on six source labels transfer to texts of the seventh, unseen source
  (leave-one-source-out)?
- **Typography:** does the fingerprint survive when curly quotes, apostrophes and long dashes are normalised to
  ASCII (character model), or when those punctuation rates are removed (stylometric model)?
'''),
md('''## 7. Key results'''),
code(r'''
def get(s, m, st):
    r = s[(s.model == m) & (s.stage == st)].iloc[0]
    return f"{r.macro_f1:.3f} [{r.macro_f1_ci_low:.3f}, {r.macro_f1_ci_high:.3f}]"
lines = [f"- K = {K} translators (chance {CHANCE:.3f}); {pool.item_id.nunique()} matched documents, "
         f"{pool.article_id.nunique()} articles, {len(pool)} texts."]
for m in MODELS:
    lines.append(f"- **{m}**: T1 {get(sc, m, 'T1')}, T2 {get(sc, m, 'T2')}, T3 {get(sc, m, 'T3')}.")
display(Markdown("\n".join(lines)))
preds.to_parquet(OUT / "predictions.parquet", index=False)
json.dump({"classes": TR, "chance": CHANCE, "models": list(MODELS), "best_model": best, "seed": SEED,
           "folds": CFG["cv_folds"], "repeats": NR, "cohort": "matched", "texts": int(len(pool))},
          open(OUT / "config.json", "w"), indent=1)
'''),
md('''RQ3_INTERPRETATION'''),
]

INTERP = {}
_interp_file = Path(__file__).with_name("interpretations.md")
if _interp_file.exists():
    cur = None
    for line in _interp_file.read_text().splitlines(keepends=True):
        if line.startswith("@@ "):
            cur = line[3:].strip()
            INTERP[cur] = ""
        elif cur:
            INTERP[cur] += line


def fill(cells):
    for c in cells:
        if c.cell_type == "markdown" and c.source.strip() in INTERP:
            c.source = INTERP[c.source.strip()].strip()
    return cells


if __name__ == "__main__":
    write("01_RQ1_Translation_Decay.ipynb", fill(RQ1))
    write("02_RQ2_Source_Attribution.ipynb", fill(RQ2))
    write("03_RQ3_Translator_Fingerprint.ipynb", fill(RQ3))
