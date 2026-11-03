"""Usage (from Sam/): python scripts/build_report.py . SoT_Results_Report.html

Build a self-contained HTML report of Sam's SoT results (RQ1-RQ3) from the committed CSVs.
No JavaScript, no external assets: charts are CSS bars, hover uses title attributes."""
import sys, html
from pathlib import Path
import pandas as pd

ROOT = Path(sys.argv[1])          # Sam/
OUT = Path(sys.argv[2])
R = ROOT / "results"

NAME = {"chatgpt": "ChatGPT", "palm": "PaLM", "dipper": "Dipper", "pegasus(full)": "Pegasus",
        "madlad-3b": "MADLAD-400-3B", "nllb": "NLLB-200-1.3B", "qwen3-4b": "Qwen3-4B", "original": "Original (T0)"}
TRANS = {"madlad-3b", "nllb", "qwen3-4b"}
PARA = ["chatgpt", "palm", "dipper", "pegasus(full)"]


def esc(s):
    return html.escape(str(s))


def bars(rows, lo=0.0, hi=1.0, legend=None, fmt="{:.2f}"):
    """rows: list of (label, [(value, series_class, tooltip), ...]). Horizontal bars on one shared scale."""
    out = ['<div class="chart">']
    if legend:
        out.append('<div class="legend">' + "".join(
            f'<span><i class="sw {c}"></i>{esc(t)}</span>' for c, t in legend) + "</div>")
    for label, items in rows:
        out.append(f'<div class="row"><div class="lab">{esc(label)}</div><div class="bars">')
        for v, cls, tip in items:
            w = max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100
            out.append(f'<div class="bar-line" title="{esc(tip)}"><div class="track">'
                       f'<div class="bar {cls}" style="width:{w:.1f}%"></div></div>'
                       f'<span class="val">{fmt.format(v)}</span></div>')
        out.append("</div></div>")
    out.append(f'<div class="axis"><span>{lo:g}</span><span>{(lo+hi)/2:g}</span><span>{hi:g}</span></div></div>')
    return "\n".join(out)


def cell(v):
    if isinstance(v, float):
        return f"{v:.2f}"
    return v


def table(df, cls=""):
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(cell(v))}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


# ---------- RQ1 ----------
dc = pd.read_csv(R / "sot_analysis/decay_curves.csv")
def dv(var, rnd, met):
    r = dc[(dc.variant == var) & (dc["round"] == rnd) & (dc.metric == met)]
    return r.iloc[0] if len(r) else None

rq1_metrics = [("bleu", "BLEU"), ("word_retention", "Word retention"), ("chrf", "chrF"),
               ("bertscore_f1_rescaled", "BERTScore F1"), ("mpnet_cosine", "MPNet"), ("e5_cosine", "E5"), ("style", "Style")]
avail = set(dc.metric)
rq1_metrics = [(m, n) for m, n in rq1_metrics if m in avail]
rows = []
for v in PARA:
    rows.append([NAME[v]] + [f"{dv(v, 3, m)['mean']:.2f}" for m, _ in rq1_metrics])
t_rq1 = table(pd.DataFrame(rows, columns=["Paraphraser"] + [n for _, n in rq1_metrics]))

c_rq1 = bars([(NAME[v], [
    (dv(v, 3, "bertscore_f1_rescaled")["mean"], "s-content",
     f"{NAME[v]} T3 content (BERTScore): {dv(v,3,'bertscore_f1_rescaled')['mean']:.3f} "
     f"[{dv(v,3,'bertscore_f1_rescaled')['ci_low']:.3f}, {dv(v,3,'bertscore_f1_rescaled')['ci_high']:.3f}]"),
    (dv(v, 3, "style")["mean"], "s-style",
     f"{NAME[v]} T3 style: {dv(v,3,'style')['mean']:.3f} [{dv(v,3,'style')['ci_low']:.3f}, {dv(v,3,'style')['ci_high']:.3f}]"),
]) for v in PARA], legend=[("s-content", "Content (BERTScore F1)"), ("s-style", "Style (combined)")])

sv = pd.read_csv(R / "sot_analysis/style_vs_content.csv")
gap_rows = []
for v in PARA:
    g = sv[sv.variant == v].set_index("round")
    gap_rows.append([NAME[v]] + [f"{g.loc[k,'mean']:+.3f}" for k in (1, 2, 3)] +
                    [f"[{g.loc[3,'ci_low']:.3f}, {g.loc[3,'ci_high']:.3f}]"])
t_gap = table(pd.DataFrame(gap_rows, columns=["Paraphraser", "Gap T1", "Gap T2", "Gap T3", "T3 95% CI"]))

tv = pd.read_csv(R / "sot_analysis/translation_vs_paraphrase_T3_gap.csv")
tv_cols = tv.columns.tolist()

# family means at T3 from per-variant curves
tp = pd.read_csv(R / "sot_analysis/translation_vs_paraphrase.csv")
fam_metrics = [("bleu", "BLEU"), ("chrf", "chrF"), ("bertscore_f1_rescaled", "BERTScore F1"), ("mpnet_cosine", "MPNet"),
               ("e5_cosine", "E5"), ("entity_recall", "Entity recall"), ("number_recall", "Number recall"),
               ("sent_coverage", "Sentence coverage")]
t3 = tp[tp["round"] == 3]
fam_rows = []
for r in tv.itertuples():
    n, tr, pa = r.metric, r.translation, r.paraphrase
    fam_rows.append((n, [(tr, "s-trans", f"{n}, translation (3 models) at T3: {tr:.3f}"),
                         (pa, "s-para", f"{n}, paraphrase (4 models) at T3: {pa:.3f}")]))
c_tv = bars(fam_rows, legend=[("s-trans", "Round-trip translation (3 models)"), ("s-para", "Paraphrase (4 models)")])
# gap table from committed CSV (has CIs)
gcol = {c.lower(): c for c in tv_cols}
tvt = tv.copy(); tvt["gap [95% CI]"] = [f"{g:+.2f} [{a:.2f}, {b:.2f}]" for g,a,b in zip(tv.gap,tv.ci_low,tv.ci_high)]
tvt = tvt[["metric","translation","paraphrase","gap [95% CI]","n_docs"]].round(2)
tvt.columns = ["Metric","Translation","Paraphrase","Gap [95% CI]","Documents"]
t_tv = table(tvt)

conv = pd.read_csv(R / "sot_analysis/convergence_round1_share.csv")
conv["variant"] = conv.variant.map(NAME)
conv.columns = ["Paraphraser","Round 1 share of T3 loss (chrF)","Round 1 share (BERTScore)"]
for c in conv.columns[1:]: conv[c] = (conv[c]*100).round().astype(int).astype(str) + "%"
t_conv = table(conv)

steps = pd.read_csv(R / "sot_analysis/translation_vs_paraphrase_steps.csv")

# ---------- RQ2 ----------
sc = pd.read_csv(R / "rq2_attribution/scores.csv")
m = sc[(sc.model == "tfidf_char") & (sc.ablation == "opening_removed")]
t0 = m[m.rewriter == "original"].iloc[0]
rq2_order = m[m.stage == "T3"].sort_values("retained", ascending=False)
c_rq2 = bars([(NAME[r.rewriter], [(r.retained, "s-trans" if r.rewriter in TRANS else "s-para",
               f"{NAME[r.rewriter]} T3: macro-F1 {r.macro_f1:.3f} [{r.macro_f1_ci_low:.3f}, {r.macro_f1_ci_high:.3f}], retained {r.retained:.2f}")])
              for r in rq2_order.itertuples()],
             legend=[("s-trans", "Round-trip translator"), ("s-para", "Paraphraser")])

fg = pd.read_csv(R / "rq2_attribution/family_gap_T3.csv")
fg = fg[fg.ablation=="opening_removed"].copy()
MODN = {"tfidf_char":"Char n-grams","tfidf_word":"Word n-grams","stylometric":"Stylometric","length_only":"Length only"}
fg["gap [95% CI]"] = [f"{g:+.2f} [{a:.2f}, {b:.2f}]" for g,a,b in zip(fg.gap,fg.ci_low,fg.ci_high)]
fg = fg.assign(model=fg.model.map(MODN))[["model","translation","paraphrase","gap [95% CI]"]].round(2)
fg.columns = ["Model","Translation","Paraphrase","Gap [95% CI]"]
t_fg = table(fg)

fs = pd.read_csv(R / "rq2_attribution/full_scores.csv")
fm = fs[(fs.model == "tfidf_char") & (fs.ablation == "opening_removed")]
allp = fm[fm.dataset == "all"]
curve_rows = [["Original", f"{allp[allp.rewriter=='original'].macro_f1.iloc[0]:.2f}", "", "", "", "1.00", ""]]
for v in sorted(PARA, key=lambda v: -allp[(allp.rewriter == v) & (allp.stage == "T3")].retained.iloc[0]):
    g = allp[allp.rewriter == v].set_index("stage")
    curve_rows.append([NAME[v], "", f"{g.loc['T1','macro_f1']:.2f}", f"{g.loc['T2','macro_f1']:.2f}",
                       f"{g.loc['T3','macro_f1']:.2f}", f"{g.loc['T3','retained']:.2f}",
                       f"{g.loc['T3','gain_over_length']+0:+.2f}".replace("-0.00","0.00")])
t_full = table(pd.DataFrame(curve_rows, columns=["", "T0", "T1", "T2", "T3", "Retained at T3", "Gain over length at T3"]))

ds = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp", "all"]
heat = ['<div class="tw"><table class="heat"><thead><tr><th>Retained at T3</th>' +
        "".join(f"<th>{d}</th>" for d in ds) + "</tr></thead><tbody>"]
for v in PARA:
    heat.append(f"<tr><td>{NAME[v]}</td>")
    for d in ds:
        x = fm[(fm.dataset == d) & (fm.rewriter == v) & (fm.stage == "T3")].retained.iloc[0]
        a = max(0.0, min(1.0, x))
        heat.append(f'<td class="hc" style="--a:{a:.2f}" title="{NAME[v]}, {d}: {x:.3f}">{x:.2f}</td>')
    heat.append("</tr>")
heat.append("</tbody></table></div>")
t_heat = "".join(heat)

to = pd.read_csv(R / "rq2_attribution/rewriter_takeover.csv")
fto = pd.read_csv(R / "rq2_attribution/full_rewriter_takeover.csv")

trunc = pd.read_csv(R / "rq2_attribution/length_control_truncated.csv")

# ---------- RQ3 ----------
s3 = pd.read_csv(R / "rq3_fingerprint/scores_main.csv")
pooled = s3[s3.stage == "T1-T3"]
MOD = {"tfidf_char": "Char n-grams", "tfidf_word": "Word n-grams", "stylometric": "Stylometric (187)",
       "stylometric_no_sentence": "Stylometric, no sentence features", "length_only": "Length only"}
rq3_rows = []
for mname in MOD:
    p = pooled[(pooled.model == mname) & (pooled.family == "paraphrase")].iloc[0]
    t = pooled[(pooled.model == mname) & (pooled.family == "translation")].iloc[0]
    rq3_rows.append((MOD[mname], [
        (t.corrected, "s-trans", f"{MOD[mname]}, translators (K=3): corrected {t.corrected:.3f} [{t.corrected_ci_low:.3f}, {t.corrected_ci_high:.3f}], raw macro-F1 {t.macro_f1:.3f}"),
        (p.corrected, "s-para", f"{MOD[mname]}, paraphrasers (K=4): corrected {p.corrected:.3f} [{p.corrected_ci_low:.3f}, {p.corrected_ci_high:.3f}], raw macro-F1 {p.macro_f1:.3f}"),
    ]))
c_rq3 = bars(rq3_rows, legend=[("s-trans", "Translators (3)"), ("s-para", "Paraphrasers (4)")])
fd = pd.read_csv(R / "rq3_fingerprint/family_difference.csv")
fd["model"] = fd.model.map(MOD)
fd.columns = ["Model", "Paraphrase (corrected)", "Translation (corrected)", "Difference", "CI low", "CI high"]
fd["Difference [95% CI]"] = [f"{d:+.2f} [{a:.2f}, {b:.2f}]" for d,a,b in zip(fd.Difference,fd["CI low"],fd["CI high"])]
t_fd = table(fd[["Model","Paraphrase (corrected)","Translation (corrected)","Difference [95% CI]"]].round(2))

pc = pd.read_csv(R / "rq3_fingerprint/per_class.csv")
pc = pc[pc.model.isin(["tfidf_char", "stylometric"])].copy()
pcw = pc.pivot(index="class", columns="model", values="f1").reindex(PARA)
pcw.index = [NAME[i] for i in pcw.index]
pcw = pcw.rename(columns={"tfidf_char": "F1, char n-grams", "stylometric": "F1, stylometric"}).round(2).reset_index()
pcw = pcw.rename(columns={"index": "Paraphraser"})
t_pc = table(pcw)

desc = pd.read_csv(R / "rq3_fingerprint/length_typography_descriptives.csv")
d3 = desc[desc.stage == "T3"].copy()
d3["rewriter"] = d3.rewriter.map(lambda x: NAME.get(x, x))
d3 = d3[["rewriter", "word_ratio", "sent_len_mean", "comma rate", "mattr", "share with curly quotes/dashes"]]
d3.columns = ["Paraphraser (T3)", "Length vs T0", "Mean sentence length", "Comma rate", "MATTR", "Curly quotes/dashes"]
t_desc = table(d3.round(2))

ty = pd.read_csv(R / "rq3_fingerprint/typography_controls.csv")
ty.columns = ["Control","Paraphrase: full","Paraphrase: control","Translation: full","Translation: control"]
t_ty = table(ty.round(2))

# ---------- page ----------
CSS = """
:root{--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e3df;--track:#f0efec;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s7:#6250d6;--heat:42,120,214;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--card:#222221;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#8f8e88;--line:#3a3a37;--track:#2d2d2b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s7:#9085e9;--heat:57,135,229;--accent:#3987e5}}
:root[data-theme="dark"]{--surface:#1a1a19;--card:#222221;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e88;--line:#3a3a37;
--track:#2d2d2b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s7:#9085e9;--heat:57,135,229;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{min-width:0;max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px}
h2{font-size:21px;margin:44px 0 8px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:24px 0 6px}
p,li{color:var(--ink2)} strong{color:var(--ink)}
.sub{color:var(--muted);margin:0 0 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:20px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.card .h{font-size:17px;font-weight:600;color:var(--ink);margin:4px 0}
.card p{margin:0;font-size:14px}
.note{background:var(--card);border-left:3px solid var(--accent);padding:10px 14px;border-radius:0 8px 8px 0;margin:14px 0}
.tw{overflow-x:auto;max-width:100%;margin:10px 0 6px}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:12.5px}
td{color:var(--ink)}
.hc{background:rgba(var(--heat),calc(var(--a)*.55))}
.chart{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px 10px;margin:12px 0}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--ink2);margin-bottom:10px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px}
.row{display:grid;grid-template-columns:minmax(110px,190px) minmax(0,1fr);gap:10px;align-items:center;padding:4px 0}
.lab{font-size:13.5px;color:var(--ink)}
.bars{display:flex;flex-direction:column;gap:2px}
.bar-line{display:grid;grid-template-columns:minmax(0,1fr) 44px;gap:8px;align-items:center;cursor:default}
.track{height:12px;background:var(--track);border-radius:0 4px 4px 0}
.bar{height:12px;border-radius:0 4px 4px 0}
.val{font-size:12.5px;color:var(--ink2);font-variant-numeric:tabular-nums}
.bar-line:hover .bar{filter:brightness(1.1)} .bar-line:hover .val{color:var(--ink)}
.axis{display:flex;justify-content:space-between;margin-left:calc(min(190px,30%) + 10px);margin-right:52px;font-size:11.5px;color:var(--muted)}
.s-trans{background:var(--s1)} .s-para{background:var(--s2)} .s-content{background:var(--s3)} .s-style{background:var(--s7)}
.cap{font-size:13px;color:var(--muted);margin:4px 0 14px}
code{font-size:13px;background:var(--track);padding:1px 5px;border-radius:4px}
@media (max-width:560px){.row{grid-template-columns:minmax(0,1fr);gap:2px}.axis{margin-left:0}h1{font-size:25px}}
"""

body = f"""
<main>
<h1>Ship of Theseus: rewriting and what survives</h1>
<p class="sub">CS 6120 · Sam Wong · Ship of Theseus paraphrased corpus (Tripto et al.) · RQ1 decay, RQ2 source attribution, RQ3 rewriter fingerprint</p>

<p>When a text is rewritten three times in a row (T0 → T1 → T2 → T3), how much of its wording, meaning, style and source identity survives?
And does English → Spanish → English round-trip translation behave differently from English paraphrasing
(ChatGPT, PaLM, Dipper, Pegasus)?</p>

<div class="cards">
<div class="card"><div class="k">RQ1 · decay</div><div class="h">Meaning outlasts wording and style</div>
<p>Most of the loss happens in round 1 (71–90%). Style is lost faster than content for every paraphraser by T3.</p></div>
<div class="card"><div class="k">RQ1 · translation</div><div class="h">Translation preserves more on every metric</div>
<p>On the same 178 XSum documents: BERTScore 0.81 vs 0.58 at T3. Translation settles after one round; paraphrase keeps drifting.</p></div>
<div class="card"><div class="k">RQ2 · attribution</div><div class="h">Paraphrase erases more of the source</div>
<p>Translators keep 72–85% of the source signal, most paraphrasers 29–55%. ChatGPT relabels 55% of non-OpenAI texts as OpenAI.</p></div>
<div class="card"><div class="k">RQ3 · fingerprint</div><div class="h">Both families leave one, of different kinds</div>
<p>Paraphrasers are recognised by style and length, and their fingerprint grows each round. Translators are recognised by wording and characters.</p></div>
</div>

<h2>Data and method</h2>
<div class="tw"><table><tbody>
<tr><td>Corpus</td><td style="text-align:left;white-space:normal">7 datasets (cmv, eli5, sci_gen, tldr, wp, xsum, yelp); 7 sources (Human + 6 LLMs)</td></tr>
<tr><td>Sample</td><td style="text-align:left;white-space:normal"><strong>16,682 originals</strong> that every paraphraser completed to T3, on 2,996 articles; 583,870 pairs</td></tr>
<tr><td>Translation set</td><td style="text-align:left;white-space:normal">Samer Ahmed's XSum round trips (MADLAD-400-3B, NLLB-200-1.3B, Qwen3-4B; 3 rounds). Matched set: <strong>178 documents, 31 articles</strong> that every rewriter completed</td></tr>
<tr><td>Statistical unit</td><td style="text-align:left;white-space:normal">the original article. 95% CIs from an article-cluster bootstrap; classifier folds grouped by article</td></tr>
<tr><td>Retained scale</td><td style="text-align:left;white-space:normal"><strong>1 = identical to T0, 0 = no closer than an unrelated T0 text from the same dataset</strong> (floor = mean of 200 unrelated T0 pairs per dataset). Similarities: (s − floor)/(1 − floor); distances: (floor − d)/floor</td></tr>
<tr><td>Validation</td><td style="text-align:left;white-space:normal">Samer's originals are byte-identical to ours; our code reproduces 11 of his per-pair metrics and 100% of his RQ2 T0 predictions</td></tr>
</tbody></table></div>
<p class="cap">Hover any bar or heat cell for the exact value and its 95% CI.</p>

<h2>RQ1 · How words, meaning, style and information decay</h2>
<h3>Retained at T3 (all 7 datasets)</h3>
{t_rq1}
<p>Wording goes first and meaning stays: embedding similarity (MPNet 0.68–0.86) is far above BLEU (0.07–0.31). PaLM preserves the most
and Dipper the least. Embedding cosines compress differences that BERTScore and lexical metrics separate.</p>

<h3>Style decays faster than content</h3>
{c_rq1}
<p class="cap">Retained at T3, mean over 16,682 originals. Content = baseline-rescaled BERTScore F1; style = combined 11-component score (surface, punctuation, function words, syntax, character n-grams).</p>
{t_gap}
<p class="cap">Decay gap = retained content − retained style. Positive means style is lost faster.</p>
<p>All four T3 gaps exclude zero, and the gap grows over rounds (Pegasus +0.005 at T1 → +0.063 at T3). ChatGPT has the widest gap:
it keeps meaning but rewrites in its own voice. Punctuation is the most fragile style family for every paraphraser.</p>

<h3>Most of the change happens in round 1</h3>
{t_conv}
<p>71–90% of the T3 loss is already present at T1. Dipper is the exception that keeps rewriting about 85% of words every round.</p>

<h3>Round-trip translation vs paraphrase (same 178 XSum documents)</h3>
{c_tv}
<p class="cap">Family means of retained scores at T3. Translation = MADLAD, NLLB, Qwen3-4B; paraphrase = ChatGPT, PaLM, Dipper, Pegasus.</p>
{t_tv}
<p><strong>Translation preserves more on every metric.</strong> The gap is largest for exact wording and numbers, smallest for overall meaning.
NLLB preserves the most of any rewriter. It isn't a clean split: PaLM (BERTScore 0.76) sits close to Qwen3-4B (0.78).
After round 1, translators change only 2–10% of words per round, while paraphrasers keep changing 15–42%.</p>

<h2>RQ2 · Can the original source still be identified?</h2>
<p>A classifier trained only on originals (7 sources, chance macro-F1 0.143) is applied to rewrites. Samer's protocol, ported verbatim.
The ~30-word opening shared by all sources is removed. <strong>Retained attribution</strong> = (F1 at Tk − chance)/(F1 at T0 − chance).</p>

<h3>Matched XSum: translators vs paraphrasers at T3</h3>
{c_rq2}
<p class="cap">Character n-gram classifier, retained attribution at T3. T0 macro-F1 = {t0.macro_f1:.2f}.</p>
{t_fg}
<p class="cap">Family gap at T3 (translation − paraphrase, macro-F1), paired article bootstrap.</p>
<p>Translation keeps more of the source on every model, and all three CIs exclude zero. Pegasus (0.74) overlaps the translators,
consistent with it being the most conservative paraphraser in RQ1.</p>

<h3>Full corpus (paraphrase only, 7 datasets, 13M predictions)</h3>
{t_full}
{t_heat}
<p>The XSum ranking holds everywhere. With about 2,000 training originals per dataset, originals are much easier to attribute
(0.70 vs 0.47), and the share of signal paraphrases erase is larger than the small set suggested. The decline continues past round 1.</p>

<h3>Much of what survives paraphrasing is length</h3>
<p>Dipper's text models are no better than length alone at T3 (gain 0.00). With every text cut to 50 words, only Pegasus keeps a
real signal (0.34 vs chance 0.14 on the full corpus; ChatGPT 0.16, PaLM 0.22, Dipper 0.19).</p>

<h3>ChatGPT replaces the source with its own style</h3>
<div class="tw"><table><thead><tr><th>Share attributed to OpenAI (non-OpenAI texts, T3)</th><th>Originals</th><th>ChatGPT</th><th>PaLM</th><th>Dipper</th><th>Pegasus</th></tr></thead>
<tbody><tr><td>Matched XSum</td><td>3%</td><td><strong>34%</strong></td><td>13%</td><td>9%</td><td>11%</td></tr>
<tr><td>Full corpus</td><td>4%</td><td><strong>55%</strong></td><td>18%</td><td>6%</td><td>5%</td></tr></tbody></table></div>
<p>From 33% (yelp) to 74% (eli5) of ChatGPT rewrites go to the OpenAI class. Its low attribution score is a systematic relabelling
toward its own family, not random noise. PaLM shows no analogue.</p>

<h2>RQ3 · Do rewriters leave a fingerprint?</h2>
<p>Given only a rewritten text, can a classifier tell which model produced it? 178 matched XSum documents, 5×5 article-grouped CV,
trained on pooled T1–T3. The families have different numbers of classes, so scores are chance-corrected:
(F1 − 1/K)/(1 − 1/K), 0 = chance, 1 = perfect.</p>
{c_rq3}
<p class="cap">Chance-corrected macro-F1, pooled T1–T3.</p>
{t_fd}
<p class="cap">Paraphrase − translation difference, paired article bootstrap.</p>
<ul>
<li><strong>Translators are recognised by wording and characters</strong> (char n-grams 0.59 vs 0.54; not a reliable difference).</li>
<li><strong>Paraphrasers are recognised by style and length:</strong> stylometric +0.21 [0.15, 0.26], length alone +0.19 [0.14, 0.23].</li>
<li><strong>The paraphrase fingerprint grows each round</strong> (cross-round stylometric 0.53 corrected at T1 → 0.67 at T3); the translator fingerprint is flat.</li>
<li><strong>It doesn't rely on typography</strong> and transfers to unseen sources (0.39–0.68 corrected with one source held out).</li>
</ul>
<h3>What carries the paraphrase fingerprint</h3>
{t_desc}
{t_pc}
{t_ty}
<p>Pegasus shortens text to 67% with short sentences and few commas; Dipper shortens to 77%, lowers vocabulary diversity and adds curly
quotes; ChatGPT trims to 85% with a stable, comma-rich style. Translators keep length at 92–99%, so only finer lexical and
orthographic choices separate them. MADLAD's curly quotes are a large part of the translator signal.</p>

<h2>Caveats</h2>
<ul>
<li>Six of the seven sources are LLM-generated, so a paraphraser may push a text toward a style it already had.</li>
<li>The translation comparison is small and narrow: XSum only, 31 articles, one language pair (Spanish), greedy decoding. MT models translate sentence by sentence, Qwen whole documents.</li>
<li>Word retention and style are not compared for translation, because the translated texts are not in the repo.</li>
<li>Information metrics are heuristics: a paraphrased entity name or reformatted number counts as a loss.</li>
<li>RQ2/RQ3 use fixed, untuned classical models: they measure relative survival of the signal, not the best achievable attribution.</li>
<li>Invalid generations are counted as attrition and excluded, never repaired.</li>
</ul>
<p class="cap">Source: <code>Sam/notebooks/sot_analysis.ipynb</code>, <code>rq2_source_attribution.ipynb</code>, <code>rq3_rewriter_fingerprint.ipynb</code>; tables in <code>Sam/results/</code>; metric definitions in <code>Sam/docs/metric_standard.md</code>.</p>
</main>
"""

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ship of Theseus Results</title>
<style>{CSS}</style></head>
<body>{body}</body></html>
"""
OUT.write_text(page)
print("wrote", OUT, len(page))
