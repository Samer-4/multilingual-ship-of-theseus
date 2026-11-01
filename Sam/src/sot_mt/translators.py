"""Translator adapters with one interface: translate(text, src, tgt) -> TranslationResult.

  MadladTranslator   google/madlad400-3b-mt; the pilot worker's exact input handling ("<2xx> " prefix,
                     madlad_pack segmentation at 300 tokens, batches of 4 segments, greedy, 512 new tokens,
                     EOS-based cap detection, outputs joined with a space)
  NllbTranslator     facebook/nllb-200-*; src_lang on the tokenizer, forced_bos_token_id = target code,
                     one sentence per segment
  ChatTranslator     Qwen, Phi-4-mini and Llama instruct models; system prompt + user text, whole document, greedy,
                     max_new_tokens = max(16, min(2*n_in + 64, max_model_len - n_prompt - 8)) as in the
                     legacy Qwen run, <think> blocks stripped
  MockTranslator     deterministic, no model, for tests and local dry runs

Weights come from the Hugging Face hub (settings.weights absent) or from a Kaggle Model attached to the kernel
(settings.weights: {source: kaggle_models, handle: owner/model/framework/variation/version}); every load records
what was actually loaded (load_report: weights id, default generation config, device, GPU memory).
Model weights are only ever read from the local Hugging Face cache unless allow_download=True, and only
one adapter may hold a loaded model at a time unless allow_multiple_models=True. torch and transformers are
imported inside _load()/_generate(), so this module imports without them.

Error contract:
  * deterministic problems (segmentation impossible, a segment decodes to nothing) -> TranslationResult with
    error set: the runner stores an INVALID record (final, blocks the chain, never retried);
  * transient problems (CUDA OOM after batch halving, other runtime errors) -> raised: the runner logs a
    failed attempt (not a record) and retries it in a later run, up to max_attempts.
A result is only returned once EVERY segment has been translated: there are no partial outputs.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from . import segmentation
from .files import config_id

THINK = re.compile(r"<think>.*?</think>", re.S)
_LOADED: dict = {}          # translator name -> adapter holding a loaded model


class MissingCredentialsError(RuntimeError):
    pass


class ImportOnlyTranslatorError(RuntimeError):
    pass


class ModelAlreadyLoadedError(RuntimeError):
    pass


class TransientTranslationError(RuntimeError):
    pass


class NoGPUError(RuntimeError):
    pass


def require_cuda() -> None:
    """Real adapters run on cuda:0. Fail before any download or load when no GPU is visible."""
    import torch
    if not torch.cuda.is_available():
        raise NoGPUError("no CUDA GPU visible: on Kaggle set Accelerator = GPU (Settings > Accelerator) and "
                         "restart the session; for a pipeline check without a GPU use MOCK = True")


def dtype_kwarg(transformers_version: str) -> str:
    """from_pretrained keyword for the dtype: `torch_dtype` up to 4.55 (as in MADLAD round 1, 4.46.3),
    `dtype` from 4.56 on (`torch_dtype` still works in 5.x but logs a deprecation warning)."""
    major, minor = (int(x) for x in re.findall(r"\d+", transformers_version)[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


@dataclass
class TranslationResult:
    text: Optional[str]
    finish: str                         # "stop" | "length" (some output hit max_new_tokens) | "error"
    error: Optional[str] = None         # deterministic engine error code, e.g. "empty_segment"
    meta: dict = field(default_factory=dict)


def is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def strip_think(text: str) -> str:
    return THINK.sub("", text).strip()


def llm_max_new_tokens(n_in: int, n_prompt: int, max_model_len: int) -> int:
    """The legacy Qwen rule (degree_zero_chain.ipynb), with max_model_len as the context cap."""
    return max(16, min(2 * n_in + 64, max_model_len - n_prompt - 8))


def chat_messages(system: str, text: str, system_role: bool = True) -> list:
    if system_role:
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]
    return [{"role": "user", "content": system + "\n\n" + text}]


def generation_config(tcfg: dict, translator: str, prompt_variant: str) -> tuple:
    """(generation_config_id, config) for a pathway. Same construction that produced the Stage 2 id."""
    t = tcfg["translators"][translator]
    if prompt_variant not in t["prompt_variants"]:
        raise KeyError(f"{translator} has no prompt_variant {prompt_variant!r}")
    cfg = {"translator": translator, "prompt_variant": prompt_variant, "model_checkpoint": t["model_checkpoint"],
           "prompt": t["prompt_variants"][prompt_variant], "settings": t["settings"]}
    return config_id(cfg), cfg


# Decoding is plain greedy as declared in translators.yaml: penalties a checkpoint's generation_config.json may
# set by default are neutralised explicitly (they are recorded in load_report.default_generation_config).
NEUTRAL_DECODING = {"repetition_penalty": 1.0, "no_repeat_ngram_size": 0}


class ModelNotAttachedError(RuntimeError):
    """settings.weights names a Kaggle Model that is not mounted under /kaggle/input (attach it, or accept its
    licence on Kaggle first)."""


def find_kaggle_model(handle: str, root=None) -> Optional[str]:
    """Directory of an attached Kaggle Model `owner/model/framework/variation/version` that holds config.json."""
    from pathlib import Path
    root = Path(root or os.environ.get("SAM_KAGGLE_INPUT", "/kaggle/input"))
    want = [x.lower() for x in handle.split("/")[1:]]
    for cfg in sorted(root.glob("**/config.json")):
        if [x.lower() for x in cfg.parent.parts[-4:]] == want:
            return str(cfg.parent)
    return None


def weights_fingerprint(path: str) -> str:
    """sha256 of the file list (names + sizes) and of the small config/tokenizer files of a local model folder."""
    import hashlib
    from pathlib import Path
    h = hashlib.sha256()
    for f in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        h.update(f"{f.relative_to(path)}|{f.stat().st_size}\n".encode())
        if f.suffix == ".json" or f.name.startswith("tokenizer"):
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class Translator:
    """Common interface. Subclasses implement _load() and translate()."""

    def __init__(self, tcfg: dict, translator: str, prompt_variant: str, allow_download: bool = False,
                 allow_multiple_models: bool = False, token: Optional[str] = None):
        self.translator, self.prompt_variant = translator, prompt_variant
        self.entry = tcfg["translators"][translator]
        self.model_checkpoint = self.entry["model_checkpoint"]
        self.model_revision = self.entry.get("model_revision")
        self.settings = self.entry["settings"]
        self.generation_config_id, self.generation_config = generation_config(tcfg, translator, prompt_variant)
        self.allow_download, self.allow_multiple_models, self.token = allow_download, allow_multiple_models, token
        self.weights = self.settings.get("weights") or {"source": "huggingface"}
        self.load_path = None                # resolved on first use (hub id, or the attached Kaggle Model folder)
        self.model = self.tokenizer = None
        self.load_report = None

    # -- model lifecycle -------------------------------------------------------------------------
    def load(self) -> None:
        if self.model is not None:
            return
        others = [n for n, a in _LOADED.items() if a is not self]
        if others and not self.allow_multiple_models:
            raise ModelAlreadyLoadedError(f"{others} already loaded; unload it first or set allow_multiple_models")
        self._load()
        _LOADED[self.translator] = self

    def unload(self) -> None:
        self.model = self.tokenizer = None
        _LOADED.pop(self.translator, None)
        self._clear_cache()

    def _load(self) -> None:
        raise NotImplementedError

    def resolve_load_path(self) -> str:
        if self.load_path is None:
            if self.weights["source"] == "kaggle_models":
                found = find_kaggle_model(self.weights["handle"])
                if not found:
                    raise ModelNotAttachedError(f"Kaggle Model {self.weights['handle']} is not attached (model_sources "
                                                "in kernel-metadata.json; its licence must be accepted on Kaggle)")
                self.load_path = found
            else:
                self.load_path = self.model_checkpoint
        return self.load_path

    def weights_report(self) -> dict:
        """What was loaded: the hub commit, or a fingerprint of the Kaggle Model folder; the checkpoint's own
        generation defaults; where the model sits and the GPU memory it takes."""
        rep = {"weights_source": self.weights["source"], "weights_handle": self.weights.get("handle"),
               "load_path": self.load_path, "resolved_revision": getattr(self.model.config, "_commit_hash", None)}
        rep["weights_id"] = (weights_fingerprint(self.load_path) if self.weights["source"] == "kaggle_models"
                             else rep["resolved_revision"])
        try:
            rep["default_generation_config"] = self.model.generation_config.to_diff_dict()
        except Exception:  # noqa: BLE001
            rep["default_generation_config"] = None
        rep["device"] = str(getattr(self.model, "device", None))
        rep["param_dtype"] = str(next(self.model.parameters()).dtype)
        try:
            import torch
            rep["gpu"] = [{"name": torch.cuda.get_device_name(i), "allocated_gb": round(torch.cuda.memory_allocated(i) / 2**30, 2)}
                          for i in range(torch.cuda.device_count())]
        except Exception:  # noqa: BLE001
            rep["gpu"] = None
        return rep

    def _hf_kwargs(self) -> dict:
        if self.weights["source"] == "kaggle_models":
            return {"local_files_only": True}
        kw = {"local_files_only": not self.allow_download}
        if self.model_revision:
            kw["revision"] = self.model_revision
        if self.token:
            kw["token"] = self.token
        return kw

    def _torch_dtype(self):
        import torch
        return {"float16": torch.float16, "half": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[self.settings.get("dtype", "float16")]

    def _dtype_kw(self) -> dict:
        import transformers
        return {dtype_kwarg(transformers.__version__): self._torch_dtype()}

    def load_tokenizer(self):
        """Tokenizer only (no GPU, no model weights); _load() reuses it."""
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.resolve_load_path(), **self._hf_kwargs())
        return self.tokenizer

    @staticmethod
    def _clear_cache() -> None:
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def translate(self, text: str, src: str, tgt: str) -> TranslationResult:
        raise NotImplementedError

    def identity(self) -> dict:
        return {"translator": self.translator, "prompt_variant": self.prompt_variant,
                "model_checkpoint": self.model_checkpoint, "model_revision": self.model_revision,
                "generation_config_id": self.generation_config_id}


# -- sequence-to-sequence MT (MADLAD, NLLB) ---------------------------------------------------------

class Seq2SeqTranslator(Translator):
    segmentation_method = "madlad_pack"

    @property
    def budget(self) -> int:
        s = self.settings
        return (s.get("chunking") or s.get("segmentation"))["token_budget"]

    @property
    def batch_size(self) -> int:
        return int(self.settings.get("batch_size", 1))

    def _load(self) -> None:
        require_cuda()
        from transformers import AutoModelForSeq2SeqLM
        self.load_tokenizer()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.resolve_load_path(), **self._dtype_kw(),
                                                           low_cpu_mem_usage=True, **self._hf_kwargs()
                                                           ).to("cuda:0").eval()

    def count_tokens(self, text: str, src: str, tgt: str) -> int:
        raise NotImplementedError

    def _generate(self, segments: List[str], src: str, tgt: str) -> List[tuple]:
        """[(decoded_text, ended_with_eos, n_output_tokens)] for one batch of segments."""
        raise NotImplementedError

    def translate(self, text: str, src: str, tgt: str) -> TranslationResult:
        started = time.monotonic()
        count = lambda s: self.count_tokens(s, src, tgt)  # noqa: E731
        try:
            segs = segmentation.segment(text, self.segmentation_method, count, self.budget)
        except segmentation.SegmentationError as e:
            return TranslationResult(None, "error", error="segmentation_error",
                                     meta={"segmentation_error": str(e), "seconds": round(time.monotonic() - started, 2)})
        meta = {"segmentation": segmentation.describe(segs, self.segmentation_method, count, self.budget),
                "batch_sizes": [], "oom_retries": 0}
        outs, i, bs = [], 0, self.batch_size
        while i < len(segs):
            batch = segs[i:i + bs]
            try:
                res = self._generate(batch, src, tgt)
            except Exception as e:  # noqa: BLE001
                if is_oom(e) and bs > 1:
                    self._clear_cache()
                    bs, meta["oom_retries"] = max(1, bs // 2), meta["oom_retries"] + 1
                    continue
                if is_oom(e):
                    self._clear_cache()
                    raise TransientTranslationError(f"cuda_oom at batch size 1 (segment {i})") from e
                raise
            if len(res) != len(batch):
                raise TransientTranslationError(f"{len(res)} outputs for {len(batch)} segments")
            meta["batch_sizes"].append(len(batch))
            outs.extend(res)
            i += len(batch)
        meta["segment_output_tokens"] = [n for _, _, n in outs]
        meta["segment_hit_eos"] = [bool(e) for _, e, _ in outs]
        meta["seconds"] = round(time.monotonic() - started, 2)
        if any(not t for t, _, _ in outs):
            return TranslationResult(None, "error", error="empty_segment", meta=meta)
        finish = "stop" if all(e for _, e, _ in outs) else "length"
        return TranslationResult(segmentation.reconstruct([t for t, _, _ in outs]), finish, meta=meta)

    def _run_generate(self, enc) -> list:
        import torch
        d = self.settings["decoding"]
        with torch.inference_mode():
            return self.model.generate(**enc, num_beams=d["num_beams"], do_sample=d["do_sample"],
                                       max_new_tokens=d["max_new_tokens"], **self._extra_generate_kwargs)

    _extra_generate_kwargs: dict = {}

    def _eos_ids(self) -> set:
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        return set(eos) if isinstance(eos, (tuple, list, set)) else {int(eos)}

    def _decode(self, gen) -> List[tuple]:
        eos, out = self._eos_ids(), []
        for ids in gen:
            ids = ids.tolist()
            pad = self.tokenizer.pad_token_id
            n = sum(1 for t in ids if t != pad)
            out.append((self.tokenizer.decode(ids, skip_special_tokens=True).strip(), any(t in eos for t in ids), n))
        return out


def _same(a, b, equal) -> Optional[bool]:
    try:
        return bool(a is b or equal(a, b))
    except Exception:  # noqa: BLE001  shapes/devices differ
        return False


def t5_embedding_report(model, equal=None) -> dict:
    """How the T5 input embeddings are wired after loading (MADLAD is a T5 model)."""
    if equal is None:
        import torch
        equal = torch.equal
    sh, enc, dec = model.shared, model.encoder.embed_tokens, model.decoder.embed_tokens
    rep = {"tie_word_embeddings": getattr(model.config, "tie_word_embeddings", None),
           "encoder_is_shared": enc is sh, "decoder_is_shared": dec is sh,
           "encoder_equals_shared": _same(enc.weight, sh.weight, equal),
           "decoder_equals_shared": _same(dec.weight, sh.weight, equal),
           "decoder_equals_encoder": _same(dec.weight, enc.weight, equal),
           "commit_hash": getattr(model.config, "_commit_hash", None)}
    lm = getattr(model, "lm_head", None)
    if lm is not None:
        rep["lm_head_equals_shared"] = _same(lm.weight, sh.weight, equal)
    try:
        rep["param_dtype"] = str(sh.weight.dtype)
        rep["wo_dtype"] = str(model.decoder.block[0].layer[-1].DenseReluDense.wo.weight.dtype)
    except Exception:  # noqa: BLE001
        pass
    return rep


def tie_t5_input_embeddings(model, source: str = "shared") -> None:
    """Make the encoder, the decoder and `shared` use ONE embedding module, as transformers 4.46.3 builds T5
    (`T5Stack(config, self.shared)`). transformers 5.x builds separate modules and only ties them through
    `_tied_weights_keys`, which it skips when the checkpoint stores different values ("... so we will NOT
    tie them"); MADLAD's checkpoint does, and the decoder then runs on a different matrix than in round 1.
    source: "shared" (the T5 mapping's source) or "decoder" (the checkpoint's decoder.embed_tokens)."""
    emb = {"shared": model.shared, "decoder": model.decoder.embed_tokens}[source]
    model.shared = emb
    model.encoder.embed_tokens = emb
    model.decoder.embed_tokens = emb


class EngineWiringError(RuntimeError):
    """The model's embeddings are not wired as the runtime patch requires; nothing may be translated."""


class MadladTranslator(Seq2SeqTranslator):
    segmentation_method = "madlad_pack"
    load_report: Optional[dict] = None

    def __init__(self, tcfg, translator, prompt_variant, **kw):
        super().__init__(tcfg, translator, prompt_variant, **kw)
        # configs/translators.yaml: runtime_patches (outside the frozen entry; the config id is unchanged)
        self.runtime_patch = dict((tcfg.get("runtime_patches") or {}).get(translator) or {})
        self.embedding_tie = self.runtime_patch.get("embedding_tie", "none")
        self.wiring: Optional[str] = None
        self._orig: Optional[dict] = None

    def _load(self) -> None:
        super()._load()
        m = self.model
        lm = getattr(m, "lm_head", None)
        self._orig = {"shared": m.shared, "encoder": m.encoder.embed_tokens, "decoder": m.decoder.embed_tokens,
                      "lm_head": None if lm is None else lm.weight}
        before = t5_embedding_report(m)
        one_module = before["encoder_is_shared"] and before["decoder_is_shared"]   # e.g. transformers 4.46.3
        self.rewire("none" if one_module else self.embedding_tie)
        self.load_report = {"runtime_patch": self.runtime_patch.get("patch_id"), "embedding_tie": self.embedding_tie,
                            "applied": self.wiring, "already_one_module": one_module, "before": before,
                            "after": t5_embedding_report(m)}

    def rewire(self, mode: str) -> None:
        """Put back the modules exactly as loaded, then apply `mode` ("none" | "shared" | "decoder") and verify
        it: encoder, decoder and `shared` are one module taken from the named source, lm_head untouched.
        Diagnostics switch wirings only through this method, so no state leaks between variants."""
        if self.model is None or self._orig is None:
            raise EngineWiringError("model not loaded")
        m, o = self.model, self._orig
        m.shared, m.encoder.embed_tokens, m.decoder.embed_tokens = o["shared"], o["encoder"], o["decoder"]
        if mode not in {"none", "shared", "decoder"}:
            raise EngineWiringError(f"unknown embedding wiring {mode!r}")
        if mode != "none":
            tie_t5_input_embeddings(m, mode)
            src = o[mode]
            if not (m.shared is src and m.encoder.embed_tokens is src and m.decoder.embed_tokens is src):
                raise EngineWiringError(f"wiring {mode!r} not applied")
        if o["lm_head"] is not None and m.lm_head.weight is not o["lm_head"]:
            raise EngineWiringError("lm_head changed while rewiring the input embeddings")
        self.wiring = mode

    def unload(self) -> None:
        super().unload()
        self._orig, self.wiring = None, None

    def translate(self, text, src, tgt):
        if self.model is not None and self.wiring is None:
            raise EngineWiringError("embedding wiring was never verified for this load")
        res = super().translate(text, src, tgt)
        res.meta["embedding_wiring"] = self.wiring
        res.meta["runtime_patch"] = self.runtime_patch.get("patch_id") if self.wiring not in (None, "none") else None
        return res

    def count_tokens(self, text, src, tgt):
        return len(self.tokenizer(f"<2{tgt}> " + text, add_special_tokens=True).input_ids)

    def _generate(self, segments, src, tgt):
        enc = self.tokenizer([f"<2{tgt}> " + x for x in segments], padding=True, return_tensors="pt",
                             truncation=False).to("cuda:0")
        return self._decode(self._run_generate(enc))


def seq2seq_embedding_report(model, equal=None) -> dict:
    """Are the encoder/decoder input embeddings and the output projection the one shared matrix (as the NLLB
    checkpoints are trained)? The MADLAD failure under transformers 5 was exactly this not holding."""
    if equal is None:
        import torch
        equal = torch.equal
    sh, enc, dec = model.get_input_embeddings(), model.get_encoder().embed_tokens, model.get_decoder().embed_tokens
    out = model.get_output_embeddings()
    return {"tie_word_embeddings": getattr(model.config, "tie_word_embeddings", None),
            "encoder_equals_shared": _same(enc.weight, sh.weight, equal),
            "decoder_equals_shared": _same(dec.weight, sh.weight, equal),
            "lm_head_equals_shared": None if out is None else _same(out.weight, sh.weight, equal)}


class NllbTranslator(Seq2SeqTranslator):
    segmentation_method = "sentence"

    def _load(self) -> None:
        super()._load()
        emb = seq2seq_embedding_report(self.model)
        self.load_report = {**self.weights_report(), "embeddings": emb}
        if emb["tie_word_embeddings"] and not all(v is not False for k, v in emb.items() if k != "tie_word_embeddings"):
            raise EngineWiringError(f"NLLB embeddings are not tied after loading: {emb}")

    def _code(self, lang):
        return self.settings["lang_codes"][lang]

    def count_tokens(self, text, src, tgt):
        self.tokenizer.src_lang = self._code(src)
        return len(self.tokenizer(text).input_ids)

    def _generate(self, segments, src, tgt):
        self.tokenizer.src_lang = self._code(src)
        enc = self.tokenizer(segments, padding=True, return_tensors="pt", truncation=False).to("cuda:0")
        self._extra_generate_kwargs = {"forced_bos_token_id": self.tokenizer.convert_tokens_to_ids(self._code(tgt)),
                                       **NEUTRAL_DECODING}
        return self._decode(self._run_generate(enc))


# -- instruction-tuned LLMs (Qwen, Llama) -----------------------------------------------------------

class ChatTranslator(Translator):

    def system_prompt(self, src: str, tgt: str) -> str:
        return self.entry["prompt_variants"][self.prompt_variant][f"system_{src}_{tgt}"]

    def _load(self) -> None:
        require_cuda()
        from transformers import AutoModelForCausalLM
        self.load_tokenizer()
        # one GPU: a 3-4B fp16 model fits a 16 GB T4; splitting it over both T4s only adds transfers
        self.model = AutoModelForCausalLM.from_pretrained(self.resolve_load_path(), **self._dtype_kw(),
                                                          device_map={"": 0}, low_cpu_mem_usage=True,
                                                          **self._hf_kwargs()).eval()
        self.load_report = {**self.weights_report(),
                            "chat_template_kwargs": self.settings.get("chat_template_kwargs") or {}}

    def _prompt(self, system: str, text: str) -> str:
        # fixed template variables (Llama 3.x inserts today's date unless date_string is given)
        kw = self.settings.get("chat_template_kwargs") or {}
        try:
            return self.tokenizer.apply_chat_template(chat_messages(system, text), tokenize=False,
                                                      add_generation_prompt=True, **kw)
        except Exception:  # noqa: BLE001  templates without a system role (same fallback as the legacy run)
            return self.tokenizer.apply_chat_template(chat_messages(system, text, system_role=False),
                                                      tokenize=False, add_generation_prompt=True, **kw)

    def _generate(self, system: str, text: str) -> tuple:
        """(raw_text, hit_eos, n_in, n_prompt, n_out, max_new_tokens) for one document."""
        import torch
        prompt = self._prompt(system, text)
        n_in = len(self.tokenizer.encode(text, add_special_tokens=False))
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        n_prompt = enc.input_ids.shape[1]
        max_new = llm_max_new_tokens(n_in, n_prompt, int(self.settings["max_model_len"]))
        with torch.inference_mode():
            out = self.model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None,
                                      max_new_tokens=max_new, pad_token_id=self.tokenizer.pad_token_id
                                      or self.tokenizer.eos_token_id, **NEUTRAL_DECODING)
        new = out[0, n_prompt:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos) if isinstance(eos, (list, tuple, set)) else {eos}
        hit = any(t in eos for t in new)
        return self.tokenizer.decode(new, skip_special_tokens=True), hit, n_in, n_prompt, len(new), max_new

    def translate(self, text: str, src: str, tgt: str) -> TranslationResult:
        started = time.monotonic()
        try:
            raw, hit, n_in, n_prompt, n_out, max_new = self._generate(self.system_prompt(src, tgt), text)
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                self._clear_cache()
                raise TransientTranslationError("cuda_oom") from e
            raise
        meta = {"segmentation": {"method": "none", "n_segments": 1}, "n_in_tokens": n_in, "n_prompt_tokens": n_prompt,
                "n_out_tokens": n_out, "max_new_tokens": max_new, "hit_eos": hit,
                "think_stripped": bool(THINK.search(raw)), "seconds": round(time.monotonic() - started, 2)}
        return TranslationResult(strip_think(raw), "stop" if hit else "length", meta=meta)


# -- mock ---------------------------------------------------------------------------------------------

class MockTranslator(Translator):
    """Deterministic fake. Output depends only on the input text and direction, so reruns are identical.
    `script` is a function (text, src, tgt, call_number) -> behaviour, or a dict keyed by call number:
    "loop", "blank", "length", "empty_segment", "raise", "oom", "interrupt" (None = normal)."""

    is_mock = True                       # the runner prefixes its run_id with "MOCK-"

    def __init__(self, tcfg, translator, prompt_variant, script=None, **kw):
        super().__init__(tcfg, translator, prompt_variant, **kw)
        self.script, self.calls = script or {}, 0

    def _load(self):
        self.model = self.tokenizer = "mock"

    def translate(self, text, src, tgt):
        from .files import sha256_text
        self.calls += 1
        h = sha256_text(f"{self.translator}|{text}")[:8]      # different pathways give different text
        action = (self.script(text, src, tgt, self.calls) if callable(self.script)
                  else self.script.get(self.calls))
        if action == "interrupt":
            raise KeyboardInterrupt("mock interruption")
        if action == "raise":
            raise TransientTranslationError("mock runtime error")
        if action == "oom":
            raise TransientTranslationError("cuda_oom")
        if action == "empty_segment":
            return TranslationResult(None, "error", error="empty_segment", meta={"mock": True})
        n = len(text.split())
        out = {"es": f"El documento {h} es una traducción de {n} palabras al español.",
               "en": f"The document {h} is a translation of {n} words into English."}[tgt]
        if action == "loop":
            out = out + " " + ", ".join(["bucle"] * 40)
        if action == "blank":
            out = "  "
        meta = {"segmentation": {"method": "none", "n_segments": 1}, "mock": True}
        return TranslationResult(out, "length" if action == "length" else "stop", meta=meta)


ADAPTERS = {"madlad-3b": MadladTranslator, "nllb": NllbTranslator, "qwen3-4b": ChatTranslator,
            "llama": ChatTranslator, "phi4-mini": ChatTranslator}


def build_translator(tcfg: dict, translator: str, prompt_variant: str, allow_download: bool = False,
                     use_hf_token: bool = False, allow_multiple_models: bool = False) -> Translator:
    """Construct (not load) an adapter. Gated models need use_hf_token=True AND an HF_TOKEN in the env."""
    entry = tcfg["translators"].get(translator)
    if entry is None:
        raise KeyError(f"unknown translator {translator!r}; known: {sorted(tcfg['translators'])}")
    if entry.get("import_only"):
        raise ImportOnlyTranslatorError(f"{translator} is import-only (no adapter for its engine)")
    token = None
    if entry.get("requires_hf_token"):
        token = hf_token()
        if not use_hf_token:
            raise MissingCredentialsError(f"{translator} is gated: set USE_HF_TOKEN=True to allow Hugging Face access")
        if not token:
            raise MissingCredentialsError(f"{translator} is gated: no HF_TOKEN in the environment (Kaggle Secret)")
    elif use_hf_token:
        token = hf_token()
    return ADAPTERS[translator](tcfg, translator, prompt_variant, allow_download=allow_download,
                                allow_multiple_models=allow_multiple_models, token=token)
