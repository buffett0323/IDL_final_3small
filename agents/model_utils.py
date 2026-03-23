"""Utilities for loading and driving seq2seq / causal translation models."""

from __future__ import annotations

import unicodedata

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# Default local cache for model downloads
DEFAULT_CACHE_DIR = "/data/user_data/haolingp/models"

# Few-shot prompt for causal LM translation (EN → ZH)
_CAUSAL_ENZH_FEW_SHOT = (
    "Translate English to Chinese.\n\n"
    "English: Macedonians go to polls in referendum on changing country's name.\n"
    "Chinese: 马其顿人就更改国名举行公投。\n\n"
    "English: Orlando Bloom and Miranda Kerr still love each other.\n"
    "Chinese: 奥兰多·布鲁姆和米兰达·可儿仍然彼此相爱。\n\n"
    "English: {source}\n"
    "Chinese:"
)


def load_translation_model(
    model_name, device, source_lang=None, target_lang=None, cache_dir=None
):
    """Load a Hugging Face seq2seq translation model with optional lang codes."""
    cache = cache_dir or DEFAULT_CACHE_DIR
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache)
    model.eval()
    model.to(device)

    if source_lang and hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = source_lang

    forced_bos_token_id = None
    if target_lang:
        if hasattr(tokenizer, "lang_code_to_id"):
            forced_bos_token_id = tokenizer.lang_code_to_id.get(target_lang)
        else:
            # NLLB tokenizers use convert_tokens_to_ids for language codes
            tid = tokenizer.convert_tokens_to_ids(target_lang)
            if tid != tokenizer.unk_token_id:
                forced_bos_token_id = tid

    return tokenizer, model, forced_bos_token_id


def load_causal_translation_model(model_name_or_path, device, cache_dir=None):
    """Load a causal LM (decoder-only) for translation, e.g. Qwen3-4B-Base.

    Returns (tokenizer, model, prompt_template) where prompt_template is a
    str.format-style string with a single ``{source}`` placeholder.
    """
    cache = cache_dir or DEFAULT_CACHE_DIR
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, cache_dir=cache, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache,
        dtype="auto",
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return tokenizer, model, _CAUSAL_ENZH_FEW_SHOT


def is_cjk(char: str) -> bool:
    """Return True if *char* is a CJK ideograph or CJK punctuation."""
    cp = ord(char)
    # CJK Unified Ideographs and common extension blocks
    if (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    ):
        return True
    # CJK punctuation / fullwidth forms
    if (0x3000 <= cp <= 0x303F) or (0xFF00 <= cp <= 0xFFEF):
        return True
    return False


def split_chinese_chars(text: str) -> list[str]:
    """Split text into emission units for Chinese target.

    Each CJK character becomes one unit.  Contiguous non-CJK characters
    (e.g. numbers, Latin) are grouped as a single unit.  Whitespace is
    consumed as a separator and never emitted on its own.
    """
    units: list[str] = []
    buf: list[str] = []
    for ch in text:
        if is_cjk(ch):
            if buf:
                units.append("".join(buf))
                buf = []
            units.append(ch)
        elif ch in (" ", "\t", "\n"):
            if buf:
                units.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        units.append("".join(buf))
    return units


def segment_chinese_reference(text: str) -> str:
    """Insert spaces between CJK characters so SimulEval counts them correctly."""
    return " ".join(split_chinese_chars(text))


def add_language_args(parser):
    """Add generic source/target language arguments for multilingual models."""
    parser.add_argument(
        "--source-lang", type=str, default=None,
        help="Optional source language code for multilingual tokenizers.",
    )
    parser.add_argument(
        "--target-lang", type=str, default=None,
        help="Optional target language code for multilingual tokenizers.",
    )


def build_generate_kwargs(forced_bos_token_id, **kwargs):
    """Attach optional multilingual generation kwargs."""
    if forced_bos_token_id is not None:
        kwargs["forced_bos_token_id"] = forced_bos_token_id
    return kwargs
