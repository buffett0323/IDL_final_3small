"""Utilities for loading and driving seq2seq translation models."""

from __future__ import annotations

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_translation_model(model_name, device, source_lang=None, target_lang=None):
    """Load a Hugging Face seq2seq translation model with optional lang codes."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    if source_lang and hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = source_lang

    forced_bos_token_id = None
    if target_lang and hasattr(tokenizer, "lang_code_to_id"):
        forced_bos_token_id = tokenizer.lang_code_to_id.get(target_lang)

    return tokenizer, model, forced_bos_token_id


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
