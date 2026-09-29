"""Loading a code LLM at FP16, 8-bit or 4-bit precision (bitsandbytes)."""

from __future__ import annotations

from typing import Tuple

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
PRECISIONS = ("fp16", "int8", "int4")


def load_model(model_id: str, precision: str) -> Tuple[object, object]:
    """Return ``(model, tokenizer)`` with the weights stored at ``precision``.

    * ``fp16`` - half precision, no quantization (the reference model)
    * ``int8`` - LLM.int8() weight quantization
    * ``int4`` - 4-bit NF4 weight quantization with FP16 compute
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")

    import torch
    import transformers
    from packaging import version
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # `torch_dtype` was renamed to `dtype` in transformers 4.56.
    dtype_arg = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    kwargs = {"device_map": "auto", dtype_arg: torch.float16}
    if precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer
