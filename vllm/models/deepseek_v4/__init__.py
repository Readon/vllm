"""DeepSeek V4 model — hardware-isolated entry point.

The actual implementation lives under ``nvidia/`` and ``amd/``; this module
picks the right one for the current platform and re-exports the public
classes used by the model registry and quantization config lookup.

Model classes are lazily imported so that just importing
``DeepseekV4FP8Config`` (used during startup quantization detection) does not
trigger the full NVIDIA import chain, which has unconditional dependencies
on sm_90+-only packages like ``flash_mla``.
"""

import typing as _typing

from .quant_config import DeepseekV4FP8Config

__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]


_LAZY_IMPORTS: dict[str, type] | None = None


def _resolve_lazy_imports() -> dict[str, type]:
    global _LAZY_IMPORTS
    if _LAZY_IMPORTS is not None:
        return _LAZY_IMPORTS

    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from .amd.dspark import DSparkDeepseekV4ForCausalLM  # type: ignore[assignment]
        from .amd.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
        from .amd.mtp import DeepSeekV4MTP  # type: ignore[assignment]
    elif current_platform.is_xpu():
        from .xpu.dspark import DSparkDeepseekV4ForCausalLM  # type: ignore[assignment]
        from .xpu.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
        from .xpu.mtp import DeepSeekV4MTP  # type: ignore[assignment]
    else:
        from .nvidia.dspark import DSparkDeepseekV4ForCausalLM  # type: ignore[assignment]
        from .nvidia.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
        from .nvidia.mtp import DeepSeekV4MTP  # type: ignore[assignment]

    _LAZY_IMPORTS = {
        "DSparkDeepseekV4ForCausalLM": DSparkDeepseekV4ForCausalLM,
        "DeepseekV4ForCausalLM": DeepseekV4ForCausalLM,
        "DeepSeekV4MTP": DeepSeekV4MTP,
    }
    return _LAZY_IMPORTS


def __getattr__(name: str) -> _typing.Any:
    imports = _resolve_lazy_imports()
    try:
        return imports[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )


# Make ``__dir__`` and ``__all__`` agree for static analysis tools.
dir = __all__  # type: ignore[assignment]
