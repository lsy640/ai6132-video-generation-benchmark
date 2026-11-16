# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Load only the requested pipeline; TI2V does not require S2V/Animate extras."""
from importlib import import_module

_PIPELINES = {
    'WanI2V': '.image2video',
    'WanS2V': '.speech2video',
    'WanT2V': '.text2video',
    'WanTI2V': '.textimage2video',
    'WanAnimate': '.animate',
}
__all__ = list(_PIPELINES)


def __getattr__(name):
    if name in _PIPELINES:
        value = getattr(import_module(_PIPELINES[name], __name__), name)
    elif name in ('configs', 'distributed', 'modules'):
        value = import_module('.' + name, __name__)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
