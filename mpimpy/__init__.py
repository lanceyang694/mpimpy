# this is pimpy package
__all__ = ["bitslicedpe", "diffpairdpe", "fpmemdpe"]
__version__ = "0.1.1"

from .memmat import bitslicedpe
from .memmatdp import diffpairdpe
from .memmatfp import fpmemdpe