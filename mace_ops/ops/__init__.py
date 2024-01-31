import os
import torch
import sys
_HERE = os.path.realpath(os.path.dirname(__file__))


def _lib_path():
    path = os.path.join(os.path.join(_HERE, "../lib"), "libmace_ops.so")

    if os.path.isfile(path):
        return path

    raise ImportError("Could not find libmace_ops shared library at " + path)

torch.classes.load_library(_lib_path())