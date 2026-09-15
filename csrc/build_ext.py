"""Build C++ extensions (GAE + FDA) using PyTorch's AOT path."""

from pathlib import Path
from torch.utils.cpp_extension import CppExtension, BuildExtension
from setuptools import setup

ROOT = Path(__file__).resolve().parent

setup(
    name="freq_nav_cpp",
    ext_modules=[
        CppExtension(
            name="gae_cpp",
            sources=[str(ROOT / "gae.cpp")],
        ),
        CppExtension(
            name="fda_cpp",
            sources=[str(ROOT / "fda.cpp")],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
