"""Build C++ extensions (GAE, FDA, freq_adapt, mock_env) using PyTorch's AOT path."""

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
        CppExtension(
            name="freq_adapt_cpp",
            sources=[str(ROOT / "freq_adapt.cpp")],
        ),
        CppExtension(
            name="mock_env_cpp",
            sources=[str(ROOT / "mock_env.cpp")],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
