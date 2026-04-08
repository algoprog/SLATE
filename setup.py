"""SLATE: Step-Level Advantage estimation for Truncated Exploration."""

import os
from pathlib import Path
from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read verl version
verl_version_file = os.path.join(os.path.dirname(__file__), "verl", "version", "version")
if os.path.exists(verl_version_file):
    with open(verl_version_file) as f:
        verl_version = f.read().strip()
else:
    verl_version = "0.1.0"

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "hydra-core",
    "numpy<2.0.0",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0",
    "torchdata",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "transformers",
    "wandb",
    "packaging>=20.0",
    "tensorboard",
    "requests",
    "aiohttp",
    "tqdm",
]

extras_require = {
    "gpu": ["liger-kernel", "flash-attn"],
    "vllm": ["vllm>=0.8.5,<=0.11.0"],
    "retrieval": ["uvicorn", "fastapi", "faiss-gpu"],
    "test": ["pytest"],
}

setup(
    name="slate-rl",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    url="https://github.com/anonymous/SLATE",
    license="CC-BY-NC-SA-4.0",
    author="",
    author_email="",
    description="SLATE: Step-Level Advantage estimation for Truncated Exploration",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
)
