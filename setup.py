# Root setup.py for Universal Translation System
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent


def read_reqs(rel_path: str) -> list[str]:
    path = ROOT / rel_path
    if not path.exists():
        return []
    reqs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            # Inline include support
            inc = line.split(" ", 1)[1].strip()
            reqs.extend(read_reqs(Path(rel_path).parent.joinpath(inc).as_posix()))
            continue
        reqs.append(line)
    return reqs

base_requires = read_reqs("requirements/base.txt")

extras = {
    "train": read_reqs("requirements/train.txt"),
    "serve": read_reqs("requirements/serve.txt"),
    "decoder": read_reqs("requirements/decoder.txt"),
    "coordinator": read_reqs("requirements/coordinator.txt"),
    "export": read_reqs("requirements/export.txt"),
    "tui": ["textual>=0.52.0", "pynvml>=11.5.0", "rich>=13.0.0"],
    "dev": read_reqs("requirements/dev.txt"),
}

# Convenience bundle
extras["all"] = sorted(set(sum(extras.values(), [])))

setup(
    name="universal-translation-system",
    version="1.0.0",
    description="Universal Translation System: Edge encoding, cloud decoding, multi-platform SDKs",
    author="Mohammad Zeeshan",
    author_email="mohammad.zeeshan@code-with-zeeshan.com",
    url="https://github.com/code-with-zeeshan/universal-translation-system",
    packages=find_packages(),
    install_requires=base_requires,
    extras_require=extras,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "uts-tui=tui.app:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
