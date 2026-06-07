# universal-decoder-node/setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="universal-decoder-node",
    version="1.0.0",
    author="Universal Translation System",
    author_email="your.email@example.com",
    description="High-performance translation decoder service for Universal Translation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/code-with-zeeshan/universal-translation-system/tree/main/universal-decoder-node",
    packages=find_packages(include=["udn", "udn.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0,<2.0.0",
        "msgpack>=1.0.0",
        "lz4>=4.3.2",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.20.0",
        "litserve>=0.12.0",
        "pydantic>=2.0.0",
        "click>=8.1.0",
        "requests>=2.31.0",
        "PyYAML>=6.0.0",
        "docker>=6.1.0",
        "prometheus-client>=0.19.0",
        "PyJWT>=2.8.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.0",
    ],
    extras_require={
        "perf": [
            "orjson>=3.9.10",
            "uvloop>=0.19.0; platform_system != 'Windows'",
            "tenacity>=8.2.3",
            "watchdog>=2.3.0",
            "zeroconf>=0.132.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "gpu": [
            "triton>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "udn=udn.cli:cli",
            "udn-litserve=udn.litserve_decoder:main",
        ],
    },
    include_package_data=True,
    package_data={
        "udn": ["*.yaml", "*.yml"],
    },
)
