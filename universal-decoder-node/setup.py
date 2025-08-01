# universal-decoder-node/setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="universal-decoder-node",
    version="0.1.0",
    author="Universal Translation System",
    author_email="your.email@example.com",
    description="High-performance translation decoder service for Universal Translation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/code-with-zeeshan/universal-decoder-node",
    packages=find_packages(),
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
        "numpy>=1.21.0",
        "msgpack>=1.0.0",
        "lz4>=4.3.2",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "click>=8.1.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "docker>=6.1.0",
        "prometheus-client>=0.18.0",
        "pyjwt>=2.8.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.0",
    ],
    extras_require={
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
            "universal-decoder-node=universal_decoder_node.cli:cli",
            "udn=universal_decoder_node.cli:cli",  # Short alias
        ],
    },
    include_package_data=True,
    package_data={
        "universal_decoder_node": ["*.yaml", "*.yml"],
    },
)