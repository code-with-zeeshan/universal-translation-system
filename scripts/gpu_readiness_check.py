#!/usr/bin/env python3
"""
GPU Readiness Checker for Universal Translation System (Ubuntu/Debian focused)

Validates:
- PyTorch CUDA availability and GPU list (if torch installed), shows torch and CUDA versions
- nvidia-smi availability
- Docker is installed and accessible
- NVIDIA Container Toolkit by running a CUDA base image with --gpus all
- Optional: run a minimal PyTorch CUDA container to validate runtime

Usage:
  python scripts/gpu_readiness_check.py [--try-pytorch-container]
"""
from __future__ import annotations
import platform
import shutil
import subprocess
import sys
from typing import List, Tuple


def run(cmd: List[str], timeout: int = 15) -> Tuple[int, str, str]:
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out.strip(), err.strip()
    except Exception as e:
        return 1, "", str(e)


def check_torch_cuda() -> str:
    try:
        import torch  # type: ignore
        parts = [f"Torch {torch.__version__}"]
        try:
            parts.append(f"CUDA {torch.version.cuda}")
        except Exception:
            parts.append("CUDA N/A")
        has_cuda = torch.cuda.is_available()
        parts.append("CUDA available: " + ("yes" if has_cuda else "no"))
        if has_cuda:
            parts.append(f"GPUs={torch.cuda.device_count()}")
            names = []
            for i in range(torch.cuda.device_count()):
                names.append(torch.cuda.get_device_name(i))
            parts.append("Names=" + ", ".join(names))
        return " | ".join(parts)
    except Exception as e:
        return f"PyTorch not importable or no CUDA build: {e}"


def check_nvidia_smi() -> str:
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi: not found in PATH"
    code, out, err = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if code == 0:
        return f"nvidia-smi OK: {out}" if out else "nvidia-smi OK"
    return f"nvidia-smi error: {err or out}"


def check_docker() -> str:
    if shutil.which("docker") is None:
        return "Docker: not found in PATH"
    code, out, err = run(["docker", "--version"])
    if code == 0:
        return f"Docker: {out}"
    return f"Docker error: {err or out}"


def check_nvidia_container_toolkit() -> str:
    if shutil.which("docker") is None:
        return "NVIDIA Toolkit: docker not found"
    code, out, err = run([
        "docker", "run", "--rm", "--gpus", "all",
        "nvidia/cuda:12.2.0-base-ubuntu22.04", "nvidia-smi"
    ], timeout=25)
    if code == 0:
        return "NVIDIA Container Toolkit: OK (docker --gpus all works)"
    return f"NVIDIA Container Toolkit: FAILED ({err or out})"


def try_pytorch_container() -> str:
    if shutil.which("docker") is None:
        return "PyTorch container test: docker not found"
    # Use an upstream CUDA-enabled PyTorch image
    code, out, err = run([
        "docker", "run", "--rm", "--gpus", "all",
        "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
        "python", "-c",
        "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
    ], timeout=35)
    if code == 0:
        return f"PyTorch container test: OK -> {out}"
    return f"PyTorch container test: FAILED ({err or out})"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU Readiness Checker")
    parser.add_argument("--try-pytorch-container", action="store_true", help="Also run a minimal PyTorch CUDA container test")
    args = parser.parse_args()

    print("\n=== GPU Readiness Checker (Ubuntu/Debian) ===")
    print("- " + check_torch_cuda())
    print("- " + check_nvidia_smi())
    print("- " + check_docker())
    print("- " + check_nvidia_container_toolkit())

    if args.try_pytorch_container:
        print("- " + try_pytorch_container())

    sysname = platform.system().lower()
    if "linux" in sysname:
        print("\nIf Toolkit check failed:")
        print("- Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html")
        print("- Ensure your user can run docker without sudo, or run via sudo for tests")
    else:
        print("\nNon-Linux host detected. For Windows/macOS, prefer WSL2 or Linux hosts for reliable GPU access.")


if __name__ == "__main__":
    main()