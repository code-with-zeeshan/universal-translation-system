#!/usr/bin/env python3
"""
scripts/build_models.py
Orchestrate model building, conversion, and optimization for CI/CD pipelines.
"""
import argparse
import subprocess
import sys

def build_encoder():
    print("[Encoder] Building encoder core (C++/ONNX)...")
    subprocess.run(["docker", "build", "-f", "docker/encoder.Dockerfile", "-t", "universal-encoder-core:latest", "."], check=True)
    print("[Encoder] Build complete.")

def build_decoder():
    print("[Decoder] Building decoder (Litserve/PyTorch)...")
    subprocess.run(["docker", "build", "-f", "cloud_decoder/Dockerfile", "-t", "universal-decoder:latest", "."], check=True)
    print("[Decoder] Build complete.")

def convert_models():
    print("[Model Conversion] Converting and optimizing models for deployment...")
    subprocess.run(["python", "training/convert_models.py"], check=True)
    print("[Model Conversion] Conversion complete.")

def main():
    parser = argparse.ArgumentParser(description="Build and convert models for Universal Translation System.")
    parser.add_argument('--encoder', action='store_true', help='Build encoder core')
    parser.add_argument('--decoder', action='store_true', help='Build decoder')
    parser.add_argument('--convert', action='store_true', help='Convert/optimize models')
    parser.add_argument('--all', action='store_true', help='Build everything')
    args = parser.parse_args()

    if args.all or args.encoder:
        build_encoder()
    if args.all or args.decoder:
        build_decoder()
    if args.all or args.convert:
        convert_models()
    if not (args.encoder or args.decoder or args.convert or args.all):
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
