#!/usr/bin/env python3
"""Entry point: python -m tui.app [--pipeline] [--train] [--config ...]"""
from utils.error_boundary import run_safely
from tui.app import main

if __name__ == "__main__":
    raise SystemExit(run_safely(main))
