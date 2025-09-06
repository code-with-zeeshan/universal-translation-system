#!/usr/bin/env python3
"""
First-time success checker for the Universal Translation System.

- Verifies core services are reachable on localhost with expected ports
- Optionally hits /metrics endpoints
- Optionally performs a sample translation via Coordinator (best-effort)
- Verifies model/vocab artifacts exist on disk
- On failures, can show recent docker logs for failing services
- Can emit a JSON report for CI

Usage:
  python scripts/first_time_success.py \
    [--retries 3] [--timeout 5] \
    [--check-metrics] [--check-translate] \
    [--json-report out.json] [--show-logs-on-fail 150]
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:
    print("[WARN] 'requests' not installed. Install it for HTTP checks: pip install requests", file=sys.stderr)
    requests = None  # type: ignore


# ----- tiny .env parser -----
def parse_dotenv(path: str = ".env") -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not os.path.exists(path):
        return env
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except Exception:
        pass
    return env


def get_env_or_default(key: str, default: str, dotenv: Dict[str, str]) -> str:
    return os.environ.get(key) or dotenv.get(key) or default


# ----- checks -----
def http_check(url: str, timeout: float) -> Tuple[bool, Optional[int]]:
    if requests is None:
        return False, None
    try:
        r = requests.get(url, timeout=timeout)
        return (r.status_code < 500), r.status_code
    except Exception:
        return False, None


def tcp_check(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def post_json(url: str, payload: Dict, timeout: float, headers: Optional[Dict[str, str]] = None) -> Tuple[bool, Optional[int], Optional[Dict]]:
    if requests is None:
        return False, None, None
    try:
        r = requests.post(url, json=payload, timeout=timeout, headers=headers or {})
        ok = r.status_code < 500
        data = None
        try:
            data = r.json()
        except Exception:
            pass
        return ok, r.status_code, data
    except Exception:
        return False, None, None


def docker_logs(service: str, tail: int = 100) -> str:
    cmd = ["docker", "compose", "logs", "--no-color", "--tail", str(tail), service]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = p.communicate(timeout=20)
        return out
    except Exception as e:
        return f"<failed to get logs: {e}>"


def verify_artifacts(models_dir: Path, vocabs_dir: Path) -> Dict[str, bool]:
    results = {
        "models_dir_exists": models_dir.exists(),
        "vocabs_dir_exists": vocabs_dir.exists(),
        "production_model_present": False,
        "vocabs_non_empty": False,
    }
    if results["models_dir_exists"]:
        # Accept either ONNX or TorchScript or PT
        candidates = [
            models_dir / "production" / "encoder.onnx",
            models_dir / "production" / "encoder.pt",
            models_dir / "production" / "model.onnx",
            models_dir / "production" / "model.pt",
        ]
        results["production_model_present"] = any(p.exists() for p in candidates)
    if results["vocabs_dir_exists"]:
        try:
            results["vocabs_non_empty"] = any(vocabs_dir.iterdir())
        except Exception:
            results["vocabs_non_empty"] = False
    return results


def main():
    p = argparse.ArgumentParser(description="First-time success checker")
    p.add_argument("--retries", type=int, default=3, help="Number of retries for each check")
    p.add_argument("--timeout", type=float, default=5.0, help="Per-attempt timeout in seconds")
    p.add_argument("--check-metrics", action="store_true", help="Also hit /metrics endpoints where available")
    p.add_argument("--check-translate", action="store_true", help="Attempt a sample translation via coordinator (best-effort)")
    p.add_argument("--json-report", type=str, help="Write detailed JSON report to this path")
    p.add_argument("--show-logs-on-fail", type=int, nargs="?", const=100, help="Show last N lines of docker logs for failing services")
    args = p.parse_args()

    dotenv = parse_dotenv()

    ENCODER_PORT = int(get_env_or_default("ENCODER_PORT", "8000", dotenv))
    DECODER_PORT = int(get_env_or_default("DECODER_PORT", "8001", dotenv))
    COORDINATOR_PORT = int(get_env_or_default("COORDINATOR_PORT", "8002", dotenv))
    PROMETHEUS_PORT = int(get_env_or_default("PROMETHEUS_PORT", "9090", dotenv))
    GRAFANA_PORT = int(get_env_or_default("GRAFANA_PORT", "3000", dotenv))
    REDIS_PORT = int(get_env_or_default("REDIS_PORT", "6379", dotenv))

    # Basic reachability checks
    checks = [
        ("Encoder",      f"http://localhost:{ENCODER_PORT}/health",   "http",  ENCODER_PORT, "encoder"),
        ("Decoder",      f"http://localhost:{DECODER_PORT}/health",   "http",  DECODER_PORT, "decoder"),
        ("Coordinator",  f"http://localhost:{COORDINATOR_PORT}/health","http",  COORDINATOR_PORT, "coordinator"),
        ("Prometheus",   f"http://localhost:{PROMETHEUS_PORT}/",      "http",  PROMETHEUS_PORT, "prometheus"),
        ("Grafana",      f"http://localhost:{GRAFANA_PORT}/login",     "http",  GRAFANA_PORT, "grafana"),
        ("Redis",        f"tcp://localhost:{REDIS_PORT}",              "tcp",   REDIS_PORT, "redis"),
    ]

    # Metrics checks (optional): decoder and coordinator expose /metrics
    metrics_checks = [
        ("DecoderMetrics",     f"http://localhost:{DECODER_PORT}/metrics",    "http",  DECODER_PORT, "decoder"),
        ("CoordinatorMetrics", f"http://localhost:{COORDINATOR_PORT}/metrics","http",  COORDINATOR_PORT, "coordinator"),
    ]

    print("\n=== First-time Success Check ===")
    print("Using ports: ENCODER=%d DECODER=%d COORDINATOR=%d PROMETHEUS=%d GRAFANA=%d REDIS=%d" %
          (ENCODER_PORT, DECODER_PORT, COORDINATOR_PORT, PROMETHEUS_PORT, GRAFANA_PORT, REDIS_PORT))

    report: Dict[str, Dict] = {"checks": {}, "metrics": {}, "artifacts": {}, "translate": {}}
    failures: List[str] = []

    # Run reachability checks
    for name, target, kind, port, service in checks:
        ok = False
        status_code: Optional[int] = None
        for _ in range(0, args.retries):
            if kind == "http":
                ok, status_code = http_check(target, args.timeout)
            else:
                ok = tcp_check("localhost", port, args.timeout)
            if ok:
                break
            time.sleep(0.5)
        print(f"{'✅' if ok else '❌'} {name:<18} -> {target}{'' if status_code is None else f' [{status_code}]'}")
        report["checks"][name] = {"ok": ok, "target": target, "status": status_code}
        if not ok:
            failures.append(service)

    # Metrics checks
    if args.check_metrics:
        for name, target, _, _, service in metrics_checks:
            ok, status_code = http_check(target, args.timeout)
            # metrics failure is non-fatal but reported
            print(f"{'✅' if ok else '⚠️'} {name:<18} -> {target}{'' if status_code is None else f' [{status_code}]'}")
            report["metrics"][name] = {"ok": ok, "target": target, "status": status_code}

    # Artifacts checks
    artifacts = verify_artifacts(Path("models"), Path("vocabs"))
    report["artifacts"] = artifacts
    art_ok = all([
        artifacts["models_dir_exists"],
        artifacts["vocabs_dir_exists"],
        artifacts["production_model_present"],
        artifacts["vocabs_non_empty"],
    ])
    print(f"{'✅' if art_ok else '⚠️'} Artifacts -> models_dir={artifacts['models_dir_exists']} production_model={artifacts['production_model_present']} vocabs_dir={artifacts['vocabs_dir_exists']} vocabs_non_empty={artifacts['vocabs_non_empty']}")

    # Optional sample translation via coordinator
    if args.check_translate:
        sample_payload = {"text": "Hello world", "source_lang": "en", "target_lang": "es"}
        # Try common endpoints
        tried = []
        for path in ("/translate", "/api/v1/translate"):
            url = f"http://localhost:{COORDINATOR_PORT}{path}"
            ok, status, data = post_json(url, sample_payload, args.timeout)
            tried.append({"url": url, "ok": ok, "status": status, "response": data})
            if ok:
                break
        report["translate"] = {"attempts": tried}
        if tried and tried[-1]["ok"]:
            print(f"✅ Sample translate via coordinator -> {tried[-1]['url']} [{tried[-1]['status']}]")
        else:
            # Non-fatal: often requires auth or full artifact setup
            print("⚠️ Sample translate via coordinator did not succeed (may require auth/artifacts). See report for details.")

    # Show docker logs on failure
    if failures and args.show_logs_on_fail:
        tail = int(args.show_logs_on_fail)
        print("\n--- Recent Docker Logs (failing services) ---")
        for service in sorted(set(failures)):
            print(f"\n### {service} (last {tail} lines) ###")
            print(docker_logs(service, tail=tail))

    # JSON report
    if args.json_report:
        try:
            with open(args.json_report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport written to {args.json_report}")
        except Exception as e:
            print(f"[WARN] Failed to write JSON report: {e}")

    # Exit code: fail only if core reachability failed
    if failures:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()