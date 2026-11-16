#!/usr/bin/env python3
"""Single entry point for the latest cell-marker pipeline and frozen baseline."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="Check or execute a configured experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--variant", choices=("latest", "frozen", "binary-spa-e"), default="latest",
                     help="latest: cluster evidence + E cell-marker check; frozen: original B2.16")
    run.add_argument("--check-only", action="store_true")
    run.add_argument("--resume", action="store_true")
    sub.add_parser("verify", help="Verify packaged source integrity")
    sub.add_parser("stages", help="List the frozen scientific stages")
    args = parser.parse_args()
    if args.action == "verify":
        target = ROOT / "base/tools/verify_bundle.py"
        command = [sys.executable, str(target)]
    elif args.action == "stages":
        command = [sys.executable, str(ROOT / "base/run_stage.py"), "list"]
    else:
        if args.check_only and args.resume:
            parser.error("--check-only and --resume cannot be combined")
        target = (ROOT / "base/run_all.py" if args.variant == "frozen" else
                  ROOT / "experimental_cell_gate/run_all_binary_spa.py")
        command = [sys.executable, str(target), "--config", str(args.config)]
        if args.check_only:
            command.append("--check-only")
        if args.resume:
            command.append("--resume")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
