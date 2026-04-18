#!/usr/bin/env python3
"""Gate short hidden-dist runs and suggest promotion candidates.

Usage:
  python gate_hidden_dist_runs.py
  python gate_hidden_dist_runs.py --tag short_ablation --json-out gate_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import wandb

try:
    from wandb.apis.public import Api as WandbPublicApi
except Exception:  # pragma: no cover - fallback for older installs
    WandbPublicApi = None


@dataclass
class RunGate:
    run_id: str
    name: str
    state: str
    best_step: Optional[int]
    best_disc: Optional[float]
    best_by_3: Optional[float]
    best_by_6: Optional[float]
    decision: str
    reason: str


def best_disc_values(history: Iterable[dict]) -> tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
    best_step = None
    best_disc = None
    best_by_3 = None
    best_by_6 = None

    for row in history:
        disc = row.get("test/disc_mean")
        step = row.get("custom_step")
        if disc is None or step is None:
            continue
        try:
            disc = float(disc)
            step = int(step)
        except (TypeError, ValueError):
            continue

        if best_disc is None or disc < best_disc:
            best_disc = disc
            best_step = step
        if step <= 3 and (best_by_3 is None or disc < best_by_3):
            best_by_3 = disc
        if step <= 6 and (best_by_6 is None or disc < best_by_6):
            best_by_6 = disc

    return best_step, best_disc, best_by_3, best_by_6


def decide(best_by_3: Optional[float], best_by_6: Optional[float], gate_b: float, gate_c: float) -> tuple[str, str]:
    if best_by_3 is None:
        return "PENDING", "No eval points at custom_step<=3 yet"
    if best_by_3 >= gate_b:
        return "STOP", f"best_by_3={best_by_3:.4f} >= gate_b={gate_b:.4f}"
    if best_by_6 is None:
        return "HOLD", "Passed gate_b but no eval points at custom_step<=6 yet"
    if best_by_6 < gate_c:
        return "PROMOTE", f"best_by_6={best_by_6:.4f} < gate_c={gate_c:.4f}"
    return "HOLD", f"best_by_6={best_by_6:.4f} >= gate_c={gate_c:.4f}"


def fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="azencot-group")
    parser.add_argument("--project", default="ts_corrupted")
    parser.add_argument("--tag", default="short_ablation")
    parser.add_argument("--require-tag", action="append", default=["hidden_dist"])
    parser.add_argument("--gate-b", type=float, default=0.20, help="Gate B threshold at custom_step<=3")
    parser.add_argument("--gate-c", type=float, default=0.10, help="Gate C threshold at custom_step<=6")
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if hasattr(wandb, "Api"):
        api = wandb.Api()
    elif WandbPublicApi is not None:
        api = WandbPublicApi()
    else:
        print(
            "W&B Public API is unavailable in this environment. "
            "Install/upgrade the `wandb` package in the runtime used for gating.",
            file=sys.stderr,
        )
        return
    runs = api.runs(
        f"{args.entity}/{args.project}",
        filters={"tags": {"$in": [args.tag]}},
        per_page=args.max_runs,
    )

    required = set(args.require_tag)
    gate_rows: list[RunGate] = []
    for run in runs:
        tags = set(run.tags or [])
        if not required.issubset(tags):
            continue

        history = run.scan_history(keys=["custom_step", "test/disc_mean"])
        best_step, best_disc, best_by_3, best_by_6 = best_disc_values(history)
        decision, reason = decide(best_by_3, best_by_6, args.gate_b, args.gate_c)

        gate_rows.append(
            RunGate(
                run_id=run.id,
                name=run.name,
                state=run.state,
                best_step=best_step,
                best_disc=best_disc,
                best_by_3=best_by_3,
                best_by_6=best_by_6,
                decision=decision,
                reason=reason,
            )
        )

    gate_rows.sort(key=lambda r: ({"PROMOTE": 0, "HOLD": 1, "PENDING": 2, "STOP": 3}.get(r.decision, 9), r.best_disc or 9.9))

    header = f"{'run_id':10} {'state':10} {'best@step':12} {'best<=3':8} {'best<=6':8} {'decision':8} reason"
    print(header)
    print("-" * len(header))
    for row in gate_rows:
        best_step_txt = "-" if row.best_disc is None else f"{row.best_disc:.4f}@{row.best_step}"
        print(
            f"{row.run_id:10} {row.state:10} {best_step_txt:12} {fmt(row.best_by_3):8} {fmt(row.best_by_6):8} {row.decision:8} {row.reason}"
        )

    promote = [r.run_id for r in gate_rows if r.decision == "PROMOTE"]
    print(f"\nPROMOTE count: {len(promote)}")
    if promote:
        print("PROMOTE run IDs:", ", ".join(promote))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "entity": args.entity,
                    "project": args.project,
                    "tag": args.tag,
                    "gate_b": args.gate_b,
                    "gate_c": args.gate_c,
                    "rows": [row.__dict__ for row in gate_rows],
                    "promote_run_ids": promote,
                },
                f,
                indent=2,
            )
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
