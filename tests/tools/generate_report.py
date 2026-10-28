#!/usr/bin/env python3
# Copyright 2026 Scaleway, Aqora, Quantum Commons
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regenerates the repo-root ``LOSS_REPORT.md`` from the conversion batteries.

Reads the static battery registries (edges, declared losses, unsupported
conversions, feature coverage) and, optionally, per-paradigm observed-loss JSON
dumps (``--observed``) to fill the ``Observed (run)`` / ``Expected (run)``
columns. The run columns are only rendered when observed data is provided;
empty run cells render as ``-``.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "tests" / "src" / "circuit_conversion"),
    str(ROOT / "tests" / "src" / "result_conversion"),
]

import circuit_testing as ct  # noqa: E402
import result_testing as rt  # noqa: E402


def _esc(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _load_observed(
    paths: List[str],
) -> Dict[str, Dict[str, List[Tuple[str, ...]]]]:
    """Merges observed-loss JSON dumps: edge id -> loss -> observations."""
    merged: Dict[str, Dict[str, List[Tuple[str, ...]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in paths:
        with open(path) as fh:
            data = json.load(fh)
        for edge, categories in data.items():
            for category, entry in categories.items():
                for obs in entry.get("observations", []):
                    merged[edge][category].append(tuple(obs))
    return merged


def _run_cell(observations: List[Tuple[str, ...]], value_index: int) -> str:
    """One run-level column cell from the observations of a row.

    A single value renders inline; multiple values collapse into a GitHub-native
    ``<details>`` block kept on one physical line (a blank line inside a table
    cell would terminate the row in the GitHub Markdown renderer).
    """
    values = set()
    for obs in observations:
        if len(obs) <= value_index:
            continue
        value = obs[value_index]
        if not value:
            continue
        field = obs[4] if len(obs) > 4 else None
        cell = f"`{_esc(field)}` = `{_esc(value)}`" if field else f"`{_esc(value)}`"
        values.add(cell)
    if not values:
        return "-"
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    count = f"{len(ordered)} values"
    body = "<br>".join(ordered)
    return f"<details><summary>{count}</summary><br>{body}</details>"


def _loss_table(
    edges,
    descriptions,
    observed: Dict[str, Dict[str, List[Tuple[str, ...]]]],
    with_columns: bool,
) -> str:
    headers = ["Edge", "Loss", "Expected", "Actual", "Explanation"]
    if with_columns:
        headers += ["Observed (run)", "Expected (run)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for edge in sorted(edges, key=lambda e: e.id):
        for loss in sorted(edge.known_losses):
            desc = descriptions.get(loss)
            row = [
                f"`{_esc(edge.id)}`",
                f"`{loss}`",
                _esc(desc.expected) if desc else "-",
                _esc(desc.actual) if desc else "-",
                _esc(desc.why) if desc else "-",
            ]
            if with_columns:
                obs = observed.get(edge.id, {}).get(loss, [])
                row.append(_run_cell(obs, 2))  # observed value
                row.append(_run_cell(obs, 3))  # expected value
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _unsupported_table(entries) -> str:
    lines = [
        "| Unsupported conversion | Paradigm | Behavior |",
        "|---|---|---|",
    ]
    for name, _op, paradigm in sorted(entries, key=lambda x: (x[2], x[0])):
        lines.append(f"| `{_esc(name)}` | {paradigm} | raises (`Exception`) |")
    return "\n".join(lines)


def _feature_table(rows) -> str:
    lines = ["| Feature | Status | Note |", "|---|---|---|"]
    for axis, status, note in rows:
        lines.append(f"| {_esc(axis)} | {_esc(status)} | {_esc(note)} |")
    return "\n".join(lines)


def _paradigm_block(
    name: str,
    edges,
    descriptions,
    observed: Dict[str, Dict[str, List[Tuple[str, ...]]]],
    with_columns: bool,
) -> List[str]:
    paradigm_edges = sorted(
        (e for e in edges if e.paradigm == name), key=lambda e: e.id
    )
    lossy = [e for e in paradigm_edges if e.known_losses]
    lossless = [e for e in paradigm_edges if not e.known_losses]
    out = [f"### {name} paradigm", "", _loss_table(lossy, descriptions, observed, with_columns)]
    if lossless:
        ids = ", ".join(f"`{_esc(e.id)}`" for e in lossless)
        out += ["", f"*Lossless edges:* {ids}."]
    out.append("")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observed",
        action="append",
        default=[],
        metavar="JSON",
        help="observed-loss JSON dump (repeatable)",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "LOSS_REPORT.md", help="output path"
    )
    args = parser.parse_args()

    observed = _load_observed(args.observed)
    with_columns = bool(args.observed)

    circuit_edges = ct.build_edges(ct.NONE)
    result_edges = rt.build_edges(rt.NONE)

    out = [
        "# qio conversion report — losses, limitations & unsupported conversions",
        "",
        "> Auto-generated from the conversion test batteries (`loss_report()` + "
        "static registries). Do not edit by hand — regenerate with "
        "`make -C tests report` (`report-observed` adds the run columns). "
        "Auto-refreshed in CI on every merge to `main`.",
        "",
        "## Circuit conversions (`QuantumProgram`)",
        "",
        "Each row = one conversion edge × one declared information loss. A loss "
        "not declared on an edge is checked `exact` (any deviation fails the "
        "test). Edges without declared losses are lossless round trips (listed "
        "after each paradigm's table).",
        "",
    ]
    for paradigm in ("gate", "analog", "photonic"):
        out += _paradigm_block(
            paradigm, circuit_edges, ct._LOSS_DESCRIPTIONS, observed, with_columns
        )
    out += [
        "### Unsupported conversions",
        "",
        "Each entry must raise (`Exception`), enforced by the negative tests.",
        "",
        _unsupported_table(ct.UNSUPPORTED_CONVERSIONS),
        "",
        "### Feature coverage",
        "",
        _feature_table(ct.CIRCUIT_FEATURE_COVERAGE),
        "",
        "## Result conversions (`QuantumProgramResult`)",
        "",
        "The counts histogram is always compared bit-exact; the rows below are "
        "the tolerated metadata losses.",
        "",
    ]
    for paradigm in ("gate", "analog", "photonic"):
        out += _paradigm_block(
            paradigm, result_edges, rt._LOSS_DESCRIPTIONS, observed, with_columns
        )
    out += [
        "### SDK conventions (tolerated, no information loss)",
        "",
        "| Field | Expected | Actual | Explanation |",
        "|---|---|---|---|",
    ]
    for key in ("measurement_key", "bit_order"):
        desc = rt._LOSS_DESCRIPTIONS.get(key)
        if desc:
            out.append(
                f"| `{key}` | {_esc(desc.expected)} | {_esc(desc.actual)} "
                f"| {_esc(desc.why)} |"
            )
    out += [
        "",
        "### Unsupported conversions",
        "",
        "The complement of the declared read graph. Each entry must raise "
        "(`Exception`).",
        "",
        _unsupported_table(rt.UNSUPPORTED_CONVERSIONS),
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
