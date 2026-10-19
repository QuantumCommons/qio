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
"""Session hooks for the circuit conversion battery.

``pytest_terminal_summary`` prints the information losses observed during the
run (the ones declared as inherent to each conversion path). For each observed
loss it explains the category (``loss_description``) and lists the affected
circuits with the concrete deviation recorded as detail. This makes the
conversion losses explicit in the output instead of being silently tolerated.
"""

import pytest

import circuit_testing as ct


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    report = ct.loss_report()
    terminalreporter.section("conversion information losses observed (declared & inherent)")
    if not report:
        terminalreporter.write_line("  none")
        return

    terminalreporter.write_line("\n  circuit feature coverage (axis | status | note):")
    width = max(len(axis) for axis, _, _ in ct.CIRCUIT_FEATURE_COVERAGE)
    for axis, status, note in ct.CIRCUIT_FEATURE_COVERAGE:
        terminalreporter.write_line(f"    {axis:{width}s} | {status:19s} | {note}")

    if not report:
        terminalreporter.write_line("  conversion information losses: none declared/observed")
        return

    categories = sorted({category for edges in report.values() for category in edges})
    terminalreporter.write_line("\n  loss categories & meaning:")
    for category in categories:
        terminalreporter.write_line(f"    - {category:28s}")
        terminalreporter.write_line(f"{ct.loss_description(category)}")

    terminalreporter.write_line("\n  observations (edge | category | affected circuits):")
    for edge_id, categories_map in sorted(report.items()):
        first = True
        for category, entry in sorted(categories_map.items()):
            circuits = ", ".join(sorted(entry["circuits"]))
            if first:
                terminalreporter.write_line(f"    {edge_id:35s} {category:28s} {circuits}")
                first = False
            else:
                terminalreporter.write_line(f"    {'':35s} {category:28s} {circuits}")
            for circuit, detail in sorted(entry["observations"]):
                suffix = f' : "{detail}"' if detail else ""
                terminalreporter.write_line(f"      - {circuit}{suffix}")