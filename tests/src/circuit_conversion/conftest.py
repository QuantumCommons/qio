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
run (the ones declared as inherent to each conversion path). This makes the
conversion losses explicit in the output instead of being silently tolerated.
"""

import pytest

import circuit_testing as ct


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    report = ct.loss_report()
    rows = [
        (edge_id, category, ", ".join(sorted(circuits)))
        for edge_id, categories in sorted(report.items())
        for category, circuits in sorted(categories.items())
    ]
    terminalreporter.section("conversion information losses observed (declared & inherent)")
    if not rows:
        terminalreporter.write_line("  none")
        return
    for edge_id, category, circuits in rows:
        terminalreporter.write_line(f"  {edge_id:28s} {category:22s} {circuits}")
