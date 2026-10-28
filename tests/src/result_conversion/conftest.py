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
"""Session hooks for the result conversion battery.

``pytest_terminal_summary`` prints the information losses observed during the
run (the ones declared as inherent to each conversion path). For each observed
loss it explains the category (``loss_description``) and lists the affected
results with the concrete deviation recorded as detail.
"""

import pytest

import result_testing as rt


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    report = rt.loss_report()
    # Always dump when QIO_LOSS_REPORT_JSON is set, even for an empty run,
    # so each paradigm writes its JSON and the generator can merge cleanly.
    _dump_loss_report(report)
    terminalreporter.section(
        "result conversion information losses observed (declared & inherent)"
    )
    if not report:
        terminalreporter.write_line("  none")
        return

    categories = sorted({category for edges in report.values() for category in edges})
    terminalreporter.write_line("\n  loss categories & meaning:")
    for category in categories:
        terminalreporter.write_line(f"    - {category:28s}")
        terminalreporter.write_line(f"{rt.loss_description(category)}")

    terminalreporter.write_line(
        "\n  observations (edge | category | affected results):"
    )
    for edge_id, categories_map in sorted(report.items()):
        first = True
        for category, entry in sorted(categories_map.items()):
            results = ", ".join(sorted(entry["results"]))
            if first:
                terminalreporter.write_line(
                    f"\n    {edge_id:65s} {category:28s} {results}"
                )
                first = False
            else:
                terminalreporter.write_line(f"    {'':65s} {category:28s} {results}")
            for result, detail, _o, _e, _f in sorted(
                entry["observations"], key=_obs_key
            ):
                suffix = f' : "{detail}"' if detail else ""
                terminalreporter.write_line(f"      - {result}{suffix}")


def _obs_key(obs):
    return tuple(str(x) for x in obs)


def _dump_loss_report(report) -> None:
    """Writes the session loss report to ``QIO_LOSS_REPORT_JSON`` when set."""
    import json
    import os

    path = os.environ.get("QIO_LOSS_REPORT_JSON")
    if not path:
        return
    payload = {
        edge_id: {
            category: {
                "results": sorted(entry["results"]),
                "observations": sorted(entry["observations"], key=_obs_key),
            }
            for category, entry in categories.items()
        }
        for edge_id, categories in report.items()
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
