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
import copy
from datetime import datetime

from qiskit.providers import JobStatus
from qiskit.result import Result


def sanitize_qiskit_result(result):
    if result is None:
        return result

    if hasattr(result, "status"):
        if isinstance(result.status, JobStatus) or hasattr(result.status, "name"):
            result.status = result.status.name
        elif not isinstance(result.status, (str, type(None))):
            result.status = str(result.status)

    if hasattr(result, "date") and result.date is not None:
        if isinstance(result.date, datetime):
            result.date = result.date.isoformat()
        elif not isinstance(result.date, str):
            result.date = str(result.date)

    if hasattr(result, "results") and result.results:
        for exp in result.results:
            if hasattr(exp, "status") and (
                isinstance(exp.status, JobStatus) or hasattr(exp.status, "name")
            ):
                exp.status = exp.status.name
            if hasattr(exp, "date") and isinstance(exp.date, datetime):
                exp.date = exp.date.isoformat()

    return result


def convert(qiskit_result: Result) -> dict:
    return sanitize_qiskit_result(copy.deepcopy(qiskit_result)).to_dict()
