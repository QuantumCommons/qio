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
from mimiqcircuits import QCSResults


def convert(qcsr: QCSResults, **kwargs) -> dict:
    """
    Convert a mimiqcircuits.qcsresults object into a dict.
    """

    raw_counts = qcsr.histogram()
    histogram = {key.to01(): int(val) for key, val in raw_counts.items()}

    result_dict = {
        "simulator": getattr(qcsr, "simulator", None),
        "version": getattr(qcsr, "version", None),
        "timings": getattr(qcsr, "timings", None),
        "fidelity_estimate": getattr(qcsr, "fidelity_estimate", None),
        "average_multi_qubit_gate_error_estimate": getattr(
            qcsr, "average_multi_qubit_gate_error_estimate", None
        ),
        "executions": getattr(qcsr, "executions", None),
        "samples": getattr(qcsr, "samples", None),
        "amplitudes": getattr(qcsr, "amplitudes", None),
        "histogram": histogram,
    }

    if kwargs:
        result_dict.update(kwargs)

    return result_dict
