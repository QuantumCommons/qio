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

import datetime
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData


def _make_expresult_from_mimiq_qcsr(qcsr: dict, **kwargs) -> ExperimentResult:

    histogram = qcsr.get("histogram", {})
    shots = kwargs.get(
        "shots", qcsr.get("shots", qcsr.get("samples", qcsr.get("executions", 0)))
    )

    num_qubits = kwargs.get("num_qubits")
    if num_qubits is None and histogram:
        num_qubits = len(list(histogram.keys())[0])
    else:
        num_qubits = num_qubits or 0

    exp_result_data = ExperimentResultData(counts=histogram)

    if "amplitudes" in qcsr and qcsr["amplitudes"]:
        exp_result_data.statevector = qcsr["amplitudes"]

    return ExperimentResult(
        shots=shots,
        success=True,
        status="DONE",
        data=exp_result_data,
        header={
            "n_qubits": num_qubits,
            "memory_slots": num_qubits,
            "name": "quantanium_circuit",
        },
    )


def convert(qcsr: dict, **kwargs) -> Result:
    """
    Convert a mimiqcircuits.qcsresults object into a qiskit.result format.
    """
    kwargs = kwargs or {}

    backend_name = kwargs.pop("backend_name", qcsr.get("simulator", "Quantanium"))
    backend_version = kwargs.pop("backend_version", qcsr.get("backend_version", "1.0"))
    job_id = kwargs.pop("job_id", "unknown")
    qobj_id = kwargs.pop("qobj_id", job_id)
    success = kwargs.pop("success", True)
    status = kwargs.pop("status", "DONE")
    date = kwargs.pop("date", datetime.datetime.now().isoformat())

    kwargs.pop("results", None)

    return Result(
        backend_name=backend_name,
        backend_version=backend_version,
        job_id=job_id,
        qobj_id=qobj_id,
        success=success,
        results=[_make_expresult_from_mimiq_qcsr(qcsr, **kwargs)],
        date=date,
        status=status,
        **kwargs,
    )
