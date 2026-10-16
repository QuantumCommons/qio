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
"""Qiskit result adapter for the battery: input builder, format check, metadata
probes and the per-source facts (dict converter, wrap/read classmethods,
inherent losses)."""

from typing import Any, Dict, Optional, Sequence

from qio.core import (
    QuantumProgramResult,
    QuantumProgramResultSerializationFormat as Serialization,
)

from reference_results import ReferenceResult
from result_testing import (
    LOSS_QISKIT_METADATA,
    LOSS_QISKIT_STATEVECTOR,
    Probe,
    ResultSDK,
    _assert_counts,
    _run_probes,
)


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _qiskit_statevector(result: Any) -> Optional[Sequence[float]]:
    try:
        sv = result.data(0).statevector
    except Exception:
        return None
    return list(sv) if sv is not None else None


QISKIT_PROBES = (
    Probe(LOSS_QISKIT_METADATA, "backend_name", lambda r: r.backend_name, lambda ref: ref.backend_name),
    Probe(LOSS_QISKIT_METADATA, "backend_version", lambda r: r.backend_version, lambda ref: ref.backend_version),
    Probe(LOSS_QISKIT_METADATA, "job_id", lambda r: r.job_id, lambda ref: ref.job_id),
    Probe(LOSS_QISKIT_METADATA, "qobj_id", lambda r: r.qobj_id, lambda ref: ref.qobj_id),
    Probe(LOSS_QISKIT_METADATA, "date", lambda r: r.date, lambda ref: ref.date, normalize=_iso),
    Probe(
        LOSS_QISKIT_STATEVECTOR,
        "statevector",
        _qiskit_statevector,
        lambda ref: list(ref.statevector) if ref.statevector is not None else None,
        normalize=lambda v: [float(x) for x in v] if v is not None else None,
    ),
)


def _normalize_hex_counts(raw: Dict[Any, Any], n_qubits: int) -> Dict[str, int]:
    """Normalizes possibly-hex qiskit counts to plain bitstrings."""
    out: Dict[str, int] = {}
    for bitstring, count in raw.items():
        count = int(count)
        if isinstance(bitstring, str) and bitstring.startswith("0x"):
            bitstring = format(int(bitstring, 16), f"0{n_qubits}b")
        else:
            bitstring = str(bitstring)
        out[bitstring] = out.get(bitstring, 0) + count
    return out


class QiskitSDK(ResultSDK):
    kind = "qiskit"
    display = "qiskit.Result"
    fmt = Serialization.QISKIT_RESULT_JSON_V1
    fmt_label = "qiskit_result_json_v1"
    dict_keys = ("results",)
    to_dict_module = "qiskit_to_dict"
    from_dict = QuantumProgramResult.from_qiskit_result_dict
    wrap = QuantumProgramResult.from_qiskit_result
    read = QuantumProgramResult.to_qiskit_result
    # The qiskit job metadata + statevector cannot cross the plain-JSON
    # intermediate, so every edge *arriving* on a qiskit target declares them.
    target_losses = (LOSS_QISKIT_METADATA, LOSS_QISKIT_STATEVECTOR)
    probes = QISKIT_PROBES

    def build(self, reference: ReferenceResult):
        from qiskit.result import Result
        from qiskit.result.models import ExperimentResult, ExperimentResultData

        memory = []
        for bitstring, count in reference.counts.items():
            memory.extend([bitstring] * count)

        experiment = ExperimentResult(
            shots=reference.shots,
            success=True,
            status="DONE",
            data=ExperimentResultData(
                counts=dict(reference.counts),
                memory=memory,
                statevector=(
                    list(reference.statevector) if reference.statevector is not None else None
                ),
            ),
            header={
                "name": reference.name,
                "n_qubits": reference.n_qubits,
                "memory_slots": reference.n_qubits,
                "qreg_sizes": [["q", reference.n_qubits]],
                "creg_sizes": [["m", reference.n_qubits]],
                "metadata": {"source": "qio-result-fixture"},
            },
        )

        return Result(
            backend_name=reference.backend_name,
            backend_version=reference.backend_version,
            job_id=reference.job_id,
            qobj_id=reference.qobj_id,
            date=reference.date,
            status="COMPLETED",
            success=True,
            results=[experiment],
        )

    def check(self, result, expected: ReferenceResult = None) -> None:
        from qiskit.result import Result

        assert isinstance(result, Result), f"expected qiskit Result, got {type(result)}"
        if expected is not None:
            _assert_counts(_qiskit_counts(result, expected), expected)
            _run_probes(self.probes, result, expected)


def _qiskit_counts(result, expected: ReferenceResult = None) -> Dict[str, int]:
    experiment = result.to_dict()["results"][0]
    header = experiment.get("header", {}) or {}
    raw = dict((experiment.get("data", {}) or {}).get("counts", {}) or {})
    n_qubits = header.get("n_qubits")
    if n_qubits is None and raw:
        n_qubits = max(
            (len(str(k)) for k in raw if not (isinstance(k, str) and k.startswith("0x"))),
            default=0,
        )
    n_qubits = int(n_qubits or 0)
    return _normalize_hex_counts(raw, n_qubits)


SDK = QiskitSDK()
