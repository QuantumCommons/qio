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
"""Static reference result fixtures for the result conversion battery.

Unlike the SDK tests (which simulate circuits at test time), each
:class:`ReferenceResult` describes an **already executed** run as a static
counts histogram (``counts``: ``{bitstring: count}``) plus a shot count. It is
exposed identically in every SDK result representation - Cirq, Qiskit, CUDA-Q,
MIMIQ - through a single generator ``get_reference_result``.

The bitstring convention is fixed across all SDK fixtures: bitstring character
``k`` (from the left, ``0``-indexed) is the measured value of qubit ``k`` on
the source circuit. Every fixture is built from the *same* canonical bitstring
set, so every converter (which only re-serializes the bitstrings) must preserve
them exactly: the result battery compares counts histograms bit-exactly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class ReferenceResult:
    name: str
    n_qubits: int
    shots: int
    counts: Dict[str, int]
    description: str = field(default="")

    def __post_init__(self) -> None:
        total = sum(self.counts.values())
        if total != self.shots:
            raise ValueError(
                f"{self.name}: counts sum {total} != shots {self.shots}"
            )
        for bitstring in self.counts:
            if len(bitstring) != self.n_qubits:
                raise ValueError(
                    f"{self.name}: bitstring {bitstring!r} has length "
                    f"{len(bitstring)}, expected {self.n_qubits}"
                )

    # SDK builders - all static, no execution involved.
    #
    # Cirq: one measurement key per qubit (``m<k>``), matching the way
    # ``cirq_to_qiskit`` / ``cirq_to_mimiq`` reconstruct bitstrings from the
    # per-key records (sorted keys ``m0, m1, ...`` -> bitstring ``b0 b1 ...``).

    def cirq(self) -> "cirq.Result":
        import cirq
        import numpy as np

        measurements = {
            f"m{k}": np.zeros((self.shots, 1), dtype=np.int8)
            for k in range(self.n_qubits)
        }
        index = 0
        for bitstring, count in self.counts.items():
            bits = [int(b) for b in bitstring]
            for _ in range(count):
                for k in range(self.n_qubits):
                    measurements[f"m{k}"][index, 0] = bits[k]
                index += 1
        if index != self.shots:
            raise AssertionError(f"{self.name}: {index} shots != {self.shots}")
        result = cirq.ResultDict(
            params=cirq.ParamResolver({}), measurements=measurements
        )
        # ``ParamResolver`` is not JSON serializable: mirror the SDK tests
        # (``test_cirq.py`` sets ``_params = None``) so the result dict can be
        # compressed through qio.
        result._params = None
        return result

    def cirq_dict(self) -> dict:
        """The Cirq result serialized as a dict (cirq_to_dict output)."""
        return self.cirq()._json_dict_()

    # Qiskit: a Result assembled from the counts histogram (no backend).

    def qiskit(self) -> "qiskit.result.Result":
        from qiskit.result import Result

        return Result.from_dict(
            {
                "backend_name": "qio_test_simulator",
                "backend_version": "1.0",
                "qobj_id": "qio-result-test",
                "job_id": "qio-result-test",
                "success": True,
                "status": "COMPLETED",
                "results": [
                    {
                        "shots": self.shots,
                        "success": True,
                        "status": "DONE",
                        "header": {
                            "name": self.name,
                            "n_qubits": self.n_qubits,
                            "memory_slots": self.n_qubits,
                            "qreg_sizes": [["q", self.n_qubits]],
                            "creg_sizes": [["m", self.n_qubits]],
                        },
                        "data": {"counts": dict(self.counts)},
                    }
                ],
            }
        )

    def qiskit_dict(self) -> dict:
        """The Qiskit result serialized as a dict (qiskit_to_dict output)."""
        return self.qiskit().to_dict()

    # CUDA-Q: a SampleResult reconstructed by ``deserialize`` from a
    # hand-built serialized blob (the wire format documented by
    # ``cudaq_sample_to_qiskit``): register name, then per bitstring the
    # triplet ``[value, bit_size, count]``.

    def _cudaq_serialize(self) -> List[int]:
        register_name = "q"
        data: List[int] = [len(register_name)]
        data.extend(ord(ch) for ch in register_name)
        data.append(len(self.counts))
        for bitstring, count in self.counts.items():
            data.append(int(bitstring, 2))
            data.append(len(bitstring))
            data.append(count)
        return data

    def cudaq(self) -> "cudaq.SampleResult":
        import cudaq

        sample_result = cudaq.SampleResult()
        sample_result.deserialize(self._cudaq_serialize())
        return sample_result

    # MIMIQ: a QCSResults populated with the classical states of the run.

    def mimiq(self) -> "mimiqcircuits.QCSResults":
        from bitarray import frozenbitarray
        from mimiqcircuits import QCSResults

        cstates: List[frozenbitarray] = []
        for bitstring, count in self.counts.items():
            cstates.extend([frozenbitarray(bitstring)] * count)
        return QCSResults(
            simulator="linalg",
            version="0.1",
            cstates=cstates,
        )

    def mimiq_dict(self) -> dict:
        """The MIMIQ QCSR serialized as a dict (mimiq_to_dict output)."""
        return {
            "simulator": "linalg",
            "version": "0.1",
            "timings": None,
            "fidelity_estimate": None,
            "average_multi_qubit_gate_error_estimate": None,
            "executions": None,
            "samples": self.shots,
            "amplitudes": None,
            "histogram": dict(self.counts),
        }


REFERENCE_RESULTS: Sequence[ReferenceResult] = [
    ReferenceResult(
        name="bell2",
        n_qubits=2,
        shots=1000,
        counts={"00": 509, "11": 491},
        description="2-qubit Bell-state sampling.",
    ),
    ReferenceResult(
        name="x11",
        n_qubits=2,
        shots=1000,
        counts={"11": 1000},
        description="Deterministic 2-qubit result: x(0), cx(0, 1) -> 11.",
    ),
    ReferenceResult(
        name="ghz3",
        n_qubits=3,
        shots=800,
        counts={"000": 401, "111": 399},
        description="3-qubit GHZ-state sampling.",
    ),
    ReferenceResult(
        name="single1",
        n_qubits=1,
        shots=100,
        counts={"1": 100},
        description="Deterministic single-qubit result.",
    ),
    ReferenceResult(
        name="mixed01",
        n_qubits=2,
        shots=600,
        counts={"01": 301, "10": 299},
        description="Asymmetric 2-qubit result (exercises bit ordering).",
    ),
]

_REFERENCE_RESULT_INDEX = {c.name: c for c in REFERENCE_RESULTS}


def get_reference_result(name: str) -> ReferenceResult:
    """Statically build a controlled reference result by name."""
    if name not in _REFERENCE_RESULT_INDEX:
        raise KeyError(
            f"unknown reference result {name!r}; available: "
            f"{sorted(_REFERENCE_RESULT_INDEX)}"
        )
    return _REFERENCE_RESULT_INDEX[name]
