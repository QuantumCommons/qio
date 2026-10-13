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
MIMIQ, Perceval, Pulser - through the per-SDK adapters in ``result_sdks/``
(``ResultSDK.build(reference)``); this module itself is pure data and imports
no SDK.

The bitstring convention is fixed across all SDK fixtures: bitstring character
``k`` (from the left, ``0``-indexed) is the measured value of qubit ``k`` on
the source circuit. Every fixture is built from the *same* canonical bitstring
set, so every converter (which only re-serializes the bitstrings) must preserve
them exactly: the result battery compares counts histograms bit-exactly.

Every reference also carries as much *execution metadata* as its format can
hold (backend identity, job identifiers, measurement date, register name,
statevector, MIMIQ ``zstates``/``fidelities``/``avggateerrors``/``timings``).
Because the whole battery is driven from the SDK **object** (the intermediate
dicts are produced by qio's own ``<sdk>_to_dict`` converters), filling these
fields lets the battery make the information loss of each conversion path
explicit in the report.

Serialization note: the intermediate representations are plain JSON/zlib, so
fields that are not JSON-safe can never cross the intermediate boundary. They
are kept out of the *input* fixtures on purpose and only checked in memory on
the output objects:

* cirq ``params`` - ``cirq.ParamResolver`` is not JSON-serializable
  (``ResultDict._json_dict_`` emits a live resolver; see bellow).
* qiskit ``statevector`` - stored here as a real-valued amplitude list, which
  ``Result.to_dict()`` keeps JSON-safe.
* mimiq ``amplitudes`` - keyed by ``bitarray`` and complex-valued; not JSON
  safe. When a reference carries a ``statevector``, the qiskit->mimiq edge
  reconstructs these amplitudes in memory and the checker validates them.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ReferenceResult:
    name: str
    n_qubits: int
    shots: int
    counts: Dict[str, int]
    description: str = field(default="")
    # Execution identity (SDK-neutral facts of the "executed" run).
    backend_name: str = "qio-test-simulator"
    backend_version: str = "1.0"
    job_id: str = "qio-result-job"
    qobj_id: str = "qio-result-qobj"
    date: str = "2026-01-01T00:00:00+00:00"
    # CUDA-Q measurement register name (default when sampling the qubit qreg).
    register_name: str = "q"
    # Real-valued reference statevector (length 2^n_qubits), JSON-safe through
    # qiskit Result.to_dict(). Length n bitstring index = qubit 0 is MSB.
    statevector: Optional[Sequence[float]] = None
    # MIMIQ QCSResults-only metadata.
    fidelities: Optional[Sequence[float]] = None
    avggateerrors: Optional[Sequence[float]] = None
    zstates: Optional[Sequence[str]] = None
    timings: Optional[Dict[str, float]] = None

    def _expected_amplitudes(self) -> Dict[str, complex]:
        """Amplitudes reconstructed from the statevector (qiskit->mimiq path)."""
        amplitudes = {}
        if self.statevector is not None:
            for index, amp in enumerate(self.statevector):
                if abs(complex(amp)) > 1e-10:
                    bitstring = format(index, f"0{self.n_qubits}b")
                    amplitudes[bitstring] = complex(amp)
        return amplitudes

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
        if self.statevector is not None:
            expect = 2**self.n_qubits
            if len(self.statevector) != expect:
                raise ValueError(
                    f"{self.name}: statevector has {len(self.statevector)} "
                    f"amplitudes, expected {expect}"
                )
            if not all(isinstance(a, (int, float)) for a in self.statevector):
                raise ValueError(
                    f"{self.name}: statevector must be real-valued (JSON-safe)"
                )


_RT2 = 0.7071067811865475


def _ghz3_statevector() -> List[float]:
    sv = [0.0] * 8
    sv[0] = _RT2
    sv[7] = _RT2
    return sv


REFERENCE_RESULTS: Tuple[ReferenceResult, ...] = (
    ReferenceResult(
        name="bell2",
        n_qubits=2,
        shots=1000,
        counts={"00": 509, "11": 491},
        description="2-qubit Bell-state sampling.",
        statevector=[_RT2, 0.0, 0.0, _RT2],
        fidelities=[0.995],
        avggateerrors=[0.002],
        zstates=["00"],
        timings={"total": 0.0123, "apply": 0.0091},
    ),
    ReferenceResult(
        name="x11",
        n_qubits=2,
        shots=1000,
        counts={"11": 1000},
        description="Deterministic 2-qubit result: x(0), cx(0, 1) -> 11.",
        statevector=[0.0, 0.0, 0.0, 1.0],
        fidelities=[1.0],
        avggateerrors=[0.0],
        zstates=["11"],
        timings={"total": 0.0087, "apply": 0.0066},
    ),
    ReferenceResult(
        name="ghz3",
        n_qubits=3,
        shots=800,
        counts={"000": 401, "111": 399},
        description="3-qubit GHZ-state sampling.",
        statevector=_ghz3_statevector(),
        fidelities=[0.98],
        avggateerrors=[0.01],
        zstates=["000"],
        timings={"total": 0.0231, "apply": 0.0184},
    ),
    ReferenceResult(
        name="single1",
        n_qubits=1,
        shots=100,
        counts={"1": 100},
        description="Deterministic single-qubit result.",
        statevector=[0.0, 1.0],
        fidelities=[1.0],
        avggateerrors=[0.0],
        zstates=["1"],
        timings={"total": 0.0031, "apply": 0.0022},
    ),
    ReferenceResult(
        name="mixed01",
        n_qubits=2,
        shots=600,
        counts={"01": 301, "10": 299},
        description="Asymmetric 2-qubit result (exercises bit ordering).",
        statevector=[0.0, _RT2, _RT2, 0.0],
        fidelities=[0.99],
        avggateerrors=[0.003],
        zstates=["01"],
        timings={"total": 0.0119, "apply": 0.0083},
    ),
)

_REFERENCE_RESULT_INDEX = {c.name: c for c in REFERENCE_RESULTS}


def get_reference_result(name: str) -> ReferenceResult:
    """Statically build a controlled reference result by name."""
    if name not in _REFERENCE_RESULT_INDEX:
        raise KeyError(
            f"unknown reference result {name!r}; available: "
            f"{sorted(_REFERENCE_RESULT_INDEX)}"
        )
    return _REFERENCE_RESULT_INDEX[name]
