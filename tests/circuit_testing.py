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
"""Modular test helpers for the circuit conversion battery.

Design:
- **One format checker per SDK / per serialization format** : ``check_cirq``,
  ``check_qiskit``, ``check_cudaq``, ``check_mimiq``, ``check_program``,
  ``check_qasm``, ``check_cirq_json``. Each validates the format of the object
  and that the expected known information (``spec``, a :class:`ReferenceCircuit`)
  holds.
- **Generic driver** : ``convert(circuit, converter, ...)`` takes the input
  circuit and the conversion function to call as an argument, and runs the
  verifications (before: format of the input; after: output checks). It is
  composed into multi-step pipelines by ``run_path``.
- **Declarative registries** : ``build_edges()`` returns the full conversion
  graph (SDK -> format -> SDK) as :class:`ConversionEdge` rows; adding a
  conversion is adding one call. ``UNSUPPORTED_CONVERSIONS`` hosts the negative
  cases.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from qio.core import (
    QuantumProgram,
    QuantumProgramCompressionFormat,
    QuantumProgramSerializationFormat,
)

from reference_circuits import ReferenceCircuit, get_reference_circuit

Compression = QuantumProgramCompressionFormat
Serialization = QuantumProgramSerializationFormat
NONE = Compression.NONE
ZLIB = Compression.ZLIB_BASE64_V1


# ---------------------------------------------------------------------------
# Format checkers (one per SDK / serialization format)
# Signature: check(result, expected: Optional[ReferenceCircuit]) -> None
# ---------------------------------------------------------------------------
def check_cirq(result, expected: Optional[ReferenceCircuit] = None) -> None:
    import cirq

    assert isinstance(result, cirq.Circuit), f"expected cirq.Circuit, got {type(result)}"
    if expected is not None:
        assert len(result.all_qubits()) == expected.n_qubits, (
            len(result.all_qubits()),
            expected.n_qubits,
        )


def check_qiskit(result, expected: Optional[ReferenceCircuit] = None) -> None:
    from qiskit import QuantumCircuit

    assert isinstance(result, QuantumCircuit), f"expected QuantumCircuit, got {type(result)}"
    if expected is not None:
        assert result.num_qubits == expected.n_qubits, (
            result.num_qubits,
            expected.n_qubits,
        )


def check_cudaq(result, expected: Optional[ReferenceCircuit] = None) -> None:
    # A CUDA-Q kernel cannot be introspected without executing it; the actual
    # behavior is validated by the "counts" oracle.
    assert result is not None, "CUDA-Q kernel is None"


def check_mimiq(result, expected: Optional[ReferenceCircuit] = None) -> None:
    import mimiqcircuits

    assert result is not None, "MIMIQ circuit is None"
    assert isinstance(result, mimiqcircuits.Circuit), f"expected mimiq Circuit, got {type(result)}"
    num_qubits = getattr(result, "num_qubits", None)
    if callable(num_qubits):
        num_qubits = num_qubits()
    if expected is not None and num_qubits is not None:
        assert num_qubits == expected.n_qubits, (num_qubits, expected.n_qubits)


def check_program(serialization_format: Serialization) -> Callable:
    """Validates a QuantumProgram intermediate: format matches and serialization
    is not empty."""

    def _check(result, expected: Optional[ReferenceCircuit] = None) -> None:
        assert isinstance(result, QuantumProgram), type(result)
        assert result.serialization_format == serialization_format, (
            result.serialization_format,
            serialization_format,
        )
        assert bool(result.serialization), "empty serialization"

    return _check


def check_qasm(version: int) -> Callable:
    """Validates the serialized OpenQASM ``version`` (2 or 3) content: it must
    parse and, when an expected circuit is provided, declare the right qubit
    count."""

    def _check(result, expected: Optional[ReferenceCircuit] = None) -> None:
        source = result.serialization if isinstance(result, QuantumProgram) else result
        assert source, "empty QASM string"
        num_qubits = _qasm_num_qubits(source, version)
        if expected is not None:
            assert num_qubits == expected.n_qubits, (num_qubits, expected.n_qubits)

    return _check


def check_cirq_json(result, expected: Optional[ReferenceCircuit] = None) -> None:
    import cirq

    source = result.serialization if isinstance(result, QuantumProgram) else result
    assert source, "empty Cirq JSON"
    circuit = cirq.read_json(json_text=source)
    assert isinstance(circuit, cirq.Circuit), type(circuit)
    if expected is not None:
        assert len(circuit.all_qubits()) == expected.n_qubits, (
            len(circuit.all_qubits()),
            expected.n_qubits,
        )


def _qasm_num_qubits(source: str, version: int) -> int:
    if version == 2:
        from qiskit import qasm2

        return qasm2.loads(source).num_qubits

    import openqasm3
    from openqasm3 import ast

    program = openqasm3.parse(source)
    num_qubits = 0
    for statement in program.statements:
        if isinstance(statement, ast.QubitDeclaration):
            magnitude = getattr(statement.size, "value", None)
            num_qubits += int(magnitude) if magnitude is not None else 1
    if num_qubits == 0:
        raise AssertionError(f"no qubit declaration found in QASM {version}")
    return num_qubits


# ---------------------------------------------------------------------------
# Equivalence oracles
# ---------------------------------------------------------------------------
def _cirq_unitary(circuit) -> Any:
    import cirq
    from cirq import MeasurementGate

    operations = [
        op for moment in circuit for op in moment if not isinstance(op.gate, MeasurementGate)
    ]
    return cirq.unitary(cirq.Circuit(operations))


def _qiskit_unitary(qc) -> Any:
    from qiskit.quantum_info import Operator

    return Operator(qc.remove_final_measurements(inplace=False)).data


def _reverse_bit_order(matrix) -> Any:
    """Permutes a 2^n x 2^n matrix so that qubit 0 becomes the most significant
    bit (Cirq ordering) instead of the least significant (Qiskit ordering)."""
    import numpy as np

    arr = np.asarray(matrix)
    size = arr.shape[0]
    num_qubits = int(round(np.log2(size)))
    permutation = [
        sum(((i >> k) & 1) << (num_qubits - 1 - k) for k in range(num_qubits))
        for i in range(size)
    ]
    return arr[np.ix_(permutation, permutation)]


def assert_unitaries_close(reference, actual) -> None:
    """Asserts two circuit unitaries are equal up to a global phase. Uses a
    bit-reversal fallback so Cirq- and Qiskit-ordered matrices are compared
    consistently."""
    if _unit_close(reference, actual):
        return
    if _unit_close(reference, _reverse_bit_order(actual)):
        return
    raise AssertionError("converted circuit unitary differs from the reference")


def _unit_close(reference, actual, atol: float = 1e-8) -> bool:
    import numpy as np

    ref = np.asarray(reference, dtype=complex)
    act = np.asarray(actual, dtype=complex)
    if ref.shape != act.shape:
        return False
    inner = np.vdot(ref, act)
    if abs(inner) < 1e-12:
        return bool(np.allclose(ref, act, atol=atol))
    phase = inner / abs(inner)
    return bool(np.allclose(phase * ref, act, atol=atol, rtol=1e-6))


def assert_counts(kernel, expected: ReferenceCircuit) -> None:
    import cudaq

    assert expected.expected_counts is not None, f"{expected.name} is not deterministic"
    counts = dict(cudaq.sample(kernel, shots_count=500).items())
    actual_keys = set(counts.keys())
    assert actual_keys == set(expected.expected_counts.keys()), (actual_keys, expected)


# ---------------------------------------------------------------------------
# Generic driver + pipeline
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    fn: Callable
    kwargs: Dict[str, Any] = field(default_factory=dict)
    checks: Tuple[Callable, ...] = ()


@dataclass(frozen=True)
class ConversionEdge:
    id: str
    input_fn: Callable[[ReferenceCircuit], Any]
    steps: Tuple[Step, ...]
    oracle: str = "unitary"  # "unitary" | "counts" | "structural"


def convert(circuit, converter, *args, checks=(), expected=None, **kwargs) -> Any:
    """Generic conversion driver.

    Args:
        circuit: the input circuit (format checked upfront).
        converter: the conversion function to call (e.g. ``QuantumProgram.from_*_circuit``).
        checks: post-conversion verifications, each ``check(result, expected)``.
        expected: known information (a ReferenceCircuit) checks are validated against.
    """
    assert circuit is not None, "input circuit is required"
    result = converter(circuit, *args, **kwargs)
    for check in checks:
        check(result, expected)
    return result


def run_path(edge: ConversionEdge, circuit: ReferenceCircuit) -> Any:
    """Runs a full conversion path (input -> steps) and applies its oracle."""
    current = edge.input_fn(circuit)
    for step in edge.steps:
        current = convert(current, step.fn, expected=circuit, checks=step.checks, **step.kwargs)

    if edge.oracle == "unitary":
        actual = _to_unitary(current)
        assert_unitaries_close(_cirq_unitary(circuit.cirq()), actual)
    elif edge.oracle == "counts":
        assert_counts(current, circuit)
    elif edge.oracle == "structural":
        # structural checks were already run by the steps
        pass
    else:
        raise ValueError(f"unknown oracle: {edge.oracle}")
    return current


def _to_unitary(circuit) -> Any:
    import cirq
    from qiskit import QuantumCircuit

    if isinstance(circuit, cirq.Circuit):
        return _cirq_unitary(circuit)
    if isinstance(circuit, QuantumCircuit):
        return _qiskit_unitary(circuit)
    raise TypeError(f"cannot extract unitary from {type(circuit)}")


# ---------------------------------------------------------------------------
# Declarative registries
# ---------------------------------------------------------------------------
_INPUT_FNS = {
    "cirq": lambda c: c.cirq(),
    "qiskit": lambda c: c.qiskit(),
}


def build_edges(compression: Compression) -> Tuple[ConversionEdge, ...]:
    """Builds the conversion graph (SDK -> format -> SDK) for one compression."""
    content_checks_ok = compression == NONE
    producer = QuantumProgram.from_cirq_circuit
    edges = []

    def add(
        edge_id: str,
        input_kind: str,
        write_fn: Callable,
        fmt: Serialization,
        read_fn: Optional[Callable] = None,
        read_check: Optional[Callable] = None,
        oracle: str = "unitary",
    ) -> None:
        checks = (check_program(fmt),) + (
            _format_content_checks(fmt) if content_checks_ok else ()
        )
        steps = [
            Step(write_fn, {"dest_format": fmt, "compression_format": compression}, checks)
        ]
        if read_fn is not None:
            steps.append(Step(read_fn, {}, (read_check,)))
        edges.append(
            ConversionEdge(edge_id, _INPUT_FNS[input_kind], tuple(steps), oracle=oracle)
        )

    add("cirq->qasm2->cirq", "cirq", producer, Serialization.QASM_V2, QuantumProgram.to_cirq_circuit, check_cirq)
    add("cirq->qasm3->cirq", "cirq", producer, Serialization.QASM_V3, QuantumProgram.to_cirq_circuit, check_cirq)
    add("cirq->cirqjson->cirq", "cirq", producer, Serialization.CIRQ_CIRCUIT_JSON_V1, QuantumProgram.to_cirq_circuit, check_cirq)
    add("cirq->qasm2->qiskit", "cirq", producer, Serialization.QASM_V2, QuantumProgram.to_qiskit_circuit, check_qiskit)
    add("cirq->qasm3->qiskit", "cirq", producer, Serialization.QASM_V3, QuantumProgram.to_qiskit_circuit, check_qiskit)
    add("cirq->qasm3->cudaq", "cirq", producer, Serialization.QASM_V3, QuantumProgram.to_cudaq_kernel, check_cudaq, oracle="counts")
    add("cirq->qasm2->mimiq", "cirq", producer, Serialization.QASM_V2, QuantumProgram.to_mimiq_circuit, check_mimiq, oracle="structural")

    producer = QuantumProgram.from_qiskit_circuit
    add("qiskit->qasm2->qiskit", "qiskit", producer, Serialization.QASM_V2, QuantumProgram.to_qiskit_circuit, check_qiskit)
    add("qiskit->qasm3->qiskit", "qiskit", producer, Serialization.QASM_V3, QuantumProgram.to_qiskit_circuit, check_qiskit)
    add("qiskit->qasm2->cirq", "qiskit", producer, Serialization.QASM_V2, QuantumProgram.to_cirq_circuit, check_cirq)
    add("qiskit->qasm3->cirq", "qiskit", producer, Serialization.QASM_V3, QuantumProgram.to_cirq_circuit, check_cirq)
    add("qiskit->qasm3->cudaq", "qiskit", producer, Serialization.QASM_V3, QuantumProgram.to_cudaq_kernel, check_cudaq, oracle="counts")
    add("qiskit->qasm2->cudaq", "qiskit", producer, Serialization.QASM_V2, QuantumProgram.to_cudaq_kernel, check_cudaq, oracle="counts")
    add("qiskit->qasm2->mimiq", "qiskit", producer, Serialization.QASM_V2, QuantumProgram.to_mimiq_circuit, check_mimiq, oracle="structural")

    return tuple(edges)


def _format_content_checks(fmt: Serialization) -> Tuple[Callable, ...]:
    if fmt in (Serialization.QASM_V2,):
        return (check_qasm(2),)
    if fmt in (Serialization.QASM_V3,):
        return (check_qasm(3),)
    if fmt in (Serialization.CIRQ_CIRCUIT_JSON_V1,):
        return (check_cirq_json,)
    return ()


# Negative/edge cases: each entry is (id, callable) and must raise an Exception.
def _bell_cirq():
    return get_reference_circuit("bell").cirq()


def _bell_qiskit():
    return get_reference_circuit("bell").qiskit()


def _bell_cudaq():
    return get_reference_circuit("bell").cudaq_kernel()


def _cirq_program(dest_format: Serialization) -> QuantumProgram:
    return QuantumProgram.from_cirq_circuit(_bell_cirq(), dest_format=dest_format)


UNSUPPORTED_CONVERSIONS = (
    ("from_cirq_circuit -> QASM_V1", lambda: QuantumProgram.from_cirq_circuit(_bell_cirq(), dest_format=Serialization.QASM_V1)),
    ("from_qiskit_circuit -> QASM_V1", lambda: QuantumProgram.from_qiskit_circuit(_bell_qiskit(), dest_format=Serialization.QASM_V1)),
    ("from_cudaq_kernel -> QASM_V3", lambda: QuantumProgram.from_cudaq_kernel(_bell_cudaq(), dest_format=Serialization.QASM_V3)),
    ("from_cudaq_kernel -> CIRQJSON", lambda: QuantumProgram.from_cudaq_kernel(_bell_cudaq(), dest_format=Serialization.CIRQ_CIRCUIT_JSON_V1)),
    ("to_qasm2_circuit on QASM_V3", lambda: _cirq_program(Serialization.QASM_V3).to_qasm2_circuit()),
    ("to_mimiq_circuit on QASM_V3", lambda: _cirq_program(Serialization.QASM_V3).to_mimiq_circuit()),
    ("to_mimiq_circuit on CIRQJSON", lambda: _cirq_program(Serialization.CIRQ_CIRCUIT_JSON_V1).to_mimiq_circuit()),
    ("to_cudaq_kernel on CIRQJSON", lambda: _cirq_program(Serialization.CIRQ_CIRCUIT_JSON_V1).to_cudaq_kernel()),
)
