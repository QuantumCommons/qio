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
"""Gate-based programming paradigm of the conversion battery.

Concrete :class:`Paradigm` of the *gate-based* circuits: the static reference
circuits (``bell``, ``ghz``, ``rotations``, ... defined here as
:class:`GateSpec`), the conversion graph
(``cirq/qiskit -> qasm2/qasm3/cirqjson -> {cirq, qiskit, cudaq, mimiq}``) and
every gate-oriented SDK check (Cirq / Qiskit / OpenQASM / MIMIQ / CirqJSON +
the no-execution CUDA-Q loops).
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from qio.core import QuantumProgram, QuantumProgramSerializationFormat as Serialization

from reference_circuits import (
    PARADIGM_GATE,
    GateSpec,
    Param,
    ReferenceCircuit,
    _ops,
)

from circuit_testing import (
    Compression,
    LOSS_CIRCUIT_METADATA,
    LOSS_CIRCUIT_NAME,
    LOSS_CUDAQ_GATE_DECOMPOSITION,
    LOSS_GLOBAL_PHASE,
    LOSS_MEASUREMENT_MAPPING,
    LOSS_QUBIT_TOPOLOGY,
    LOSS_ROTATION_PRECISION,
    Paradigm,
    _info_and,
    _resolve_serialization,
    check_program,
    get_reference_circuit,
    record_loss,
)

# --- Operation-chain normalization + comparison --------------------------
#
# Every checked object (intermediate content or final SDK circuit) is reduced
# to the canonical ``Operation = (gate_name, params, qubit_indices)`` form and
# compared against the reference specification (``expected.gates``).


def _qubit_order(circuit) -> Dict[Any, int]:
    """Qubit -> canonical index, ordered by label so the reference gate indices
    (0..n-1) are independent of the qubit type (LineQubit, NamedQubit or
    GridQubit)."""
    return {q: i for i, q in enumerate(sorted(circuit.all_qubits(), key=str))}


# Reversed reference mapping: cirq gate class (or ZPowGate exponent) -> name.
_ROTATION_GATES = ("rx", "ry", "rz")

_CIRQ_GATE_NAMES = {
    "HPowGate": "h",
    "_PauliX": "x",
    "_PauliY": "y",
    "_PauliZ": "z",
    "CXPowGate": "cx",
    "CZPowGate": "cz",
    "Rx": "rx",
    "Ry": "ry",
    "Rz": "rz",
}


def _zpow_exponent_name(exponent: float) -> str:
    for name, expect in (("s", 0.5), ("sdg", -0.5), ("t", 0.25), ("tdg", -0.25)):
        if abs(exponent - expect) < 1e-12:
            return name
    return "z"


def _cirq_gate_params(gate: Any) -> Tuple[Param, ...]:
    import math

    name = type(gate).__name__
    if name in ("Rx", "Ry", "Rz"):
        rads = getattr(gate, "_rads", None)
        if rads is None:
            rads = float(gate.exponent) * math.pi
        if isinstance(rads, (int, float)):
            return (float(rads),)
        # Sympy symbol / symbolic expression: canonicalize to the same "~<name>"
        # marker used by the reference circuit spec.
        return (f"~{rads}",)
    return ()


def _cirq_topology(circuit) -> Tuple[str, ...]:
    """Canonical qubit labels of a Cirq circuit, in reference index order."""
    import cirq

    order = _qubit_order(circuit)
    labels = [None] * len(order)
    for qubit, index in order.items():
        if isinstance(qubit, cirq.LineQubit):
            label = f"q{qubit.x}"
        elif isinstance(qubit, cirq.GridQubit):
            label = f"({qubit.row},{qubit.col})"
        else:  # cirq.NamedQubit / subclasses
            label = str(qubit)
        labels[index] = label
    return tuple(labels)


def _cirq_gate_name(gate: Any) -> str:
    name = type(gate).__name__
    if name == "ZPowGate":
        return _zpow_exponent_name(gate.exponent)
    if name not in _CIRQ_GATE_NAMES:
        raise AssertionError(f"unmapped cirq gate {name!r}")
    return _CIRQ_GATE_NAMES[name]


def _cirq_operations(circuit) -> List[Tuple[str, Tuple[Param, ...], Tuple[int, ...]]]:
    import cirq

    order = _qubit_order(circuit)
    ops = []
    for op in circuit.all_operations():
        indices = tuple(order[q] for q in op.qubits)
        if isinstance(op.gate, cirq.MeasurementGate):
            # Split multi-qubit measurements into one canonical "measure" op
            # per qubit so merged terminal measurements match the reference.
            ops.extend(("measure", (), (i,)) for i in indices)
            continue
        ops.append((_cirq_gate_name(op.gate), _cirq_gate_params(op.gate), indices))
    return ops


def _qiskit_operations(qc) -> List[Tuple[str, Tuple[Param, ...], Tuple[int, ...]]]:
    ops = []
    for instr in qc.data:
        operation = instr.operation
        indices = tuple(qc.find_bit(q).index for q in instr.qubits)
        if operation.name == "measure":
            ops.append(("measure", (), indices))
        else:
            params = tuple(_param_marker(p) for p in operation.params)
            ops.append((operation.name, params, indices))
    return ops


def _param_marker(param: Any) -> Param:
    """Canonicalizes a gate parameter for the op-chain comparison: floats stay
    floats, symbolic expressions (Qiskit Parameter / Sympy symbol) collapse to
    the same "~<name>" marker used by the reference circuit spec."""
    if isinstance(param, (int, float)):
        return float(param)
    return f"~{param}"


_MIMIQ_GATE_NAMES = {
    "GateH": "h",
    "GateX": "x",
    "GateY": "y",
    "GateZ": "z",
    "GateS": "s",
    "GateSDG": "sdg",
    "GateT": "t",
    "GateTDG": "tdg",
    "GateRX": "rx",
    "GateRY": "ry",
    "GateRZ": "rz",
    "GateCX": "cx",
    "GateCZ": "cz",
    "Measure": "measure",
}


def _mimiq_operations(circuit) -> List[Tuple[str, Tuple[Param, ...], Tuple[int, ...]]]:
    ops = []
    for instr in circuit.instructions:
        gate = instr.operation
        name = type(gate).__name__
        if name not in _MIMIQ_GATE_NAMES:
            raise AssertionError(f"unmapped mimiq gate {name!r}")
        name = _MIMIQ_GATE_NAMES[name]
        indices = tuple(int(q) for q in instr.get_qubits())
        if name == "measure":
            ops.append(("measure", (), indices))
        else:
            params = (
                tuple(float(p) for p in gate.getparams())
                if name in _ROTATION_GATES
                else ()
            )
            ops.append((name, params, indices))
    return ops


def _qasm_operations(
    source: str, version: int
) -> List[Tuple[str, Tuple[Param, ...], Tuple[int, ...]]]:
    # Independent parsers (Qiskit's own), not the qio converters under test.
    from qiskit import qasm2, qasm3

    qc = qasm2.loads(source) if version == 2 else qasm3.loads(source)
    return _qiskit_operations(qc)


def _cirqjson_operations(
    source: str,
) -> List[Tuple[str, Tuple[Param, ...], Tuple[int, ...]]]:
    import cirq

    return _cirq_operations(cirq.read_json(json_text=source))


def _params_close(
    a: Tuple[Param, ...], b: Tuple[Param, ...], atol: float = 1e-6
) -> bool:
    import math

    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, str) or isinstance(y, str):
            # Symbolic markers compare by identity ("~theta" == "~theta").
            if x != y:
                return False
            continue
        # Exact equality short-circuits; isclose covers the ~10-digit rounding
        # introduced by ``cirq.to_qasm`` when it serializes rotation angles.
        if not (x == y or math.isclose(x, y, rel_tol=atol, abs_tol=atol)):
            return False
    return True


def check_operation_sequence(
    actual: List[Tuple[str, Tuple[Param, ...], Tuple[int, ...]]],
    expected_gates,
    atol: float = 1e-6,
) -> None:
    """Exact-first operation-chain check against the reference gate list.

    The gate chain (non-measurement operations) must match in order: names and
    qubit indices exactly, rotation parameters exactly - a parameter deviation
    that stays within ``atol`` is the ``rotation_precision`` loss (declare it
    on edges serialized by Cirq), anything larger is a bug. ``measure`` ops
    must form exactly the same multiset (commuting terminal measures, order
    carries no information): any missing/extra measurement is a bug.
    """
    gate_actual = [op for op in actual if op[0] != "measure"]
    gate_expected = [op for op in expected_gates if op[0] != "measure"]
    assert len(gate_actual) == len(gate_expected), (
        len(gate_actual),
        len(gate_expected),
    )
    for (name_a, params_a, indices_a), (name_b, params_b, indices_b) in zip(
        gate_actual, gate_expected
    ):
        assert name_a == name_b, (name_a, name_b)
        assert indices_a == indices_b, (indices_a, indices_b)
        if params_a != params_b:
            assert _params_close(params_a, params_b, atol), (
                f"rotation parameters differ beyond tolerance: "
                f"{params_a} vs {params_b}"
            )
            record_loss(
                LOSS_ROTATION_PRECISION,
                detail=(
                    f"{name_a} on {indices_a}: "
                    f"{tuple(round(float(p), 10) for p in params_a)} -> "
                    f"{tuple(round(float(p), 10) for p in params_b)}"
                ),
            )

    measure_actual = sorted(op for op in actual if op[0] == "measure")
    measure_expected = sorted(op for op in expected_gates if op[0] == "measure")
    assert measure_actual == measure_expected, (measure_actual, measure_expected)


# --- Circuit-level information checks (name, global phase, metadata,
# topology, measurement mapping). Unlike the gate chain these are soft by
# design: any deviation is recorded as the declared loss - OpenQASM genuinely
# cannot carry them, so they are declared on the QASM/CUDA-Q edges and the
# report documents exactly what a round trip drops. Only the classical-bit
# *count* is a hard check (a conversion must never lose a measurement's
# storage).


def _qiskit_measure_map(qc) -> Dict[int, int]:
    """Measure statement as {qubit index -> classical-bit index}."""
    mapping = {}
    for instr in qc.data:
        if instr.operation.name != "measure":
            continue
        qubit = qc.find_bit(instr.qubits[0]).index
        clbit = qc.find_bit(instr.clbits[0]).index if instr.clbits else None
        mapping[qubit] = clbit
    return mapping


def _expected_measure_keys(expected: ReferenceCircuit) -> Tuple[str, ...]:
    """The measurement keys the canonical builders emit (``m<indices>``)."""
    keys = sorted(
        "m" + "".join(str(i) for i in indices)
        for gate, _, indices in expected.gates
        if gate == "measure"
    )
    return tuple(keys)


def _check_qiskit_circuit_info(result, expected: ReferenceCircuit) -> None:
    if expected.name and result.name != expected.name:
        record_loss(
            LOSS_CIRCUIT_NAME, detail=f"{expected.name!r} -> {result.name!r}"
        )
    phase = float(result.global_phase)
    if abs(phase - expected.global_phase) > 1e-9:
        record_loss(LOSS_GLOBAL_PHASE, detail=f"{expected.global_phase} -> {phase}")
    if expected.metadata:
        actual_meta = dict(result.metadata or {})
        if actual_meta != dict(expected.metadata):
            record_loss(
                LOSS_CIRCUIT_METADATA, detail=f"{actual_meta} != {expected.metadata}"
            )
    measured = expected.measurement_indices()
    if not measured:
        return
    # A conversion must preserve the classical storage: hard check.
    assert result.num_clbits == len(measured), (result.num_clbits, len(measured))
    mapping = _qiskit_measure_map(result)
    expected_map = {i: i for i in measured}
    if mapping != expected_map:
        record_loss(
            LOSS_MEASUREMENT_MAPPING,
            detail=f"qubit->clbit {expected_map} -> {mapping}",
        )


def _check_cirq_circuit_info(result, expected: ReferenceCircuit) -> None:
    if expected.qubit_scheme != "line":
        labels = _cirq_topology(result)
        expected_labels = expected.topology_labels()
        if labels != expected_labels:
            record_loss(
                LOSS_QUBIT_TOPOLOGY,
                detail=f"labels {expected_labels} -> {labels}",
            )
    if not expected.has_measurements():
        return
    keys = tuple(sorted(k.name for k in result.all_measurement_key_objs()))
    expected_keys = _expected_measure_keys(expected)
    if keys != expected_keys:
        record_loss(
            LOSS_MEASUREMENT_MAPPING,
            detail=f"measurement keys {expected_keys} -> {keys}",
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


# --- SDK format checks ----------------------------------------------------


def check_cirq(result, expected: Optional[ReferenceCircuit] = None) -> None:
    import cirq

    assert isinstance(
        result, cirq.Circuit
    ), f"expected cirq.Circuit, got {type(result)}"
    if expected is not None:
        assert len(result.all_qubits()) == expected.n_qubits, (
            len(result.all_qubits()),
            expected.n_qubits,
        )
        check_operation_sequence(_cirq_operations(result), expected.gates)
        _check_cirq_circuit_info(result, expected)


def check_qiskit(result, expected: Optional[ReferenceCircuit] = None) -> None:
    """Strict check for a *created* Qiskit circuit: qubit count and the full
    operation sequence must match the reference specification, and the
    circuit-level information (name/global phase/metadata/measure mapping) is
    compared as recorded (declared) losses - QASM round trips are known to
    regenerate the name and drop phase/metadata."""
    from qiskit import QuantumCircuit

    assert isinstance(
        result, QuantumCircuit
    ), f"expected QuantumCircuit, got {type(result)}"
    if expected is not None:
        assert result.num_qubits == expected.n_qubits, (
            result.num_qubits,
            expected.n_qubits,
        )
        check_operation_sequence(_qiskit_operations(result), expected.gates)
        _check_qiskit_circuit_info(result, expected)


def check_cudaq(result, expected: Optional[ReferenceCircuit] = None) -> None:
    # A CUDA-Q kernel cannot be introspected without executing it, so the
    # kernel is only asserted to exist here. Its behavior is validated *around*
    # the leg instead: by the "counts" oracle on the direct paths, or by the
    # end-of-chain SDK circuit (qubit count + ``unitary`` oracle) on the
    # no-execution round-trip loops (see ``add_cudaq_loop``).
    assert result is not None, "CUDA-Q kernel is None"


def check_mimiq(result, expected: Optional[ReferenceCircuit] = None) -> None:
    import mimiqcircuits

    assert result is not None, "MIMIQ circuit is None"
    assert isinstance(
        result, mimiqcircuits.Circuit
    ), f"expected mimiq Circuit, got {type(result)}"
    num_qubits = getattr(result, "num_qubits", None)
    if callable(num_qubits):
        num_qubits = num_qubits()
    if expected is not None and num_qubits is not None:
        assert num_qubits == expected.n_qubits, (num_qubits, expected.n_qubits)
    if expected is not None:
        check_operation_sequence(_mimiq_operations(result), expected.gates)


def check_qir(result, expected: Optional[ReferenceCircuit] = None) -> None:
    """Validates the serialized ``QIR_V1`` content: it must parse as a valid
    PyQIR module (``verify`` passes) and declare the reference qubit count.

    QIR is a leaf format in the battery (no consumption edges), so this is the
    only place its content is checked; the producers' output therefore only
    needs to be *well-formed* QIR (the ``unitary`` oracle cannot close a QIR
    leg without a QIR interpreter)."""
    from pyqir import Context, Module

    source = _resolve_serialization(result)
    assert isinstance(source, str), f"expected QIR LLVM IR text, got {type(source)}"
    assert source.strip(), "empty QIR LLVM IR string"
    module = Module.from_ir(Context(), source, "module")
    error = module.verify()
    assert error is None, f"QIR module does not verify: {error}"
    if expected is None:
        return
    num_qubits = _qir_num_qubits(source)
    if num_qubits is not None:
        assert num_qubits == expected.n_qubits, (num_qubits, expected.n_qubits)


def _qir_num_qubits(source: str) -> Optional[int]:
    """Best-effort qubit count of a QIR module.

    The qBraid-derived producers write qubit ``0`` as ``ptr null`` and qubit
    ``i > 0`` as ``ptr inttoptr (i64 i to ptr)``; CUDA-Q allocates a QIR array
    (``@__quantum__rt__qubit_allocate_array``). Returns ``None`` when the
    qubit cardinality cannot be derived.
    """
    import re

    m = re.search(
        r"@__quantum__rt__qubit_allocate_array\((?:i32|i64)\s*(\d+)", source
    )
    if m:
        return int(m.group(1))
    indices = {int(x) for x in re.findall(r"inttoptr \(i64 (\d+) to ptr\)", source)}
    if indices:
        return max(indices) + 1
    if "@__quantum__qis__" in source:
        # A circuit whose only qubit is 0 never emits an inttoptr reference.
        return 1
    return None


def check_qasm(version: int) -> Callable:
    """Validates the serialized OpenQASM ``version`` (2 or 3) content: it must
    parse, declare the right qubit count and reproduce the reference operation
    chain."""

    def _check(result, expected: Optional[ReferenceCircuit] = None) -> None:
        source = _resolve_serialization(result)
        assert source, "empty QASM string"
        num_qubits = _qasm_num_qubits(source, version)
        if expected is not None:
            assert num_qubits == expected.n_qubits, (num_qubits, expected.n_qubits)
            check_operation_sequence(_qasm_operations(source, version), expected.gates)

    return _check


def check_qasm_cudaq_documented(version: int) -> Callable:
    """Content check for the OpenQASM ``version`` re-emitted after a CUDA-Q leg.

    The source must parse and declare the right qubit count (hard failures).
    The exact operation chain is then compared, but *any* deviation is recorded
    as the declared :data:`LOSS_CUDAQ_GATE_DECOMPOSITION` loss - with the
    concrete deviation as detail - instead of failing the test: CUDA-Q owns the
    gate basis when re-emitting OpenQASM 2 (``cz -> h+cx+h``, ``s/t -> rz/rx``
    sequences, per-qubit classical registers). Semantic correctness is
    guaranteed by the ``unitary`` oracle closing every cudaq loop.
    """

    def _check(result, expected: Optional[ReferenceCircuit] = None) -> None:
        source = _resolve_serialization(result)
        assert source, "empty QASM string"
        num_qubits = _qasm_num_qubits(source, version)
        if expected is not None:
            assert num_qubits == expected.n_qubits, (num_qubits, expected.n_qubits)
            try:
                check_operation_sequence(
                    _qasm_operations(source, version), expected.gates
                )
            except AssertionError as exc:
                record_loss(
                    LOSS_CUDAQ_GATE_DECOMPOSITION,
                    detail=f"op chain deviation: {exc}",
                )

    return _check


def check_cirq_light(result, expected: Optional[ReferenceCircuit] = None) -> None:
    """Lenient cirq check: instance + declared qubit count + circuit-level
    information. Used on the final SDK of the cudaq loops: the cudaq leg's
    structure is already documented by ``check_qasm_cudaq_documented`` on the
    re-emitted QASM2, so the op chain is not re-checked here, but the
    circuit-level information losses still belong in the report."""
    import cirq

    assert isinstance(result, cirq.Circuit), f"expected cirq.Circuit, got {type(result)}"
    if expected is not None:
        assert len(result.all_qubits()) == expected.n_qubits, (
            len(result.all_qubits()),
            expected.n_qubits,
        )
        _check_cirq_circuit_info(result, expected)


def check_qiskit_light(result, expected: Optional[ReferenceCircuit] = None) -> None:
    """Lenient qiskit check: instance + declared qubit count + circuit-level
    information. Used on the final SDK of the cudaq loops (see
    ``check_cirq_light``)."""
    from qiskit import QuantumCircuit

    assert isinstance(
        result, QuantumCircuit
    ), f"expected QuantumCircuit, got {type(result)}"
    if expected is not None:
        assert result.num_qubits == expected.n_qubits, (
            result.num_qubits,
            expected.n_qubits,
        )
        _check_qiskit_circuit_info(result, expected)


def check_cirq_json(result, expected: Optional[ReferenceCircuit] = None) -> None:
    import cirq

    source = _resolve_serialization(result)
    assert source, "empty Cirq JSON"
    circuit = cirq.read_json(json_text=source)
    assert isinstance(circuit, cirq.Circuit), type(circuit)
    if expected is not None:
        assert len(circuit.all_qubits()) == expected.n_qubits, (
            len(circuit.all_qubits()),
            expected.n_qubits,
        )
        check_operation_sequence(_cirq_operations(circuit), expected.gates)
        _check_cirq_circuit_info(circuit, expected)


# --- Negative / edge cases (each entry raises an Exception) --------------


def _bell_cirq():
    return get_reference_circuit("bell").cirq()


def _bell_qiskit():
    return get_reference_circuit("bell").qiskit()


def _bell_cudaq():
    return get_reference_circuit("bell").cudaq_kernel()


def _cirq_program(dest_format: Serialization) -> QuantumProgram:
    return QuantumProgram.from_cirq_circuit(_bell_cirq(), dest_format=dest_format)


def _param_qiskit():
    return get_reference_circuit("parametrized").qiskit()


# --- The paradigm ---------------------------------------------------------


class GateParadigm(Paradigm):
    """The gate-based paradigm: qubit circuits over Cirq / Qiskit / OpenQASM /
    MIMIQ (and the no-execution CUDA-Q legs), closed by the ``unitary`` /
    ``counts`` / ``structural`` oracles."""

    name = PARADIGM_GATE
    circuits = (
        ReferenceCircuit(
            GateSpec(
                name="bell",
                n_qubits=2,
                gates=_ops(
                    ("h", (), (0,)),
                    ("cx", (), (0, 1)),
                    ("measure", (), (0,)),
                    ("measure", (), (1,)),
                ),
                expected_counts={"00": None, "11": None},
                description="2-qubit Bell state: h + cx + measure.",
            ),
        ),
        ReferenceCircuit(
            GateSpec(
                name="ghz",
                n_qubits=3,
                gates=_ops(
                    ("h", (), (0,)),
                    ("cx", (), (0, 1)),
                    ("cx", (), (0, 2)),
                    ("measure", (), (0,)),
                    ("measure", (), (1,)),
                    ("measure", (), (2,)),
                ),
                expected_counts={"000": None, "111": None},
                description="3-qubit GHZ state.",
            ),
        ),
        ReferenceCircuit(
            GateSpec(
                name="rotations",
                n_qubits=2,
                gates=_ops(
                    ("rx", (0.3,), (0,)),
                    ("ry", (0.5,), (1,)),
                    ("cz", (), (0, 1)),
                    ("rz", (0.7,), (0,)),
                    ("measure", (), (0,)),
                    ("measure", (), (1,)),
                ),
                description="2-qubit circuit with parameterized rotations and cz.",
            ),
        ),
        ReferenceCircuit(
            GateSpec(
                name="x_cx",
                n_qubits=2,
                gates=_ops(
                    ("x", (), (0,)),
                    ("cx", (), (0, 1)),
                    ("measure", (), (0,)),
                    ("measure", (), (1,)),
                ),
                expected_counts={"11": None},
                description="Deterministic 2-qubit circuit: x(0), cx(0, 1).",
            ),
        ),
        ReferenceCircuit(
            GateSpec(
                name="gates4",
                n_qubits=4,
                gates=_ops(
                    ("h", (), (0,)),
                    ("x", (), (1,)),
                    ("s", (), (2,)),
                    ("t", (), (3,)),
                    ("cx", (), (0, 1)),
                    ("cz", (), (2, 3)),
                    ("measure", (), (0,)),
                    ("measure", (), (1,)),
                    ("measure", (), (2,)),
                    ("measure", (), (3,)),
                ),
                description="4-qubit circuit mixing single- and two-qubit gates.",
            ),
        ),
        ReferenceCircuit(
            GateSpec(
                name="phased_meta",
                n_qubits=2,
                gates=_ops(
                    ("h", (), (0,)),
                    ("cx", (), (0, 1)),
                    ("z", (), (0,)),
                    ("measure", (), (0,)),
                    ("measure", (), (1,)),
                ),
                expected_counts={"00": None, "11": None},
                global_phase=0.5,
                metadata={"id": 42, "ansatz": "rx"},
                description=(
                    "Deterministic circuit carrying non-trivial circuit-level "
                    "information: global_phase and a Qiskit metadata dict."
                ),
            ),
        ),
        ReferenceCircuit(
            GateSpec(
                name="topology",
                n_qubits=2,
                gates=_ops(
                    ("h", (), (0,)),
                    ("cx", (), (0, 1)),
                    ("measure", (), (0,)),
                    ("measure", (), (1,)),
                ),
                expected_counts={"00": None, "11": None},
                qubit_scheme="named",
                description=(
                    "Bell circuit on named qubits (a, b); the named topology is only "
                    "preserved by CirqJSON, QASM flattens it to a linear array."
                ),
            ),
        ),
        ReferenceCircuit(
            GateSpec(
                name="parametrized",
                n_qubits=1,
                gates=_ops(
                    ("rx", ("~theta",), (0,)),
                ),
                oracle="structural",
                supported_edge_ids=(
                    "qiskit->qasm3->qiskit",
                    "cirq->cirqjson->cirq",
                ),
                description=(
                    "Unbound parameterized circuit rx(theta). Symbolic parameters only "
                    "survive QASM3 (input float) and CirqJSON; QASM2 raises on unbound "
                    "params and the cirq qasm importer cannot parse them."
                ),
            ),
        ),
    )
    input_fns = {
        "cirq": lambda c: c.cirq(),
        "qiskit": lambda c: c.qiskit(),
        "mimiq": lambda c: c.mimiq(),
        "cudaq": lambda c: c.cudaq_kernel(),
    }

    def content_checks(self, fmt: Serialization) -> Tuple[Callable, ...]:
        if fmt in (Serialization.QASM_V2,):
            return (check_qasm(2),)
        if fmt in (Serialization.QASM_V3,):
            return (check_qasm(3),)
        if fmt in (Serialization.CIRQ_CIRCUIT_JSON_V1,):
            return (check_cirq_json,)
        if fmt in (Serialization.QIR_V1,):
            return (check_qir,)
        return ()

    def build_edges(self, compression: Compression):  # noqa: C901 - declarative graph
        producer = QuantumProgram.from_cirq_circuit
        edges = []

        def register(
            edge_id: str,
            input_kind: str,
            steps: Sequence["Paradigm.Step"],
            oracle: str = "unitary",
            known_losses: Tuple[str, ...] = (),
        ) -> None:
            edges.append(
                self.ConversionEdge(
                    edge_id,
                    input_kind,
                    self.input_fns[input_kind],
                    tuple(steps),
                    oracle=oracle,
                    known_losses=known_losses,
                    paradigm=self.name,
                )
            )

        def add(
            edge_id: str,
            input_kind: str,
            write_fn: Callable,
            fmt: Serialization,
            read_fn: Optional[Callable] = None,
            read_check: Optional[Callable] = None,
            oracle: str = "unitary",
            known_losses: Tuple[str, ...] = (),
        ) -> None:
            checks = (check_program(fmt),) + self.content_checks(fmt)
            steps = [
                self.Step(
                    write_fn,
                    {"dest_format": fmt, "compression_format": compression},
                    checks,
                )
            ]
            if read_fn is not None:
                steps.append(self.Step(read_fn, {}, (read_check,)))
            register(edge_id, input_kind, steps, oracle, known_losses)

        def add_cudaq_loop(
            edge_id_base: str,
            input_kind: str,
            write_fn: Callable,
            write_fmt: Serialization,
            known_losses: Tuple[str, ...] = (),
        ) -> None:
            """Registers a no-execution cudaq round trip back to each SDK:
            ``input -> qasm(write_fmt) -> cudaq kernel -> qasm2 -> {cirq, qiskit}``.

            A CUDA-Q kernel cannot be introspected without executing it
            (``check_cudaq`` stays a mere existence check), so the information is
            verified *around* the leg: the input serialization is checked exactly
            (``write_checks``, before CUDA-Q sees it), then the re-emitted QASM2 is
            checked by :func:`check_qasm_cudaq_documented` - any gate-basis rewrite
            CUDA-Q performs (``cz -> h+cx+h``, ``s/t -> rz/rx``) is *recorded* in
            the loss report as a declared :data:`LOSS_CUDAQ_GATE_DECOMPOSITION`
            loss instead of failing, and the end-of-chain SDK circuit is only
            required to declare the right qubit count (complete). Semantic
            correctness is enforced by the ``unitary`` oracle that closes the loop:
            a drop or alteration of a gate would change the unitary and fail.
            """
            write_checks = (check_program(write_fmt),) + self.content_checks(write_fmt)
            back_checks = (
                check_program(Serialization.QASM_V2),
                check_qasm_cudaq_documented(2),
            )
            terminal_converters = {
                "cirq": QuantumProgram.to_cirq_circuit,
                "qiskit": QuantumProgram.to_qiskit_circuit,
            }
            terminal_checks = {
                "cirq": check_cirq_light,
                "qiskit": check_qiskit_light,
            }
            for terminal in ("cirq", "qiskit"):
                steps = [
                    self.Step(
                        write_fn,
                        {"dest_format": write_fmt, "compression_format": compression},
                        write_checks,
                    ),
                    self.Step(QuantumProgram.to_cudaq_kernel, {}, (check_cudaq,)),
                    self.Step(
                        QuantumProgram.from_cudaq_kernel,
                        {
                            "dest_format": Serialization.QASM_V2,
                            "compression_format": compression,
                        },
                        back_checks,
                    ),
                    self.Step(
                        terminal_converters[terminal],
                        {},
                        (terminal_checks[terminal],),
                    ),
                ]
                register(
                    f"{edge_id_base}->{terminal}",
                    input_kind,
                    steps,
                    oracle="unitary",
                    known_losses=known_losses,
                )

        add(
            "cirq->qasm2->cirq",
            "cirq",
            producer,
            Serialization.QASM_V2,
            QuantumProgram.to_cirq_circuit,
            check_cirq,
            known_losses=_info_and(LOSS_ROTATION_PRECISION),
        )
        add(
            "cirq->qasm3->cirq",
            "cirq",
            producer,
            Serialization.QASM_V3,
            QuantumProgram.to_cirq_circuit,
            check_cirq,
            known_losses=_info_and(LOSS_ROTATION_PRECISION),
        )
        add(
            "cirq->cirqjson->cirq",
            "cirq",
            producer,
            Serialization.CIRQ_CIRCUIT_JSON_V1,
            QuantumProgram.to_cirq_circuit,
            check_cirq,
        )
        add(
            "cirq->qasm2->qiskit",
            "cirq",
            producer,
            Serialization.QASM_V2,
            QuantumProgram.to_qiskit_circuit,
            check_qiskit,
            known_losses=_info_and(LOSS_ROTATION_PRECISION),
        )
        add(
            "cirq->qasm3->qiskit",
            "cirq",
            producer,
            Serialization.QASM_V3,
            QuantumProgram.to_qiskit_circuit,
            check_qiskit,
            known_losses=_info_and(LOSS_ROTATION_PRECISION),
        )
        add(
            "cirq->qasm3->cudaq",
            "cirq",
            producer,
            Serialization.QASM_V3,
            QuantumProgram.to_cudaq_kernel,
            check_cudaq,
            oracle="counts",
            known_losses=(LOSS_ROTATION_PRECISION,),
        )
        add(
            "cirq->qasm2->mimiq",
            "cirq",
            producer,
            Serialization.QASM_V2,
            QuantumProgram.to_mimiq_circuit,
            check_mimiq,
            oracle="structural",
            known_losses=(LOSS_ROTATION_PRECISION,),
        )
        # cudaq round trips without execution: cudaq -> qasm2 -> {cirq, qiskit}
        # closes on the unitary oracle, so no simulation is needed to check for
        # losses (see ``add_cudaq_loop``). The declared losses are inherent to the
        # cudaq leg: Cirq truncates rotation angles on write, CUDA-Q rewrites the
        # gate basis on re-emission, and the circuit-info losses come from both the
        # QASM serializations and the per-qubit creg split.
        add_cudaq_loop(
            "cirq->qasm2->cudaq->qasm2",
            "cirq",
            producer,
            Serialization.QASM_V2,
            known_losses=_info_and(
                LOSS_ROTATION_PRECISION, LOSS_CUDAQ_GATE_DECOMPOSITION
            ),
        )
        add_cudaq_loop(
            "cirq->qasm3->cudaq->qasm2",
            "cirq",
            producer,
            Serialization.QASM_V3,
            known_losses=_info_and(
                LOSS_ROTATION_PRECISION, LOSS_CUDAQ_GATE_DECOMPOSITION
            ),
        )

        producer = QuantumProgram.from_qiskit_circuit
        add(
            "qiskit->qasm2->qiskit",
            "qiskit",
            producer,
            Serialization.QASM_V2,
            QuantumProgram.to_qiskit_circuit,
            check_qiskit,
            known_losses=_info_and(),
        )
        add(
            "qiskit->qasm3->qiskit",
            "qiskit",
            producer,
            Serialization.QASM_V3,
            QuantumProgram.to_qiskit_circuit,
            check_qiskit,
            known_losses=_info_and(),
        )
        add(
            "qiskit->qasm2->cirq",
            "qiskit",
            producer,
            Serialization.QASM_V2,
            QuantumProgram.to_cirq_circuit,
            check_cirq,
            known_losses=_info_and(),
        )
        add(
            "qiskit->qasm3->cirq",
            "qiskit",
            producer,
            Serialization.QASM_V3,
            QuantumProgram.to_cirq_circuit,
            check_cirq,
            known_losses=_info_and(),
        )
        add(
            "qiskit->qasm3->cudaq",
            "qiskit",
            producer,
            Serialization.QASM_V3,
            QuantumProgram.to_cudaq_kernel,
            check_cudaq,
            oracle="counts",
        )
        add(
            "qiskit->qasm2->cudaq",
            "qiskit",
            producer,
            Serialization.QASM_V2,
            QuantumProgram.to_cudaq_kernel,
            check_cudaq,
            oracle="counts",
        )
        add(
            "qiskit->qasm2->mimiq",
            "qiskit",
            producer,
            Serialization.QASM_V2,
            QuantumProgram.to_mimiq_circuit,
            check_mimiq,
            oracle="structural",
        )
        add_cudaq_loop(
            "qiskit->qasm2->cudaq->qasm2",
            "qiskit",
            producer,
            Serialization.QASM_V2,
            known_losses=_info_and(
                LOSS_ROTATION_PRECISION, LOSS_CUDAQ_GATE_DECOMPOSITION
            ),
        )
        add_cudaq_loop(
            "qiskit->qasm3->cudaq->qasm2",
            "qiskit",
            producer,
            Serialization.QASM_V3,
            known_losses=_info_and(
                LOSS_ROTATION_PRECISION, LOSS_CUDAQ_GATE_DECOMPOSITION
            ),
        )

        producer = QuantumProgram.from_mimiq_circuit
        add(
            "mimiq->qasm2->qiskit",
            "mimiq",
            producer,
            Serialization.QASM_V2,
            QuantumProgram.to_qiskit_circuit,
            check_qiskit,
            known_losses=_info_and(),
        )
        add(
            "mimiq->qasm2->cirq",
            "mimiq",
            producer,
            Serialization.QASM_V2,
            QuantumProgram.to_cirq_circuit,
            check_cirq,
            known_losses=_info_and(),
        )

        add(
            "cirq->qir",
            "cirq",
            QuantumProgram.from_cirq_circuit,
            Serialization.QIR_V1,
            oracle="structural",
        )
        add(
            "qiskit->qir",
            "qiskit",
            QuantumProgram.from_qiskit_circuit,
            Serialization.QIR_V1,
            oracle="structural",
        )
        add(
            "cudaq->qir",
            "cudaq",
            QuantumProgram.from_cudaq_kernel,
            Serialization.QIR_V1,
            oracle="structural",
        )

        def _qasm2_to_qiskit(circuit_str):
            from qio.utils.conversion.program.qasm2_to_qiskit import convert

            return convert(circuit_str)

        def _qasm3_to_qiskit(circuit_str):
            from qio.utils.conversion.program.qasm3_to_qiskit import convert

            return convert(circuit_str)

        qasm2_write_checks = (check_program(Serialization.QASM_V2),) + self.content_checks(
            Serialization.QASM_V2
        )
        qasm3_write_checks = (check_program(Serialization.QASM_V3),) + self.content_checks(
            Serialization.QASM_V3
        )
        register(
            "qiskit->qasm3->qasm2->qiskit",
            "qiskit",
            (
                self.Step(
                    QuantumProgram.from_qiskit_circuit,
                    {
                        "dest_format": Serialization.QASM_V3,
                        "compression_format": compression,
                    },
                    qasm3_write_checks,
                ),
                self.Step(QuantumProgram.to_qasm2_circuit, {}, (check_qasm(2),)),
                self.Step(_qasm2_to_qiskit, {}, (check_qiskit,)),
            ),
            oracle="unitary",
            known_losses=_info_and(),
        )
        register(
            "qiskit->qasm2->qasm3->qiskit",
            "qiskit",
            (
                self.Step(
                    QuantumProgram.from_qiskit_circuit,
                    {
                        "dest_format": Serialization.QASM_V2,
                        "compression_format": compression,
                    },
                    qasm2_write_checks,
                ),
                self.Step(QuantumProgram.to_qasm3_circuit, {}, (check_qasm(3),)),
                self.Step(_qasm3_to_qiskit, {}, (check_qiskit,)),
            ),
            oracle="unitary",
            known_losses=_info_and(),
        )

        return tuple(edges)

    unsupported_conversions = (
        (
            "from_cirq_circuit -> QASM_V1",
            lambda: QuantumProgram.from_cirq_circuit(
                _bell_cirq(), dest_format=Serialization.QASM_V1
            ),
        ),
        (
            "from_qiskit_circuit -> QASM_V1",
            lambda: QuantumProgram.from_qiskit_circuit(
                _bell_qiskit(), dest_format=Serialization.QASM_V1
            ),
        ),
        (
            "from_qiskit_circuit (unbound params) -> QASM_V2",
            lambda: QuantumProgram.from_qiskit_circuit(
                _param_qiskit(), dest_format=Serialization.QASM_V2
            ),
        ),
        (
            "from_cudaq_kernel -> QASM_V3",
            lambda: QuantumProgram.from_cudaq_kernel(
                _bell_cudaq(), dest_format=Serialization.QASM_V3
            ),
        ),
        (
            "from_cudaq_kernel -> CIRQJSON",
            lambda: QuantumProgram.from_cudaq_kernel(
                _bell_cudaq(), dest_format=Serialization.CIRQ_CIRCUIT_JSON_V1
            ),
        ),
        (
            "to_qasm2_circuit on CIRQJSON",
            lambda: _cirq_program(Serialization.CIRQ_CIRCUIT_JSON_V1).to_qasm2_circuit(),
        ),
        (
            "to_qasm3_circuit on CIRQJSON",
            lambda: _cirq_program(Serialization.CIRQ_CIRCUIT_JSON_V1).to_qasm3_circuit(),
        ),
        (
            "to_mimiq_circuit on QASM_V3",
            lambda: _cirq_program(Serialization.QASM_V3).to_mimiq_circuit(),
        ),
        (
            "to_mimiq_circuit on CIRQJSON",
            lambda: _cirq_program(Serialization.CIRQ_CIRCUIT_JSON_V1).to_mimiq_circuit(),
        ),
        (
            "to_cudaq_kernel on CIRQJSON",
            lambda: _cirq_program(Serialization.CIRQ_CIRCUIT_JSON_V1).to_cudaq_kernel(),
        ),
        (
            "to_pulser_sequence on QASM_V2",
            lambda: QuantumProgram.from_qiskit_circuit(
                _bell_qiskit(), dest_format=Serialization.QASM_V2
            ).to_pulser_sequence(),
        ),
    )
