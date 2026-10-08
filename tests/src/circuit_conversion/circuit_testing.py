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
  and that its operation chain matches the reference specification
  (``expected.gates``, a :class:`ReferenceCircuit`).
- **Operation-chain verification** : every intermediate (``QuantumProgram``
  QASM2/QASM3/CirqJSON content) and final SDK circuit is normalized to the
  canonical ``(gate, params, qubit_indices)`` form and compared against
  ``expected.gates``. Gate names, qubit indices and the measurement multiset
  are compared exactly; rotation parameters are compared exactly, with the
  ``rotation_precision`` loss tolerated on the paths serialized by Cirq
  (``cirq.to_qasm`` truncates rotation angles to ~10 significant digits, so
  exact floats are unreachable there).
- **CUDA-Q round trips** : a CUDA-Q kernel cannot be introspected without
  executing it, so the no-execution loops (``add_cudaq_loop``) only assert the
  kernel exists and validate the information *around* the leg: the input
  serialization exactly, then the re-emitted QASM2. Exact operation-chain
  equality is not asserted on the CUDA-Q output as a hard failure because
  CUDA-Q legitimately normalizes the gate basis when re-emitting OpenQASM 2
  (``cz -> h+cx+h``, ``s/t -> rz/rx`` sequences, per-qubit classical
  registers); instead any such deviation is *recorded* in the loss report as
  the declared ``cudaq_gate_decomposition`` loss, and the final SDK circuit is
  checked for qubit count (complete) + the ``unitary`` oracle (correct).
- **Information-loss policy** : each :class:`ConversionEdge` declares the
  losses inherent to its path (``known_losses``). The checks are exact-first:
  any loss category that is *not* declared fails the test (regression), a
  declared one is tolerated and recorded. ``loss_report()`` + the
  ``conftest.py`` terminal summary surface the observed losses at the end of
  the session.
- **Circuit-level information** : beyond the gate chain, the final SDK checks
  also track what survives of the circuit metadata - circuit name, global
  phase, Qiskit metadata dict, qubit topology (named/grid labels) and the
  measurement mapping (classical-bit slots / Cirq measurement keys). QASM and
  CUDA-Q paths declare these as inherent ``known_losses``; CirqJSON is exact.
- **Feature coverage** : :data:`CIRCUIT_FEATURE_COVERAGE` records which of the
  circuit features (qubit topology, classical control flow, metadata & global
  phase, symbolic parameters, calibrations, measurement mapping) the battery
  actually exercises. The ``conftest.py`` terminal summary prints it as a
  matrix so the report states what it does and does not verify.
- **Input circuits are NOT checked** : they are built statically and trusted
  (``ReferenceCircuit`` builders), so ``run_path`` converts them without any
  verification; only intermediate and final objects are checked.
- **Generic driver** : ``convert(circuit, converter, ...)`` takes the input
  circuit and the conversion function to call as an argument, and runs the
  post-conversion checks. It is composed into multi-step pipelines by
  ``run_path``.
- **Declarative registries** : ``build_edges()`` returns the full conversion
  graph (SDK -> format -> SDK) as :class:`ConversionEdge` rows; adding a
  conversion is adding one call. ``UNSUPPORTED_CONVERSIONS`` hosts the negative
  cases.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from qio.core import (
    QuantumProgram,
    QuantumProgramCompressionFormat,
    QuantumProgramSerializationFormat,
)

from reference_circuits import Param, ReferenceCircuit, get_reference_circuit

Compression = QuantumProgramCompressionFormat
Serialization = QuantumProgramSerializationFormat
NONE = Compression.NONE
ZLIB = Compression.ZLIB_BASE64_V1


# check format (one per SDK / serialization format)


# Operation-chain normalization + comparison.
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


# Information-loss tracking (objectives: highlight inherent conversion losses,
# catch undeclared ones as regressions).
#
# Each ConversionEdge declares the losses inherent to its path
# (``known_losses``). Every check is exact-first: any deviation is classified
# into a loss category; a category that is not declared on the current edge
# fails the test (regression), a declared one is tolerated and recorded in the
# session report (``loss_report()``).
#
# Categories:
# - ``rotation_precision``: Cirq's QASM export truncates rotation angles to
#   ~10 significant digits (``rx(pi*0.0954929659)``), so rotation parameters
#   are not bit-exact after any round trip serialized by Cirq. The gate name,
#   qubit indices and measurement structure still match exactly.
#
# Single-qubit terminal measurements are compared as a *multiset*: they
# commute and Cirq packs them into a shared moment (then orders them by qubit
# index), so their ordering carries no information and is not tracked as a
# loss. Any missing or extra measurement is a hard failure.

LOSS_ROTATION_PRECISION = "rotation_precision"
LOSS_CUDAQ_GATE_DECOMPOSITION = "cudaq_gate_decomposition"
LOSS_CIRCUIT_NAME = "circuit_name"
LOSS_GLOBAL_PHASE = "global_phase"
LOSS_CIRCUIT_METADATA = "circuit_metadata"
LOSS_QUBIT_TOPOLOGY = "qubit_topology"
LOSS_MEASUREMENT_MAPPING = "measurement_mapping"
LOSS_SYMBOLIC_PARAMS = "symbolic_parameters"

# Circuit-level information that QASM (2/3) and the CUDA-Q gate-basis rewrite
# inherently drop; declared as known losses on the QASM/CUDA-Q edges. The
# CirqJSON edge declares none of them: CirqJSON is exact for these properties.
QASM_CIRCUIT_INFO_LOSSES = (
    LOSS_CIRCUIT_NAME,
    LOSS_GLOBAL_PHASE,
    LOSS_CIRCUIT_METADATA,
    LOSS_QUBIT_TOPOLOGY,
    LOSS_MEASUREMENT_MAPPING,
)


def _info_and(*base: str) -> Tuple[str, ...]:
    """``known_losses`` for edges whose terminal SDK is checked for the
    circuit-level information: the OpenQASM-inherent circuit-info losses are
    always declared (QASM cannot carry name/phase/metadata/topology/mapping or
    rewrites the measurement mapping), plus the given ``base`` losses."""
    return QASM_CIRCUIT_INFO_LOSSES + tuple(base)

_LOSS_DESCRIPTIONS = {
    LOSS_ROTATION_PRECISION: (
        "rotation angles are serialized by Cirq's QASM export at ~10 significant "
        "digits, so exact floats are unreachable after a Cirq-serialized round "
        "trip; gate names, qubit indices and measurement structure still match"
    ),
    LOSS_CUDAQ_GATE_DECOMPOSITION: (
        "CUDA-Q re-emits OpenQASM 2 in a different gate basis (e.g. cz -> h+cx+h, "
        "s/t -> rz/rx) and splits measurements into per-qubit classical "
        "registers; the qubit count is preserved and the end-of-chain unitary is "
        "checked by the unitary oracle, so this is a structural - not semantic - "
        "rewrite"
    ),
    LOSS_CIRCUIT_NAME: (
        "the circuit name (ReferenceCircuit.name, carried by the Qiskit builder) "
        "is not part of OpenQASM: every QASM round trip regenerates it (e.g. "
        "'circuit-42'); Cirq circuits carry no name at all"
    ),
    LOSS_GLOBAL_PHASE: (
        "Qiskit circuit.global_phase is not expressible in OpenQASM (no syntax in "
        "QASM2, dropped by QASM3 export), so a non-zero phase is lost on any QASM "
        "round trip; the unitary oracle intentionally ignores global phases, so "
        "this is tracked here instead"
    ),
    LOSS_CIRCUIT_METADATA: (
        "the Qiskit circuit.metadata dict is dropped by every intermediate format "
        "(QASM/CirqJSON/CUDA-Q); only a natively-built Qiskit circuit carries it"
    ),
    LOSS_QUBIT_TOPOLOGY: (
        "qubit topology (Cirq NamedQubit/GridQubit labels, register names) is "
        "flattened to a linear q[i] array by OpenQASM exports; only the CirqJSON "
        "intermediate preserves the named/grid labels, and only the Cirq SDK can "
        "re-read them"
    ),
    LOSS_MEASUREMENT_MAPPING: (
        "the classical side of measurements is not preserved verbatim: the "
        "qubit -> classical-bit assignment and the Cirq measurement keys can be "
        "reordered or merged by QASM/CUDA-Q round trips (e.g. per-qubit keys m0/m1 "
        "collapse to one merged register, or a cirq -> qasm -> qiskit path swaps "
        "the clbit slots); the number of classical bits is a hard check"
    ),
    LOSS_SYMBOLIC_PARAMS: (
        "unbound (symbolic) parameters survive OpenQASM 3 (input float) and "
        "CirqJSON, but OpenQASM 2 cannot represent them (export raises) and the "
        "cirq OpenQASM importer cannot parse them; the parametrized circuit is "
        "therefore only exercised on the QASM3 and CirqJSON edges"
    ),
}

_LOSS_REPORT: Dict[str, Dict[str, Dict[str, Any]]] = {}
_LOSS_CTX = {"id": None, "circuit": None, "known": frozenset()}


def record_loss(category: str, detail: Optional[str] = None) -> None:
    """Records a declared information loss - optionally with a concrete
    ``detail`` of what was observed - or fails an undeclared one."""
    if category not in _LOSS_CTX["known"]:
        raise AssertionError(
            f"undeclared information loss {category!r} on edge "
            f"{_LOSS_CTX['id']!r} (circuit {_LOSS_CTX['circuit']!r}); declare it "
            "in the edge's known_losses or fix the converter"
        )
    entry = _LOSS_REPORT.setdefault(_LOSS_CTX["id"], {}).setdefault(
        category, {"circuits": set(), "observations": set()}
    )
    entry["circuits"].add(_LOSS_CTX["circuit"])
    entry["observations"].add((_LOSS_CTX["circuit"], detail))


def set_loss_context(edge: "ConversionEdge", circuit: ReferenceCircuit) -> None:
    _LOSS_CTX["id"] = edge.id
    _LOSS_CTX["circuit"] = circuit.name
    _LOSS_CTX["known"] = frozenset(edge.known_losses)


def loss_report() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Session report: edge id -> {loss category: {circuits, observations}}."""
    return _LOSS_REPORT


def loss_description(category: str) -> str:
    """Human-readable explanation of a loss category (used by the report)."""
    return _LOSS_DESCRIPTIONS.get(category, "")


# Feature coverage: which of the circuit-level information axes the battery
# actually exercises, and how. Surfaced by the conftest terminal summary as a
# matrix so the report states what it does / does not verify.
# Status: "round-trip tested" | "tested as unsupported" | "not exercised".
CIRCUIT_FEATURE_COVERAGE = (
    (
        "qubit topology & naming",
        "round-trip tested",
        "named/grid Cirq qubits are preserved by CirqJSON but flattened to q[i] "
        "by OpenQASM (declared 'qubit_topology'); Qiskit/CUDA-Q have no native "
        "equivalent",
    ),
    (
        "classical control flow (dynamic circuits)",
        "not exercised",
        "no dynamic reference circuits; Qiskit IfElseOp/WhileOp and QASM3 "
        "control flow are outside the normalized gate set (would need new "
        "builders and checks)",
    ),
    (
        "circuit metadata & global phase",
        "round-trip tested",
        "Qiskit global_phase and metadata dict are dropped by every QASM round "
        "trip (declared 'global_phase'/'circuit_metadata'); the unitary oracle "
        "intentionally ignores phases",
    ),
    (
        "symbolic parameters",
        "tested as unsupported",
        "QASM3 ('input float') and CirqJSON preserve unbound parameters; QASM2 "
        "raises on unbound params and the cirq QASM importer cannot parse them, "
        "so the parametrized circuit runs on the QASM3/CirqJSON edges only",
    ),
    (
        "calibrations & pulse",
        "not exercised",
        "Qiskit Pulse calibrations and QASM3 defcal require converter-level "
        "support; no reference circuit carries them",
    ),
    (
        "measurement mapping & memory slots",
        "round-trip tested",
        "qubit->clbit assignment and Cirq measurement keys are compared "
        "(declared 'measurement_mapping'); the classical-bit count is a hard "
        "check",
    ),
)


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


def _resolve_serialization(result) -> str:
    """Uncompresses a QuantumProgram serialization when needed."""
    if isinstance(result, QuantumProgram):
        source = result.serialization
        if result.compression_format == ZLIB:
            from qio.utils.compression import zlib_to_str

            source = zlib_to_str(source)
        return source
    return result


# Circuit-level information checks (name, global phase, metadata, topology,
# measurement mapping). Unlike the gate chain these are soft by design: any
# deviation is recorded as the declared loss - OpenQASM genuinely cannot carry
# them, so they are declared on the QASM/CUDA-Q edges and the report documents
# exactly what a round trip drops. Only the classical-bit *count* is a hard
# check (a conversion must never lose a measurement's storage).


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


# Equivalence oracles


def _cirq_unitary(circuit) -> Any:
    import cirq
    from cirq import MeasurementGate

    operations = [
        op
        for moment in circuit
        for op in moment
        if not isinstance(op.gate, MeasurementGate)
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


# Generic driver + pipeline


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
    known_losses: Tuple[str, ...] = ()  # inherent information losses of the path


def convert(circuit, converter, *args, checks=(), expected=None, **kwargs) -> Any:
    """Generic conversion driver.

    Args:
        circuit: the input circuit (statically built and trusted - not checked).
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
    set_loss_context(edge, circuit)
    current = edge.input_fn(circuit)
    for step in edge.steps:
        current = convert(
            current, step.fn, expected=circuit, checks=step.checks, **step.kwargs
        )

    oracle = circuit.oracle or edge.oracle
    if oracle == "unitary":
        actual = _to_unitary(current)
        assert_unitaries_close(_cirq_unitary(circuit.cirq()), actual)
    elif oracle == "counts":
        assert_counts(current, circuit)
    elif oracle == "structural":
        # structural checks were already run by the steps
        pass
    else:
        raise ValueError(f"unknown oracle: {oracle}")
    return current


def _to_unitary(circuit) -> Any:
    import cirq
    from qiskit import QuantumCircuit

    if isinstance(circuit, cirq.Circuit):
        return _cirq_unitary(circuit)
    if isinstance(circuit, QuantumCircuit):
        return _qiskit_unitary(circuit)
    raise TypeError(f"cannot extract unitary from {type(circuit)}")


# Declarative registries

INPUT_FNS = {
    "cirq": lambda c: c.cirq(),
    "qiskit": lambda c: c.qiskit(),
}


def build_edges(compression: Compression) -> Tuple[ConversionEdge, ...]:
    """Builds the conversion graph (SDK -> format -> SDK) for one compression.
    Intermediate content checks are performed for both ``NONE`` and ``ZLIB``
    (the serialization is decompressed before parsing)."""
    producer = QuantumProgram.from_cirq_circuit
    edges = []

    def register(
        edge_id: str,
        input_kind: str,
        steps: Sequence[Step],
        oracle: str = "unitary",
        known_losses: Tuple[str, ...] = (),
    ) -> None:
        edges.append(
            ConversionEdge(
                edge_id,
                INPUT_FNS[input_kind],
                tuple(steps),
                oracle=oracle,
                known_losses=known_losses,
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
        checks = (check_program(fmt),) + _format_content_checks(fmt)
        steps = [
            Step(
                write_fn,
                {"dest_format": fmt, "compression_format": compression},
                checks,
            )
        ]
        if read_fn is not None:
            steps.append(Step(read_fn, {}, (read_check,)))
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
        write_checks = (check_program(write_fmt),) + _format_content_checks(write_fmt)
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
                Step(
                    write_fn,
                    {"dest_format": write_fmt, "compression_format": compression},
                    write_checks,
                ),
                Step(QuantumProgram.to_cudaq_kernel, {}, (check_cudaq,)),
                Step(
                    QuantumProgram.from_cudaq_kernel,
                    {
                        "dest_format": Serialization.QASM_V2,
                        "compression_format": compression,
                    },
                    back_checks,
                ),
                Step(
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


def _param_qiskit():
    return get_reference_circuit("parametrized").qiskit()


UNSUPPORTED_CONVERSIONS = (
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
        "to_qasm2_circuit on QASM_V3",
        lambda: _cirq_program(Serialization.QASM_V3).to_qasm2_circuit(),
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
)
