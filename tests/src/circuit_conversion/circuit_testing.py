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
"""Generic conversion-battery machinery, organized by programming paradigm.

Design:
- **Paradigm-abstract driver** : the abstract :class:`Paradigm` base hosts the
  machinery every paradigm shares - the conversion pipeline (:class:`Step`,
  :class:`Paradigm.ConversionEdge`, :meth:`Paradigm.convert`,
  :meth:`Paradigm.run_path`), the equivalence oracles (``unitary``, ``counts``,
  ``structural``, plus paradigm-specific ones registered in
  :data:`EXTRA_ORACLES`), the loss-tracking report and the per-circuit run
  filter (:meth:`Paradigm.circuit_runnable`). Concrete paradigms
  (:class:`~gate_paradigm.GateParadigm`,
  :class:`~analog_paradigm.AnalogParadigm`,
  :class:`~photonic_paradigm.PhotonicParadigm`) live in sibling files and
  provide their own reference circuits, edge builder, content checks, negative
  cases and feature-coverage rows.
- **Facade registries** : the module-level names the rest of the battery uses
  (``build_edges``, ``run_path``, ``UNSUPPORTED_CONVERSIONS``,
  ``CIRCUIT_FEATURE_COVERAGE``, ``REFERENCE_CIRCUITS``,
  ``get_reference_circuit``, ``loss_report``, ...) aggregate every registered
  paradigm, so adding a paradigm to ``:data:`PARADIGMS`` `` `` extends the whole
  battery.
- **One format checker per SDK / per serialization format** : the checks live
  next to their paradigm (``check_cirq``/``check_qiskit``/``check_cudaq``/
  ``check_mimiq``/``check_qasm``/``check_cirq_json`` in ``gate_paradigm``,
  ``check_pulser``/``check_pulser_json`` in ``analog_paradigm``,
  ``check_perceval``/``check_perceval_json`` in ``photonic_paradigm``). Each
  validates the format of the object *and* that its operation chain matches the
  reference specification (``expected.gates``, a :class:`ReferenceCircuit`).
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
- **Information-loss policy** : each :class:`Paradigm.ConversionEdge` declares
  the losses inherent to its path (``known_losses``). The checks are
  exact-first: any loss category that is *not* declared fails the test
  (regression), a declared one is tolerated and recorded.
  ``loss_report()`` + the ``conftest.py`` terminal summary surface the observed
  losses at the end of the session.
- **Circuit-level information** : beyond the gate chain, the final SDK checks
  also track what survives of the circuit metadata - circuit name, global
  phase, Qiskit metadata dict, qubit topology (named/grid labels) and the
  measurement mapping (classical-bit slots / Cirq measurement keys). QASM and
  CUDA-Q paths declare these as inherent ``known_losses``; CirqJSON is exact.
- **Feature coverage** : :data:`CIRCUIT_FEATURE_COVERAGE` records which of the
  circuit features (qubit topology, classical control flow, metadata & global
  phase, symbolic parameters, calibrations, measurement mapping, photonic
  mode-based circuits) the battery actually exercises. The ``conftest.py``
  terminal summary prints it as a matrix so the report states what it does and
  does not verify.
- **Input circuits are NOT checked** : they are built statically and trusted
  (``ReferenceCircuit`` builders), so ``run_path`` converts them without any
  verification; only intermediate and final objects are checked.
- **Generic driver** : ``convert(circuit, converter, ...)`` takes the input
  circuit and the conversion function to call as an argument, and runs the
  post-conversion checks. It is composed into multi-step pipelines by
  ``run_path``.
- **Declarative registries** : each paradigm's ``build_edges()`` returns its
  conversion graph (SDK -> format -> SDK) as :class:`Paradigm.ConversionEdge`
  rows; adding a conversion is adding one call. ``UNSUPPORTED_CONVERSIONS``
  aggregates the negative cases of every paradigm.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from qio.core import (
    QuantumProgram,
    QuantumProgramCompressionFormat,
    QuantumProgramSerializationFormat,
)

from reference_circuits import ReferenceCircuit

Compression = QuantumProgramCompressionFormat
Serialization = QuantumProgramSerializationFormat
NONE = Compression.NONE
ZLIB = Compression.ZLIB_BASE64_V1


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
LOSS_ANALOG_DEVICE = "analog_device"

# Inherent limitations of the *self-hosted* qasm2 <-> qasm3 text normalization
# that the ``cudaq_to_qasm3`` and ``qasm3_to_mimiq`` converters are chained on
# (they only re-express straight-line, literal-parameter stdgates programs).
# They are declared on the mixing legs (``known_losses``) and recorded - with a
# concrete detail - by :func:`check_qasm_normalization_limitations` whenever the
# corresponding construct is actually present in the serialized text, so the
# limitation is surfaced by the loss report instead of failing or being silently
# swallowed.
LOSS_QASM_CONTROL_FLOW = "qasm_control_flow"
LOSS_QASM_GPHASE = "qasm_gphase"
LOSS_QASM_GATE_SET = "qasm_stdlib_gate"

QASM_NORMALIZATION_LIMIT_LOSSES = (
    LOSS_QASM_CONTROL_FLOW,
    LOSS_QASM_GPHASE,
    LOSS_QASM_GATE_SET,
)

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
    LOSS_ANALOG_DEVICE: (
        "the Pulser abstract repr is device-independent, so in principle the "
        "device object could be lost/rewritten by a 'from_abstract_repr' round "
        "trip. With pulser 1.9.1 + the reference devices the device is "
        "preserved exactly (device equality and channel objects are checked), "
        "so this category is currently never recorded - it is a declared guard "
        "that surfaces (tolerated, not failing) any future pulser/device "
        "version that would drop the device from the serialization"
    ),
    LOSS_QASM_CONTROL_FLOW: (
        "the cudaq_to_qasm3 / qasm3_to_mimiq converters ride the self-hosted "
        "qasm2 <-> qasm3 text normalization, which handles straight-line gate "
        "programs only: classical control flow (if/else, for/while/switch) and "
        "gate-definition blocks / subroutines are not re-expressed between the "
        "two grammars (e.g. QASM3 'if (c[0])' is not transcribed to the QASM2 "
        "'if (c[0]==1)' form), and block statements shatter on the ';' split"
    ),
    LOSS_QASM_GPHASE: (
        "the QASM3 -> QASM2 leg silently drops global-phase statements "
        "(gphase(...) with a '// dropped' comment): QASM2 has no "
        "representation for them, so a non-zero global phase is lost on the "
        "qasm3 -> mimiq path"
    ),
    LOSS_QASM_GATE_SET: (
        "the qasm2 <-> qasm3 text normalization only renames u1/cu1 <-> p/cp; "
        "it neither decomposes nor maps any other stdlib gate whose spelling "
        "differs between the two grammars (u2/u3/cu3/cu, ...), so such gates "
        "are passed through unmapped and may not exist in the target grammar's "
        "include file"
    ),
}

_LOSS_REPORT: Dict[str, Dict[str, Dict[str, Any]]] = {}
_LOSS_CTX = {"id": None, "circuit": None, "known": frozenset()}

# Generic feature-coverage rows shared by every paradigm (they describe qio
# circuit-level features as exercised on the gate network). Paradigm-specific
# rows (analog pulses, photonic mode-based circuits) are contributed by the
# paradigms themselves (see ``Paradigm.feature_coverage``).
GENERIC_FEATURE_COVERAGE = (
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
        "qasm2<->qasm3 text normalization",
        "straight-line only",
        "the cudaq_to_qasm3 / qasm3_to_mimiq converters chain on the "
        "self-hosted text converters: literal-parameter stdgates round-trip; "
        "classical control flow, gphase (silently dropped on the QASM3->QASM2 "
        "leg) and non-mapped stdlib gates are declared 'qasm_control_flow' / "
        "'qasm_gphase' / 'qasm_stdlib_gate'",
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
        "measurement mapping & memory slots",
        "round-trip tested",
        "qubit->clbit assignment and Cirq measurement keys are compared "
        "(declared 'measurement_mapping'); the classical-bit count is a hard "
        "check",
    ),
)


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


def set_loss_context(
    edge: "Paradigm.ConversionEdge", circuit: ReferenceCircuit
) -> None:
    _LOSS_CTX["id"] = edge.id
    _LOSS_CTX["circuit"] = circuit.name
    _LOSS_CTX["known"] = frozenset(edge.known_losses)


def loss_report() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Session report: edge id -> {loss category: {circuits, observations}}."""
    return _LOSS_REPORT


def loss_description(category: str) -> str:
    """Human-readable explanation of a loss category (used by the report)."""
    return _LOSS_DESCRIPTIONS.get(category, "")


def _resolve_serialization(result) -> str:
    """Uncompresses a QuantumProgram serialization when needed."""
    if isinstance(result, QuantumProgram):
        source = result.serialization
        if result.compression_format == ZLIB:
            from qio.utils.compression import zlib_to_str

            source = zlib_to_str(source)
        return source
    return result


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


# Equivalence oracles (generic: they only rely on the final SDK object and the
# reference circuit). Paradigm-specific oracles (e.g. the photonic SLOS
# simulation) register into ``EXTRA_ORACLES``.


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


def assert_unitaries_close(reference, actual) -> None:
    """Asserts two circuit unitaries are equal up to a global phase. Uses a
    bit-reversal fallback so Cirq- and Qiskit-ordered matrices are compared
    consistently."""
    if _unit_close(reference, actual):
        return
    if _unit_close(reference, _reverse_bit_order(actual)):
        return
    raise AssertionError("converted circuit unitary differs from the reference")


def assert_counts(kernel, expected: ReferenceCircuit) -> None:
    import cudaq

    assert expected.expected_counts is not None, f"{expected.name} is not deterministic"
    counts = dict(cudaq.sample(kernel, shots_count=500).items())
    actual_keys = set(counts.keys())
    assert actual_keys == set(expected.expected_counts.keys()), (actual_keys, expected)


def _to_unitary(circuit) -> Any:
    import cirq
    from qiskit import QuantumCircuit

    if isinstance(circuit, cirq.Circuit):
        return _cirq_unitary(circuit)
    if isinstance(circuit, QuantumCircuit):
        return _qiskit_unitary(circuit)
    raise TypeError(f"cannot extract unitary from {type(circuit)}")


# Paradigm-specific equivalence oracles (name -> callable(current, expected)).
# Registered by the paradigm modules at import time.
EXTRA_ORACLES: Dict[str, Callable[[Any, ReferenceCircuit], None]] = {}


class Paradigm(ABC):
    """Abstract base of a programming paradigm of the conversion battery.

    It hosts the *generic* machinery: the conversion pipeline
    (:class:`Step`, :class:`Paradigm.ConversionEdge`, :meth:`convert`,
    :meth:`run_path`), the equivalence oracles, the content-check registry hook
    and the per-circuit run filter. Concrete paradigms implement
    :meth:`build_edges` and provide their own circuits, input builders,
    negative cases and feature-coverage rows.
    """

    name: str = ""
    circuits: Tuple[ReferenceCircuit, ...] = ()
    input_fns: Dict[str, Callable[[ReferenceCircuit], Any]] = {}
    unsupported_conversions: Tuple[Tuple[str, Callable], ...] = ()
    feature_coverage: Tuple[Tuple[str, str, str], ...] = ()

    @dataclass(frozen=True)
    class ConversionEdge:
        """One row of the conversion graph: ``input -> steps``.

        ``input_kind`` names the SDK the input circuit is built for (used to
        look up the input builder); ``paradigm`` is the programming paradigm
        owning the edge (the circuit only runs on edges of its own paradigm).
        """

        id: str
        input_kind: str
        input_fn: Callable[[ReferenceCircuit], Any]
        steps: Tuple["Paradigm.Step", ...]
        oracle: str = (
            "unitary"  # "unitary" | "counts" | "structural" | <paradigm-specific>
        )
        known_losses: Tuple[str, ...] = ()  # inherent information losses of the path
        paradigm: str = "gate"  # programming paradigm the edge operates on

    @dataclass(frozen=True)
    class Step:
        fn: Callable
        kwargs: Dict[str, Any] = field(default_factory=dict)
        checks: Tuple[Callable, ...] = ()

    @staticmethod
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

    @abstractmethod
    def build_edges(
        self, compression: Compression
    ) -> Tuple["Paradigm.ConversionEdge", ...]:
        """Builds this paradigm's conversion graph (SDK -> format -> SDK) for
        one compression. Intermediate content checks are performed for both
        ``NONE`` and ``ZLIB`` (the serialization is decompressed before
        parsing)."""

    def content_checks(self, fmt: Serialization) -> Tuple[Callable, ...]:
        """Content checkers of the intermediate serialization ``fmt``."""
        return ()

    def run_oracle(self, oracle: str, current: Any, circuit: ReferenceCircuit) -> None:
        """Applies the equivalence oracle of the edge/circuit on the final
        object ``current`` of the conversion path."""
        if oracle == "unitary":
            actual = _to_unitary(current)
            assert_unitaries_close(_cirq_unitary(circuit.cirq()), actual)
        elif oracle == "counts":
            assert_counts(current, circuit)
        elif oracle == "structural":
            # structural checks were already run by the steps
            pass
        else:
            fn = EXTRA_ORACLES.get(oracle)
            if fn is None:
                raise ValueError(f"unknown oracle: {oracle}")
            fn(current, circuit)

    def run_path(
        self, edge: "Paradigm.ConversionEdge", circuit: ReferenceCircuit
    ) -> Any:
        """Runs a full conversion path (input -> steps) and applies its oracle."""
        set_loss_context(edge, circuit)
        current = edge.input_fn(circuit)
        for step in edge.steps:
            current = self.convert(
                current, step.fn, expected=circuit, checks=step.checks, **step.kwargs
            )
        self.run_oracle(circuit.oracle or edge.oracle, current, circuit)
        return current

    def circuit_runnable(
        self, circuit: ReferenceCircuit, edge: "Paradigm.ConversionEdge"
    ) -> bool:
        """Whether the circuit runs on the edge: same paradigm (and the edge is
        not excluded by ``supported_edge_ids``) and - for "counts" edges - the
        circuit has known expected counts."""
        if not circuit.supports_edge(edge):
            return False
        oracle = circuit.oracle or edge.oracle
        if oracle == "counts" and circuit.expected_counts is None:
            return False
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} paradigm={self.name!r}>"


# Facade / registries -----------------------------------------------------
# Aggregated from the concrete paradigms (populated at the bottom of the
# module, after the paradigm modules are imported).

PARADIGMS: List[Paradigm] = []
PARADIGM_BY_NAME: Dict[str, Paradigm] = {}


def build_edges(compression: Compression) -> Tuple[Paradigm.ConversionEdge, ...]:
    """Builds the full conversion graph (every paradigm, SDK -> format -> SDK)
    for one compression."""
    edges: List[Paradigm.ConversionEdge] = []
    for paradigm in PARADIGMS:
        edges.extend(paradigm.build_edges(compression))
    return tuple(edges)


def run_path(edge: Paradigm.ConversionEdge, circuit: ReferenceCircuit) -> Any:
    """Runs a full conversion path, dispatching to the edge's paradigm."""
    paradigm = PARADIGM_BY_NAME.get(edge.paradigm)
    if paradigm is None:
        raise ValueError(
            f"no registered paradigm for edge {edge.id!r} (paradigm {edge.paradigm!r})"
        )
    return paradigm.run_path(edge, circuit)


def circuit_runnable(edge: Paradigm.ConversionEdge, circuit: ReferenceCircuit) -> bool:
    """Whether the circuit runs on the edge, per the edge's paradigm."""
    return PARADIGM_BY_NAME[edge.paradigm].circuit_runnable(circuit, edge)


def get_reference_circuit(name: str) -> ReferenceCircuit:
    """Statically build a controlled input circuit by name."""
    if name not in REFERENCE_CIRCUIT_INDEX:
        raise KeyError(
            f"unknown reference circuit {name!r}; available: "
            f"{sorted(REFERENCE_CIRCUIT_INDEX)}"
        )
    return REFERENCE_CIRCUIT_INDEX[name]


# Public aliases kept for backwards compatibility with the per-module API.
ConversionEdge = Paradigm.ConversionEdge
Step = Paradigm.Step
convert = Paradigm.convert


# Populate the paradigm registry. The imports live at the bottom so the
# paradigm modules only ever see already-defined names when they import this
# module (no circular-import surprises): they use the base `Paradigm`, the
# generic helpers and the loss machinery, all defined above.
from gate_paradigm import GateParadigm  # noqa: E402
from analog_paradigm import AnalogParadigm  # noqa: E402
from photonic_paradigm import PhotonicParadigm  # noqa: E402


def _register(paradigm: Paradigm) -> None:
    PARADIGMS.append(paradigm)
    PARADIGM_BY_NAME[paradigm.name] = paradigm


GATE_PARADIGM = GateParadigm()
ANALOG_PARADIGM = AnalogParadigm()
PHOTONIC_PARADIGM = PhotonicParadigm()

for _paradigm in (GATE_PARADIGM, ANALOG_PARADIGM, PHOTONIC_PARADIGM):
    _register(_paradigm)

# Aggregated registries: the rest of the battery consumes these module-level
# names and automatically extends when a paradigm/edge/circuit is added.
UNSUPPORTED_CONVERSIONS = tuple(
    case for _paradigm in PARADIGMS for case in _paradigm.unsupported_conversions
)

CIRCUIT_FEATURE_COVERAGE = GENERIC_FEATURE_COVERAGE + tuple(
    row for _paradigm in PARADIGMS for row in _paradigm.feature_coverage
)

REFERENCE_CIRCUITS = tuple(
    circuit for _paradigm in PARADIGMS for circuit in _paradigm.circuits
)
REFERENCE_CIRCUIT_INDEX = {circuit.name: circuit for circuit in REFERENCE_CIRCUITS}
