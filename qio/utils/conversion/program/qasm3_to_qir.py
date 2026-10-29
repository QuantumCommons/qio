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
# Portions of this file are derived from qBraid/qbraid-qir
# (https://github.com/qBraid/qbraid-qir), Copyright 2025 qBraid, licensed under
# the Apache License, Version 2.0 (modified); see the repository NOTICE file
# for the full attribution.
from __future__ import annotations

import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING, Type, Union

import openqasm3
import pyqasm
import pyqir
from openqasm3.ast import (
    BinaryExpression,
    BooleanLiteral,
    BranchingStatement,
    ClassicalDeclaration,
    Identifier,
    Include,
    IndexExpression,
    IndexedIdentifier,
    IntegerLiteral,
    Program,
    QubitDeclaration,
    QuantumBarrier,
    QuantumGate,
    QuantumMeasurementStatement,
    QuantumPhase,
    QuantumReset,
    Span,
    Statement,
    UnaryExpression,
    UnaryOperator,
)
from pyqir import Context, Module, qir_module, qis, rt

if TYPE_CHECKING:
    from pyqasm.modules import QasmModule

logger = logging.getLogger(__name__)

#: QASM3 values for the two QIR profiles we support.
PROFILE_BASE = "Base"
PROFILE_ADAPTIVE = "AdaptiveExecution"


@dataclass
class ProfileCapabilities:
    """Data class representing profile capabilities."""

    forward_branching: bool = False
    conditional_execution: bool = False
    qubit_reuse: bool = False
    measurement_tracking: bool = False
    grouped_output_recording: bool = False
    preserves_register_structure: bool = False
    inverted_bit_order: bool = False


@dataclass
class ProfileRestrictions:
    """Data class representing profile restrictions."""

    subset_barriers_allowed: bool = True
    must_cover_all_qubits: bool = False


class Profile(ABC):
    """Abstract base class defining the behaviour of a QIR profile."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.capabilities = ProfileCapabilities()
        self.restrictions = ProfileRestrictions()

    @abstractmethod
    def get_measurement_function(self) -> Callable:
        """Return the measurement function for this profile."""

    @abstractmethod
    def get_reset_function(self) -> Callable:
        """Return the reset function for this profile."""

    @abstractmethod
    def get_barrier_function(self) -> Callable:
        """Return the barrier function for this profile."""

    @abstractmethod
    def get_conditional_function(self) -> Callable:
        """Return the conditional branching function for this profile."""

    @abstractmethod
    def allow_qubit_use_after_measurement(self) -> bool:
        """Return whether this profile allows qubit operations after measurement."""

    @abstractmethod
    def record_output_method(self, visitor: "QIRVisitor") -> None:
        """Record the circuit output using this profile's output schema."""


class BaseProfile(Profile):
    """Basic (non-adaptive) QIR profile."""

    def __init__(self):
        super().__init__("Base", "1.0.0")
        self.capabilities.conditional_execution = True
        self.restrictions.subset_barriers_allowed = False
        self.restrictions.must_cover_all_qubits = True

    def get_measurement_function(self) -> Callable:
        return qis.mz

    def get_reset_function(self) -> Callable:
        return qis.reset

    def get_barrier_function(self) -> Callable:
        return qis.barrier

    def get_conditional_function(self) -> Callable:
        return qis.if_result

    def allow_qubit_use_after_measurement(self) -> bool:
        return False

    def record_output_method(self, visitor: QIRVisitor) -> None:
        """Basic output recording - one result per measurement actually performed.

        Records only results a measurement actually wrote. Recording one per *qubit*
        instead emitted reads of results the program never measured: a circuit with no
        measurement declares ``required_num_results=0`` yet recorded one result per
        qubit, and qir-runner reported the uninitialised state it found there as if it
        were a measurement outcome.
        """
        if not visitor._record_output:
            return
        assert visitor._llvm_module is not None
        assert visitor._builder is not None
        i8p = pyqir.PointerType(pyqir.IntType(visitor._llvm_module.context, 8))
        for i in sorted(visitor._measured_results):
            result_ref = pyqir.result(visitor._llvm_module.context, i)
            rt.result_record_output(
                visitor._builder, result_ref, pyqir.Constant.null(i8p)
            )


class AdaptiveProfile(Profile):
    """Adaptive execution QIR profile with register-preserving output recording."""

    def __init__(self):
        super().__init__("AdaptiveExecution", "1.0.0")
        self.capabilities.forward_branching = True
        self.capabilities.conditional_execution = True
        self.capabilities.qubit_reuse = True
        self.capabilities.measurement_tracking = True
        self.capabilities.grouped_output_recording = True
        self.capabilities.preserves_register_structure = True
        self.capabilities.inverted_bit_order = True

        self.restrictions.subset_barriers_allowed = False
        self.restrictions.must_cover_all_qubits = True

    def get_measurement_function(self) -> Callable:
        return qis.mz

    def get_reset_function(self) -> Callable:
        return qis.reset

    def get_barrier_function(self) -> Callable:
        return qis.barrier

    def get_conditional_function(self) -> Callable:
        return qis.if_result

    def allow_qubit_use_after_measurement(self) -> bool:
        return True

    def record_output_method(self, visitor: QIRVisitor) -> None:
        """Adaptive profile output recording - preserves register structure.

        Records only results a measurement actually wrote, which is what
        ``_clbit_labels`` alone cannot tell us: it is populated when a register is
        *declared*, so a bit that was never measured still has a label. Recording it
        makes the runtime report an uninitialised result as a measurement outcome.

        A register with no measured bits is skipped entirely rather than recorded as an
        empty array, and the array length counts the bits actually recorded -- emitting
        the declared width would promise elements that never follow.
        """
        if not visitor._record_output:
            return
        assert visitor._llvm_module is not None
        assert visitor._builder is not None
        i8p = pyqir.PointerType(pyqir.IntType(visitor._llvm_module.context, 8))
        null_ptr = pyqir.Constant.null(i8p)
        recorded_ids = set()

        # Record output grouped by register to preserve structure
        for reg_name, reg_size in visitor._global_creg_size_map.items():
            # Collect first (in inverted order), so the array length matches what follows.
            measured_ids = []
            for i in range(reg_size - 1, -1, -1):
                bit_id = visitor._clbit_labels.get(f"{reg_name}_{i}")
                if bit_id is None or bit_id in recorded_ids:
                    continue
                if bit_id in visitor._measured_results:
                    measured_ids.append(bit_id)

            if not measured_ids:
                continue

            rt.array_record_output(
                visitor._builder,
                pyqir.const(
                    pyqir.IntType(visitor._llvm_module.context, 64), len(measured_ids)
                ),
                null_ptr,
            )
            for bit_id in measured_ids:
                result_ref = pyqir.result(visitor._llvm_module.context, bit_id)
                rt.result_record_output(visitor._builder, result_ref, null_ptr)
                recorded_ids.add(bit_id)


class ProfileRegistry:
    """Registry mapping profile names to their implementations."""

    _profiles: dict[str, Profile] = {
        "Base": BaseProfile(),
        "AdaptiveExecution": AdaptiveProfile(),
    }

    @classmethod
    def get_profile(cls, name: str) -> Profile:
        """Return the profile registered under ``name``, if any."""
        try:
            return cls._profiles[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown profile: {name!r}. Valid profiles are: {sorted(cls._profiles)}"
            ) from exc

    @classmethod
    def list_profiles(cls) -> list[str]:
        """Return the names of all registered profiles."""
        return list(cls._profiles.keys())


class QIRVisitor(ABC):
    """Holds the low-level state shared by every QIR building visitor."""

    def __init__(self) -> None:
        self._record_output: bool = True
        self._llvm_module: Optional[pyqir.Module] = None
        self._builder: Optional[pyqir.Builder] = None
        self._clbit_labels: dict[str, int] = {}
        self._global_creg_size_map: dict[str, int] = {}
        # Result ids actually written by a measurement. Output recording is driven by
        # this rather than by a qubit or clbit count: a %Result only holds a value once
        # mz writes it, so recording an id absent from this set reads uninitialised
        # runtime state. Subclasses that emit measurements must populate it.
        self._measured_results: set[int] = set()


def _identity(builder, qubit):
    """Identity gate - no operation."""


def _sqrt_x(builder, qubit):
    """√X gate implemented as H·S·H."""
    qis.h(builder, qubit)
    qis.s(builder, qubit)
    qis.h(builder, qubit)


def _sqrt_x_dagger(builder, qubit):
    """√X† gate implemented as H·S†·H."""
    qis.h(builder, qubit)
    qis.s_adj(builder, qubit)
    qis.h(builder, qubit)


# Gate maps are keyed by (lower-cased) QASM3 gate name, as emitted after unrolling.
PYQIR_ONE_QUBIT_OP_MAP: dict[str, Callable] = {
    "h": qis.h,
    "x": qis.x,
    "y": qis.y,
    "z": qis.z,
    "s": qis.s,
    "t": qis.t,
    "sdg": qis.s_adj,
    "tdg": qis.t_adj,
    "id": _identity,
    "sx": _sqrt_x,
    "sxdg": _sqrt_x_dagger,
}

PYQIR_ONE_QUBIT_ROTATION_MAP: dict[str, Callable] = {
    "rx": qis.rx,
    "ry": qis.ry,
    "rz": qis.rz,
}

PYQIR_TWO_QUBIT_OP_MAP: dict[str, Callable] = {
    "cx": qis.cx,
    "cnot": qis.cx,
    "cz": qis.cz,
    "swap": qis.swap,
}

PYQIR_THREE_QUBIT_OP_MAP: dict[str, Callable] = {
    "ccx": qis.ccx,
    "ccnot": qis.ccx,
}

# Maps each gate category to the number of qubits its operations act on.
QASM_OP_MAPPINGS: list[tuple[dict[str, Callable], int]] = [
    (PYQIR_ONE_QUBIT_OP_MAP, 1),
    (PYQIR_ONE_QUBIT_ROTATION_MAP, 1),
    (PYQIR_TWO_QUBIT_OP_MAP, 2),
    (PYQIR_THREE_QUBIT_OP_MAP, 3),
]


def map_qasm_op_to_pyqir_callable(op_name: str) -> tuple[Callable, int]:
    """Map a (lower-cased) QASM gate name to its PyQIR callable and qubit count.

    Raises:
        Qasm3ConversionError: If the QASM operation is unsupported or undeclared.
    """
    op_name = op_name.lower()
    for mapping, qubits in QASM_OP_MAPPINGS:
        if op_name in mapping:
            return mapping[op_name], qubits
    raise Qasm3ConversionError(f"Unsupported / undeclared QASM operation: {op_name}")


_PHYSICAL_QUBIT_RE = re.compile(r"^\$(\d+)$")


def _physical_qubit_index(node: Any) -> Optional[int]:
    """Return the hardware index of a physical qubit reference, else None.

    Physical qubits are written "$n" in OpenQASM 3 and survive unrolling as a plain
    ``Identifier`` (they have no declaring register), unlike virtual qubits which
    become ``IndexedIdentifier``. The index is carried in the name: "$3" is qubit 3.
    """
    if not isinstance(node, Identifier):
        return None
    match = _PHYSICAL_QUBIT_RE.match(node.name)
    return int(match.group(1)) if match else None


class QasmQIRVisitor(QIRVisitor):
    """A profile-aware visitor for converting OpenQASM 3 programs to QIR.

    Args:
        profile_name: Name of the QIR profile to use. Defaults to "Base".
        initialize_runtime: If True, quantum runtime will be initialized. Defaults to True.
        record_output: If True, output of the circuit will be recorded. Defaults to True.
        external_gates: List of custom gates that should not be unrolled.
        emit_barrier_calls: If True, barrier calls will be emitted. Defaults to True.
    """

    @staticmethod
    def _valid_profiles() -> list[str]:
        return ProfileRegistry.list_profiles()

    def __init__(
        self,
        profile_name: str = PROFILE_BASE,
        initialize_runtime: bool = True,
        record_output: bool = True,
        external_gates: Optional[list[str]] = None,
        emit_barrier_calls: bool = True,
    ):
        super().__init__()

        # Resolve the profile before doing any work so invalid names fail early.
        if profile_name not in ProfileRegistry.list_profiles():
            raise ValueError(
                f"Unknown profile: {profile_name!r}. Valid profiles are: "
                f"{sorted(ProfileRegistry.list_profiles())}"
            )
        self._profile: Profile = ProfileRegistry.get_profile(profile_name)

        self._llvm_module: pyqir.Module = None  # type: ignore[assignment]
        self._builder: pyqir.Builder = None  # type: ignore[assignment]
        self._entry_point: str = ""
        self._qubit_labels: dict[str, int] = {}
        self._clbit_labels: dict[str, int] = {}
        self._global_qreg_size_map: dict[str, int] = {}
        self._global_creg_size_map: dict[str, int] = {}
        self._barrier_qubits: set[pyqir.Constant] = set()
        self._required_qubit_count: int = 0

        self._initialize_runtime: bool = initialize_runtime
        self._record_output: bool = record_output
        self._emit_barrier_calls: bool = emit_barrier_calls

        # Qubits whose measurement state must be tracked for use-after-measurement checks.
        if self._profile.allow_qubit_use_after_measurement() is False:
            self._measured_qubits: dict[int, bool] = {}

        # External gates, mapped lazily to their PyQIR function declarations.
        self._external_gates_map: dict[str, Optional[pyqir.Function]] = {
            external_gate: None for external_gate in (external_gates or [])
        }

    @property
    def profile(self) -> Profile:
        """Return the active profile."""
        return self._profile

    @property
    def entry_point(self) -> str:
        return self._entry_point

    def visit_qasm3_module(self, module: QasmQIRModule) -> None:
        """Build the QIR entry point for the given QASM3 module."""
        qasm3_module = module.qasm_program
        logger.debug(
            "Visiting Qasm3 module '%s' (%d qubits) with profile '%s'",
            module.name,
            qasm3_module.num_qubits,
            self._profile.name,
        )

        self._llvm_module = module.llvm_module
        context = self._llvm_module.context
        qir_profiles = "adaptive" if self._profile.name == PROFILE_ADAPTIVE else "base"

        # pyqasm registers physical qubits during unrolling and folds them into num_qubits:
        # indices are absolute hardware addresses, so a program touching only "$7" reports 8
        # qubits, which is what the entry point has to declare for "$7" to be a valid QIR id.
        self._required_qubit_count = qasm3_module.num_qubits

        entry = pyqir.entry_point(
            self._llvm_module,
            module.name,
            self._required_qubit_count,
            qasm3_module.num_clbits,
            qir_profiles=qir_profiles,
        )

        self._entry_point = entry.name
        self._builder = pyqir.Builder(context)
        self._builder.insert_at_end(pyqir.BasicBlock(context, "entry", entry))

        if self._initialize_runtime:
            i8p = pyqir.PointerType(pyqir.IntType(context, 8))
            rt.initialize(self._builder, pyqir.Constant.null(i8p))

    def finalize(self) -> None:
        self._check_and_apply_barrier()  # apply an incomplete barrier left at program end
        assert self._builder is not None
        self._builder.ret(None)

    def record_output(self, module: QasmQIRModule) -> None:
        """Record output using profile-specific method."""
        self._profile.record_output_method(self)

    def _visit_register(
        self, register: Union[QubitDeclaration, ClassicalDeclaration]
    ) -> None:
        """Assign a global bit index to every bit of a register declaration."""
        logger.debug("Visiting register '%s'", str(register))
        is_qubit = isinstance(register, QubitDeclaration)

        current_size = len(self._qubit_labels) if is_qubit else len(self._clbit_labels)
        if is_qubit:
            register_size = 1 if register.size is None else register.size.value
            register_name = register.qubit.name
        else:
            register_size = (
                1 if register.type.size is None else register.type.size.value
            )
            register_name = register.identifier.name

        size_map = (
            self._global_qreg_size_map if is_qubit else self._global_creg_size_map
        )
        label_map = self._qubit_labels if is_qubit else self._clbit_labels

        size_map[register_name] = register_size
        for i in range(register_size):
            label_map[f"{register_name}_{i}"] = current_size + i

        logger.debug("Added labels for register '%s'", str(register))

    def _get_op_bits(self, operation: Any, qubits: bool = True) -> list[pyqir.Constant]:
        """Return the QIR qubits (or results) an operation acts on.

        QASM3 programs are unrolled before visiting, so every register operand is an
        ``IndexedIdentifier``; physical qubits ("$n") are the exception and carry their
        hardware index in the identifier itself.
        """
        qir_bits: list[pyqir.Constant] = []

        if isinstance(operation, QuantumMeasurementStatement):
            # _visit_measurement is the sole handler for measurement statements and has
            # already rejected a missing target, so operation.target is non-None here.
            bit_list = [operation.measure.qubit] if qubits else [operation.target]
        else:
            bit_list = (
                operation.qubits
                if isinstance(operation.qubits, list)
                else [operation.qubits]
            )

        for bit in bit_list:
            physical_id = _physical_qubit_index(bit) if qubits else None
            if physical_id is not None:
                qir_bits.append(pyqir.qubit(self._llvm_module.context, physical_id))
                continue

            if not isinstance(bit, IndexedIdentifier):
                kind = "qubit" if qubits else "classical bit"
                raise_qasm3_error(
                    f"Unsupported {kind} operand of type {type(bit).__name__}",
                    span=getattr(bit, "span", None),
                )

            reg_name = bit.name.name
            assert isinstance(bit.indices, list) and len(bit.indices) == 1
            assert isinstance(bit.indices[0], list) and len(bit.indices[0]) == 1
            assert isinstance(bit.indices[0][0], IntegerLiteral)
            bit_id = bit.indices[0][0].value

            label_map = self._qubit_labels if qubits else self._clbit_labels
            label = label_map[f"{reg_name}_{bit_id}"]

            qir_bits.append(
                pyqir.qubit(self._llvm_module.context, label)
                if qubits
                else pyqir.result(self._llvm_module.context, label)
            )

        return qir_bits

    def _check_qubit_use_after_measurement(
        self, qubit_ids: list[pyqir.Constant]
    ) -> None:
        """Enforce the profile's qubit use-after-measurement policy."""
        if self._profile.allow_qubit_use_after_measurement():
            return
        for qubit_id in qubit_ids:
            qubit_id_result = pyqir.ptr_id(qubit_id)
            if qubit_id_result is not None and self._measured_qubits.get(
                qubit_id_result, False
            ):
                raise_qasm3_error(
                    f"Base Profile violation: Cannot use qubit {qubit_id_result} after measurement"
                )

    def _visit_measurement(self, statement: QuantumMeasurementStatement) -> None:
        """Visit a measurement statement."""
        logger.debug("Visiting measurement statement '%s'", str(statement))

        if statement.target is None:
            raise_qasm3_error(
                "Measurement result must be assigned to a classical bit, "
                "e.g. 'c[0] = measure q[0];'",
                span=statement.span,
            )
        source_ids = self._get_op_bits(statement, qubits=True)
        target_ids = self._get_op_bits(statement, qubits=False)
        measurement_func = self._profile.get_measurement_function()

        for src_id, tgt_id in zip(source_ids, target_ids):
            if self._profile.allow_qubit_use_after_measurement() is False:
                self._measured_qubits[pyqir.ptr_id(src_id)] = True

            measurement_func(self._builder, src_id, tgt_id)

            result_id = pyqir.ptr_id(tgt_id)
            if result_id is not None:
                self._measured_results.add(result_id)

    def _visit_reset(self, statement: QuantumReset) -> None:
        """Visit a reset statement."""
        logger.debug("Visiting reset statement '%s'", str(statement))
        qubit_ids = self._get_op_bits(statement, True)
        reset_func = self._profile.get_reset_function()

        for qid in qubit_ids:
            if self._profile.allow_qubit_use_after_measurement() is False:
                qubit_id_result = pyqir.ptr_id(qid)
                if qubit_id_result is not None:
                    self._measured_qubits[qubit_id_result] = False
            reset_func(self._builder, qid)

    def _barrier_applicable(self) -> bool:
        """Return whether the accumulated barrier can be emitted for this profile."""
        if self._profile.restrictions.subset_barriers_allowed:
            return True

        # Physical qubits belong to no declared register, so the register size map is
        # empty for those programs and the entry point's qubit count is the only measure
        # of how many qubits a full barrier has to cover.
        total_qubit_count = max(
            sum(self._global_qreg_size_map.values()), self._required_qubit_count
        )
        return len(self._barrier_qubits) == total_qubit_count

    def _check_and_apply_barrier(self) -> None:
        """Emit a pending barrier if the profile allows it, else raise."""
        if not self._barrier_qubits:
            return

        if self._barrier_applicable():
            if self._emit_barrier_calls:
                self._profile.get_barrier_function()(self._builder)
            self._barrier_qubits.clear()
        elif self._emit_barrier_calls:
            raise_qasm3_error(
                "Barrier operation on a qubit subset is not supported in pyqir",
                err_type=NotImplementedError,
            )

    def _visit_barrier(self, barrier: QuantumBarrier) -> None:
        """Collect the qubits covered by a barrier statement."""
        self._barrier_qubits.update(self._get_op_bits(barrier, qubits=True))

        # try to apply barrier in case all qubits are covered here itself
        if self._barrier_applicable():
            if self._emit_barrier_calls:
                self._profile.get_barrier_function()(self._builder)
            self._barrier_qubits.clear()

    def _get_op_parameters(self, operation: QuantumGate) -> list[float]:
        """Return the floating-point parameters of a parametric gate."""
        return [param.value for param in operation.arguments if hasattr(param, "value")]

    def _visit_basic_gate_operation(self, operation: QuantumGate) -> None:
        """Visit a gate operation, emitting its QIR call."""
        logger.debug("Visiting basic gate operation '%s'", str(operation))
        op_name: str = operation.name.name.lower()
        op_qubits = self._get_op_bits(operation)

        # Profile-aware qubit usage check
        self._check_qubit_use_after_measurement(op_qubits)

        qir_func, expected_qubit_count = map_qasm_op_to_pyqir_callable(op_name)
        if len(op_qubits) != expected_qubit_count:
            raise_qasm3_error(
                f"Gate {op_name} expects {expected_qubit_count} qubit(s), got {len(op_qubits)}"
            )

        if op_name in PYQIR_ONE_QUBIT_ROTATION_MAP:
            op_parameters = self._get_op_parameters(operation)
            if not op_parameters:
                raise_qasm3_error(f"Parametric gate {op_name} requires parameters")
            qir_func(self._builder, *op_parameters, *op_qubits)
        else:
            if operation.arguments:
                raise_qasm3_error(
                    f"Non-parametric gate {op_name} should not have parameters"
                )
            qir_func(self._builder, *op_qubits)

    def _visit_external_gate_operation(self, operation: QuantumGate) -> None:
        """Visit a custom gate that was declared external (not unrolled)."""
        logger.debug("Visiting external gate operation '%s'", str(operation))
        op_name: str = operation.name.name
        op_qubits = self._get_op_bits(operation)

        self._check_qubit_use_after_measurement(op_qubits)

        if operation.modifiers:
            raise_qasm3_error(
                "Modifiers on externally linked gates are not supported in pyqir",
                err_type=NotImplementedError,
            )

        context = self._llvm_module.context
        qir_function = self._external_gates_map[op_name]
        if qir_function is None:
            # First time seeing this external gate -> declare its external function.
            qubit_ptr = pyqir.PointerType(pyqir.Type.void(context))
            arguments = [pyqir.Type.double(context)] * len(operation.arguments)
            arguments += [qubit_ptr] * len(op_qubits)
            qir_function = pyqir.Function(
                pyqir.FunctionType(pyqir.Type.void(context), arguments),
                pyqir.Linkage.EXTERNAL,
                f"__quantum__qis__{op_name}__body",
                self._llvm_module,
            )
            self._external_gates_map[op_name] = qir_function

        if operation.arguments:
            op_parameters = list(map(float, self._get_op_parameters(operation)))
            self._builder.call(qir_function, [*op_parameters, *op_qubits])
        else:
            self._builder.call(qir_function, op_qubits)

    def _visit_generic_gate_operation(self, operation: QuantumGate) -> None:
        """Route a gate operation to its external or built-in handler."""
        if operation.name.name in self._external_gates_map:
            self._visit_external_gate_operation(operation)
        else:
            self._visit_basic_gate_operation(operation)

    def _get_branch_params(self, condition: Any) -> tuple[str, int, bool]:
        """Extract (register name, register id, positive branch) from a branching condition."""

        def validate_index_expression(expression):
            assert isinstance(expression, IndexExpression)
            assert isinstance(expression.collection, Identifier)
            assert isinstance(expression.index, list) and len(expression.index) == 1
            assert isinstance(expression.index[0], IntegerLiteral)

        if isinstance(condition, UnaryExpression):
            validate_index_expression(condition.expression)
            return (
                condition.expression.collection.name,
                condition.expression.index[0].value,
                condition.op != UnaryOperator["!"],
            )
        if isinstance(condition, BinaryExpression):
            assert isinstance(
                condition.rhs, BooleanLiteral
            ), "Invalid branching condition"
            validate_index_expression(condition.lhs)
            return (
                condition.lhs.collection.name,
                condition.lhs.index[0].value,
                condition.rhs.value,
            )
        if isinstance(condition, IndexExpression):
            assert isinstance(condition.index, list) and len(condition.index) == 1
            return (condition.collection.name, condition.index[0].value, True)
        return "", -1, True

    def _visit_branching_statement(self, statement: BranchingStatement) -> None:
        """Visit an if/else branching statement."""
        logger.debug(
            "Visiting branching statement with profile '%s'", self._profile.name
        )

        if not self._profile.capabilities.conditional_execution:
            raise_qasm3_error(
                f"Profile '{self._profile.name}' does not support conditional execution",
                err_type=NotImplementedError,
            )

        condition = statement.condition
        if_block = statement.if_block
        else_block = statement.else_block
        reg_name, reg_id, positive_branch = self._get_branch_params(condition)

        if not positive_branch:
            if_block, else_block = else_block, if_block

        def _visit_statement_block(block):
            if block:
                for stmt in block:
                    self.visit_statement(stmt)

        conditional_func = self._profile.get_conditional_function()
        zero_callback: Callable[[], None] = lambda: _visit_statement_block(else_block)
        one_callback: Callable[[], None] = lambda: _visit_statement_block(if_block)

        conditional_func(
            self._builder,
            pyqir.result(
                self._llvm_module.context, self._clbit_labels[f"{reg_name}_{reg_id}"]
            ),
            zero=zero_callback,
            one=one_callback,
        )

    def visit_statement(self, statement: Statement) -> None:
        """Dispatch a single statement to its handler."""
        logger.debug("Visiting statement '%s'", str(statement))

        visit_map = {
            Include: lambda x: None,  # No operation
            QubitDeclaration: self._visit_register,
            ClassicalDeclaration: self._visit_register,
            QuantumMeasurementStatement: self._visit_measurement,
            QuantumReset: self._visit_reset,
            QuantumBarrier: self._visit_barrier,
            QuantumGate: self._visit_generic_gate_operation,
            BranchingStatement: self._visit_branching_statement,
            QuantumPhase: lambda x: None,  # No operation
        }

        if not isinstance(statement, QuantumBarrier):
            self._check_and_apply_barrier()

        visitor_function = visit_map.get(type(statement))
        if visitor_function:
            visitor_function(statement)
        else:
            raise_qasm3_error(
                f"Unsupported statement of type {type(statement)}", span=statement.span
            )

    def ir(self) -> str:
        return str(self._llvm_module)

    def bitcode(self) -> bytes:
        return self._llvm_module.bitcode


class Qasm3ConversionError(Exception):
    """Raised when converting an OpenQASM 3 program to QIR fails."""


def raise_qasm3_error(
    message: Optional[str] = None,
    err_type: Type[Exception] = Qasm3ConversionError,
    span: Optional[Span] = None,
    raised_from: Optional[Exception] = None,
) -> None:
    """Raise a QASM3 conversion error, optionally chaining from another exception.

    Args:
        message: The error message.
        err_type: The type of error to raise.
        span: The span (location) in the QASM file where the error occurred.
        raised_from: Optional exception from which this error was raised (chaining).
    """
    if span:
        logging.error(
            "Error at line %s, column %s in QASM file",
            span.start_line,
            span.start_column,
        )
    if raised_from:
        raise err_type(message) from raised_from
    raise err_type(message)


def generate_module_id() -> str:
    """Generate a unique QIR module ID."""
    return f"program-{uuid.uuid1()}"


class QasmQIRModule:
    """An OpenQASM 3 program bundled with the PyQIR module being built.

    Args:
        name: Name of the module.
        qasm_module: The pyqasm QASM3 module.
        llvm_module: The PyQIR module.
    """

    def __init__(
        self,
        name: str,
        qasm_module: "QasmModule",
        llvm_module: pyqir.Module,
    ):
        self._name = name
        self._llvm_module = llvm_module
        self._qasm_program = qasm_module

    @property
    def name(self) -> str:
        """Return the name of the module."""
        return self._name

    @property
    def llvm_module(self) -> pyqir.Module:
        """Return the PyQIR Module instance."""
        return self._llvm_module

    @property
    def qasm_program(self) -> "QasmModule":
        """Return the pyqasm QASM3 program."""
        return self._qasm_program

    @property
    def num_qubits(self) -> int:
        return self.qasm_program.num_qubits

    @property
    def num_clbits(self) -> int:
        return self.qasm_program.num_clbits

    def accept(self, visitor: QasmQIRVisitor) -> None:
        """Visit the preprocessed program statements, then record output and finalize."""
        visitor.visit_qasm3_module(self)
        for statement in self.qasm_program.unrolled_ast.statements:
            visitor.visit_statement(statement)
        visitor.record_output(self)
        visitor.finalize()


def convert(
    program: Union[Program, str],
    name: Optional[str] = None,
    external_gates: Optional[list[str]] = None,
    profile: str = PROFILE_BASE,
    **kwargs,
) -> str:
    """Convert an OpenQASM 3 program to QIR LLVM IR text.

    Args:
        program: The OpenQASM 3 program to convert (as an AST or a string).
        name: Identifier for the created QIR module. Auto-generated if not provided.
        external_gates: List of custom gate names that are not natively recognized by
            pyqasm but should be treated as valid during program unrolling.
        profile: The QIR profile to use: "Base" or "AdaptiveExecution". Defaults to "Base".

    Keyword Args:
        initialize_runtime (bool): Whether to perform quantum runtime environment
            initialization, defaults to `True`.
        record_output (bool): Whether to record output calls for registers, defaults to `True`.
        emit_barrier_calls (bool): Whether to emit barrier calls, defaults to `True`.

    Returns:
        The QIR LLVM IR text (``str``) representation of the input OpenQASM 3 program.

    Raises:
        TypeError: If the input is not a valid OpenQASM 3 program.
        ValueError: If the profile is unknown.
        Qasm3ConversionError: If the conversion fails.
    """
    if isinstance(program, Program):
        program = openqasm3.dumps(program)
    elif not isinstance(program, str):
        raise TypeError(
            "Input quantum program must be of type openqasm3.ast.Program or str."
        )

    qasm3_module = pyqasm.loads(program)
    qasm3_module.unroll(external_gates=external_gates)
    if name is None:
        name = generate_module_id()
    llvm_module = qir_module(Context(), name)

    final_module = QasmQIRModule(name, qasm3_module, llvm_module)

    # Validate the profile before doing any work so invalid names fail early.
    if profile not in ProfileRegistry.list_profiles():
        raise ValueError(
            f"Unknown profile: {profile!r}. Valid profiles are: {sorted(ProfileRegistry.list_profiles())}"
        )

    visitor = QasmQIRVisitor(
        profile_name=profile, external_gates=external_gates, **kwargs
    )
    final_module.accept(visitor)

    err = llvm_module.verify()
    if err is not None:
        raise Qasm3ConversionError(err)
    return str(llvm_module)
