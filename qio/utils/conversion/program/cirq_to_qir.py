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
from __future__ import annotations

import hashlib
import logging
from typing import Callable, List, Optional, Sequence, Type, Union

import cirq
import numpy as np
from cirq import Moment, protocols, transformers
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers import create_transformer_with_kwargs, merge_k_qubit_gates
import pyqir
from pyqir import (
    BasicBlock,
    Builder,
    Constant,
    Context,
    IntType,
    Module,
    PointerType,
    qir_module,
    qis,
    rt,
)

logger = logging.getLogger(__name__)


def _identity(builder, qubits):
    """Identity gate - no operation."""


def measure_x(builder, qubit, result):
    """Measure in the X basis by rotating to the Z basis first."""
    qis.h(builder, qubit)
    qis.mz(builder, qubit, result)


def measure_y(builder, qubit, result):
    """Measure in the Y basis by rotating to the Z basis first."""
    qis.s_adj(builder, qubit)
    qis.h(builder, qubit)
    qis.mz(builder, qubit, result)


# NOTE: Upper/lower case matters here, and was set to match the Cirq gate/op string representation.
PYQIR_OP_MAP: dict[str, Callable] = {
    # Identity Gate
    "I": _identity,
    # Single-Qubit Clifford Gates
    "H": qis.h,
    "X": qis.x,
    "Y": qis.y,
    "Z": qis.z,
    # Single-Qubit Rotation Gates
    "Rx": qis.rx,
    "Ry": qis.ry,
    "Rz": qis.rz,
    # Single-Qubit Non-Clifford Gates
    "S": qis.s,
    "T": qis.t,
    "S**-1": qis.s_adj,
    "T**-1": qis.t_adj,
    # Two-Qubit Gates
    "SWAP": qis.swap,
    "CNOT": qis.cx,
    "CZ": qis.cz,
    # Three-Qubit Gates
    "TOFFOLI": qis.ccx,
    # Classical Gates/Operations
    "measure_z": qis.mz,
    "measure_x": measure_x,
    "measure_y": measure_y,
    "reset": qis.reset,
}


def map_cirq_op_to_pyqir_callable(
    operation: cirq.Operation,
) -> tuple[Callable, str]:
    """Map a Cirq operation to its corresponding PyQIR callable.

    Args:
        operation: The Cirq operation to map.

    Returns:
        A tuple containing the corresponding PyQIR callable and a string identifying
        the gate/operation type.

    Raises:
        CirqConversionError: If the operation or gate is not supported.
    """
    if isinstance(operation, cirq.ops.GateOperation):
        gate = operation.gate

        if isinstance(gate, cirq.ops.MeasurementGate):
            op_name = "measure_z"

        elif isinstance(gate, cirq.ops.PauliMeasurementGate):
            op_name = str(gate.observable())
            op_name = op_name.removeprefix("+").removeprefix("-")
            if op_name not in ["X", "Y", "Z"]:
                raise CirqConversionError(
                    f"Multi-qubit gate {op_name} not supported for measurement."
                )
            op_name = f"measure_{op_name.lower()}"

        elif isinstance(gate, (cirq.ops.Rx, cirq.ops.Ry, cirq.ops.Rz)):
            op_name = gate.__class__.__name__
        elif isinstance(gate, cirq.ops.Pauli):
            op_name = gate.__class__.__name__[-1]  # X, Y or Z
        elif isinstance(gate, (cirq.ops.XPowGate, cirq.ops.YPowGate, cirq.ops.ZPowGate)):
            if gate.exponent == 1 or (
                isinstance(gate, cirq.ZPowGate) and gate.exponent in [0.25, -0.25, 0.5, -0.5]
            ):
                op_name = str(gate)  # X, Y, Z, S, T, S**-1, T**-1
            else:
                op_name = f"R{gate.__class__.__name__[0].lower()}"  # Rotations
        else:
            op_name = str(gate)
    else:
        op_name = str(operation)

    try:
        return PYQIR_OP_MAP[op_name], op_name
    except KeyError as err:
        raise CirqConversionError(f"Cirq gate {operation} not supported.") from err


class BasicCirqVisitor:
    """Converts cirq.Circuit elements to a PyQIR module.

    Args:
        initialize_runtime: If True, quantum runtime will be initialized. Defaults to True.
        record_output: If True, output of the circuit will be recorded. Defaults to True.
    """

    def __init__(self, initialize_runtime: bool = True, record_output: bool = True):
        self._module: pyqir.Module = None  # type: ignore[assignment]
        self._builder: pyqir.Builder = None  # type: ignore[assignment]
        self._entry_point: str = ""
        self._qubit_labels: dict[cirq.Qid, int] = {}
        # Result ids actually written by a measurement. Output recording is driven by
        # this rather than by the qubit count: a %Result only holds a value once a
        # measurement writes it, so recording a qubit that was never measured reads
        # uninitialised runtime state.
        self._measured_results: set[int] = set()
        # Maps cirq measurement keys to the ordered result slots they were written to.
        # Populated as measurement operations are visited; used to resolve the control
        # keys of classically-conditioned operations.
        self._result_slots_by_key: dict[cirq.MeasurementKey, list[int]] = {}
        self._initialize_runtime = initialize_runtime
        self._record_output = record_output

    def visit_cirq_module(self, module: CirqModule) -> None:
        """Initialize the PyQIR builder for the given module."""
        logger.debug("Visiting Cirq module '%s' (%d)", module.name, module.num_qubits)
        self._module = module.module
        context = self._module.context
        entry = pyqir.entry_point(self._module, module.name, module.num_qubits, module.num_clbits)

        self._entry_point = entry.name
        self._builder = Builder(context)
        self._builder.insert_at_end(BasicBlock(context, "entry", entry))

        if self._initialize_runtime:
            i8p = PointerType(IntType(context, 8))
            nullptr = Constant.null(i8p)
            rt.initialize(self._builder, nullptr)

    @property
    def entry_point(self) -> str:
        """Return the entry point name."""
        return self._entry_point

    def _check_initialized(self) -> None:
        """Raise if the visitor has not been initialized via visit_cirq_module."""
        if self._module is None or self._builder is None:
            raise RuntimeError(
                "Visitor has not been initialized. Call visit_cirq_module() first."
            )

    def finalize(self) -> None:
        """Finalize the QIR module by adding a return instruction."""
        self._check_initialized()
        self._builder.ret(None)

    def record_output(self) -> None:
        """Record the results of every measurement performed in the circuit."""
        if not self._record_output:
            return
        self._check_initialized()

        i8p = PointerType(IntType(self._module.context, 8))
        null_ptr = Constant.null(i8p)
        for i in sorted(self._measured_results):
            result_ref = pyqir.result(self._module.context, i)
            rt.result_record_output(self._builder, result_ref, null_ptr)

    def visit_register(self, qids: list[cirq.Qid]) -> None:
        """Assign global qubit labels to the given qids."""
        logger.debug("Visiting qids '%s'", str(qids))
        if not all(isinstance(x, cirq.Qid) for x in qids):
            raise TypeError("All elements in the list must be of type cirq.Qid.")
        self._qubit_labels.update({bit: n + len(self._qubit_labels) for n, bit in enumerate(qids)})

    def visit_operation(self, operation: cirq.Operation) -> None:
        """Visit and convert a single cirq operation to QIR."""
        self._check_initialized()
        qlabels = [self._qubit_labels[bit] for bit in operation.qubits]

        qubits = [pyqir.qubit(self._module.context, n) for n in qlabels]
        results = [pyqir.result(self._module.context, n) for n in qlabels]

        def handle_measurement(pyqir_func):
            logger.debug("Visiting measurement operation '%s'", str(operation))
            if isinstance(operation.gate, cirq.MeasurementGate):
                self._result_slots_by_key[operation.gate.key] = list(qlabels)
            for qubit, result, label in zip(qubits, results, qlabels, strict=True):
                self._measured_results.add(label)
                pyqir_func(self._builder, qubit, result)

        # dealing with conditional gates
        if isinstance(operation, cirq.ClassicallyControlledOperation):
            conditions = []
            for condition in operation._conditions:  # list of KeyConditions
                slots = self._result_slots_by_key.get(condition.key)
                if not slots:
                    raise CirqConversionError(
                        f"No measurement found for control key {condition.key!r}."
                    )
                index = condition.index if condition.index is not None else 0
                if index >= len(slots):
                    raise CirqConversionError(
                        f"Control key {condition.key!r} has no result slot {index}."
                    )
                conditions.append(pyqir.result(self._module.context, slots[index]))

            regular_op = operation.without_classical_controls()
            temp_pyqir_func, op_str = map_cirq_op_to_pyqir_callable(regular_op)

            if op_str in ["Rx", "Ry", "Rz"]:
                pyqir_func = lambda: temp_pyqir_func(  # noqa: E731
                    self._builder,
                    regular_op.gate._rads,  # type: ignore[union-attr]
                    *qubits,
                )
            else:
                pyqir_func = lambda: temp_pyqir_func(self._builder, *qubits)  # noqa: E731

            def _branch(conds, pyqir_func):
                if len(conds) == 0:
                    temp_id, _ = map_cirq_op_to_pyqir_callable(cirq.I)
                    return lambda: temp_id(self._builder, *qubits)
                return qis.if_result(
                    self._builder,
                    conds[0],
                    zero=_branch(conds[1:], pyqir_func),
                    one=pyqir_func,
                )

            _branch(conditions, pyqir_func)
        else:
            pyqir_func, op_str = map_cirq_op_to_pyqir_callable(operation)

            if op_str.startswith("measure"):
                handle_measurement(pyqir_func)
            elif op_str in ["Rx", "Ry", "Rz"]:
                pyqir_func(self._builder, operation.gate._rads, *qubits)  # type: ignore[union-attr]
            else:
                pyqir_func(self._builder, *qubits)

    def ir(self) -> str:
        """Return the QIR as a string."""
        self._check_initialized()
        return str(self._module)


def _add_rads_attribute(
    circuit: cirq.Circuit,
    *,
    context: cirq.TransformerContext = cirq.TransformerContext(),  # pylint: disable=unused-argument
) -> cirq.Circuit:
    """Attach a `_rads` attribute to every rotation gate in the circuit.

    The QIR visitor expects each single-qubit rotation gate to expose a `_rads`
    attribute holding its rotation angle in radians. Newer Cirq versions removed
    this private field, so it is recomputed here as ``exponent * pi``.
    """
    new_moments = []
    for moment in circuit:
        new_ops = []
        for op in moment.operations:
            gate = op.gate
            if isinstance(gate, (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate)):
                try:
                    gate._rads = float(gate.exponent * np.pi)  # type: ignore[union-attr]
                except AttributeError:
                    pass
            new_ops.append(op)
        new_moments.append(Moment(new_ops))
    return cirq.Circuit(new_moments)


class QirTargetGateSet(cirq.TwoQubitCompilationTargetGateset):
    """Target gateset that maps the supported Cirq gates onto the QIR basis set."""

    def __init__(
        self,
        *,
        atol: float = 1e-8,
        allow_partial_czs: bool = False,
        additional_gates: Sequence[Union[Type["cirq.Gate"], "cirq.Gate", "cirq.GateFamily"]] = (),
        preserve_moment_structure: bool = True,
    ) -> None:
        super().__init__(
            cirq.IdentityGate,
            cirq.HPowGate,
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.ZPowGate,
            cirq.SWAP,
            cirq.CNOT,
            cirq.CZ,
            cirq.TOFFOLI,
            cirq.ResetChannel,
            cirq.MeasurementGate,
            cirq.PauliMeasurementGate,
            *additional_gates,
            name="QirTargetGateset",
            preserve_moment_structure=preserve_moment_structure,
        )
        self.allow_partial_czs = allow_partial_czs
        self.atol = atol
        self._supported_single_qubit_set = (
            cirq.IdentityGate,
            cirq.ResetChannel,
            cirq.MeasurementGate,
            cirq.PauliMeasurementGate,
        )
        self._supported_power_gate_set = (
            cirq.HPowGate,
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.ZPowGate,
        )
        self._supported_two_qubit_set = (cirq.CNotPowGate, cirq.CZPowGate, cirq.SwapPowGate)
        self._supported_multi_qubit_set = cirq.CCXPowGate

    @property
    def preprocess_transformers(self) -> List["cirq.TRANSFORMER"]:
        """Override to keep TOFFOLI undecomposed."""
        return [
            create_transformer_with_kwargs(
                transformers.expand_composite,
                # Don't decompose operations that are in our gateset OR have <= 2 qubits
                no_decomp=lambda op: protocols.num_qubits(op) <= 2 or op in self,
            ),
            create_transformer_with_kwargs(
                merge_k_qubit_gates.merge_k_qubit_unitaries,
                k=2,
                rewriter=lambda op: op.with_tags(self._intermediate_result_tag),
            ),
        ]

    @property
    def postprocess_transformers(self) -> List["cirq.TRANSFORMER"]:
        return [_add_rads_attribute]  # type: ignore[list-item]

    def _decompose_single_qubit_operation(
        self, op: "cirq.Operation", moment_idx: int  # pylint: disable=unused-argument
    ) -> Union[DecomposeResult, None]:
        # Unwrap TaggedOperation and CircuitOperation to get the actual gate
        actual_op = op
        if isinstance(actual_op, cirq.TaggedOperation):
            actual_op = actual_op.sub_operation
        if isinstance(actual_op, cirq.CircuitOperation):
            ops = list(actual_op.circuit.all_operations())
            if len(ops) == 1:
                actual_op = ops[0]
        gate = actual_op.gate if hasattr(actual_op, "gate") else actual_op

        # Check if gate is already in our target gateset.
        # For Pow gates, any exponent is fine (they're all supported).
        if isinstance(gate, self._supported_power_gate_set):
            yield actual_op
            return None
        if isinstance(gate, self._supported_single_qubit_set):
            yield actual_op
            return None

        # Decompose everything else
        qubit = op.qubits[0]
        mat = cirq.unitary(op)
        for gate_result in cirq.single_qubit_matrix_to_gates(mat, self.atol):
            yield gate_result(qubit)
        return None

    def _decompose_two_qubit_operation(
        self, op: "cirq.Operation", _
    ) -> Union["cirq.OP_TREE", None]:
        # Unwrap TaggedOperation and CircuitOperation to get the actual gate
        actual_op = op
        if isinstance(actual_op, cirq.TaggedOperation):
            actual_op = actual_op.sub_operation
        if isinstance(actual_op, cirq.CircuitOperation):
            ops = list(actual_op.circuit.all_operations())
            if len(ops) == 1:
                actual_op = ops[0]
        gate = actual_op.gate if hasattr(actual_op, "gate") else actual_op

        # CNOT, CZ, SWAP are already in the target set
        # (they're CNotPowGate, CZPowGate, SwapPowGate with exponent=1)
        if isinstance(gate, self._supported_two_qubit_set):
            yield actual_op
            return None

        # Decompose unsupported gates
        if not cirq.has_unitary(op):
            return NotImplemented

        yield cirq.two_qubit_matrix_to_cz_operations(
            op.qubits[0],
            op.qubits[1],
            cirq.unitary(op),
            allow_partial_czs=self.allow_partial_czs,
            atol=self.atol,
        )
        return None

    def _decompose_multi_qubit_operation(
        self, op: cirq.Operation, _
    ) -> Union[DecomposeResult, None]:
        """Decompose operations acting on more than 2 qubits using this gateset."""
        # If already valid (in our gateset), leave it as is.
        if self.validate(op):
            yield op
            return None

        # If not valid, try to unwrap and check the actual gate
        actual_op = op
        if isinstance(actual_op, cirq.TaggedOperation):
            actual_op = actual_op.sub_operation
        if isinstance(actual_op, cirq.CircuitOperation):
            ops_list = list(actual_op.circuit.all_operations())
            if len(ops_list) == 1:
                actual_op = ops_list[0]
        gate = actual_op.gate if hasattr(actual_op, "gate") else actual_op

        if isinstance(gate, self._supported_multi_qubit_set):
            yield actual_op
        return NotImplemented


def preprocess_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
    """Preprocess a Cirq circuit so that every operation is supported by the QIR conversion."""
    gateset = QirTargetGateSet()
    qubit_map = {qubit: cirq.LineQubit(i) for i, qubit in enumerate(circuit.all_qubits())}
    line_qubit_circuit = circuit.transform_qubits(lambda q: qubit_map[q])

    # Circuits with conditional operations skip the full optimization and only
    # apply the postprocessors.
    has_conditional = any(
        isinstance(op, cirq.ClassicallyControlledOperation)
        for op in line_qubit_circuit.all_operations()
    )

    if has_conditional:
        processed: cirq.Circuit = line_qubit_circuit
        for transformer in gateset.postprocess_transformers:
            result = transformer(processed)
            processed = cirq.Circuit(result.all_operations())
        return processed

    try:
        return cirq.optimize_for_target_gateset(line_qubit_circuit, gateset=gateset)
    except Exception as exc:
        raise CirqConversionError("Failed to preprocess circuit.") from exc


class CirqConversionError(Exception):
    """Raised when a Cirq program cannot be converted to QIR."""


def generate_module_id(circuit: cirq.Circuit) -> str:
    """Generate a stable, short module ID from a Cirq circuit.

    The circuit is serialized to JSON, hashed with SHA-256, and the hash is
    truncated to a concise alphanumeric string prefixed with 'circuit-'.
    """
    serialized_circuit = cirq.to_json(circuit)
    hash_object = hashlib.sha256(serialized_circuit.encode())
    hash_hex = hash_object.hexdigest()
    alphanumeric_hash = "".join(filter(str.isalnum, hash_hex))
    truncated_hash = alphanumeric_hash[:7]
    return f"circuit-{truncated_hash}"


class CirqModule:
    """A Cirq circuit prepared for QIR conversion.

    Args:
        name: Name of the module.
        module: QIR Module instance.
        circuit: The underlying Cirq circuit.
        num_qubits: Number of qubits in the circuit.
    """

    def __init__(
        self,
        name: str,
        module: Module,
        circuit: cirq.Circuit,
        num_qubits: int,
    ):
        self._name = name
        self._module = module
        self._circuit = circuit
        self._num_qubits = num_qubits
        self._num_clbits = num_qubits  # create one classical result per qubit

    @property
    def name(self) -> str:
        """Return the name of the module."""
        return self._name

    @property
    def module(self) -> Module:
        """Return the QIR Module instance."""
        return self._module

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the circuit."""
        return self._num_qubits

    @property
    def num_clbits(self) -> int:
        """Return the number of classical result slots."""
        return self._num_clbits

    @classmethod
    def from_circuit(
        cls, circuit: cirq.Circuit, name: str, module: Optional[Module] = None
    ) -> "CirqModule":
        """Construct a CirqModule from a cirq.Circuit and an optional QIR Module."""
        if module is None:
            module = qir_module(Context(), name)
        return cls(
            name=name,
            module=module,
            circuit=circuit,
            num_qubits=len(circuit.all_qubits()),
        )

    def accept(self, visitor: BasicCirqVisitor) -> None:
        """Visit the circuit's qubits then its operations."""
        visitor.visit_cirq_module(self)
        visitor.visit_register(list(self._circuit.all_qubits()))
        for operation in self._circuit.all_operations():
            visitor.visit_operation(operation)
        visitor.record_output()
        visitor.finalize()


def convert(circuit: cirq.Circuit, name: Optional[str] = None, **kwargs) -> Module:
    """Convert a Cirq circuit to a PyQIR module.

    Args:
        circuit: The Cirq circuit to convert.
        name: Identifier for the created QIR module. Auto-generated if not provided.

    Keyword Args:
        initialize_runtime (bool): Whether to perform quantum runtime environment initialization,
            default `True`.
        record_output (bool): Whether to record output calls for registers, default `True`.

    Returns:
        The QIR ``pyqir.Module`` representation of the input Cirq circuit.

    Raises:
        TypeError: If the input is not a valid Cirq circuit.
        ValueError: If the input circuit is empty.
        CirqConversionError: If the conversion fails.
    """
    if not isinstance(circuit, cirq.Circuit):
        raise TypeError("Input quantum program must be of type cirq.Circuit.")

    if len(circuit) == 0:
        raise ValueError("Input quantum circuit must consist of at least one operation.")

    if name is None:
        name = generate_module_id(circuit)

    try:
        circuit = preprocess_circuit(circuit)
    except Exception as err:  # pylint: disable=broad-exception-caught
        raise CirqConversionError("Failed to preprocess circuit.") from err

    llvm_module = qir_module(Context(), name)
    module = CirqModule.from_circuit(circuit, name, llvm_module)

    visitor = BasicCirqVisitor(**kwargs)
    module.accept(visitor)

    error = llvm_module.verify()
    if error is not None:
        raise CirqConversionError(error)
    return llvm_module
