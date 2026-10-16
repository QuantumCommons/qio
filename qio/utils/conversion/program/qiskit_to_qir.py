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
# (https://github.com/qBraid/qbraid-qir), Copyright 2026 qBraid, licensed under
# the Apache License, Version 2.0 (modified). The derived code further
# originates from microsoft/qiskit-qir (MIT, Copyright (c) Microsoft
# Corporation) as documented in qBraid's qiskit/NOTICE.md; see the repository
# NOTICE file for the full attribution.
from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import pyqir
from pyqir import (
    BasicBlock,
    Builder,
    Constant,
    Context,
    IntType,
    Module,
    PointerType,
    entry_point,
    qir_module,
    qis,
    rt,
)

from qiskit import ClassicalRegister, QuantumRegister
from qiskit.circuit import Clbit, Instruction, Qubit, QuantumCircuit
from qiskit.compiler import transpile as qiskit_transpile

logger = logging.getLogger(__name__)


def _identity(builder, qubit):
    """Identity gate - no operation."""


# Single-qubit non-parametric gates, keyed by (lower-cased) Qiskit instruction name.
PYQIR_ONE_QUBIT_OP_MAP: dict[str, Callable] = {
    "h": qis.h,
    "x": qis.x,
    "y": qis.y,
    "z": qis.z,
    "s": qis.s,
    "sdg": qis.s_adj,
    "t": qis.t,
    "tdg": qis.t_adj,
    "id": _identity,
    "reset": qis.reset,
}

# Single-qubit rotation gates, keyed by Qiskit instruction name.
PYQIR_ONE_QUBIT_ROTATION_MAP: dict[str, Callable] = {
    "rx": qis.rx,
    "ry": qis.ry,
    "rz": qis.rz,
}

PYQIR_TWO_QUBIT_OP_MAP: dict[str, Callable] = {
    "cx": qis.cx,
    "cz": qis.cz,
    "swap": qis.swap,
}

PYQIR_THREE_QUBIT_OP_MAP: dict[str, Callable] = {
    "ccx": qis.ccx,
}

# Qiskit carries measurement in the instruction name rather than the gate.
MEASUREMENT_GATES = frozenset({"measure", "m", "mz"})

NOOP_INSTRUCTIONS = frozenset({"delay"})

SUPPORTED_INSTRUCTIONS: list[str] = sorted(
    set(PYQIR_ONE_QUBIT_OP_MAP)
    | set(PYQIR_ONE_QUBIT_ROTATION_MAP)
    | set(PYQIR_TWO_QUBIT_OP_MAP)
    | set(PYQIR_THREE_QUBIT_OP_MAP)
    | MEASUREMENT_GATES
    | {"barrier"}
    | NOOP_INSTRUCTIONS
)

QISKIT_BASIS_GATES: list[str] = sorted(
    (set(PYQIR_ONE_QUBIT_OP_MAP) - {"reset"})
    | set(PYQIR_ONE_QUBIT_ROTATION_MAP)
    | set(PYQIR_TWO_QUBIT_OP_MAP)
    | set(PYQIR_THREE_QUBIT_OP_MAP)
    | {"measure", "reset"}
)


class QiskitConversionError(Exception):
    """Raised when a Qiskit circuit cannot be converted to QIR."""


class BasicQiskitVisitor:
    """Converts Qiskit circuit elements to a PyQIR module.

    Args:
        initialize_runtime: If True, quantum runtime will be initialized. Defaults to True.
        record_output: If True, output of the circuit will be recorded. Defaults to True.
        emit_barrier_calls: If True, barrier instructions will be emitted. Defaults to False.
    """

    def __init__(
        self,
        initialize_runtime: bool = True,
        record_output: bool = True,
        emit_barrier_calls: bool = False,
    ):
        self._module: pyqir.Module = None  # type: ignore[assignment]
        self._qiskit_module: QiskitModule | None = None
        self._builder: pyqir.Builder = None  # type: ignore[assignment]
        self._entry_point: str = ""
        self._qubit_labels: dict[Qubit, int] = {}
        self._clbit_labels: dict[Clbit, int] = {}
        # Result ids actually written by a measurement. Output recording is driven by
        # this rather than by the register layout: a %Result only holds a value once
        # mz writes it, so recording an unmeasured bit reads uninitialised runtime state.
        self._measured_results: set[int] = set()
        self._initialize_runtime = initialize_runtime
        self._record_output = record_output
        self._emit_barrier_calls = emit_barrier_calls

    def visit_qiskit_module(self, module: QiskitModule) -> None:
        """Initialize the PyQIR builder for the given module."""
        logger.debug(
            "Visiting Qiskit module '%s' (%d qubits, %d clbits)",
            module.name,
            module.num_qubits,
            module.num_clbits,
        )
        self._qubit_labels.clear()
        self._clbit_labels.clear()

        if module.module is None:
            raise ValueError("QiskitModule must have a PyQIR module set before visiting.")
        self._module = module.module
        self._qiskit_module = module
        context = self._module.context
        entry = entry_point(self._module, module.name, module.num_qubits, module.num_clbits)

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
        """Raise if the visitor has not been initialized via visit_qiskit_module."""
        if self._module is None or self._builder is None:
            raise RuntimeError(
                "Visitor has not been initialized. Call visit_qiskit_module() first."
            )

    def finalize(self) -> None:
        """Finalize the QIR module by adding a return instruction."""
        self._check_initialized()
        self._builder.ret(None)

    def record_output(self) -> None:
        """Record output for classical registers using the circuit's register layout."""
        if not self._record_output:
            return
        self._check_initialized()

        if self._qiskit_module is None:
            raise RuntimeError(
                "No QiskitModule has been visited. Call visit_qiskit_module() first."
            )

        i8p = PointerType(IntType(self._module.context, 8))
        null_ptr = Constant.null(i8p)

        # Qiskit inverts the ordering of results within each register but keeps the
        # overall register ordering. Only results a measurement actually wrote are
        # recorded; an unwritten %Result would report uninitialised runtime state.
        logical_id_base = 0
        for size in self._qiskit_module.reg_sizes:
            measured_ids = [
                logical_id_base + i
                for i in range(size - 1, -1, -1)
                if logical_id_base + i in self._measured_results
            ]
            logical_id_base += size
            if not measured_ids:
                continue

            rt.array_record_output(
                self._builder,
                pyqir.const(IntType(self._module.context, 64), len(measured_ids)),
                null_ptr,
            )
            for result_id in measured_ids:
                result_ref = pyqir.result(self._module.context, result_id)
                rt.result_record_output(self._builder, result_ref, null_ptr)

    def visit_register(self, register: Union[QuantumRegister, ClassicalRegister]) -> None:
        """Visit a register and assign global bit indices to its bits."""
        logger.debug("Visiting register '%s'", register.name)
        if isinstance(register, QuantumRegister):
            self._qubit_labels.update(
                {bit: n + len(self._qubit_labels) for n, bit in enumerate(register)}
            )
            logger.debug("Added labels for qubits %s", list(register))
        elif isinstance(register, ClassicalRegister):
            self._clbit_labels.update(
                {bit: n + len(self._clbit_labels) for n, bit in enumerate(register)}
            )
        else:
            raise QiskitConversionError(f"Register of type {type(register)} not supported.")

    def _process_composite_instruction(
        self,
        instruction: Instruction,
        qargs: tuple[Qubit, ...],
        cargs: tuple[Clbit, ...],
    ) -> None:
        """Process a composite (decomposable) instruction by recursing into its definition."""
        subcircuit = instruction.definition
        logger.debug(
            "Processing composite instruction %s with qubits %s",
            instruction.name,
            qargs,
        )

        if len(qargs) != subcircuit.num_qubits:
            raise QiskitConversionError(
                f"Composite instruction {instruction.name} called with wrong number of qubits; "
                f"{subcircuit.num_qubits} expected, {len(qargs)} provided"
            )
        if len(cargs) != subcircuit.num_clbits:
            raise QiskitConversionError(
                f"Composite instruction {instruction.name} called with wrong number of clbits; "
                f"{subcircuit.num_clbits} expected, {len(cargs)} provided"
            )

        for circuit_instruction in subcircuit.data:
            inst = circuit_instruction.operation
            i_qargs = circuit_instruction.qubits
            i_cargs = circuit_instruction.clbits
            mapped_qbits = tuple(qargs[subcircuit.qubits.index(i)] for i in i_qargs)
            mapped_clbits = tuple(cargs[subcircuit.clbits.index(i)] for i in i_cargs)
            logger.debug(
                "Processing sub-instruction %s with mapped qubits %s",
                inst.name,
                mapped_qbits,
            )
            self.visit_instruction(inst, mapped_qbits, mapped_clbits)

    def visit_instruction(
        self,
        instruction: Instruction,
        qargs: tuple[Qubit, ...],
        cargs: tuple[Clbit, ...],
    ) -> None:
        """Visit and convert a single instruction to QIR."""
        self._check_initialized()
        qlabels = [self._qubit_labels[bit] for bit in qargs]
        clabels = [self._clbit_labels[bit] for bit in cargs]
        qubits = [pyqir.qubit(self._module.context, n) for n in qlabels]
        results = [pyqir.result(self._module.context, n) for n in clabels]

        logger.debug(
            "Visiting instruction '%s' (%s)",
            instruction.name,
            ", ".join(str(label) for label in qlabels + clabels),
        )

        op_name = instruction.name.lower()

        if op_name in NOOP_INSTRUCTIONS:
            return

        if op_name in MEASUREMENT_GATES:
            for qubit, result, label in zip(qubits, results, clabels, strict=True):
                self._measured_results.add(label)
                qis.mz(self._builder, qubit, result)
        elif op_name == "barrier":
            if self._emit_barrier_calls:
                qis.barrier(self._builder)
        elif op_name in PYQIR_ONE_QUBIT_OP_MAP:
            PYQIR_ONE_QUBIT_OP_MAP[op_name](self._builder, qubits[0])
        elif op_name in PYQIR_ONE_QUBIT_ROTATION_MAP:
            PYQIR_ONE_QUBIT_ROTATION_MAP[op_name](
                self._builder, float(instruction.params[0]), qubits[0]
            )
        elif op_name in PYQIR_TWO_QUBIT_OP_MAP:
            PYQIR_TWO_QUBIT_OP_MAP[op_name](self._builder, qubits[0], qubits[1])
        elif op_name in PYQIR_THREE_QUBIT_OP_MAP:
            PYQIR_THREE_QUBIT_OP_MAP[op_name](self._builder, *qubits)
        elif instruction.definition is not None:
            self._process_composite_instruction(instruction, qargs, cargs)
        else:
            raise QiskitConversionError(
                f"Gate '{instruction.name}' is not supported. "
                f"Please transpile using supported gates: {SUPPORTED_INSTRUCTIONS}"
            )

    def ir(self) -> str:
        """Return the QIR as a string."""
        self._check_initialized()
        return str(self._module)

    def bitcode(self) -> bytes:
        """Return the QIR as bitcode."""
        self._check_initialized()
        return self._module.bitcode


def generate_module_id(circuit: QuantumCircuit) -> str:
    """Return a module ID for the given circuit."""
    return circuit.name if circuit.name else "main"


class QiskitModule:
    """A Qiskit circuit prepared for QIR conversion.

    Attributes:
        circuit: The original Qiskit QuantumCircuit.
        name: The name of the module.
        module: The PyQIR Module being built.
        num_qubits: Number of qubits in the circuit.
        num_clbits: Number of classical bits in the circuit.
        reg_sizes: Sizes of each classical register in the circuit.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        module: Optional[Module],
        num_qubits: int,
        num_clbits: int,
        reg_sizes: list[int],
    ):
        self._circuit = circuit
        self._name = name
        self._module = module
        self._num_qubits = num_qubits
        self._num_clbits = num_clbits
        self._reg_sizes = reg_sizes

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @property
    def name(self) -> str:
        return self._name

    @property
    def module(self) -> Optional[Module]:
        return self._module

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_clbits(self) -> int:
        return self._num_clbits

    @property
    def reg_sizes(self) -> list[int]:
        return self._reg_sizes

    @classmethod
    def from_circuit(
        cls, circuit: QuantumCircuit, module: Optional[Module] = None
    ) -> "QiskitModule":
        """Create a QiskitModule from a Qiskit QuantumCircuit."""
        return cls(
            circuit=circuit,
            name=generate_module_id(circuit),
            module=module,
            num_qubits=circuit.num_qubits,
            num_clbits=circuit.num_clbits,
            reg_sizes=[len(creg) for creg in circuit.cregs],
        )

    def accept(self, visitor: BasicQiskitVisitor) -> None:
        """Visit all registers and instructions in this module."""
        visitor.visit_qiskit_module(self)
        for register in circuit_registers(self._circuit):
            visitor.visit_register(register)
        for instruction in self._circuit.data:
            visitor.visit_instruction(
                instruction.operation, instruction.qubits, instruction.clbits
            )
        visitor.record_output()
        visitor.finalize()


def circuit_registers(circuit: QuantumCircuit):
    """Yield the quantum then classical registers of a circuit."""
    yield from circuit.qregs
    yield from circuit.cregs


def convert(
    circuit: QuantumCircuit,
    name: Optional[str] = None,
    transpile: bool = False,
    **kwargs,
) -> str:
    """Convert a Qiskit QuantumCircuit to QIR LLVM IR text.

    Args:
        circuit: The Qiskit QuantumCircuit to convert.
        name: Identifier for the created QIR module. Auto-generated if not provided.
        transpile: If True, transpile the circuit to the supported basis gate set
            before conversion. This enables conversion of circuits containing gates
            not directly supported in QIR. Defaults to False.

    Keyword Args:
        initialize_runtime (bool): Whether to perform quantum runtime environment initialization,
            default `True`.
        record_output (bool): Whether to record output calls for registers, default `True`.
        emit_barrier_calls (bool): Whether to emit barrier calls in the QIR, default `False`.

    Returns:
        The QIR LLVM IR text (``str``) representation of the input Qiskit circuit.

    Raises:
        TypeError: If the input is not a valid Qiskit QuantumCircuit.
        ValueError: If the input circuit is empty.
        QiskitConversionError: If the conversion fails.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("Input quantum program must be of type qiskit.QuantumCircuit.")

    if len(circuit.data) == 0:
        raise ValueError("Input quantum circuit must consist of at least one operation.")

    if transpile:
        circuit = qiskit_transpile(circuit, basis_gates=QISKIT_BASIS_GATES)

    if name is None:
        name = generate_module_id(circuit)

    llvm_module = qir_module(Context(), name)
    module = QiskitModule.from_circuit(circuit, llvm_module)

    visitor = BasicQiskitVisitor(**kwargs)
    module.accept(visitor)

    error = llvm_module.verify()
    if error is not None:
        raise QiskitConversionError(error)

    return str(llvm_module)
