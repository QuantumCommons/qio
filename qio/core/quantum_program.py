# Copyright 2025 Scaleway, Aqora, Quantum Commons
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
# limitations under the License.from enum import Enum
import json
import re

from enum import IntEnum
from typing import Dict
from collections import defaultdict

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from qio.utils import CompressionFormat, zlib_to_str, str_to_zlib, sanitize_qasm_str


class QuantumProgramSerializationFormat(IntEnum):
    UNKOWN_SERIALIZATION_FORMAT = 0
    QASM_V1 = 1
    QASM_V2 = 2
    QASM_V3 = 3
    QIR_V1 = 4
    CIRQ_CIRCUIT_JSON_V1 = 5
    PERCEVAL_CIRCUIT_JSON_V1 = 6
    PULSER_SEQUENCE_JSON_V1 = 7


@dataclass_json
@dataclass
class QuantumProgram:
    compression_format: CompressionFormat
    serialization_format: QuantumProgramSerializationFormat
    serialization: str

    @classmethod
    def from_json_dict(cls, data: Dict) -> "QuantumProgram":
        return QuantumProgram.from_dict(data)

    def to_json_dict(self) -> Dict:
        return self.to_dict()

    @classmethod
    def from_json_str(cls, data: str) -> "QuantumProgram":
        while isinstance(data, str):
            data = json.loads(data)
        return cls.from_json_dict(data)

    def to_json_str(self) -> str:
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_qiskit_circuit(
        cls,
        qiskit_circuit: "qiskit.QuantumCircuit",
        dest_format: QuantumProgramSerializationFormat = QuantumProgramSerializationFormat.QASM_V3,
        compression_format: CompressionFormat = CompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgram":
        try:
            from qiskit import qasm3, qasm2
        except ImportError:
            raise Exception("Qiskit is not installed")

        dest_format = (
            QuantumProgramSerializationFormat.QASM_V3
            if dest_format
            == QuantumProgramSerializationFormat.UNKOWN_SERIALIZATION_FORMAT
            else dest_format
        )
        compression_format = (
            CompressionFormat.NONE
            if compression_format == CompressionFormat.UNKOWN_COMPRESSION_FORMAT
            else compression_format
        )

        apply_serialization = {
            QuantumProgramSerializationFormat.QASM_V2: lambda c: qasm2.dumps(c),
            QuantumProgramSerializationFormat.QASM_V3: lambda c: sanitize_qasm_str(
                qasm3.dumps(c)
            ),
        }

        serialization = apply_serialization[dest_format](qiskit_circuit)

        apply_compression = {
            CompressionFormat.NONE: lambda s: s,
            CompressionFormat.ZLIB_BASE64_V1: lambda s: str_to_zlib(s),
        }

        compressed_serialization = apply_compression[compression_format](serialization)

        try:
            return cls(
                serialization_format=dest_format,
                compression_format=compression_format,
                serialization=compressed_serialization,
            )
        except Exception as e:
            raise Exception(
                "unsupport serialization:", dest_format, compression_format, e
            )

    def to_qiskit_circuit(self) -> "qiskit.QuantumCircuit":
        try:
            from qiskit import qasm3, qasm2, QuantumCircuit
        except ImportError:
            raise Exception("Qiskit is not installed")

        serialization = self.serialization

        if self.compression_format == CompressionFormat.ZLIB_BASE64_V1:
            serialization = zlib_to_str(serialization)

        apply_unserialization = {
            QuantumProgramSerializationFormat.QASM_V1: lambda c: QuantumCircuit.from_qasm_str(
                c
            ),
            QuantumProgramSerializationFormat.QASM_V2: lambda c: qasm2.loads(c),
            QuantumProgramSerializationFormat.QASM_V3: lambda c: qasm3.loads(c),
        }

        try:
            return apply_unserialization[self.serialization_format](serialization)
        except Exception as e:
            raise Exception(
                "unsupported unserialization:", self.serialization_format, e
            )

    @classmethod
    def from_cirq_circuit(
        cls,
        cirq_circuit: "cirq.AbstractCircuit",
        dest_format: QuantumProgramSerializationFormat = QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1,
        compression_format: CompressionFormat = CompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgram":
        try:
            import cirq
        except ImportError:
            raise Exception("Cirq is not installed")

        dest_format = (
            QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1
            if dest_format
            == QuantumProgramSerializationFormat.UNKOWN_SERIALIZATION_FORMAT
            else dest_format
        )
        compression_format = (
            CompressionFormat.NONE
            if compression_format == CompressionFormat.UNKOWN_COMPRESSION_FORMAT
            else compression_format
        )

        apply_serialization = {
            QuantumProgramSerializationFormat.QASM_V2: lambda c: c.to_qasm(
                version="2.0"
            ),
            QuantumProgramSerializationFormat.QASM_V3: lambda c: c.to_qasm(
                version="3.0"
            ),
            QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1: lambda c: cirq.to_json(
                c
            ),
        }

        serialization = apply_serialization[dest_format](cirq_circuit)

        apply_compression = {
            CompressionFormat.NONE: lambda s: s,
            CompressionFormat.ZLIB_BASE64_V1: lambda s: str_to_zlib(s),
        }

        compressed_serialization = apply_compression[compression_format](serialization)

        try:
            return cls(
                serialization_format=dest_format,
                compression_format=compression_format,
                serialization=compressed_serialization,
            )
        except Exception as e:
            raise Exception("unsupported serialization:", dest_format, e)

    def to_cirq_circuit(self) -> "cirq.Circuit":
        try:
            import cirq
        except ImportError:
            raise Exception("Cirq is not installed")

        serialization = self.serialization

        try:
            if self.compression_format == CompressionFormat.ZLIB_BASE64_V1:
                serialization = zlib_to_str(serialization)

            if self.serialization_format in [
                QuantumProgramSerializationFormat.QASM_V1,
                QuantumProgramSerializationFormat.QASM_V2,
                QuantumProgramSerializationFormat.QASM_V3,
            ]:
                from cirq.contrib.qasm_import import circuit_from_qasm
                import cirq

                def _restore_terminal_measurements(circuit: cirq.Circuit):
                    groups = defaultdict(dict)
                    ops_to_remove = []

                    pattern = re.compile(r"^m_(.+)_(?P<idx>\d+)$")

                    # 1. Identify terminal segmented measurements
                    for i, moment in enumerate(circuit):
                        for op in moment:
                            if isinstance(op.gate, cirq.MeasurementGate):
                                key = op.gate.key
                                match = pattern.match(key)

                                if match:
                                    qubit = op.qubits[0]
                                    # Check if this is the last moment in the circuit
                                    if (
                                        circuit.next_moment_operating_on([qubit], i + 1)
                                        is None
                                    ):
                                        original_name = match.group(1)
                                        index = int(match.group("idx"))
                                        groups[original_name][index] = qubit
                                        ops_to_remove.append((i, op))

                    if not groups:
                        return circuit

                    # 2. Cleanup circuit
                    # Copy to avoid modifying original in place
                    new_circuit = circuit.copy()
                    new_circuit.batch_remove(ops_to_remove)

                    # 3. Add merged measurements at the end
                    for name in sorted(groups.keys()):
                        indexed_qubits = groups[name]
                        sorted_indices = sorted(indexed_qubits.keys())
                        ordered_qubits = [indexed_qubits[idx] for idx in sorted_indices]
                        new_circuit.append(cirq.measure(*ordered_qubits, key=name))

                    return new_circuit

                return _restore_terminal_measurements(circuit_from_qasm(serialization))

            if self.serialization_format in [
                QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1,
            ]:
                from cirq import read_json

                return read_json(json_text=serialization)

        except Exception as e:
            raise Exception(
                "unsupported unserialization:", self.serialization_format, e
            )

    def to_cudaq_kernel(self) -> "cudaq.Kernel":
        try:
            import cudaq
            import pyqasm

            from openqasm3 import ast
            from cudaq import PyKernel, QuakeValue
        except ImportError as e:
            raise Exception(f"missing import: {e}")

        serialization = self.serialization

        try:
            if self.compression_format == CompressionFormat.ZLIB_BASE64_V1:
                serialization = zlib_to_str(serialization)

            if self.serialization_format == QuantumProgramSerializationFormat.QASM_V2:
                qasm_module = pyqasm.loads(serialization)
                obj_qasm3 = qasm_module.to_qasm3(as_str=True)

            elif self.serialization_format == QuantumProgramSerializationFormat.QASM_V3:
                obj_qasm3 = serialization
            else:
                raise Exception(
                    "unsupported serialization format:", self.serialization_format
                )

            def _make_gate_kernel(name: str, targs: tuple[type]) -> PyKernel:
                """Returns CUDA-Q kernel for pure standard gates (no modifiers - ctrl or adj)."""

                if name in [
                    "x",
                    "y",
                    "z",
                    "rx",
                    "ry",
                    "rz",
                    "h",
                    "s",
                    "t",
                    "sdg",
                    "tdg",
                    "u3",
                    "i",
                    "id",
                    "iden",
                ]:
                    size = 1
                elif name in ["swap"]:
                    size = 2
                else:
                    raise Exception(f"Unsupported gate: {name}")

                kernel, *qparams = cudaq.make_kernel(
                    *[cudaq.qubit for _ in range(size)], *targs
                )
                qrefs, qargs = qparams[:size], qparams[size:]

                if name in ["i", "id", "iden"]:
                    return kernel

                op = getattr(kernel, name)

                if len(targs) > 0:
                    op(*qargs, *qrefs)
                else:
                    op(*qrefs)

                return kernel

            def _openqasm3_to_cudaq(program: str | ast.Program) -> PyKernel:
                """Returns a CUDA-Q kernel representing the input OpenQASM program.

                Args:
                    qasm (str or ast.Program): OpenQASM program to convert to CUDA-Q kernel.

                Returns:
                    kernel: CUDA-Q kernel equivalent to input OpenQASM string.
                """

                module = pyqasm.loads(program)
                module.validate()
                module.unroll()

                program = module.unrolled_ast

                kernel: PyKernel = cudaq.make_kernel()
                ctx: dict[str, QuakeValue] = {}
                gate_kernels: dict[str, PyKernel] = {}

                def get_gate(name: str, targs: tuple[type]) -> PyKernel:
                    if name in gate_kernels:
                        return gate_kernels[name]

                    # Fix from qbraid: make sure we use float and not integer
                    targs = list(
                        map(lambda x: float if isinstance(x, type(int)) else x, targs)
                    )
                    gate_kernels[name] = _make_gate_kernel(name, targs)
                    return gate_kernels[name]

                def qubit_lookup(
                    qubit: ast.IndexedIdentifier | ast.Identifier,
                ) -> QuakeValue:
                    assert isinstance(
                        qubit, ast.IndexedIdentifier
                    ), f"all identifiers should've been indexed: {qubit}"

                    assert (
                        len(qubit.indices) == 1
                    ), f"multi-dim arrays are not supported: {qubit.indices}"

                    inds = qubit.indices[0]

                    assert (
                        len(inds) == 1
                    ), f"indices should've been a single integer: {inds}"
                    assert isinstance(ind := inds[0], ast.IntegerLiteral)

                    q = ctx[qubit.name.name][ind.value]
                    return q

                # pylint: disable-next=too-many-nested-blocks
                for statement in program.statements:
                    if isinstance(statement, ast.Include):
                        if statement.filename not in {"stdgates.inc", "qelib1.inc"}:
                            raise Exception(
                                f"Custom includes are unsupported: {statement}"
                            )
                    elif isinstance(statement, ast.QubitDeclaration):
                        ctx[statement.qubit.name] = kernel.qalloc(statement.size.value)
                    elif isinstance(statement, ast.ClassicalDeclaration):
                        if statement.init_expression and isinstance(
                            statement.init_expression, ast.QuantumMeasurement
                        ):
                            ctx[statement.identifier.name] = kernel.mz(
                                qubit_lookup(statement.init_expression.qubit)
                            )
                        else:
                            ctx[statement.identifier.name] = None
                    elif isinstance(statement, ast.QuantumMeasurementStatement):
                        val = kernel.mz(qubit_lookup(statement.measure.qubit))
                        if statement.target is not None:
                            assert isinstance(
                                statement.target, ast.IndexedIdentifier
                            ), f"identifiers should've been unrolled to indexed identifiers: {statement.target}"
                            ctx[statement.target.name.name] = val
                    elif isinstance(statement, ast.QuantumGate):
                        name, qubits = statement.name.name, statement.qubits

                        args = []
                        for arg in statement.arguments:
                            assert (
                                arg.value is not None
                            ), f"gate arguments should've been literals: {arg}"
                            args.append(arg.value)
                        targs = [type(a) for a in args]

                        qubit_refs = [qubit_lookup(q) for q in qubits]

                        # pyqasm unrolls multiple modifiers.
                        # ctrl isn't supported so multi-ctrl is not an issue at the moment.
                        assert len(statement.modifiers) <= 1

                        if len(statement.modifiers) == 1:
                            mod = statement.modifiers[0]
                            assert (
                                mod.modifier == ast.GateModifierName.ctrl
                            ), f"non-ctrl modifiers should've be unrolled: {mod}"

                            gate = get_gate(name, targs)
                            kernel.control(gate, qubit_refs[0], *qubit_refs[1:])
                        else:
                            if (namel := name.lower())[0] == "c" and namel[1:] in [
                                "x",
                                "y",
                                "z",
                                "rx",
                                "ry",
                                "rz",
                            ]:
                                # pyqasm doesn't unroll C{X,Y,Z} -> ctrl @ x. the below also handles this.
                                gate = get_gate(namel[1:], targs)
                                kernel.control(
                                    gate, qubit_refs[0], *qubit_refs[1:], *args
                                )
                            else:
                                gate = get_gate(name, targs)
                                # Fix from qbraid: make sure we use float and not integer
                                args = list(
                                    map(
                                        lambda x: float(x) if isinstance(x, int) else x,
                                        args,
                                    )
                                )
                                kernel.apply_call(gate, *qubit_refs, *args)

                    else:
                        raise Exception(f"Unsupported statement: {statement}")

                return kernel

            kernel = _openqasm3_to_cudaq(obj_qasm3)
        except Exception as e:
            raise Exception(
                "unsupported unserialization:", self.serialization_format, e
            )

        return kernel
