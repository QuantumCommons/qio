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
"""Unit coverage of the QIR producers and the OpenQASM version converters.

- The QIR producers (``cirq_to_qir`` / ``qiskit_to_qir`` / ``qasm3_to_qir``)
  now return the QIR LLVM IR as a ``str`` (they used to return a
  ``pyqir.Module``); every test re-parses the text with PyQIR and verifies it,
  so a producer that returns a module (or invalid text) fails.
- ``QuantumProgram.from_*_circuit(..., QIR_V1)`` serializes the produced QIR
  text (``QIR_V1`` production wiring).
- ``QuantumProgram.to_cudaq_kernel()`` reads QIR back into a CUDA-Q kernel: the
  thin profile (constant-pointer operands, from ``cirq``/``qiskit``) and the
  CUDA-Q profile (``qubit_allocate_array`` + SSA loads + ``generalizedInvoke``
  controlled gates, from ``cudaq_to_qir``) are both reconstructed and executed.
- The self-hosted QASM2 <-> QASM3 text converters (``qasm2_to_qasm3`` /
  ``qasm3_to_qasm2``) and their ``QuantumProgram`` accessors
  (``to_qasm2_circuit`` / ``to_qasm3_circuit`` normalization) are validated
  with the independent Qiskit parsers.
"""

import cirq
from pyqir import Context, Module
from qiskit import QuantumCircuit, qasm2, qasm3

from qio.core import (
    QuantumProgram,
    QuantumProgramCompressionFormat,
    QuantumProgramSerializationFormat,
)


def _cirq_bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, key="m0"),
        cirq.measure(q1, key="m1"),
    )


def _qiskit_bell():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


_QASM3_BELL = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"""

# --- QIR producers return str -------------------------------------------


def _assert_valid_qir(source: str, num_qubits: int) -> None:
    assert isinstance(source, str), f"expected str (QIR LLVM IR), got {type(source)}"
    module = Module.from_ir(Context(), source, "module")
    error = module.verify()
    assert error is None, f"QIR module does not verify: {error}"


def test_cirq_to_qir_returns_str():
    from qio.utils.conversion.program.cirq_to_qir import convert

    _assert_valid_qir(convert(_cirq_bell()), 2)


def test_qiskit_to_qir_returns_str():
    from qio.utils.conversion.program.qiskit_to_qir import convert

    _assert_valid_qir(convert(_qiskit_bell()), 2)


def test_qasm3_to_qir_returns_str():
    from qio.utils.conversion.program.qasm3_to_qir import convert

    _assert_valid_qir(convert(_QASM3_BELL), 2)


# --- QIR_V1 production wiring through QuantumProgram ---------------------


def test_from_cirq_circuit_qir_v1():
    program = QuantumProgram.from_cirq_circuit(
        _cirq_bell(),
        dest_format=QuantumProgramSerializationFormat.QIR_V1,
        compression_format=QuantumProgramCompressionFormat.NONE,
    )
    assert (
        program.serialization_format == QuantumProgramSerializationFormat.QIR_V1
    )
    _assert_valid_qir(program.serialization, 2)


def test_from_qiskit_circuit_qir_v1_compressed():
    from qio.utils.compression import zlib_to_str

    program = QuantumProgram.from_qiskit_circuit(
        _qiskit_bell(),
        dest_format=QuantumProgramSerializationFormat.QIR_V1,
    )
    assert (
        program.serialization_format == QuantumProgramSerializationFormat.QIR_V1
    )
    _assert_valid_qir(zlib_to_str(program.serialization), 2)


# --- QIR_V1 read-back through to_cudaq_kernel ---------------------------


def _sample_kernel_counts(kernel) -> set:
    import cudaq

    return set(dict(cudaq.sample(kernel, shots_count=300).items()).keys())


def test_to_cudaq_kernel_reads_qir_v1_thin_profile():
    # ``cirq_to_qir`` emits the *thin* profile (constant-pointer operands); the
    # ``QIR_V1`` reader reconstructs and executes the kernel.
    program = QuantumProgram.from_cirq_circuit(
        _cirq_bell(),
        dest_format=QuantumProgramSerializationFormat.QIR_V1,
        compression_format=QuantumProgramCompressionFormat.NONE,
    )
    assert _sample_kernel_counts(program.to_cudaq_kernel()) == {"00", "11"}


def test_to_cudaq_kernel_reads_qir_v1_cudaq_profile():
    # ``cudaq_to_qir`` emits the CUDA-Q profile (qubit_allocate_array + SSA
    # loads + controlled gates via ``generalizedInvoke``); the ``QIR_V1`` reader
    # decodes it back into an equivalent kernel.
    import cudaq

    @cudaq.kernel
    def _bell_kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits[0])
        mz(qubits[1])

    program = QuantumProgram.from_cudaq_kernel(
        _bell_kernel,
        dest_format=QuantumProgramSerializationFormat.QIR_V1,
        compression_format=QuantumProgramCompressionFormat.NONE,
    )
    assert _sample_kernel_counts(program.to_cudaq_kernel()) == {"00", "11"}


# --- QASM2 <-> QASM3 normalization ---------------------------------------


def _gate_chain(qc) -> list:
    return [
        (instr.operation.name, tuple(qc.find_bit(q).index for q in instr.qubits))
        for instr in qc.data
        if instr.operation.name != "measure"
        and instr.operation.name not in ("barrier",)
    ]


_QASM2_BELL = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""


def test_qasm2_to_qasm3_to_qasm2_roundtrip():
    from qio.utils.conversion.program.qasm2_to_qasm3 import (
        convert as qasm2_to_qasm3,
    )
    from qio.utils.conversion.program.qasm3_to_qasm2 import (
        convert as qasm3_to_qasm2,
    )

    qasm3_text = qasm2_to_qasm3(_QASM2_BELL)
    qasm3_qc = qasm3.loads(qasm3_text)
    assert qasm3_qc.num_qubits == 2
    assert qasm3_qc.num_clbits == 2
    assert _gate_chain(qasm3_qc) == [("h", (0,)), ("cx", (0, 1))]

    back = qasm3_to_qasm2(qasm3_text)
    back_qc = qasm2.loads(back)
    assert back_qc.num_qubits == 2
    assert back_qc.num_clbits == 2
    assert _gate_chain(back_qc) == [("h", (0,)), ("cx", (0, 1))]


def test_to_qasm2_circuit_normalizes_qasm3():
    program = QuantumProgram.from_qiskit_circuit(
        _qiskit_bell(),
        dest_format=QuantumProgramSerializationFormat.QASM_V3,
        compression_format=QuantumProgramCompressionFormat.NONE,
    )
    qasm2_text = program.to_qasm2_circuit()
    qc = qasm2.loads(qasm2_text)
    assert qc.num_qubits == 2
    assert qc.num_clbits == 2
    assert _gate_chain(qc) == [("h", (0,)), ("cx", (0, 1))]


def test_to_qasm3_circuit_normalizes_qasm2():
    program = QuantumProgram.from_qiskit_circuit(
        _qiskit_bell(),
        dest_format=QuantumProgramSerializationFormat.QASM_V2,
        compression_format=QuantumProgramCompressionFormat.NONE,
    )
    qasm3_text = program.to_qasm3_circuit()
    qc = qasm3.loads(qasm3_text)
    assert qc.num_qubits == 2
    assert qc.num_clbits == 2
    assert _gate_chain(qc) == [("h", (0,)), ("cx", (0, 1))]
