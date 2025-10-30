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
from enum import Enum

from dataclasses import dataclass
from dataclasses_json import dataclass_json


class QuantumProgramSerializationFormat(Enum):
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
    serialization_format: QuantumProgramSerializationFormat
    serialization: str

    @classmethod
    def from_qiskit_circuit(
        cls, qiskit_circuit: "qiskit.QuantumCircuit"
    ) -> "QuantumProgram":
        try:
            from qiskit import qasm3
        except ImportError:
            raise Exception("Qiskit is not installed")

        serialization = qasm3.dumps(qiskit_circuit)

        return cls(
            serialization_format=QuantumProgramSerializationFormat.QASM_V3,
            serialization=serialization,
            frozenset=True,
        )

    def to_qiskit_circuit(self) -> "qiskit.QuantumCircuit":
        try:
            from qiskit import qasm3, qasm2, QuantumCircuit
        except ImportError:
            raise Exception("Qiskit is not installed")

        match = {
            QuantumProgramSerializationFormat.QASM_V1: QuantumCircuit.from_qasm_str,
            QuantumProgramSerializationFormat.QASM_V2: qasm2.loads,
            QuantumProgramSerializationFormat.QASM_V3: qasm3.loads,
        }

        try:
            return match[self.serialization_format](self.serialization)
        except:
            raise Exception(
                "unsupported serialization format:", self.serialization_format
            )

    @classmethod
    def from_cirq_circuit(
        cls, cirq_circuit: "cirq.AbstractCircuit"
    ) -> "QuantumProgram":
        try:
            import cirq
        except ImportError:
            raise Exception("Cirq is not installed")

        serialization = cirq.to_json(cirq_circuit)

        return cls(
            serialization_format=QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1,
            serialization=serialization,
            frozenset=True,
        )

    def to_cirq_circuit(self) -> "cirq.Circuit":
        try:
            import cirq
        except ImportError:
            raise Exception("Cirq is not installed")

        if self.serialization_format in [
            QuantumProgramSerializationFormat.QASM_V1,
            QuantumProgramSerializationFormat.QASM_V2,
        ]:
            from cirq.contrib.qasm_import import circuit_from_qasm

            return circuit_from_qasm(self.serialization)

        if self.serialization_format in [
            QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1,
        ]:
            from cirq import read_json

            return read_json(json_text=self.serialization)

        raise Exception("unsupported serialization format:", self.serialization_format)
