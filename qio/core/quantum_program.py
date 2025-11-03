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

from enum import Enum
from typing import Dict, Union

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
    def from_json_dict(cls, data: Union[Dict, str]) -> "QuantumProgram":
        return QuantumProgram.schema().loads(data)

    def to_json_dict(self) -> Dict:
        return QuantumProgram.schema().dumps(self)

    @classmethod
    def from_json_str(cls, str: str) -> "QuantumProgram":
        data = json.loads(str)
        return cls.from_json_dict(data)

    def to_json_str(self) -> str:
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_qiskit_circuit(
        cls,
        qiskit_circuit: "qiskit.QuantumCircuit",
        dest_format: QuantumProgramSerializationFormat = QuantumProgramSerializationFormat.QASM_V3,
    ) -> "QuantumProgram":
        try:
            from qiskit import qasm3, qasm2
        except ImportError:
            raise Exception("Qiskit is not installed")

        match = {
            QuantumProgramSerializationFormat.QASM_V2: lambda c: qasm2.dumps(c),
            QuantumProgramSerializationFormat.QASM_V3: lambda c: qasm3.dumps(c),
        }

        try:
            return cls(
                serialization_format=dest_format,
                serialization=match[dest_format](qiskit_circuit),
            )
        except:
            raise

    def to_qiskit_circuit(self) -> "qiskit.QuantumCircuit":
        try:
            from qiskit import qasm3, qasm2, QuantumCircuit
        except ImportError:
            raise Exception("Qiskit is not installed")

        match = {
            QuantumProgramSerializationFormat.QASM_V1: lambda c: QuantumCircuit.from_qasm_str(
                c
            ),
            QuantumProgramSerializationFormat.QASM_V2: lambda c: qasm2.loads(c),
            QuantumProgramSerializationFormat.QASM_V3: lambda c: qasm3.loads(c),
        }

        try:
            return match[self.serialization_format](self.serialization)
        except:
            raise Exception(
                "unsupported serialization format:", self.serialization_format
            )

    @classmethod
    def from_cirq_circuit(
        cls,
        cirq_circuit: "cirq.AbstractCircuit",
        dest_format: QuantumProgramSerializationFormat = QuantumProgramSerializationFormat.QASM_V3,
    ) -> "QuantumProgram":
        try:
            import cirq
        except ImportError:
            raise Exception("Cirq is not installed")

        match = {
            QuantumProgramSerializationFormat.QASM_V2: lambda c: cirq.qasm(c),
            QuantumProgramSerializationFormat.QASM_V3: lambda c: cirq.qasm(
                c, args=cirq.QasmArgs(version="3.0")
            ),
            QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1: lambda c: cirq.to_json(
                c
            ),
        }

        try:
            return cls(
                serialization_format=dest_format,
                serialization=match[dest_format](cirq_circuit),
            )
        except:
            raise Exception("unsupported unserialization:", dest_format)

    def to_cirq_circuit(self) -> "cirq.Circuit":
        try:
            import cirq
        except ImportError:
            raise Exception("Cirq is not installed")

        if self.serialization_format in [
            QuantumProgramSerializationFormat.QASM_V1,
            QuantumProgramSerializationFormat.QASM_V2,
            QuantumProgramSerializationFormat.QASM_V3,
        ]:
            from cirq.contrib.qasm_import import circuit_from_qasm

            return circuit_from_qasm(self.serialization)

        if self.serialization_format in [
            QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1,
        ]:
            from cirq import read_json

            return read_json(json_text=self.serialization)

        raise Exception("unsupported serialization format:", self.serialization_format)
