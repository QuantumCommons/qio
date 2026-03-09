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

from enum import IntEnum
from typing import Dict

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from qio.utils import CompressionFormat, zlib_to_str, str_to_zlib


class QuantumProgramSerializationFormat(IntEnum):
    UNKNOWN_SERIALIZATION_FORMAT = 0
    QASM_V1 = 1
    QASM_V2 = 2
    QASM_V3 = 3
    QIR_V1 = 4
    CIRQ_CIRCUIT_JSON_V1 = 5
    PERCEVAL_CIRCUIT_JSON_V1 = 6
    PULSER_SEQUENCE_JSON_V1 = 7


class QuantumProgramCompressionFormat(IntEnum):
    UNKNOWN_COMPRESSION_FORMAT = 0
    NONE = 1
    ZLIB_BASE64_V1 = 2


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
            from qio.utils.conversion.program.qiskit_to_qasm2 import convert as qiskit_to_qasm2_convert
            from qio.utils.conversion.program.qiskit_to_qasm3 import convert as qiskit_to_qasm3_convert
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
            QuantumProgramSerializationFormat.QASM_V2: lambda c: qiskit_to_qasm2_convert(c),
            QuantumProgramSerializationFormat.QASM_V3: lambda c: qiskit_to_qasm3_convert(c),
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
            QuantumProgramSerializationFormat.QASM_V1: lambda c: ,
            QuantumProgramSerializationFormat.QASM_V2: lambda c: ,
            QuantumProgramSerializationFormat.QASM_V3: lambda c: ,
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
            QuantumProgramSerializationFormat.QASM_V2: lambda c: ,
            QuantumProgramSerializationFormat.QASM_V3: lambda c: x,
            QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1: ,
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


            elif self.serialization_format == QuantumProgramSerializationFormat.QASM_V3:
                obj_qasm3 = serialization
            else:
                raise Exception(
                    "unsupported serialization format:", self.serialization_format
                )

            kernel = _openqasm3_to_cudaq(obj_qasm3

        except Exception as e:
            raise Exception(
                "unsupported unserialization:", self.serialization_format, e
            )

        return kernel
