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

from qio.utils.compression import zlib_to_str, str_to_zlib


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
    compression_format: QuantumProgramCompressionFormat
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
        compression_format: QuantumProgramCompressionFormat = QuantumProgramCompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgram":
        from qio.utils.conversion.program.qiskit_to_qasm2 import (
            convert as qiskit_to_qasm2_convert,
        )
        from qio.utils.conversion.program.qiskit_to_qasm3 import (
            convert as qiskit_to_qasm3_convert,
        )

        dest_format = (
            QuantumProgramSerializationFormat.QASM_V3
            if dest_format
            == QuantumProgramSerializationFormat.UNKNOWN_SERIALIZATION_FORMAT
            else dest_format
        )
        compression_format = (
            QuantumProgramCompressionFormat.NONE
            if compression_format
            == QuantumProgramCompressionFormat.UNKNOWN_COMPRESSION_FORMAT
            else compression_format
        )

        try:
            apply_serialization = {
                QuantumProgramSerializationFormat.QASM_V2: qiskit_to_qasm2_convert,
                QuantumProgramSerializationFormat.QASM_V3: qiskit_to_qasm3_convert,
            }

            serialization = apply_serialization[dest_format](qiskit_circuit)

            if compression_format == QuantumProgramCompressionFormat.ZLIB_BASE64_V1:
                serialization = str_to_zlib(serialization)

            return cls(
                serialization_format=dest_format,
                compression_format=compression_format,
                serialization=serialization,
            )
        except Exception as e:
            raise Exception(
                "unsupport serialization:", dest_format, compression_format, e
            )

    def to_qiskit_circuit(self) -> "qiskit.QuantumCircuit":
        from qio.utils.conversion.program.qasm1_to_qiskit import (
            convert as qasm1_to_qiskit_convert,
        )
        from qio.utils.conversion.program.qasm2_to_qiskit import (
            convert as qasm2_to_qiskit_convert,
        )
        from qio.utils.conversion.program.qasm3_to_qiskit import (
            convert as qasm3_to_qiskit_convert,
        )

        serialization = self.serialization

        if self.compression_format == QuantumProgramCompressionFormat.ZLIB_BASE64_V1:
            serialization = zlib_to_str(serialization)

        try:
            apply_unserialization = {
                QuantumProgramSerializationFormat.QASM_V1: qasm1_to_qiskit_convert,
                QuantumProgramSerializationFormat.QASM_V2: qasm2_to_qiskit_convert,
                QuantumProgramSerializationFormat.QASM_V3: qasm3_to_qiskit_convert,
            }

            return apply_unserialization[self.serialization_format](serialization)
        except Exception as e:
            raise Exception(
                "unsupported unserialization:", self.serialization_format, e
            )
        
    def to_mimiq_circuit(self) -> "mimiqcircuits.Circuit":
        from qio.utils.conversion.program.qasm2_to_mimiq import (
            convert as qasm2_to_mimiq_convert,
        )

        serialization = self.serialization

        if self.compression_format == QuantumProgramCompressionFormat.ZLIB_BASE64_V1:
            serialization = zlib_to_str(serialization)

        try:
            apply_unserialization = {
                QuantumProgramSerializationFormat.QASM_V2: qasm2_to_mimiq_convert,
            }

            return apply_unserialization[self.serialization_format](serialization)
        except Exception as e:
            raise Exception("unsupported unserialization:", self.serialization_format, e)
    
    def to_qasm2_circuit(self) -> str:

        serialization = self.serialization

        if self.compression_format == QuantumProgramCompressionFormat.ZLIB_BASE64_V1:
            serialization = zlib_to_str(serialization)

        if self.serialization_format != QuantumProgramSerializationFormat.QASM_V2:
            raise Exception("unsupported unserialization:", self.serialization_format)

        return serialization
    
    @classmethod
    def from_cirq_circuit(
        cls,
        cirq_circuit: "cirq.AbstractCircuit",
        dest_format: QuantumProgramSerializationFormat = QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1,
        compression_format: QuantumProgramCompressionFormat = QuantumProgramCompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgram":
        from qio.utils.conversion.program.cirq_to_cirqjson import (
            convert as cirq_to_cirqjson_convert,
        )
        from qio.utils.conversion.program.cirq_to_qasm2 import (
            convert as cirq_to_qasm2_convert,
        )
        from qio.utils.conversion.program.cirq_to_qasm3 import (
            convert as cirq_to_qasm3_convert,
        )

        dest_format = (
            QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1
            if dest_format
            == QuantumProgramSerializationFormat.UNKNOWN_SERIALIZATION_FORMAT
            else dest_format
        )
        compression_format = (
            QuantumProgramCompressionFormat.NONE
            if compression_format
            == QuantumProgramCompressionFormat.UNKNOWN_COMPRESSION_FORMAT
            else compression_format
        )

        try:
            apply_serialization = {
                QuantumProgramSerializationFormat.QASM_V2: cirq_to_qasm2_convert,
                QuantumProgramSerializationFormat.QASM_V3: cirq_to_qasm3_convert,
                QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1: cirq_to_cirqjson_convert,
            }

            serialization = apply_serialization[dest_format](cirq_circuit)

            if compression_format == QuantumProgramCompressionFormat.ZLIB_BASE64_V1:
                serialization = str_to_zlib(serialization)

            return cls(
                serialization_format=dest_format,
                compression_format=compression_format,
                serialization=serialization,
            )
        except Exception as e:
            raise Exception("unsupported serialization:", dest_format, e)

    def to_cirq_circuit(self) -> "cirq.Circuit":
        from qio.utils.conversion.program.qasm1_to_cirq import (
            convert as qasm1_to_cirq_convert,
        )
        from qio.utils.conversion.program.qasm2_to_cirq import (
            convert as qasm2_to_cirq_convert,
        )
        from qio.utils.conversion.program.qasm3_to_cirq import (
            convert as qasm3_to_cirq_convert,
        )
        from qio.utils.conversion.program.cirqjson_to_cirq import (
            convert as cirqjson_to_cirq_convert,
        )

        serialization = self.serialization

        try:
            if (
                self.compression_format
                == QuantumProgramCompressionFormat.ZLIB_BASE64_V1
            ):
                serialization = zlib_to_str(serialization)

            apply_serialization = {
                QuantumProgramSerializationFormat.QASM_V1: qasm1_to_cirq_convert,
                QuantumProgramSerializationFormat.QASM_V2: qasm2_to_cirq_convert,
                QuantumProgramSerializationFormat.QASM_V3: qasm3_to_cirq_convert,
                QuantumProgramSerializationFormat.CIRQ_CIRCUIT_JSON_V1: cirqjson_to_cirq_convert,
            }

            return apply_serialization[self.serialization_format](serialization)

        except Exception as e:
            raise Exception(
                "unsupported unserialization:", self.serialization_format, e
            )

    @classmethod
    def from_cudaq_kernel(
        cls,
        cudaq_kernel: "cudaq.PyKernel",
        dest_format: QuantumProgramSerializationFormat = QuantumProgramSerializationFormat.QASM_V2,
        compression_format: QuantumProgramCompressionFormat = QuantumProgramCompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgram":
        from qio.utils.conversion.program.cudaq_to_qasm2 import (
            convert as cudaq_to_qasm2_convert,
        )

        dest_format = (
            QuantumProgramSerializationFormat.QASM_V2
            if dest_format
            == QuantumProgramSerializationFormat.UNKNOWN_SERIALIZATION_FORMAT
            else dest_format
        )
        compression_format = (
            QuantumProgramCompressionFormat.NONE
            if compression_format
            == QuantumProgramCompressionFormat.UNKNOWN_COMPRESSION_FORMAT
            else compression_format
        )

        try:
            apply_serialization = {
                QuantumProgramSerializationFormat.QASM_V2: cudaq_to_qasm2_convert,
            }

            serialization = apply_serialization[dest_format](cudaq_kernel)

            if compression_format == QuantumProgramCompressionFormat.ZLIB_BASE64_V1:
                serialization = str_to_zlib(serialization)

            return cls(
                serialization_format=dest_format,
                compression_format=compression_format,
                serialization=serialization,
            )
        except Exception as e:
            raise Exception("unsupported serialization:", dest_format, e)

    def to_cudaq_kernel(self) -> "cudaq.Kernel":
        from qio.utils.conversion.program.qasm2_to_cudaq import (
            convert as qasm2_to_cudaq_convert,
        )
        from qio.utils.conversion.program.qasm3_to_cudaq import (
            convert as qasm3_to_cudaq_convert,
        )

        serialization = self.serialization

        try:
            if (
                self.compression_format
                == QuantumProgramCompressionFormat.ZLIB_BASE64_V1
            ):
                serialization = zlib_to_str(serialization)

            apply_unserialization = {
                QuantumProgramSerializationFormat.QASM_V2: qasm2_to_cudaq_convert,
                QuantumProgramSerializationFormat.QASM_V3: qasm3_to_cudaq_convert,
            }

            return apply_unserialization[self.serialization_format](serialization)
        except Exception as e:
            raise Exception(
                "unsupported unserialization:", self.serialization_format, e
            )
