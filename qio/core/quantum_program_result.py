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
# limitations under the License.
import json

from typing import Union, Dict
from enum import IntEnum

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from qio.utils.compression import dict_to_zlib, zlib_to_dict


class QuantumProgramResultSerializationFormat(IntEnum):
    UNKNOWN_SERIALIZATION_FORMAT = 0
    CIRQ_RESULT_JSON_V1 = 1
    QISKIT_RESULT_JSON_V1 = 2
    CUDAQ_SAMPLE_RESULT_JSON_V1 = 3


class QuantumProgramResultCompressionFormat(IntEnum):
    UNKNOWN_COMPRESSION_FORMAT = 0
    NONE = 1
    ZLIB_BASE64_V1 = 2


@dataclass_json
@dataclass
class QuantumProgramResult:
    compression_format: QuantumProgramResultCompressionFormat
    serialization_format: QuantumProgramResultSerializationFormat
    serialization: str

    @classmethod
    def from_json_dict(cls, data: Dict) -> "QuantumProgramResult":
        return QuantumProgramResult.from_dict(data)

    def to_json_dict(self) -> Dict:
        return self.to_dict()

    @classmethod
    def from_json_str(cls, data: str) -> "QuantumProgramResult":
        while isinstance(data, str):
            data = json.loads(data)
        return cls.from_json_dict(data)

    def to_json_str(self) -> str:
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_cudaq_sample_result(
        cls,
        sample_result: "cudaq.SampleResult",
        compression_format: QuantumProgramResultCompressionFormat = QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgramResult":
        from qio.utils.conversion.program_result.cudaq_sample_to_dict import (
            convert as cudaq_sample_to_dict,
        )

        compression_format = (
            QuantumProgramResultCompressionFormat.NONE
            if compression_format
            == QuantumProgramResultCompressionFormat.UNKNOWN_COMPRESSION_FORMAT
            else compression_format
        )

        apply_compression = {
            QuantumProgramResultCompressionFormat.NONE: json.dumps,
            QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1: dict_to_zlib,
        }

        try:
            sample_serialization = cudaq_sample_to_dict(sample_result)

            return cls(
                compression_format=compression_format,
                serialization_format=QuantumProgramResultSerializationFormat.CUDAQ_SAMPLE_RESULT_JSON_V1,
                serialization=apply_compression[compression_format](
                    sample_serialization
                ),
            )
        except Exception as e:
            raise Exception("unsupport serialization:", compression_format, e)

    def to_cudaq_sample_result(self, **kwargs) -> "cudaq.SampleResult":
        from qio.utils.conversion.program_result.dict_to_cudaq_sample import (
            convert as dict_to_cudaq_sample,
        )

        apply_uncompression = {
            QuantumProgramResultCompressionFormat.NONE: json.loads,
            QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1: zlib_to_dict,
        }

        serialized_sample_result = apply_uncompression[self.compression_format](
            self.serialization
        )

        try:
            apply_unserialization = {
                QuantumProgramResultSerializationFormat.CUDAQ_SAMPLE_RESULT_JSON_V1: dict_to_cudaq_sample,
            }

            return apply_unserialization[self.serialization_format](
                serialized_sample_result
            )
        except Exception as e:
            raise Exception("unsupported unserialization:", self.serialization_format, e)

    @classmethod
    def from_qiskit_result(
        cls,
        qiskit_result: "qiskit.result.Result",
        compression_format: QuantumProgramResultCompressionFormat = QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgramResult":
        from qio.utils.conversion.program_result.qiskit_to_dict import (
            convert as qiskit_to_dict_convert,
        )

        qiskit_result_dict = qiskit_to_dict_convert(qiskit_result)

        return cls.from_qiskit_result_dict(qiskit_result_dict, compression_format)

    @classmethod
    def from_qiskit_result_dict(
        cls,
        qiskit_result_dict: Union[str, Dict],
        compression_format: QuantumProgramResultCompressionFormat = QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgramResult":
        if isinstance(qiskit_result_dict, str):
            qiskit_result_dict = json.loads(
                qiskit_result_dict
            )  # Ensure serialization is not ill-formatted

        compression_format = (
            QuantumProgramResultCompressionFormat.NONE
            if compression_format
            == QuantumProgramResultCompressionFormat.UNKNOWN_COMPRESSION_FORMAT
            else compression_format
        )

        apply_compression = {
            QuantumProgramResultCompressionFormat.NONE: json.dumps,
            QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1: dict_to_zlib,
        }

        try:
            return cls(
                compression_format=compression_format,
                serialization_format=QuantumProgramResultSerializationFormat.QISKIT_RESULT_JSON_V1,
                serialization=apply_compression[compression_format](qiskit_result_dict),
            )
        except Exception as e:
            raise Exception("unsupported serialization:", compression_format, e)

    def to_qiskit_result(self, **kwargs) -> "qiskit.result.Result":
        from qio.utils.conversion.program_result.dict_to_qiskit import (
            convert as dict_to_qiskit_convert,
        )
        from qio.utils.conversion.program_result.cirq_to_qiskit import (
            convert as cirq_to_qiskit_convert,
        )
        from qio.utils.conversion.program_result.cudaq_sample_to_qiskit import (
            convert as cudaq_sample_to_qiskit_convert,
        )

        apply_uncompression = {
            QuantumProgramResultCompressionFormat.NONE: json.loads,
            QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1: zlib_to_dict,
        }

        serialization = apply_uncompression[self.compression_format](self.serialization)

        try:
            apply_unserialization = {
                QuantumProgramResultSerializationFormat.QISKIT_RESULT_JSON_V1: dict_to_qiskit_convert,
                QuantumProgramResultSerializationFormat.CUDAQ_SAMPLE_RESULT_JSON_V1: cirq_to_qiskit_convert,
                QuantumProgramResultSerializationFormat.CIRQ_RESULT_JSON_V1: cudaq_sample_to_qiskit_convert,
            }

            return apply_unserialization[self.serialization_format](
                serialization, **kwargs
            )

        except Exception as e:
            raise Exception(
                "unsupported serialization format:", self.serialization_format
            )

    @classmethod
    def from_cirq_result(
        cls,
        cirq_result: "cirq.Result",
        compression_format: QuantumProgramResultCompressionFormat = QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgramResult":
        from qio.utils.conversion.program_result.cirq_to_dict import (
            convert as cirq_to_dict_convert,
        )

        cirq_result_dict = cirq_to_dict_convert(cirq_result)

        return cls.from_cirq_result_dict(cirq_result_dict, compression_format)

    @classmethod
    def from_cirq_result_dict(
        cls,
        cirq_result_dict: Union[str, Dict],
        compression_format: QuantumProgramResultCompressionFormat = QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgramResult":
        if isinstance(cirq_result_dict, str):
            cirq_result_dict = json.loads(
                cirq_result_dict
            )  # Ensure serialization is not ill-formatted

        compression_format = (
            QuantumProgramResultCompressionFormat.NONE
            if compression_format
            == QuantumProgramResultCompressionFormat.UNKNOWN_COMPRESSION_FORMAT
            else compression_format
        )

        apply_compression = {
            QuantumProgramResultCompressionFormat.NONE: json.dumps,
            QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1: dict_to_zlib,
        }

        try:
            serialization = apply_compression[compression_format](cirq_result_dict)

            return cls(
                compression_format=compression_format,
                serialization_format=QuantumProgramResultSerializationFormat.CIRQ_RESULT_JSON_V1,
                serialization=serialization,
            )
        except Exception as e:
            raise Exception("unsupported serialization:", e)

    def to_cirq_result(self, **kwargs) -> "cirq.Result":
        from qio.utils.conversion.program_result.dict_to_cirq import (
            convert as dict_to_cirq_convert,
        )
        from qio.utils.conversion.program_result.qiskit_to_cirq import (
            convert as qiskit_to_cirq_convert,
        )

        apply_uncompression = {
            QuantumProgramResultCompressionFormat.NONE: json.loads,
            QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1: zlib_to_dict,
        }

        result_dict = apply_uncompression[self.compression_format](self.serialization)

        try:

            apply_unserialization = {
                QuantumProgramResultSerializationFormat.CIRQ_RESULT_JSON_V1: dict_to_cirq_convert,
                QuantumProgramResultSerializationFormat.QISKIT_RESULT_JSON_V1: qiskit_to_cirq_convert,
            }

            return apply_unserialization[self.serialization_format](
                result_dict, **kwargs
            )

        except Exception as e:
            raise Exception(
                "unsupported serialization format:", self.serialization_format
            )
