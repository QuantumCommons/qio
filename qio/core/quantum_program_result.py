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
import collections
import io

from typing import Union, Sequence, Dict, Tuple, Callable, TypeVar, cast
from enum import Enum

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from qio.utils import dict_to_zlib, zlib_to_dict, CompressionFormat


class QuantumProgramResultSerializationFormat(Enum):
    UNKOWN_SERIALIZATION_FORMAT = 0
    CIRQ_RESULT_JSON_V1 = 1
    QISKIT_RESULT_JSON_V1 = 2


@dataclass_json
@dataclass
class QuantumProgramResult:
    compression_format: CompressionFormat
    serialization_format: QuantumProgramResultSerializationFormat
    serialization: str

    @classmethod
    def from_json_dict(cls, data: Union[Dict, str]) -> "QuantumProgramResult":
        return QuantumProgramResult.schema().loads(data)

    def to_json_dict(self) -> Dict:
        return QuantumProgramResult.schema().dumps(self)

    @classmethod
    def from_json_str(cls, str: str) -> "QuantumProgramResult":
        data = json.loads(str)
        return cls.from_json_dict(data)

    def to_json_str(self) -> str:
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_qiskit_result(
        cls,
        qiskit_result: "qiskit.result.Result",
        compression_format: CompressionFormat = CompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgramResult":
        try:
            from qiskit.result import Result
        except ImportError:
            raise Exception("Qiskit is not installed")

        return cls.from_qiskit_result_dict(qiskit_result.to_dict(), compression_format)

    @classmethod
    def from_qiskit_result_dict(
        cls,
        qiskit_result_dict: Union[str, Dict],
        compression_format: CompressionFormat = CompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgramResult":
        if isinstance(qiskit_result_dict, str):
            qiskit_result_dict = json.loads(
                qiskit_result_dict
            )  # Ensure serialization is not ill-formatted

        compression_format = (
            CompressionFormat.NONE
            if compression_format == CompressionFormat.UNKOWN_COMPRESSION_FORMAT
            else compression_format
        )

        apply_compression = {
            CompressionFormat.NONE: lambda d: json.dumps(d),
            CompressionFormat.ZLIB_BASE64_V1: lambda d: dict_to_zlib(d),
        }

        try:
            return cls(
                compression_format=compression_format,
                serialization_format=QuantumProgramResultSerializationFormat.QISKIT_RESULT_JSON_V1,
                serialization=apply_compression[compression_format](qiskit_result_dict),
            )
        except Exception as e:
            raise Exception("unsupport serialization:", compression_format, e)

    def to_qiskit_result(self, **kwargs) -> "qiskit.result.Result":
        try:
            from qiskit.result import Result
            from qiskit.result import Result
            from qiskit.result.models import ExperimentResult, ExperimentResultData

        except ImportError:
            raise Exception("Qiskit is not installed")

        apply_uncompression = {
            CompressionFormat.NONE: lambda d: json.loads(d),
            CompressionFormat.ZLIB_BASE64_V1: lambda d: zlib_to_dict(d),
        }

        result_dict = apply_uncompression[self.compression_format](self.serialization)

        if (
            self.serialization_format
            == QuantumProgramResultSerializationFormat.QISKIT_RESULT_JSON_V1
        ):
            data = {
                "results": result_dict["results"],
                "success": result_dict["success"],
                "header": result_dict.get("header"),
                "metadata": result_dict.get("metadata"),
            }

            if kwargs:
                data.update(kwargs)

            return Result.from_dict(data)
        elif (
            self.serialization_format
            == QuantumProgramResultSerializationFormat.CIRQ_RESULT_JSON_V1
        ):
            T = TypeVar("T")

            import numpy as np

            def __unpack_bits(
                packed_bits: str, dtype: str, shape: Sequence[int]
            ) -> np.ndarray:
                bits_bytes = bytes.fromhex(packed_bits)
                bits = np.unpackbits(np.frombuffer(bits_bytes, dtype=np.uint8))
                return bits[: np.prod(shape).item()].reshape(shape).astype(dtype)

            def __unpack_digits(
                packed_digits: str,
                binary: bool,
                dtype: Union[None, str],
                shape: Union[None, Sequence[int]],
            ):
                if binary:
                    dtype = cast(str, dtype)
                    shape = cast(Sequence[int], shape)
                    return __unpack_bits(packed_digits, dtype, shape)

                buffer = io.BytesIO()
                buffer.write(bytes.fromhex(packed_digits))
                buffer.seek(0)
                digits = np.load(buffer, allow_pickle=False)
                buffer.close()
                return digits

            def __key_to_str(key) -> str:
                if isinstance(key, str):
                    return key
                return ",".join(str(q) for q in key)

            def __big_endian_bits_to_int(bits) -> int:
                result = 0
                for e in bits:
                    result <<= 1
                    if e:
                        result |= 1
                return result

            def __tuple_of_big_endian_int(bit_groups) -> Tuple[int, ...]:
                return tuple(__big_endian_bits_to_int(bits) for bits in bit_groups)

            def __multi_measurement_histogram(
                keys,
                measurements,
                repetitions,
                fold_func: Callable[[Tuple], T] = cast(
                    Callable[[Tuple], T], __tuple_of_big_endian_int
                ),
            ):
                fixed_keys = tuple(__key_to_str(key) for key in keys)
                samples = zip(*(measurements[sub_key] for sub_key in fixed_keys))

                if len(fixed_keys) == 0:
                    samples = [()] * repetitions

                c: collections.Counter = collections.Counter()

                for sample in samples:
                    c[fold_func(sample)] += 1
                return c

            def __make_hex_from_result_array(result: Tuple):
                str_value = "".join(map(str, result))
                binary_value = bin(int(str_value))
                integer_value = int(binary_value, 2)

                return hex(integer_value)

            def __measurements(records: Dict):
                measurements = {}
                for key, data in records.items():
                    reps, instances, qubits = data.shape
                    if instances != 1:
                        raise ValueError(
                            "Cannot extract 2D measurements for repeated keys"
                        )
                    measurements[key] = data.reshape((reps, qubits))

                return measurements

            def __make_expresult_from_cirq_result(
                cirq_result_dict: Dict,
            ) -> ExperimentResult:
                raw_records = cirq_result_dict["records"]
                records = {
                    key: __unpack_digits(**val) for key, val in raw_records.items()
                }
                measurements = __measurements(records)
                repetitions = len(next(iter(records.values())))

                hist = dict(
                    __multi_measurement_histogram(
                        keys=measurements.keys(),
                        measurements=measurements,
                        repetitions=repetitions,
                    )
                )

                return ExperimentResult(
                    shots=repetitions,
                    success=True,
                    data=ExperimentResultData(
                        counts={
                            __make_hex_from_result_array(key): value
                            for key, value in hist.items()
                        },
                    ),
                )

            kwargs = kwargs or {}
            return Result(
                results=[__make_expresult_from_cirq_result(result_dict)], **kwargs
            )
        else:
            raise Exception(
                "unsupported serialization format:", self.serialization_format
            )

    @classmethod
    def from_cirq_result(
        cls,
        cirq_result: "cirq.Result",
        compression_format: CompressionFormat = CompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgramResult":
        try:
            import cirq
        except ImportError:
            raise Exception("Cirq is not installed")

        data = cirq_result._json_dict_()

        return cls.from_cirq_result_dict(data, compression_format)

    @classmethod
    def from_cirq_result_dict(
        cls,
        cirq_result_dict: Union[str, Dict],
        compression_format: CompressionFormat = CompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgramResult":
        if isinstance(cirq_result_dict, str):
            cirq_result_dict = json.loads(
                cirq_result_dict
            )  # Ensure serialization is not ill-formatted

        compression_format = (
            CompressionFormat.NONE
            if compression_format == CompressionFormat.UNKOWN_COMPRESSION_FORMAT
            else compression_format
        )

        apply_compression = {
            CompressionFormat.NONE: lambda d: json.dumps(d),
            CompressionFormat.ZLIB_BASE64_V1: lambda d: dict_to_zlib(d),
        }

        serialization = apply_compression[compression_format](cirq_result_dict)

        return cls(
            compression_format=compression_format,
            serialization_format=QuantumProgramResultSerializationFormat.CIRQ_RESULT_JSON_V1,
            serialization=serialization,
        )

    def to_cirq_result(self, **kwargs) -> "cirq.Result":
        try:
            from cirq import ResultDict
        except ImportError:
            raise Exception("Cirq is not installed")

        apply_uncompression = {
            CompressionFormat.NONE: lambda d: json.loads(d),
            CompressionFormat.ZLIB_BASE64_V1: lambda d: zlib_to_dict(d),
        }

        result_dict = apply_uncompression[self.compression_format](self.serialization)

        if (
            self.serialization_format
            != QuantumProgramResultSerializationFormat.CIRQ_RESULT_JSON_V1
        ):
            raise Exception("unsupported unserialization:", self.serialization_format)

        if kwargs:
            result_dict.update(kwargs)

        cirq_result = ResultDict._from_json_dict_(**result_dict)

        return cirq_result
