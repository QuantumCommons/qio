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

from typing import Union, Sequence, Dict, Tuple, Callable, TypeVar, cast, List
from enum import Enum

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from qio.utils import dict_to_zlib, zlib_to_dict, CompressionFormat


class QuantumProgramResultSerializationFormat(Enum):
    UNKOWN_SERIALIZATION_FORMAT = 0
    CIRQ_RESULT_JSON_V1 = 1
    QISKIT_RESULT_JSON_V1 = 2
    CUDAQ_SAMPLE_RESULT_JSON_V1 = 3


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
    def from_cudaq_sample_result(
        cls,
        sample_result: "cudaq.SampleResult",
        compression_format: CompressionFormat = CompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumProgramResult":
        compression_format = (
            CompressionFormat.NONE
            if compression_format == CompressionFormat.UNKOWN_COMPRESSION_FORMAT
            else compression_format
        )

        apply_compression = {
            CompressionFormat.NONE: lambda d: json.dumps(d),
            CompressionFormat.ZLIB_BASE64_V1: lambda d: dict_to_zlib(d),
        }

        sample_serialization = sample_result.serialize()

        try:
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
        try:
            import cudaq
        except ImportError:
            raise Exception("CUDA-Q is not installed")

        apply_uncompression = {
            CompressionFormat.NONE: lambda d: json.loads(d),
            CompressionFormat.ZLIB_BASE64_V1: lambda d: zlib_to_dict(d),
        }

        serialized_sample_result = apply_uncompression[self.compression_format](
            self.serialization
        )

        if (
            self.serialization_format
            == QuantumProgramResultSerializationFormat.CUDAQ_SAMPLE_RESULT_JSON_V1
        ):
            sample_result = cudaq.SampleResult()
            sample_result.deserialize(serialized_sample_result)

            return sample_result
        else:
            raise Exception(
                "unsupported serialization format:", self.serialization_format
            )

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
            from qiskit.result.models import ExperimentResult, ExperimentResultData

        except ImportError:
            raise Exception("Qiskit is not installed")

        apply_uncompression = {
            CompressionFormat.NONE: lambda d: json.loads(d),
            CompressionFormat.ZLIB_BASE64_V1: lambda d: zlib_to_dict(d),
        }

        serialization = apply_uncompression[self.compression_format](self.serialization)

        if (
            self.serialization_format
            == QuantumProgramResultSerializationFormat.QISKIT_RESULT_JSON_V1
        ):
            data = {
                "results": serialization["results"],
                "success": serialization["success"],
                "header": serialization.get("header"),
                "metadata": serialization.get("metadata"),
            }

            if kwargs:
                data.update(kwargs)

            return Result.from_dict(data)
        elif (
            self.serialization_format
            == QuantumProgramResultSerializationFormat.CUDAQ_SAMPLE_RESULT_JSON_V1
        ):

            def __long_to_bitstring(val: int, size: int) -> str:
                """
                Equivalent to the C++ longToBitString.
                Converts an integer to a binary string padded to the correct bit length.
                """
                # Format to binary, remove '0b' prefix, and pad with leading zeros
                return bin(val)[2:].zfill(size)

            def __extract_name(data: List[int], stride: int) -> Tuple[str, int]:
                """Extracts a string (register name) from the data array."""
                n_chars = data[stride]
                stride += 1
                name = "".join(chr(data[i]) for i in range(stride, stride + n_chars))
                stride += n_chars

                return name, stride

            def __deserialize_to_dict(data: List[int]) -> Dict[str, Dict[str, int]]:
                """
                Parses the integer array into a dictionary of registers and their counts.
                Matches the logic of ExecutionResult::deserialize and deserializeCounts.
                """
                stride = 0
                all_results = {}

                while stride < len(data):
                    # 1. Extract Register Name
                    name, stride = __extract_name(data, stride)

                    # 2. Extract Counts (deserializeCounts logic)
                    num_bitstrings = data[stride]
                    stride += 1

                    local_counts = {}
                    memory = []
                    # Each entry is a triplet: [packed_value, bit_size, count]
                    for _ in range(num_bitstrings):
                        bitstring_as_long = data[stride]
                        size_of_bitstring = data[stride + 1]
                        count = data[stride + 2]

                        bs = __long_to_bitstring(bitstring_as_long, size_of_bitstring)
                        local_counts[bs] = count
                        memory.extend([bs] * count)
                        stride += 3

                    all_results[name] = local_counts

                return all_results, memory

            parsed_data, memory = __deserialize_to_dict(serialization)
            experiment_results = []

            for reg_name, counts in parsed_data.items():
                shots = sum(counts.values())

                # Encapsulate data in Qiskit's expected format
                data_payload = ExperimentResultData(counts=counts, memory=memory)

                exp_res = ExperimentResult(
                    shots=shots,
                    success=True,
                    data=data_payload,
                    header={"name": reg_name, "memory": True},
                    status="Done",
                )
                experiment_results.append(exp_res)

            return Result(results=experiment_results, **kwargs)
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
            ) -> Tuple[collections.Counter, list]:
                fixed_keys = tuple(__key_to_str(key) for key in keys)
                samples = zip(*(measurements[sub_key] for sub_key in fixed_keys))

                if len(fixed_keys) == 0:
                    samples = [()] * repetitions

                counter = collections.Counter()
                memory = []

                for sample in samples:
                    memory.append("".join(str(a) for a in np.concatenate(sample)))
                    counter[fold_func(sample)] += 1

                return (counter, memory)

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

                counter, memory = __multi_measurement_histogram(
                    keys=measurements.keys(),
                    measurements=measurements,
                    repetitions=repetitions,
                )

                histogram = dict(counter)

                return ExperimentResult(
                    shots=repetitions,
                    success=True,
                    data=ExperimentResultData(
                        counts={
                            __make_hex_from_result_array(key): value
                            for key, value in histogram.items()
                        },
                        memory=memory,
                    ),
                )

            kwargs = kwargs or {}

            return Result(
                results=[__make_expresult_from_cirq_result(serialization)], **kwargs
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

        if kwargs:
            result_dict.update(kwargs)

        if (
            self.serialization_format
            == QuantumProgramResultSerializationFormat.CIRQ_RESULT_JSON_V1
        ):
            cirq_result = ResultDict._from_json_dict_(**result_dict)
        elif (
            self.serialization_format
            == QuantumProgramResultSerializationFormat.QISKIT_RESULT_JSON_V1
        ):
            from cirq import ResultDict
            import numpy as np

            def _extract_measurement_key(experiment_result: Dict) -> str:
                header = experiment_result.get("header", {})
                creg_sizes = header.get("creg_sizes", [])

                if creg_sizes and len(creg_sizes) > 0:
                    m_name = creg_sizes[0][0]
                    return m_name.replace("m_", "")

                return "m"

            def _to_cirq_result(data: Dict) -> ResultDict:
                experiment = data.get("results", [{}])[0]
                m_key = _extract_measurement_key(experiment)

                counts = experiment.get("data", {}).get("counts", {})
                header = experiment.get("header", {})
                qreg_sizes = header.get("qreg_sizes", [])

                if qreg_sizes and len(qreg_sizes) > 0:
                    num_qubits = qreg_sizes[0][1]
                else:
                    memory = experiment.get("memory", {})

                    if memory and len(memory) > 0:
                        num_qubits = len(memory[0])

                all_shots = []

                for bitstring_hex, count in counts.items():
                    if bitstring_hex.startswith("0x"):
                        integer_val = int(bitstring_hex, 16)
                        bitstring = format(integer_val, f"0{num_qubits}b")
                    else:
                        bitstring = bitstring_hex

                    bits = [int(b) for b in bitstring]
                    for _ in range(count):
                        all_shots.append(bits)

                measurements = {m_key: np.array(all_shots)}

                return ResultDict(
                    params=data.pop("params", None), measurements=measurements
                )

            cirq_result = _to_cirq_result(result_dict)

        return cirq_result
