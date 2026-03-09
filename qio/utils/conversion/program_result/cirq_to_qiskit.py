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
# limitations under the License.from enum import Enum
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

import numpy as np
import collections
import io

from typing import Union, Sequence, Dict, Tuple, Callable, TypeVar, cast

T = TypeVar("T")


def _unpack_bits(packed_bits: str, dtype: str, shape: Sequence[int]) -> np.ndarray:
    bits_bytes = bytes.fromhex(packed_bits)
    bits = np.unpackbits(np.frombuffer(bits_bytes, dtype=np.uint8))
    return bits[: np.prod(shape).item()].reshape(shape).astype(dtype)


def _unpack_digits(
    packed_digits: str,
    binary: bool,
    dtype: Union[None, str],
    shape: Union[None, Sequence[int]],
):
    if binary:
        dtype = cast(str, dtype)
        shape = cast(Sequence[int], shape)
        return _unpack_bits(packed_digits, dtype, shape)

    buffer = io.BytesIO()
    buffer.write(bytes.fromhex(packed_digits))
    buffer.seek(0)
    digits = np.load(buffer, allow_pickle=False)
    buffer.close()

    return digits


def _key_to_str(key) -> str:
    if isinstance(key, str):
        return key
    return ",".join(str(q) for q in key)


def _big_endian_bits_to_int(bits) -> int:
    result = 0
    for e in bits:
        result <<= 1
        if e:
            result |= 1
    return result


def _tuple_of_big_endian_int(bit_groups) -> Tuple[int, ...]:
    return tuple(_big_endian_bits_to_int(bits) for bits in bit_groups)


def _multi_measurement_histogram(
    keys,
    measurements,
    repetitions,
    fold_func: Callable[[Tuple], T] = cast(
        Callable[[Tuple], T], _tuple_of_big_endian_int
    ),
) -> Tuple[collections.Counter, list]:
    fixed_keys = tuple(_key_to_str(key) for key in keys)
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


def __make_bin_from_result_array(result: Tuple):
    str_value = "".join(map(str, result))
    return str_value


def __measurements(records: Dict):
    measurements = {}
    for key, data in records.items():
        reps, instances, qubits = data.shape
        if instances != 1:
            raise ValueError("Cannot extract 2D measurements for repeated keys")
        measurements[key] = data.reshape((reps, qubits))

    return measurements


def __make_expresult_from_cirq_result(
    cirq_result_dict: Dict,
) -> ExperimentResult:
    raw_records = cirq_result_dict["records"]
    records = {key: _unpack_digits(**val) for key, val in raw_records.items()}
    measurements = __measurements(records)
    repetitions = len(next(iter(records.values())))

    counter, memory = _multi_measurement_histogram(
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
                __make_bin_from_result_array(key): value
                for key, value in histogram.items()
            },
            memory=memory,
        ),
    )


def convert(circuit_str: str, **kwargs) -> Result:
    kwargs = kwargs or {}

    return Result(results=[__make_expresult_from_cirq_result(circuit_str)], **kwargs)
