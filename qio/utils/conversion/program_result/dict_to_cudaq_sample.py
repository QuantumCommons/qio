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
import cudaq

from typing import Dict, List, Tuple


def _long_to_bitstring(val: int, size: int) -> str:
    """
    Equivalent to the C++ longToBitString.
    Converts an integer to a binary string padded to the correct bit length.
    """
    # Format to binary, remove '0b' prefix, and pad with leading zeros
    return bin(val)[2:].zfill(size)


def _extract_name(data: List[int], stride: int) -> Tuple[str, int]:
    """Extracts a string (register name) from the data array."""
    n_chars = data[stride]
    stride += 1
    name = "".join(chr(data[i]) for i in range(stride, stride + n_chars))
    stride += n_chars

    return name, stride


def _deserialize_to_dict(
    data: List[int],
) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """
    Parses the integer array into a dictionary of registers and their counts.
    Matches the logic of ExecutionResult::deserialize and deserializeCounts.
    """
    stride = 0
    all_results = {}

    while stride < len(data):
        # 1. Extract Register Name
        name, stride = _extract_name(data, stride)

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

            bs = _long_to_bitstring(bitstring_as_long, size_of_bitstring)
            local_counts[bs] = count
            memory.extend([bs] * count)
            stride += 3

        all_results[name] = local_counts

    return all_results, memory


def _serialize_register(name: str, counts: Dict[str, int]) -> List[int]:
    """Builds the CUDA-Q SampleResult wire format for a single register."""
    data: List[int] = [len(name)]
    data.extend(ord(ch) for ch in name)
    data.append(len(counts))
    for bitstring, count in counts.items():
        data.append(int(bitstring, 2))
        data.append(len(bitstring))
        data.append(count)
    return data


_SUPPORTS_COUNTS_CTOR = None


def _supports_counts_constructor() -> bool:
    global _SUPPORTS_COUNTS_CTOR
    if _SUPPORTS_COUNTS_CTOR is None:
        try:
            cudaq.SampleResult({"0": 1})
            _SUPPORTS_COUNTS_CTOR = True
        except TypeError:
            _SUPPORTS_COUNTS_CTOR = False
    return _SUPPORTS_COUNTS_CTOR


def _ensure_counts(sample_result, parsed: Dict[str, Dict[str, int]]) -> None:
    for name, counts in parsed.items():
        if not counts:
            continue
        try:
            if sample_result.get_register_counts(name):
                continue
        except Exception:
            pass
        raise RuntimeError(f"CUDA-Q deserialize() lost counts for register '{name}'")


def convert(result_dict: dict) -> cudaq.SampleResult:
    parsed, _ = _deserialize_to_dict(result_dict)

    if _supports_counts_constructor():
        global_counts = parsed.get("__global__")
        if global_counts is not None:
            sample_result = cudaq.SampleResult(
                global_counts, {"shots": sum(global_counts.values())}
            )
            parsed = {k: v for k, v in parsed.items() if k != "__global__"}
        else:
            sample_result = cudaq.SampleResult()
        for name, counts in parsed.items():
            sample_result.deserialize(_serialize_register(name, counts))
        _ensure_counts(sample_result, parsed)
        return sample_result

    sample_result = cudaq.SampleResult()
    sample_result.deserialize(result_dict)

    _ensure_counts(sample_result, parsed)
    return sample_result
