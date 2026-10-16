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
"""CUDA-Q result adapter for the battery: input builder, format check, metadata
probes and the per-source facts. CUDA-Q has no dedicated ``*_to_dict`` step:
``from_cudaq_sample_result`` serializes internally (``to_dict_module`` is
None), so the write side is just the wrap classmethod."""

from typing import Any, Dict, List, Optional

from qio.core import (
    QuantumProgramResult,
    QuantumProgramResultSerializationFormat as Serialization,
)

from reference_results import ReferenceResult
from result_testing import LOSS_CUDAQ_REGISTER, Probe, ResultSDK, _assert_counts, _run_probes


def _cudaq_register_names(result: Any) -> Optional[set]:
    try:
        data = result.serialize()
    except Exception:
        return None
    names = set()
    stride = 0
    while stride < len(data):
        n_chars = data[stride]
        stride += 1
        name = "".join(chr(data[i]) for i in range(stride, stride + n_chars))
        stride += n_chars
        num_bitstrings = data[stride]
        stride += 1
        if num_bitstrings > 0:
            # Skip the synthetic, count-less '__global__' register CUDA-Q adds
            # on deserialize.
            names.add(name)
        stride += num_bitstrings * 3
    return names


CUDAQ_PROBES = (
    Probe(LOSS_CUDAQ_REGISTER, "register", _cudaq_register_names, lambda ref: {ref.register_name}),
)


def _cudaq_serialize(reference: ReferenceResult) -> List[int]:
    """CUDA-Q SampleResult wire format: register name, then per bitstring the
    triplet ``[value, bit_size, count]``."""
    data: List[int] = [len(reference.register_name)]
    data.extend(ord(ch) for ch in reference.register_name)
    data.append(len(reference.counts))
    for bitstring, count in reference.counts.items():
        data.append(int(bitstring, 2))
        data.append(len(bitstring))
        data.append(count)
    return data


def _parse_cudaq_serialize(data) -> Dict[str, int]:
    """Independent parser for the CUDA-Q SampleResult wire format (register
    name, then ``[value, bit_size, count]`` triplets per bitstring)."""
    stride = 0
    counts: Dict[str, int] = {}
    while stride < len(data):
        n_chars = data[stride]
        stride += 1
        stride += n_chars  # register name, already accounted for
        num_bitstrings = data[stride]
        stride += 1
        for _ in range(num_bitstrings):
            value = data[stride]
            size = data[stride + 1]
            count = data[stride + 2]
            stride += 3
            counts[format(value, f"0{size}b")] = int(count)
    return counts


class CudaqSDK(ResultSDK):
    kind = "cudaq"
    display = "cudaq.SampleResult"
    fmt = Serialization.CUDAQ_SAMPLE_RESULT_JSON_V1
    fmt_label = "cudaq_sample_result_json_v1"
    dict_keys = ()
    to_dict_module = None  # no dedicated dict step (wrap serializes internally)
    from_dict = None
    wrap = QuantumProgramResult.from_cudaq_sample_result
    read = QuantumProgramResult.to_cudaq_sample_result
    probes = CUDAQ_PROBES

    def build(self, reference: ReferenceResult):
        import cudaq

        sample_result = cudaq.SampleResult()
        sample_result.deserialize(_cudaq_serialize(reference))
        return sample_result

    def check(self, result, expected: ReferenceResult = None) -> None:
        import cudaq

        assert isinstance(result, cudaq.SampleResult), f"expected SampleResult, got {type(result)}"
        if expected is not None:
            _assert_counts(_parse_cudaq_serialize(result.serialize()), expected)
            _run_probes(self.probes, result, expected)


SDK = CudaqSDK()
