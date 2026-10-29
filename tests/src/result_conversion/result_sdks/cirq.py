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
"""Cirq result adapter for the battery: input builder, format check and the
per-source facts (dict converter, wrap/read classmethods, inherent losses)."""

from typing import Dict

from qio.core import (
    QuantumProgramResult,
    QuantumProgramResultSerializationFormat as Serialization,
)

from reference_results import ReferenceResult
from result_testing import LOSS_CIRQ_PARAMS, ResultSDK, _assert_counts


def _cirq_counts(result) -> Dict[str, int]:
    import collections

    keys = sorted(result.measurements.keys())
    if not keys:
        return {}
    repetitions = int(result.measurements[keys[0]].shape[0])
    samples = []
    for r in range(repetitions):
        bits = []
        for key in keys:
            bits.extend(int(v) for v in result.measurements[key][r])
        samples.append("".join(map(str, bits)))
    return dict(collections.Counter(samples))


class CirqSDK(ResultSDK):
    kind = "cirq"
    display = "cirq.Result"
    fmt = Serialization.CIRQ_RESULT_JSON_V1
    fmt_label = "cirq_result_json_v1"
    dict_keys = ("records",)
    to_dict_module = "cirq_to_dict"
    from_dict = QuantumProgramResult.from_cirq_result_dict
    wrap = QuantumProgramResult.from_cirq_result
    read = QuantumProgramResult.to_cirq_result
    # cirq params cannot survive a plain-JSON round trip at all, so this loss
    # is inherent to *every* edge departing from cirq.
    static_source_losses = (LOSS_CIRQ_PARAMS,)
    source_loss = (
        LOSS_CIRQ_PARAMS,
        "cirq Result params cannot be stored: ParamResolver is "
        "not JSON-serializable, so params never cross the format",
    )

    def build(self, reference: ReferenceResult):
        import cirq
        import numpy as np

        measurements = {
            f"m{k}": np.zeros((reference.shots, 1), dtype=np.int8)
            for k in range(reference.n_qubits)
        }
        index = 0
        for bitstring, count in reference.counts.items():
            bits = [int(b) for b in bitstring]
            for _ in range(count):
                for k in range(reference.n_qubits):
                    measurements[f"m{k}"][index, 0] = bits[k]
                index += 1
        result = cirq.ResultDict(
            params=cirq.ParamResolver({}), measurements=measurements
        )
        # ``cirq.ParamResolver`` is not JSON-serializable (see LOSS_CIRQ_PARAMS);
        # mirror the SDK tests and drop it on the input object.
        result._params = None
        return result

    def check(self, result, expected: ReferenceResult = None) -> None:
        import cirq

        assert isinstance(
            result, cirq.Result
        ), f"expected cirq.Result, got {type(result)}"
        if expected is not None:
            _assert_counts(_cirq_counts(result), expected)


SDK = CirqSDK()
