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
"""Perceval result adapter for the battery: input builder, format check and the
per-source facts.

This leg runs in the perceval-only pytest invocation (``-k perceval``): mimiq's
and Perceval's ``circuit.proto`` conflict in protobuf's global descriptor pool,
so the two must never be imported in the same process. The adapter must keep all
SDK imports lazy (they happen in ``build``/``check`` only)."""

from typing import Dict

from qio.core import (
    QuantumProgramResult,
    QuantumProgramResultSerializationFormat as Serialization,
)

from reference_results import ReferenceResult
from result_testing import ResultSDK, _assert_counts


def _perceval_counts(result) -> Dict[str, int]:
    # BSSamples of Fock states "|b0,b1,...>" -> bitstring "b0b1..." (mode k is
    # qubit k, matching the battery's bitstring convention).
    counts: Dict[str, int] = {}
    for sample in result:
        bitstring = str(sample).replace("|", "").replace(">", "").replace(",", "")
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


class PercevalSDK(ResultSDK):
    kind = "perceval"
    display = "perceval.BSSamples"
    fmt = Serialization.PERCEVAL_RESULT_JSON_V1
    fmt_label = "perceval_result_json_v1"
    dict_keys = ("perceval_result",)
    to_dict_module = "perceval_to_dict"
    from_dict = QuantumProgramResult.from_perceval_result_dict
    wrap = QuantumProgramResult.from_perceval_result
    read = QuantumProgramResult.to_perceval_result

    def build(self, reference: ReferenceResult):
        import numpy as np
        import perceval as pcvl

        samples = []
        for bitstring, count in reference.counts.items():
            state = pcvl.BasicState(np.array([int(b) for b in bitstring]))
            samples.extend([state] * count)
        return pcvl.BSSamples(samples)

    def check(self, result, expected: ReferenceResult = None) -> None:
        import perceval as pcvl

        assert isinstance(result, pcvl.BSSamples), f"expected BSSamples, got {type(result)}"
        if expected is not None:
            _assert_counts(_perceval_counts(result), expected)


SDK = PercevalSDK()
