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
"""Pulser result adapter for the battery: input builder, format check and the
per-source facts (dict converter, wrap/read classmethods).

Pulser does not conflict with mimiqcircuits (unlike Perceval), so this leg runs
in the main pytest invocation (``-k not perceval``)."""

from qio.core import (
    QuantumProgramResult,
    QuantumProgramResultSerializationFormat as Serialization,
)

from reference_results import ReferenceResult
from result_testing import ResultSDK, _assert_counts

# Fixed execution facts of the ``from_final_bitstrings`` builder below: the
# atom id order matches the battery's bitstring convention (character k is
# qubit k) and the sequence duration is synthetic. They are asserted on the
# output object on purpose: a converter that reordered the atoms (or dropped
# the duration) would silently change the physical meaning of the counts while
# leaving the histogram itself intact.
_TOTAL_DURATION = 1000


class PulserSDK(ResultSDK):
    kind = "pulser"
    display = "pulser.backend.Results"
    fmt = Serialization.PULSER_RESULT_JSON_V1
    fmt_label = "pulser_result_json_v1"
    dict_keys = ("atom_order", "results")
    to_dict_module = "pulser_to_dict"
    from_dict = QuantumProgramResult.from_pulser_result_dict
    wrap = QuantumProgramResult.from_pulser_result
    read = QuantumProgramResult.to_pulser_result

    def build(self, reference: ReferenceResult):
        from pulser.backend import Results

        return Results.from_final_bitstrings(
            atom_order=tuple(f"q{k}" for k in range(reference.n_qubits)),
            total_duration=_TOTAL_DURATION,
            final_bitstrings=dict(reference.counts),
        )

    def check(self, result, expected: ReferenceResult = None) -> None:
        from pulser.backend import Results

        assert isinstance(
            result, Results
        ), f"expected pulser.backend.Results, got {type(result)}"
        if expected is not None:
            _assert_counts(dict(result.final_bitstrings), expected)
            assert tuple(result.atom_order) == tuple(
                f"q{k}" for k in range(expected.n_qubits)
            ), (result.atom_order, expected.n_qubits)
            assert result.total_duration == _TOTAL_DURATION, (
                result.total_duration,
                _TOTAL_DURATION,
            )


SDK = PulserSDK()
