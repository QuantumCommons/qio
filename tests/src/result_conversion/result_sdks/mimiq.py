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
"""MIMIQ result adapter for the battery: input builder, format check, metadata
probes and the per-source facts (dict converter, wrap/read classmethods,
inherent losses)."""

from typing import Any, Dict, Optional, Sequence

from qio.core import (
    QuantumProgramResult,
    QuantumProgramResultSerializationFormat as Serialization,
)

from reference_results import ReferenceResult
from result_testing import (
    LOSS_MIMIQ_AMPLITUDES,
    LOSS_MIMIQ_METADATA,
    Probe,
    ResultSDK,
    _assert_counts,
    _run_probes,
)


def _mimiq_zstates(result: Any) -> list:
    zstates = getattr(result, "zstates", None)
    if not zstates:
        return []
    return [z.to01() if hasattr(z, "to01") else str(z) for z in zstates]


def _mimiq_amplitudes(result: Any) -> Dict[str, complex]:
    amplitudes = getattr(result, "amplitudes", None) or {}
    out = {}
    for key, value in amplitudes.items():
        bitstring = key.to01() if hasattr(key, "to01") else str(key)
        out[bitstring] = complex(value)
    return out


def _norm_seq(value: Optional[Sequence]) -> Sequence:
    return [] if value is None else list(value)


def _norm_dict(value: Optional[Dict]) -> Dict:
    return {} if value is None else dict(value)


MIMIQ_PROBES = (
    Probe(LOSS_MIMIQ_METADATA, "simulator", lambda r: getattr(r, "simulator", None), lambda ref: ref.backend_name),
    Probe(LOSS_MIMIQ_METADATA, "version", lambda r: getattr(r, "version", None), lambda ref: ref.backend_version),
    Probe(LOSS_MIMIQ_METADATA, "timings", lambda r: _norm_dict(getattr(r, "timings", None)), lambda ref: dict(ref.timings or {})),
    Probe(LOSS_MIMIQ_METADATA, "zstates", _mimiq_zstates, lambda ref: list(ref.zstates or [])),
    Probe(LOSS_MIMIQ_METADATA, "fidelities", lambda r: _norm_seq(getattr(r, "fidelities", None)), lambda ref: list(ref.fidelities or [])),
    Probe(LOSS_MIMIQ_METADATA, "avggateerrors", lambda r: _norm_seq(getattr(r, "avggateerrors", None)), lambda ref: list(ref.avggateerrors or [])),
    Probe(LOSS_MIMIQ_AMPLITUDES, "amplitudes", _mimiq_amplitudes, lambda ref: ref._expected_amplitudes()),
)


class MimiqSDK(ResultSDK):
    kind = "mimiq"
    display = "mimiqcircuits.QCSResults"
    fmt = Serialization.MIMIQ_QCSR_JSON_V1
    fmt_label = "mimiq_qcsr_json_v1"
    dict_keys = ("histogram", "simulator", "version")
    to_dict_module = "mimiq_to_dict"
    from_dict = QuantumProgramResult.from_mimiq_qcsr_dict
    wrap = QuantumProgramResult.from_mimiq_qcsr
    read = QuantumProgramResult.to_mimiq_qcsr
    # MIMIQ-only metadata + amplitudes cannot cross the plain-JSON
    # intermediate, so every edge *arriving* on a mimiq target declares them.
    target_losses = (LOSS_MIMIQ_METADATA, LOSS_MIMIQ_AMPLITUDES)
    probes = MIMIQ_PROBES

    def build(self, reference: ReferenceResult):
        from bitarray import frozenbitarray
        from mimiqcircuits import QCSResults

        cstates = []
        for bitstring, count in reference.counts.items():
            cstates.extend([frozenbitarray(bitstring)] * count)

        kwargs: dict = dict(
            simulator=reference.backend_name,
            version=reference.backend_version,
            cstates=cstates,
            fidelities=list(reference.fidelities or []),
            avggateerrors=list(reference.avggateerrors or []),
            timings=dict(reference.timings or {}),
        )
        if reference.zstates:
            kwargs["zstates"] = [frozenbitarray(z) for z in reference.zstates]

        return QCSResults(**kwargs)

    def check(self, result, expected: ReferenceResult = None) -> None:
        import mimiqcircuits

        assert isinstance(result, mimiqcircuits.QCSResults), f"expected QCSResults, got {type(result)}"
        if expected is not None:
            histogram = result.histogram()
            counts = {key.to01(): int(count) for key, count in histogram.items()}
            _assert_counts(counts, expected)
            _run_probes(self.probes, result, expected)


SDK = MimiqSDK()
