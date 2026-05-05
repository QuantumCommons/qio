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

from mimiqcircuits import QCSResults
from bitarray import frozenbitarray

def convert(result_dict: dict, **kwargs) -> QCSResults:
    """
    Reconstruct a mimiqcircuits.QCSResults object from dict.
    """
    simulator = result_dict.get("backend_name", "unknown")
    version = result_dict.get("backend_version", "unknown")
    
    
    if "cstates_override" in kwargs:
        cstates = kwargs.pop("cstates_override")
    else:
        try:
            counts = result_dict["results"][0]["data"]["counts"]
        except (KeyError, IndexError):
            counts = {}

        cstates = []
        for bitstring_str, count in counts.items():
            fba = frozenbitarray(bitstring_str)
            cstates.extend([fba] * int(count))

    fidelities = kwargs.get("fidelities", None)
    avggateerrors = kwargs.get("avggateerrors", None)
    zstates = kwargs.get("zstates", None)
    amplitudes = kwargs.get("amplitudes", None)
    timings = kwargs.get("timings", None)

    return QCSResults(
        simulator=simulator,
        version=version,
        fidelities=fidelities,
        avggateerrors=avggateerrors,
        cstates=cstates,
        zstates=zstates,
        amplitudes=amplitudes,
        timings=timings
    )