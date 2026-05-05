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

import cirq
from typing import List
from bitarray import frozenbitarray
from qio.utils.conversion.program_result.dict_to_mimiq import convert as dict_to_mimiq_convert

def convert(cirq_results: List[cirq.Result], **kwargs) -> "mimiqcircuits.QCSResults":
    """
    Convert result from cirq.Result object into mimiqcircuits.QCSResults object.
    """
    
    result = cirq_results[0]
    
    counts = {}
    cstates = []

    if result.measurements:
        meas_keys = sorted(result.measurements.keys())
        repetitions = len(result.measurements[meas_keys[0]])
        
        raw_memory = []
        for i in range(repetitions):
            bitstring = ""
            for key in meas_keys:
                bits = result.measurements[key][i]
                bitstring += "".join(str(int(b)) for b in bits)
            raw_memory.append(bitstring)
            
        cstates = [frozenbitarray(bs) for bs in raw_memory]
        
        for bs in raw_memory:
            counts[bs] = counts.get(bs, 0) + 1

    pivot_dict = {
        "backend_name": "Cirq-Simulator",
        "backend_version": cirq.__version__,
        "results": [
            {
                "data": {
                    "counts": counts
                }
            }
        ]
    }
    
    if cstates:
        kwargs["cstates_override"] = cstates
        
    return dict_to_mimiq_convert(pivot_dict, **kwargs)