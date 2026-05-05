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
import math
from qiskit.result import Result
from qio.utils.conversion.program_result.dict_to_mimiq import convert as dict_to_mimiq_convert
from bitarray import frozenbitarray

def convert(qiskit_result: Result, **kwargs) -> "mimiqcircuits.QCSResults":
    """Convert a qiskit.result.Result object into a mimiqcircuits.QCSResults object."""
    experiment_data = qiskit_result.data(0)
    counts = qiskit_result.get_counts(0) if "counts" in experiment_data else {}
    num_qubits = qiskit_result.results[0].header.n_qubits if hasattr(qiskit_result.results[0], 'header') else 0
    
    pivot_dict = {
        "backend_name": qiskit_result.backend_name,
        "backend_version": qiskit_result.backend_version,
        "results": [
            {
                "data": {
                    "counts": counts
                }
            }
        ]
    }

    if "statevector" in experiment_data:
        sv = experiment_data["statevector"]
        header = getattr(qiskit_result.results[0], 'header', None)
        num_qubits = getattr(header, 'n_qubits', int(math.log2(len(sv))))
        
        amplitudes = {}
        for i, amp in enumerate(sv):
            if abs(amp) > 1e-10:
                bitstring = format(i, f'0{num_qubits}b')
                amplitudes[frozenbitarray(bitstring)] = complex(amp)
        kwargs["amplitudes"] = amplitudes

    if "memory" in experiment_data:
        raw_memory = qiskit_result.get_memory(0)
        cstates = [frozenbitarray(bitstring) for bitstring in raw_memory]
        kwargs["cstates_override"] = cstates 

    return dict_to_mimiq_convert(pivot_dict, **kwargs)
