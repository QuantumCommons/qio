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

from qio.utils.conversion.program_result.dict_to_mimiq import (
    convert as dict_to_mimiq_convert,
)
from bitarray import frozenbitarray


def convert(result_dict: dict, **kwargs) -> "mimiqcircuits.QCSResults":
    """
    Convert a qiskit.result.Result object into a mimiqcircuits.QCSResults object.
    """
    experiment = result_dict.get("results", [{}])[0]
    raw_counts = experiment.get("data", {}).get("counts", {})

    header = experiment.get("header", {})
    qreg_sizes = header.get("qreg_sizes", [])
    num_qubits = header.get("n_qubits", None)

    if not num_qubits and qreg_sizes and len(qreg_sizes) > 0:
        num_qubits = qreg_sizes[0][1]
    else:
        memory = experiment.get("data", {}).get("memory", {})

        if memory and len(memory) > 0:
            num_qubits = len(memory[0])

    counts = {}
    for bitstring_hex, count in raw_counts.items():
        if bitstring_hex.startswith("0x"):
            integer_val = int(bitstring_hex, 16)
            bitstring = format(integer_val, f"0{num_qubits}b")
        else:
            bitstring = bitstring_hex

        counts[bitstring] = count

    pivot_dict = {
        "backend_name": result_dict.get("backend_name", "unknown"),
        "backend_version": result_dict.get("backend_version", "unknown"),
        "results": [{"data": {"counts": counts}}],
    }

    if "statevector" in experiment:
        sv = experiment["statevector"]

        amplitudes = {}
        for i, amp in enumerate(sv):
            if abs(amp) > 1e-10:
                bitstring = format(i, f"0{num_qubits}b")
                amplitudes[frozenbitarray(bitstring)] = complex(amp)
        kwargs["amplitudes"] = amplitudes

    if "memory" in experiment.get("data", {}):
        raw_memory = experiment.get("data", {}).get("memory", [])

        cstates = []
        for bitstring_hex in raw_memory:
            if bitstring_hex.startswith("0x"):
                integer_val = int(bitstring_hex, 16)
                bitstring = format(integer_val, f"0{num_qubits}b")
            else:
                bitstring = bitstring_hex

            cstates.append(frozenbitarray(bitstring))

        kwargs["cstates_override"] = cstates

    return dict_to_mimiq_convert(pivot_dict, **kwargs)
