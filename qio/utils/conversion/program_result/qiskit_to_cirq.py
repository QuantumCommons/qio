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
# limitations under the License.from enum import Enum
import numpy as np

from cirq import ResultDict


def _extract_measurement_key(experiment_result: dict) -> str:
    header = experiment_result.get("header", {})
    creg_sizes = header.get("creg_sizes", [])

    if creg_sizes and len(creg_sizes) > 0:
        m_name = creg_sizes[0][0]
        return m_name.replace("m_", "")

    return "m"


def convert(result_dict: dict) -> ResultDict:
    experiment = result_dict.get("results", [{}])[0]
    m_key = _extract_measurement_key(experiment)

    counts = experiment.get("data", {}).get("counts", {})
    header = experiment.get("header", {})
    qreg_sizes = header.get("qreg_sizes", [])
    num_qubits = header.get("n_qubits", None)

    if not num_qubits and qreg_sizes and len(qreg_sizes) > 0:
        num_qubits = qreg_sizes[0][1]
    else:
        memory = experiment.get("memory", {})

        if memory and len(memory) > 0:
            num_qubits = len(memory[0])

    all_shots = []

    for bitstring_hex, count in counts.items():
        if bitstring_hex.startswith("0x"):
            integer_val = int(bitstring_hex, 16)
            bitstring = format(integer_val, f"0{num_qubits}b")
        else:
            bitstring = bitstring_hex

        bits = [int(b) for b in bitstring]
        for _ in range(count):
            all_shots.append(bits)

    measurements = {m_key: np.array(all_shots)}

    return ResultDict(params=result_dict.pop("params", None), measurements=measurements)
