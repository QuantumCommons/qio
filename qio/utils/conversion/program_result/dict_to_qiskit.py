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
from qiskit.result import Result


def _hex_to_bitstring(value, nb_bits):
    mask = (1 << nb_bits) - 1
    return format(value & mask, f"0{nb_bits}b")


def convert(result_dict: dict, **kwargs) -> Result:
    results = result_dict["results"]

    for experiment in results:
        exp_data = experiment.get("data", {})
        counts = exp_data.get("counts", None)
        n_qubits = experiment["header"]["n_qubits"]

        if counts:
            new_counts = {}

            for bitstring, count in counts.items():
                if bitstring.startswith("0x"):
                    nbits = _hex_to_bitstring(int(bitstring, 16), n_qubits)
                else:
                    nbits = bitstring

                new_counts[nbits] = count

            exp_data["counts"] = new_counts

    data = {
        "results": results,
        "success": result_dict["success"],
        "header": result_dict.get("header"),
        "metadata": result_dict.get("metadata"),
    }

    if kwargs:
        data.update(kwargs)

    return Result.from_dict(data)
