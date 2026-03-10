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
from qiskit.result import Result


def _hex_to_bitstring(value, nb_qbits):
    """
    Converts a Qiskit hexadecimal/integer result into a formatted bitstring.
    Automatically detects and removes empty classical registers caused by measure_all().
    """
    # 1. Handle the input type (allows both '0xc' and 12)
    int_val = int(value, 16) if isinstance(value, str) else int(value)

    # 2. Check if the value exceeds the max size for the given number of qubits.
    # (1 << n_qbits) is a bitwise way to calculate 2^n_qbits.
    if int_val >= (1 << nb_qbits):
        # The result is padded. We shift right by n_qbits to drop the empty register.
        int_val = int_val >> nb_qbits

    # 3. Convert to binary, strip the '0b' prefix, and pad with leading zeros
    return bin(int_val)[2:].zfill(nb_qbits)


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
