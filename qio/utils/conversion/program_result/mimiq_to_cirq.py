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

import numpy as np
import cirq


def convert(qcsr: dict, **kwargs) -> "cirq.Result":
    """
    Convert a mimiqcircuits.qcsresults object into a cirq.Result format.
    """
    histogram = qcsr.get("histogram", {})

    raw_measurements = []

    for bitstring, count in histogram.items():
        bits = [int(b) for b in bitstring]
        for _ in range(count):
            raw_measurements.append(bits)

    num_qubits = kwargs.get("num_qubits")
    if num_qubits is None:
        num_qubits = len(list(histogram.keys())[0]) if histogram else 0

    if not raw_measurements:
        records = np.empty((0, num_qubits), dtype=np.int8)
    else:
        records = np.array(raw_measurements, dtype=np.int8)

    meas_key = kwargs.get("meas_key", "result")
    measurements = {meas_key: records}

    return cirq.ResultDict(params=cirq.ParamResolver({}), measurements=measurements)
