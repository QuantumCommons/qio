# Copyright 2025 Scaleway, Aqora, Quantum Commons
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
import os
import time
import numpy as np

import qio
import qiskit

from qiskit import QuantumCircuit


def _random_qiskit_circuit(size: int) -> QuantumCircuit:
    num_qubits = size
    num_gate = size

    qc = QuantumCircuit(num_qubits)

    for _ in range(num_gate):
        random_gate = np.random.choice(["unitary", "cx", "cy", "cz"])

        if random_gate == "cx" or random_gate == "cy" or random_gate == "cz":
            control_qubit = np.random.randint(0, num_qubits)
            target_qubit = np.random.randint(0, num_qubits)

            while target_qubit == control_qubit:
                target_qubit = np.random.randint(0, num_qubits)

            getattr(qc, random_gate)(control_qubit, target_qubit)
        else:
            for q in range(num_qubits):
                random_gate = np.random.choice(["h", "x", "y", "z"])
                getattr(qc, random_gate)(q)

    qc.measure_all()

    return qc


def test_nothing():
    pass


def test_nothing_twice():
    pass
