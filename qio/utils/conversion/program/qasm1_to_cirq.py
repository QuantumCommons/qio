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
import re

from collections import defaultdict
from cirq.contrib.qasm_import import circuit_from_qasm


def _restore_terminal_measurements(circuit: cirq.Circuit) -> cirq.Circuit:
    groups = defaultdict(dict)
    ops_to_remove = []

    pattern = re.compile(r"^m_(.+)_(?P<idx>\d+)$")

    # 1. Identify terminal segmented measurements
    for i, moment in enumerate(circuit):
        for op in moment:
            if isinstance(op.gate, cirq.MeasurementGate):
                key = op.gate.key
                match = pattern.match(key)

                if match:
                    qubit = op.qubits[0]
                    # Check if this is the last moment in the circuit
                    if circuit.next_moment_operating_on([qubit], i + 1) is None:
                        original_name = match.group(1)
                        index = int(match.group("idx"))
                        groups[original_name][index] = qubit
                        ops_to_remove.append((i, op))

    if not groups:
        return circuit

    # 2. Cleanup circuit
    # Copy to avoid modifying original in place
    new_circuit = circuit.copy()
    new_circuit.batch_remove(ops_to_remove)

    # 3. Add merged measurements at the end
    for name in sorted(groups.keys()):
        indexed_qubits = groups[name]
        sorted_indices = sorted(indexed_qubits.keys())
        ordered_qubits = [indexed_qubits[idx] for idx in sorted_indices]
        new_circuit.append(cirq.measure(*ordered_qubits, key=name))

    return new_circuit


def convert(circuit_str: str) -> cirq.Circuit:
    return _restore_terminal_measurements(circuit_from_qasm(circuit_str))
