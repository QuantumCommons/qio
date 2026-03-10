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
from qiskit import qasm3, QuantumCircuit

import re


def _sanitize_qasm_str(qasm_string: str) -> str:
    """
    Parses an OpenQASM 3 string and corrects errors where single-qubit
    gates are applied to multiple qubits in a single line.

    Args:
        qasm_string: A string containing the OpenQASM 3 circuit.

    Returns:
        A corrected OpenQASM 3 string.
    """

    # We only care about gates defined in "stdgates.inc" as single-qubit gates.
    # This list covers the most common ones (including parameterized gates).
    # We deliberately exclude multi-qubit gates (cx, swap, rzz, etc.)
    # and custom-defined gates.
    STD_SINGLE_QUBIT_GATES = {
        "id",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "p",
        "rx",
        "ry",
        "rz",
        "u",
        "u1",
        "u2",
        "u3",
    }

    # Regex to find gate applications.
    # Group 1: gate name (e.g., "ry", "id", "cx")
    # Group 2: optional parameters (e.g., "(5.802...)", "(2.976...)")
    # Group 3: qubit arguments (e.g., "q0[0]", "q0[2], q0[1]")
    gate_regex = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)(\([^)]*\))?\s+([^;]+);")

    sanitized_lines = []
    input_lines = qasm_string.split("\n")

    for line in input_lines:
        match = gate_regex.match(line)

        if match:
            gate_name = match.group(1)
            params_str = match.group(2) if match.group(2) else ""
            qubit_args_str = match.group(3)

            # Check if this is a known single-qubit gate
            if gate_name in STD_SINGLE_QUBIT_GATES:
                # Split the qubit arguments
                qubit_args = [q.strip() for q in qubit_args_str.split(",")]

                if len(qubit_args) > 1:
                    # This is the error we're looking for!
                    # Split the line into multiple single-qubit instructions.
                    indent = line.split(gate_name)[0]  # Preserve indentation
                    for qubit in qubit_args:
                        new_line = f"{indent}{gate_name}{params_str} {qubit};"
                        sanitized_lines.append(new_line)
                    continue  # Skip appending the original incorrect line

        # If no match, or if it's a valid line (e.g., multi-qubit gate
        # or correctly formed single-qubit gate), append it as-is.
        sanitized_lines.append(line)

    return "\n".join(sanitized_lines)


def convert(qiskit_circuit: QuantumCircuit) -> str:
    return _sanitize_qasm_str(qasm3.dumps(qiskit_circuit))
