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
import numpy as np

import cirq

from cirq.circuits import Circuit
from qsimcirq import QSimSimulator, QSimOptions

from qio.core import (
    QuantumComputationModel,
    QuantumComputationParameters,
    QuantumProgramResult,
    QuantumProgram,
    BackendData,
    ClientData,
)

from qio.utils import CompressionFormat


def _random_cirq_circuit(size: int) -> Circuit:
    num_qubits = size
    num_gate = size

    # Create the qubits
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]

    # Initialize the circuit
    circuit = Circuit()

    for _ in range(num_gate):
        random_gate = np.random.choice(["unitary", "cx", "cy", "cz"])

        if random_gate in ["cx", "cy", "cz"]:
            control_qubit = np.random.randint(0, num_qubits)
            target_qubit = np.random.randint(0, num_qubits)

            while target_qubit == control_qubit:
                target_qubit = np.random.randint(0, num_qubits)

            if random_gate == "cx":
                circuit.append(cirq.CNOT(qubits[control_qubit], qubits[target_qubit]))
            elif random_gate == "cy":
                circuit.append(
                    cirq.Y.controlled(1)(qubits[control_qubit], qubits[target_qubit])
                )
            elif random_gate == "cz":
                circuit.append(cirq.CZ(qubits[control_qubit], qubits[target_qubit]))
        else:
            for q in range(num_qubits):
                random_single_qubit_gate = np.random.choice(["H", "X", "Y", "Z"])
                if random_single_qubit_gate == "H":
                    circuit.append(cirq.H(qubits[q]))
                elif random_single_qubit_gate == "X":
                    circuit.append(cirq.X(qubits[q]))
                elif random_single_qubit_gate == "Y":
                    circuit.append(cirq.Y(qubits[q]))
                elif random_single_qubit_gate == "Z":
                    circuit.append(cirq.Z(qubits[q]))

    # Add measurement
    circuit.append(cirq.measure(*qubits, key="result"))

    return circuit


def test_global_cirq_flow():
    ### Client side

    qc = _random_cirq_circuit(10)
    shots = 10

    program = QuantumProgram.from_cirq_circuit(qc)
    compressed_program = QuantumProgram.from_cirq_circuit(
        qc, compress_format=CompressionFormat.ZLIB_BASE64_V1
    )

    backend_data = BackendData(
        name="qsim",
        version="1",
    )

    client_data = ClientData(
        user_agent="local",
    )

    computation_model_json = QuantumComputationModel(
        programs=[program, compressed_program],
        backend=backend_data,
        client=client_data,
    ).to_json_str()

    computation_parameters_json = QuantumComputationParameters(
        shots=shots,
    ).to_json_str()

    ### Server/Compute side
    model = QuantumComputationModel.from_json_str(computation_model_json)
    params = QuantumComputationParameters.from_json_str(computation_parameters_json)

    circuit = model.programs[0].to_cirq_circuit()
    uncomp_circuit = model.programs[1].to_cirq_circuit()

    qsim_simulator = QSimSimulator(QSimOptions(cpu_threads=4))
    result_1 = qsim_simulator.run(circuit, repetitions=params.shots)
    result_1._params = None  # ParamResolver cannot be serialized

    result_2 = qsim_simulator.run(uncomp_circuit, repetitions=params.shots)
    result_2._params = None  # ParamResolver cannot be serialized

    qresult = QuantumProgramResult.from_cirq_result(result_1).to_json_str()
    compressed_qresult = QuantumProgramResult.from_cirq_result(
        result_2, CompressionFormat.ZLIB_BASE64_V1
    ).to_json_str()

    assert qresult is not None
    assert compressed_qresult is not None

    ### Client side
    cirq_result = qresult.to_cirq_result()
    assert cirq_result is not None
    print(cirq_result)

    uncomp_cirq_result = compressed_qresult.to_cirq_result()
    assert uncomp_cirq_result is not None

    qiskit_result = qresult.to_qiskit_result()
    assert qiskit_result is not None
    print(qiskit_result)

    uncomp_qiskit_result = compressed_qresult.to_qiskit_result()
    assert uncomp_qiskit_result is not None
