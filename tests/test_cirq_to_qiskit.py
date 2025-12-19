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
from cirq.sim import Simulator

from qio.core import (
    QuantumComputationModel,
    QuantumProgram,
    QuantumProgramSerializationFormat
)

from qio.utils import CompressionFormat
from qio.utils.circuit import random_square_cirq_circuit


def test_cirq_to_qiskit_to_cirq_flow():
    original_cirq_qc = random_square_cirq_circuit(10)

    program_1 = QuantumProgram.from_cirq_circuit(
        original_cirq_qc, dest_format=QuantumProgramSerializationFormat.QASM_V3, compression_format=CompressionFormat.NONE
    )

    print("original ser", program_1.serialization)

    computation_model_json_1 = QuantumComputationModel(
        programs=[program_1],
    ).to_json_str()

    model_1 = QuantumComputationModel.from_json_str(computation_model_json_1)

    qiskit_qc = model_1.programs[0].to_qiskit_circuit()

    program_2 = QuantumProgram.from_qiskit_circuit(
        qiskit_qc, dest_format=QuantumProgramSerializationFormat.QASM_V3, compression_format=CompressionFormat.NONE
    )

    print("after qiskit ser", program_2.serialization)

    computation_model_json = QuantumComputationModel(
        programs=[program_2],
    ).to_json_str()

    model_2 = QuantumComputationModel.from_json_str(computation_model_json)

    model_2.programs[0].serialization = program_1.serialization
    print("after replacement", model_2.programs[0].serialization)
    new_cirq_qc = model_2.programs[0].to_cirq_circuit()

    sim_1 = Simulator(seed=42)
    sim_2 = Simulator(seed=42)
    repetitions = 10

    res_original = sim_1.run(original_cirq_qc, repetitions=repetitions)
    res_new = sim_2.run(new_cirq_qc, repetitions=repetitions)

    print("RESULT1", res_original)
    print("RESULT2", res_new)

    assert res_original == res_new