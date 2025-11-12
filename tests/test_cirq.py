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
from qsimcirq import QSimSimulator

from qio.core import (
    QuantumComputationModel,
    QuantumComputationParameters,
    QuantumProgramResult,
    QuantumProgram,
    BackendData,
    ClientData,
)

from qio.utils import CompressionFormat

from qio.utils.circuit import random_square_cirq_circuit


def test_global_cirq_flow():
    ### Client side
    qc = random_square_cirq_circuit(10)
    shots = 20

    program = QuantumProgram.from_cirq_circuit(
        qc, compression_format=CompressionFormat.NONE
    )
    compressed_program = QuantumProgram.from_cirq_circuit(
        qc, compression_format=CompressionFormat.ZLIB_BASE64_V1
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

    qsim_simulator = QSimSimulator()
    result_1 = qsim_simulator.run(circuit, repetitions=params.shots)
    result_1._params = None  # ParamResolver cannot be serialized

    result_2 = qsim_simulator.run(uncomp_circuit, repetitions=params.shots)
    result_2._params = None  # ParamResolver cannot be serialized

    qpr_json = QuantumProgramResult.from_cirq_result(
        result_1, compression_format=CompressionFormat.NONE
    ).to_json_str()

    compressed_qpr_json = QuantumProgramResult.from_cirq_result(
        result_2, compression_format=CompressionFormat.ZLIB_BASE64_V1
    ).to_json_str()

    assert qpr_json is not None
    assert compressed_qpr_json is not None

    ### Client side
    qpr = QuantumProgramResult.from_json_str(qpr_json)
    compressed_qpr = QuantumProgramResult.from_json_str(compressed_qpr_json)

    cirq_result = qpr.to_cirq_result()
    assert cirq_result is not None
    print("cirq result:", cirq_result)

    uncomp_cirq_result = compressed_qpr.to_cirq_result()
    assert uncomp_cirq_result is not None
    print("cirq result from compressed data:", uncomp_cirq_result)

    qiskit_result = qpr.to_qiskit_result()
    assert qiskit_result is not None
    print("qiskit result:", qiskit_result)

    uncomp_qiskit_result = compressed_qpr.to_qiskit_result()
    assert uncomp_qiskit_result is not None
    print("qiskit result from compressed data:", uncomp_qiskit_result)
