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

# from bitarray import frozenbitarray
# from mimiqcircuits import QCSResults
# Copyright 2025 Scaleway, Aqora, Quantum Commons
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (insérer la licence) ...
import pytest

pytestmark = pytest.mark.gate

import os
import tempfile

from quantanium.Quantanium import Quantanium

from qio.core import (
    QuantumComputationModel,
    QuantumComputationParameters,
    QuantumProgramResult,
    QuantumProgram,
    QuantumProgramSerializationFormat,
    QuantumProgramCompressionFormat,
    QuantumProgramResultCompressionFormat,
    BackendData,
    ClientData,
)
from qio.utils.circuit_factory import random_square_qiskit_circuit


def test_global_mimiq_flow():
    """
    Test the complete workflow: the circuit is created through Qiskit, converted into QASM2 (required format for quantanium),
    executed and then results are retrieved and analyzed.
    """
    ### Client side
    qc = random_square_qiskit_circuit(5)
    shots = 20

    program = QuantumProgram.from_qiskit_circuit(
        qc,
        dest_format=QuantumProgramSerializationFormat.QASM_V2,
        compression_format=QuantumProgramCompressionFormat.NONE,
    )
    compressed_program = QuantumProgram.from_qiskit_circuit(
        qc,
        dest_format=QuantumProgramSerializationFormat.QASM_V2,
        compression_format=QuantumProgramCompressionFormat.ZLIB_BASE64_V1,
    )

    backend_data = BackendData(name="quantanium", version="1")
    client_data = ClientData(user_agent="local")

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

    circuit = model.programs[0].to_qasm2_circuit()
    uncomp_circuit = model.programs[1].to_qasm2_circuit()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".qasm", delete=False
    ) as tmp_file:
        tmp_file.write(circuit)
        circuit_tmp_path = tmp_file.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".qasm", delete=False
    ) as tmp_file:
        tmp_file.write(uncomp_circuit)
        uncomp_circuit_tmp_path = tmp_file.name

    engine = Quantanium(False)

    try:
        result_1 = engine.execute(circuit_tmp_path, nsamples=params.shots)
        result_2 = engine.execute(uncomp_circuit_tmp_path, nsamples=params.shots)
    finally:
        if circuit_tmp_path and os.path.exists(circuit_tmp_path):
            try:
                os.remove(circuit_tmp_path)
            except OSError:
                pass
        if uncomp_circuit_tmp_path and os.path.exists(uncomp_circuit_tmp_path):
            try:
                os.remove(uncomp_circuit_tmp_path)
            except OSError:
                pass

    qpr_json = QuantumProgramResult.from_mimiq_qcsr(
        result_1, compression_format=QuantumProgramResultCompressionFormat.NONE
    ).to_json_str()

    compressed_qpr_json = QuantumProgramResult.from_mimiq_qcsr(
        result_2,
        compression_format=QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1,
    ).to_json_str()

    assert qpr_json is not None
    assert compressed_qpr_json is not None

    ### Client side
    qpr = QuantumProgramResult.from_json_str(qpr_json)
    compressed_qpr = QuantumProgramResult.from_json_str(compressed_qpr_json)

    kwargs = {}
    kwargs["backend_name"] = "quantanium"

    qiskit_result = qpr.to_qiskit_result(**kwargs)
    assert qiskit_result is not None

    uncomp_qiskit_result = compressed_qpr.to_qiskit_result(**kwargs)
    assert uncomp_qiskit_result is not None


def test_circuit_conversion_loop_to_qasm2():
    """
    Test conversion from Qiskit ciruit to QASM2.
    """
    original_qc = random_square_qiskit_circuit(4)

    prog_uncomp = QuantumProgram.from_qiskit_circuit(
        original_qc,
        dest_format=QuantumProgramSerializationFormat.QASM_V2,
        compression_format=QuantumProgramCompressionFormat.NONE,
    )
    prog_comp = QuantumProgram.from_qiskit_circuit(
        original_qc,
        dest_format=QuantumProgramSerializationFormat.QASM_V2,
        compression_format=QuantumProgramCompressionFormat.ZLIB_BASE64_V1,
    )

    qasm2_str_uncomp = prog_uncomp.to_qasm2_circuit()
    qasm2_str_comp = prog_comp.to_qasm2_circuit()

    assert isinstance(qasm2_str_uncomp, str)
    assert qasm2_str_uncomp == qasm2_str_comp

    recovered_qc_uncomp = prog_uncomp.to_qiskit_circuit()
    recovered_qc_comp = prog_comp.to_qiskit_circuit()

    assert original_qc.num_qubits == recovered_qc_uncomp.num_qubits
    assert original_qc.num_clbits == recovered_qc_uncomp.num_clbits
    assert original_qc.num_qubits == recovered_qc_comp.num_qubits
    assert original_qc.num_clbits == recovered_qc_comp.num_clbits


def test_from_mimiq_circuit():
    """
    Test the MIMIQ ingestion edge: a MIMIQ circuit is serialized to QASM2
    (from_mimiq_circuit) and the result round-trips back to the SDKs.
    """
    from mimiqcircuits.qasm import loads as mimiq_loads

    from qio.utils.conversion.program.qasm2_to_mimiq import (
        convert as qasm2_to_mimiq,
    )

    qasm2_str = """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """

    mimiq_circuit = mimiq_loads(qasm2_str)

    prog_uncomp = QuantumProgram.from_mimiq_circuit(
        mimiq_circuit,
        dest_format=QuantumProgramSerializationFormat.QASM_V2,
        compression_format=QuantumProgramCompressionFormat.NONE,
    )
    prog_comp = QuantumProgram.from_mimiq_circuit(
        mimiq_circuit,
        dest_format=QuantumProgramSerializationFormat.QASM_V2,
        compression_format=QuantumProgramCompressionFormat.ZLIB_BASE64_V1,
    )

    assert prog_uncomp.serialization_format == QuantumProgramSerializationFormat.QASM_V2

    # The uncompressed QASM2 round-trips back to an equivalent MIMIQ circuit
    # and the program feeds the Qiskit consumer.
    qasm2_uncomp = prog_uncomp.to_qasm2_circuit()
    qasm2_comp = prog_comp.to_qasm2_circuit()
    assert isinstance(qasm2_uncomp, str)
    assert qasm2_uncomp == qasm2_comp

    recovered_mimiq = qasm2_to_mimiq(qasm2_uncomp)
    assert recovered_mimiq is not None
    assert recovered_mimiq.num_qubits() == 2

    recovered_qiskit = prog_uncomp.to_qiskit_circuit()
    assert recovered_qiskit.num_qubits == 2
    assert recovered_qiskit.num_clbits == 2

    # from_mimiq_circuit -> to_mimiq_circuit round trip preserves the qubits.
    round_tripped = prog_uncomp.to_mimiq_circuit()
    assert round_tripped.num_qubits() == 2


def test_to_mimiq_circuit_reads_qasm3():
    qasm3_str = """OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    cx q[0], q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    """

    program = QuantumProgram(
        serialization_format=QuantumProgramSerializationFormat.QASM_V3,
        compression_format=QuantumProgramCompressionFormat.NONE,
        serialization=qasm3_str,
    )

    mimiq_circuit = program.to_mimiq_circuit()
    assert mimiq_circuit is not None
    assert mimiq_circuit.num_qubits() == 2


def test_mimiq_result_conversion_integrity():
    circuit = """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    """

    shots = 50
    params = QuantumComputationParameters(shots=shots)
    circuit_tmp_path = None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".qasm", delete=False
    ) as tmp_file:
        tmp_file.write(circuit)
        circuit_tmp_path = tmp_file.name

    engine = Quantanium(False)

    try:
        result = engine.execute(circuit_tmp_path, nsamples=params.shots)

        qpr = QuantumProgramResult.from_mimiq_qcsr(
            result,
            compression_format=QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1,
            shots=shots,
            num_qubits=3,
        )

        # test with qiskit result

        recovered_qiskit_result = qpr.to_qiskit_result()
        assert recovered_qiskit_result is not None
        assert recovered_qiskit_result.results[0].shots == shots

        recovered_counts = recovered_qiskit_result.get_counts()
        assert len(recovered_counts) > 0
        assert sum(recovered_counts.values()) == shots

        # test with cirq result

        recovered_cirq_result = qpr.to_cirq_result()
        assert recovered_cirq_result is not None
        assert "result" in recovered_cirq_result.measurements
        assert len(recovered_cirq_result.measurements["result"]) == shots

        recovered_counts = recovered_cirq_result.histogram(key="result")
        assert len(recovered_counts) > 0
        assert sum(recovered_counts.values()) == shots

    finally:
        if circuit_tmp_path and os.path.exists(circuit_tmp_path):
            try:
                os.remove(circuit_tmp_path)
            except OSError:
                pass
