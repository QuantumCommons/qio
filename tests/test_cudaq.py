import cudaq
import random

from qio.core import (
    QuantumProgramResult,
)

from qio.utils import CompressionFormat


@cudaq.kernel
def dummy_kernel(qubit_count: int):
    qubits = cudaq.qvector(qubit_count)
    h(qubits[0])
    for i in range(1, qubit_count):
        x.ctrl(qubits[0], qubits[i])
    mz(qubits)


def test_global_cudaq_flow():
    ### Server side
    shots = 1000
    qubit_count = 4

    results = cudaq.sample(dummy_kernel, qubit_count, shots_count=shots)

    qpr_json = QuantumProgramResult.from_cudaq_sample_result(
        results, compression_format=CompressionFormat.NONE
    ).to_json_str()

    compressed_qpr_json = QuantumProgramResult.from_cudaq_sample_result(
        results, compression_format=CompressionFormat.ZLIB_BASE64_V1
    ).to_json_str()

    assert qpr_json is not None
    assert compressed_qpr_json is not None

    ### Client side
    qpr = QuantumProgramResult.from_json_str(qpr_json)
    compressed_qpr = QuantumProgramResult.from_json_str(compressed_qpr_json)

    cudaq_result = qpr.to_cudaq_sample_result()
    assert cudaq_result is not None
    print("cudaq result:", cudaq_result)

    uncomp_cudaq_result = compressed_qpr.to_cudaq_sample_result()
    assert uncomp_cudaq_result is not None
    print("cudaq result from compressed data:", uncomp_cudaq_result)

    assert cudaq_result.get_total_shots() == uncomp_cudaq_result.get_total_shots()
    assert cudaq_result.most_probable() == uncomp_cudaq_result.most_probable()
    assert cudaq_result.serialize() == uncomp_cudaq_result.serialize()

    qiskit_result = compressed_qpr.to_qiskit_result()
    print("cudaq result as qiskit result:", qiskit_result)
    assert qiskit_result is not None


test_global_cudaq_flow()
