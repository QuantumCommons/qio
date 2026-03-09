import cudaq
import numpy as np

from cudaq import spin

from qio.core import (
    QuantumProgramResult,
    QuantumProgramResultCompressionFormat,
    QuantumProgram,
)


def test_global_cudaq_flow():
    ### Server side
    shots = 1000
    qubit_count = 4

    @cudaq.kernel
    def dummy_kernel(qubit_count: int):
        qubits = cudaq.qvector(qubit_count)
        h(qubits[0])
        for i in range(1, qubit_count):
            x.ctrl(qubits[0], qubits[i])
        mz(qubits)

    results = cudaq.sample(dummy_kernel, qubit_count, shots_count=shots)

    qpr_json = QuantumProgramResult.from_cudaq_sample_result(
        results, compression_format=QuantumProgramResultCompressionFormat.NONE
    ).to_json_str()

    compressed_qpr_json = QuantumProgramResult.from_cudaq_sample_result(
        results, compression_format=QuantumProgramResultCompressionFormat.ZLIB_BASE64_V1
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


def test_simple_kernel():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        x(q)
        mz(q)

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()

    counts = cudaq.sample(kernel_from_program, shots_count=100)

    assert len(counts) == 1
    assert "1" in counts


def test_multi_qubit_kernel():

    @cudaq.kernel
    def kernel():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        h(q0)
        cx(q0, q1)
        mz(q0)
        mz(q1)

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()

    counts = cudaq.sample(kernel_from_program, shots_count=100)

    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_qvector_kernel():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()

    counts = cudaq.sample(kernel_from_program, shots_count=100)

    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_builder_sample():

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()

    counts = cudaq.sample(kernel_from_program, shots_count=100)

    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_all_gates():

    @cudaq.kernel
    def single_qubit_gates():
        q = cudaq.qubit()
        h(q)
        x(q)
        y(q)
        z(q)
        r1(np.pi, q)
        rx(np.pi, q)
        ry(np.pi, q)
        rz(np.pi, q)
        s(q)
        t(q)
        mx(q)
        ## my(q) # not supported since the default rewriter uses `sdg`
        mz(q)

    # Test here is that this runs
    program = QuantumProgram.from_cudaq_kernel(single_qubit_gates)
    kernel_from_program = program.to_cudaq_kernel()

    counts = cudaq.sample(kernel_from_program, shots_count=100)

    assert counts

    @cudaq.kernel
    def two_qubit_gates():
        qubits = cudaq.qvector(2)
        x(qubits[0])
        swap(qubits[0], qubits[1])
        mz(qubits)

    program = QuantumProgram.from_cudaq_kernel(two_qubit_gates)
    kernel_from_program = program.to_cudaq_kernel()

    counts = cudaq.sample(kernel_from_program, shots_count=100)

    counts = cudaq.sample(two_qubit_gates, shots_count=100)

    assert len(counts) == 1
    assert "01" in counts


def test_multi_qvector():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        x(qubits)
        h(ancilla)
        mz(ancilla)

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()

    counts = cudaq.sample(kernel_from_program, shots_count=100)

    assert counts


def test_control_modifier():

    @cudaq.kernel
    def single_qubit_gates():
        qubits = cudaq.qvector(2)
        h.ctrl(qubits[0], qubits[1])
        x.ctrl(qubits[1], qubits[0])
        y.ctrl(qubits[0], qubits[1])
        z.ctrl(qubits[1], qubits[0])
        r1.ctrl(np.pi / 2, qubits[0], qubits[1])
        rx.ctrl(np.pi / 4, qubits[1], qubits[0])
        ry.ctrl(np.pi / 8, qubits[0], qubits[1])
        rz.ctrl(np.pi, qubits[1], qubits[0])
        s.ctrl(qubits[0], qubits[1])
        t.ctrl(qubits[1], qubits[0])
        mz(qubits)

    program = QuantumProgram.from_cudaq_kernel(single_qubit_gates)
    kernel_from_program = program.to_cudaq_kernel()
    counts = cudaq.sample(kernel_from_program, shots_count=100)

    @cudaq.kernel
    def two_qubit_gates():
        qubits = cudaq.qvector(3)
        x(qubits[0])
        x(qubits[1])
        swap.ctrl(qubits[0], qubits[1], qubits[2])
        mz(qubits)

    program = QuantumProgram.from_cudaq_kernel(two_qubit_gates)
    kernel_from_program = program.to_cudaq_kernel()
    counts = cudaq.sample(kernel_from_program, shots_count=100)

    assert len(counts) == 1
    assert "101" in counts

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    program = QuantumProgram.from_cudaq_kernel(bell)
    kernel_from_program = program.to_cudaq_kernel()
    counts = cudaq.sample(kernel_from_program, shots_count=100)

    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_adjoint_modifier():

    @cudaq.kernel
    def single_qubit_gates():
        q = cudaq.qubit()
        r1.adj(np.pi, q)
        rx.adj(np.pi / 2, q)
        ry.adj(np.pi / 4, q)
        rz.adj(np.pi / 8, q)
        s.adj(q)
        t.adj(q)
        mz(q)

    # Test here is that this runs
    program = QuantumProgram.from_cudaq_kernel(single_qubit_gates)
    kernel_from_program = program.to_cudaq_kernel()
    counts = cudaq.sample(kernel_from_program, shots_count=100)

    assert counts


def test_u3_decomposition():

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        u3(0.0, np.pi / 2, np.pi, qubit)
        mz(qubit)

    # Test here is that this runs
    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()
    result = cudaq.sample(kernel_from_program, shots_count=100)
    measurement_probabilities = dict(result.items())
    print(measurement_probabilities)

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        u3.ctrl(0.0, np.pi / 2, np.pi, qubits[0], qubits[1])
        mz(qubits)

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()
    counts = cudaq.sample(kernel_from_program, shots_count=100)
    print("test_u3_decomposition bell", counts)

    assert "00" in counts
    assert len(counts) == 1


def test_sample_async():
    cudaq.set_random_seed(13)

    @cudaq.kernel
    def simple():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    program = QuantumProgram.from_cudaq_kernel(simple)
    kernel_from_program = program.to_cudaq_kernel()
    future = cudaq.sample_async(kernel_from_program, shots_count=100)
    counts = future.get()
    print("test_sample_async bell", counts)

    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_observe_async():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        x(qubits[0])

    hamiltonian = spin.z(0) * spin.z(1)

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()
    future = cudaq.observe_async(kernel_from_program, hamiltonian, shots_count=1)
    result = future.get()

    assert result.expectation() == -1.0


def test_custom_operations():
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def basic_x():
        qubit = cudaq.qubit()
        custom_x(qubit)
        mz(qubit)

    program = QuantumProgram.from_cudaq_kernel(basic_x)
    kernel_from_program = program.to_cudaq_kernel()
    counts = cudaq.sample(kernel_from_program, shots_count=100)
    print("test_custom_operations", counts)

    assert len(counts) == 1 and "1" in counts


# def test_kernel_with_args():

#     @cudaq.kernel
#     def kernel(qubit_count: int):
#         qreg = cudaq.qvector(qubit_count)
#         h(qreg[0])
#         for qubit in range(qubit_count - 1):
#             x.ctrl(qreg[qubit], qreg[qubit + 1])
#         mz(qreg)

#     program = QuantumProgram.from_cudaq_kernel(kernel)
#     kernel_from_program = program.to_cudaq_kernel()
#     counts = cudaq.sample(kernel_from_program, 4, shots_count=100)
#     print("test_kernel_with_args", counts)

#     assert len(counts) == 2
#     assert "0000" in counts
#     assert "1111" in counts


# def test_kernel_subveqs():

#     @cudaq.kernel
#     def kernel():
#         qreg = cudaq.qvector(4)
#         x(qreg[1])
#         x(qreg[2])
#         v = qreg[1:3]
#         mz(v)

#     program = QuantumProgram.from_cudaq_kernel(kernel)
#     kernel_from_program = program.to_cudaq_kernel()
#     counts = cudaq.sample(kernel_from_program, shots_count=100)
#     print("test_kernel_subveqs", counts)

#     assert len(counts) == 1
#     assert "11" in counts  # got 0110


# def test_kernel_two_subveqs():

#     @cudaq.kernel
#     def kernel():
#         qreg = cudaq.qvector(4)
#         x(qreg[1])
#         x(qreg[2])
#         v1 = qreg[0:2]
#         mz(v1)
#         v2 = qreg[2:3]
#         mz(v2)

#     program = QuantumProgram.from_cudaq_kernel(kernel)
#     kernel_from_program = program.to_cudaq_kernel()
#     counts = cudaq.sample(kernel_from_program, shots_count=100)
#     print("test_kernel_two_subveqs", counts)

#     assert len(counts) == 1
#     assert "011" in counts  # got 0110


# def test_kernel_qubit_subveq():

#     @cudaq.kernel
#     def kernel():
#         qreg = cudaq.qvector(4)
#         x(qreg[1])
#         x(qreg[2])
#         v1 = qreg[0:2]
#         mz(v1)
#         v2 = qreg[2]
#         mz(v2)

#     program = QuantumProgram.from_cudaq_kernel(kernel)
#     kernel_from_program = program.to_cudaq_kernel()
#     counts = cudaq.sample(kernel_from_program, shots_count=100)
#     print("test_kernel_qubit_subveq", counts)

#     assert len(counts) == 1
#     assert "011" in counts  # got 0110


def test_multiple_measurement():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        mz(qubits[0])
        mz(qubits[1])

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()
    counts = cudaq.sample(kernel_from_program, shots_count=100)
    print("test_multiple_measurement", counts)

    assert len(counts) == 2
    assert "00" in counts
    assert "10" in counts


def test_multiple_measurement_non_consecutive():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(3)
        x(qubits[0])
        x(qubits[2])
        mz(qubits[0])
        mz(qubits[2])

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()
    counts = cudaq.sample(kernel_from_program, shots_count=100)
    print("test_multiple_measurement_non_consecutive", counts)

    assert len(counts) == 1
    assert "11" in counts


def test_qvector_slicing():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(4)
        x(q.front(2))
        mz(q)

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()
    counts = cudaq.sample(kernel_from_program, shots_count=100)
    print("test_qvector_slicing", counts)

    assert len(counts) == 1
    assert "1100" in counts


def test_toffoli():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        x(q)
        x.ctrl([q[0], q[1]], q[2])
        mz(q)

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()
    counts = cudaq.sample(kernel_from_program, shots_count=100)
    counts.dump()


def test_state_prep():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)])
        mz(q)

    program = QuantumProgram.from_cudaq_kernel(kernel)
    kernel_from_program = program.to_cudaq_kernel()
    counts = cudaq.sample(kernel_from_program, shots_count=100)
    counts.dump()

    assert "11" in counts
    assert "00" in counts
