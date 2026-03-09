from qiskit import qasm2, QuantumCircuit


def convert(qiskit_circuit: QuantumCircuit) -> str:
    return qasm2.dumps(qiskit_circuit)
