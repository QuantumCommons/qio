from qiskit import qasm3, QuantumCircuit


def convert(circuit_str: str) -> QuantumCircuit:
    return qasm3.loads(circuit_str)
