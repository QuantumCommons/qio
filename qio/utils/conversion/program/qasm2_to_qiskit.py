from qiskit import qasm2, QuantumCircuit


def convert(circuit_str: str) -> QuantumCircuit:
    return qasm2.loads(circuit_str)
