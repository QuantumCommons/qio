from qiskit import QuantumCircuit


def convert(circuit_str: str) -> QuantumCircuit:
    return QuantumCircuit.from_qasm_str(circuit_str)
