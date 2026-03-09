import cirq


def convert(cirq_circuit: cirq.Circuit) -> str:
    return cirq_circuit.to_qasm(version="3.0")
