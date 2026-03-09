import cirq

from .qasm1_to_cirq import convert as qasm1_to_cirq_convert


def convert(circuit_str: str) -> cirq.Circuit:
    return qasm1_to_cirq_convert(circuit_str)
