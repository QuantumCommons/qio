import cirq


def convert(circuit_str: str) -> cirq.Circuit:
    return cirq.read_json(json_text=circuit_str)
