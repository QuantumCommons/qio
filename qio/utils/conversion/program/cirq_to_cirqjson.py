import cirq


def convert(cirq_circuit: cirq.Circuit):
    return cirq.to_json(cirq_circuit)
