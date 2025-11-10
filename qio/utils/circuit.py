def random_square_cirq_circuit(size: int) -> "cirq.Circuit":
    return random_cirq_circuit(size, size)


def random_cirq_circuit(num_qubits: int, num_gates: int) -> "cirq.Circuit":
    import cirq
    import numpy as np
    from cirq.circuits import Circuit

    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    circuit = Circuit()

    for _ in range(num_gates):
        random_gate = np.random.choice(["unitary", "cx", "cy", "cz"])

        if random_gate in ["cx", "cy", "cz"]:
            control_qubit = np.random.randint(0, num_qubits)
            target_qubit = np.random.randint(0, num_qubits)

            while target_qubit == control_qubit:
                target_qubit = np.random.randint(0, num_qubits)

            if random_gate == "cx":
                circuit.append(cirq.CNOT(qubits[control_qubit], qubits[target_qubit]))
            elif random_gate == "cy":
                circuit.append(
                    cirq.Y.controlled(1)(qubits[control_qubit], qubits[target_qubit])
                )
            elif random_gate == "cz":
                circuit.append(cirq.CZ(qubits[control_qubit], qubits[target_qubit]))
        else:
            for q in range(num_qubits):
                random_single_qubit_gate = np.random.choice(["H", "X", "Y", "Z"])
                if random_single_qubit_gate == "H":
                    circuit.append(cirq.H(qubits[q]))
                elif random_single_qubit_gate == "X":
                    circuit.append(cirq.X(qubits[q]))
                elif random_single_qubit_gate == "Y":
                    circuit.append(cirq.Y(qubits[q]))
                elif random_single_qubit_gate == "Z":
                    circuit.append(cirq.Z(qubits[q]))

    circuit.append(cirq.measure(*qubits, key="result"))

    return circuit


def random_square_qiskit_circuit(size: int) -> "qiskit.QuantumCircuit":
    return random_qiskit_circuit(size, size)


def random_qiskit_circuit(num_qubits: int, num_gates: int) -> "qiskit.QuantumCircuit":
    import numpy as np
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_qubits)

    for _ in range(num_gates):
        random_gate = np.random.choice(["unitary", "cx", "cy", "cz"])

        if random_gate == "cx" or random_gate == "cy" or random_gate == "cz":
            control_qubit = np.random.randint(0, num_qubits)
            target_qubit = np.random.randint(0, num_qubits)

            while target_qubit == control_qubit:
                target_qubit = np.random.randint(0, num_qubits)

            getattr(qc, random_gate)(control_qubit, target_qubit)
        else:
            for q in range(num_qubits):
                random_gate = np.random.choice(["h", "x", "y", "z"])
                getattr(qc, random_gate)(q)

    qc.measure_all()

    return qc
