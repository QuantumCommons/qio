# qio
`qio` is python package to smoothly manipulate common quantum computing objects between client and server.

<p align="center">
  <img src="https://github.com/QuantumCommons/qio/tree/main/docs/logo.png" />
</p>

## Installation

We encourage installing `qio` via the pip tool (a Python package manager):

```bash
pip install qio
```

## Getting started

To leverage `qio` and get quick interoperability between your frontend and your backend, you simply need to use qio wrappers system.

```python
import cirq

from cirq.circuits import Circuit
from qsimcirq import QSimSimulator

from qio.core import (
    QuantumComputationModel,
    QuantumComputationParameters,
    QuantumProgramResult,
    QuantumProgram,
)

#############
# Client side

qc = _random_cirq_circuit(10)
shots = 100

program = QuantumProgram.from_cirq_circuit()

model_json = QuantumComputationModel(
    programs=[program],
).to_json_str()

parameters_json = QuantumComputationParameters(
    shots=shots,
).to_json_str()

# Send this to server side

###########################
# Server / Computation side

model = QuantumComputationModel.from_json_str(model_json)
params = QuantumComputationParameters.from_json_str(parameters_json)

circuit = model.programs[0].to_cirq_circuit()

qsim_simulator = QSimSimulator()

qsim_result = qsim_simulator.run(circuit, repetitions=params.shots)

program_result = QuantumProgramResult.from_cirq_result(result).to_json_str()

# Send this to back to client side

#####################
# Back to client side

cirq_result = qresult.to_cirq_result()

qiskit_result = qresult.to_qiskit_result()
```