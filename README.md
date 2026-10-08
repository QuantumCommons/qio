<p align="center"><img width="50%" src="docs/logo.png" /></p>

# qio
`qio` is a python package to smoothly manipulate quantum computing objects between client and server.

It handle all the conversion boilerplate to focus your dev on your quantum SDK and backend on real matters.

## Installation

We encourage installing `qio` via the pip tool (a Python package manager):

```bash
pip install qio
```

## Getting started

To leverage `qio` and get quick interoperability between your frontend and your backend, you simply need to use qio wrappers system.

Here a snippet code for `cirq` and `qsim`:

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

program = QuantumProgram.from_cirq_circuit(qc)

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

## qio circuit conversion mapping

Map of the SDKs integrated in `qio` and the conversion links between them
(**circuits** branch / `QuantumProgram` — *results* conversions are not
represented here).

- **Conversion flow**: an SDK node is translated into a serialization format
  (OpenQASM, QIR, Cirq JSON), then reloaded by another SDK.
- The converters are located in `qio/utils/conversion/program/`.
- `Perceval JSON` and `Pulser Sequence JSON` are declared in the
  `QuantumProgramSerializationFormat` enum but **no converter is implemented**.

```mermaid

flowchart LR

    subgraph IN["Input SDKs (write to format)"]
        direction TB
        CIRQ_IN["Cirq<br/>(cirq.Circuit)"]
        QISKIT_IN["Qiskit<br/>(qiskit.QuantumCircuit)"]
        CUDAQ_IN["CUDA Quantum<br/>(cudaq.Kernel)"]
    end

    subgraph SER["Serialization format"]
        direction TB
        QASM1["OpenQASM 1"]
        QASM2["OpenQASM 2"]
        QASM3["OpenQASM 3"]
        QIR["QIR"]
        CIRQJSON["Cirq JSON"]
    end

    subgraph OUT["Output SDKs (read from format)"]
        direction TB
        CIRQ_OUT["Cirq<br/>(cirq.Circuit)"]
        QISKIT_OUT["Qiskit<br/>(qiskit.QuantumCircuit)"]
        CUDAQ_OUT["CUDA Quantum<br/>(cudaq.Kernel)"]
        MIMIQ_OUT["MIMIQ<br/>(mimiqcircuits.Circuit)"]
    end

    subgraph DECL["Declared serialization format (unimplemented converters)"]
        direction TB
        PERCEVAL["Perceval JSON"]
        PULSER["Pulser Sequence JSON"]
    end

    CIRQ_IN -->|cirq_to_cirqjson| CIRQJSON
    CIRQ_IN -->|cirq_to_qasm2| QASM2
    CIRQ_IN -->|cirq_to_qasm3| QASM3
    QISKIT_IN -->|qiskit_to_qasm2| QASM2
    QISKIT_IN -->|qiskit_to_qasm3| QASM3
    CUDAQ_IN -->|cudaq_to_qasm2| QASM2
    CUDAQ_IN -->|cudaq_to_qir| QIR

    CIRQJSON -->|cirqjson_to_cirq| CIRQ_OUT
    QASM1 -->|qasm1_to_cirq| CIRQ_OUT
    QASM2 -->|qasm2_to_cirq| CIRQ_OUT
    QASM3 -->|qasm3_to_cirq| CIRQ_OUT
    QASM1 -->|qasm1_to_qiskit| QISKIT_OUT
    QASM2 -->|qasm2_to_qiskit| QISKIT_OUT
    QASM3 -->|qasm3_to_qiskit| QISKIT_OUT
    QASM2 -->|qasm2_to_cudaq - via qasm3_to_cudaq| CUDAQ_OUT
    QASM3 -->|qasm3_to_cudaq| CUDAQ_OUT
    QASM2 -->|qasm2_to_mimiq| MIMIQ_OUT



    classDef sdk fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#00344f,font-weight:bold,rx:20px,ry:20px;
    classDef fmt fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#062b07,font-weight:bold,rx:20px,ry:20px;
    classDef decl fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,stroke-dasharray: 5 5,color:#2a0a4a,font-weight:bold,rx:20px,ry:20px;

    class CIRQ_IN,QISKIT_IN,CUDAQ_IN,CIRQ_OUT,QISKIT_OUT,CUDAQ_OUT,MIMIQ_OUT sdk;
    class QASM1,QASM2,QASM3,QIR,CIRQJSON fmt;
    class PERCEVAL,PULSER decl;
```

### Entrypoints in `qio`

| Source SDK | Write method | Supported output formats |
|---|---|---|
| Cirq | `QuantumProgram.from_cirq_circuit` | Cirq JSON, OpenQASM 2, OpenQASM 3 |
| Qiskit | `QuantumProgram.from_qiskit_circuit` | OpenQASM 2, OpenQASM 3 |
| CUDA Quantum | `QuantumProgram.from_cudaq_kernel` | OpenQASM 2 (+ QIR via `cudaq_to_qir`) |

| Target SDK | Read method | Supported input formats |
|---|---|---|
| Cirq | `QuantumProgram.to_cirq_circuit` | OpenQASM 1/2/3, Cirq JSON |
| Qiskit | `QuantumProgram.to_qiskit_circuit` | OpenQASM 1/2/3 |
| CUDA Quantum | `QuantumProgram.to_cudaq_kernel` | OpenQASM 2/3 |
| MIMIQ | `QuantumProgram.to_mimiq_circuit` | OpenQASM 2 |

## qio result conversion mapping

Map of the SDKs integrated in `qio` and the conversion links between them
(**results** branch / `QuantumProgramResult`).

- **Conversion flow**: an SDK result is serialized into a dedicated JSON format
  (`*_RESULT_JSON_V1`), then reloaded by an SDK (its own or another one).
- Unlike circuits, there is **no shared format** (no QASM): each SDK has its own
  JSON format, and cross conversions directly link one JSON format to another
  SDK.
- The converters are located in `qio/utils/conversion/program_result/`.

```mermaid

flowchart LR

    subgraph IN["Input SDKs (write to format)"]
        direction TB
        CIRQ_IN["Cirq<br/>(cirq.Result)"]
        QISKIT_IN["Qiskit<br/>(qiskit.result.Result)"]
        CUDAQ_IN["CUDA Quantum<br/>(cudaq.SampleResult)"]
        MIMIQ_IN["MIMIQ<br/>(mimiqcircuits.QCSResults)"]
    end

    subgraph SER["Result serialization format"]
        direction TB
        CIRQJSON["Cirq result JSON<br/>(CIRQ_RESULT_JSON_V1)"]
        QISKITJSON["Qiskit result JSON<br/>(QISKIT_RESULT_JSON_V1)"]
        CUDAQJSON["CUDA Quantum sample JSON<br/>(CUDAQ_SAMPLE_RESULT_JSON_V1)"]
        MIMIQJSON["MIMIQ QCSR JSON<br/>(MIMIQ_QCSR_JSON_V1)"]
    end

    subgraph OUT["Output SDKs (read from format)"]
        direction TB
        CIRQ_OUT["Cirq<br/>(cirq.Result)"]
        QISKIT_OUT["Qiskit<br/>(qiskit.result.Result)"]
        CUDAQ_OUT["CUDA Quantum<br/>(cudaq.SampleResult)"]
        MIMIQ_OUT["MIMIQ<br/>(mimiqcircuits.QCSResults)"]
    end

    CIRQ_IN -->|cirq_to_dict| CIRQJSON
    QISKIT_IN -->|qiskit_to_dict| QISKITJSON
    CUDAQ_IN -->|cudaq_sample_to_dict| CUDAQJSON
    MIMIQ_IN -->|mimiq_to_dict| MIMIQJSON

    CIRQJSON -->|dict_to_cirq| CIRQ_OUT
    QISKITJSON -->|dict_to_qiskit| QISKIT_OUT
    CUDAQJSON -->|dict_to_cudaq_sample| CUDAQ_OUT
    MIMIQJSON -->|dict_to_mimiq| MIMIQ_OUT

    CIRQJSON -->|cirq_to_qiskit| QISKIT_OUT
    CIRQJSON -->|cirq_to_mimiq| MIMIQ_OUT
    QISKITJSON -->|qiskit_to_cirq| CIRQ_OUT
    QISKITJSON -->|qiskit_to_mimiq| MIMIQ_OUT
    CUDAQJSON -->|cudaq_sample_to_qiskit| QISKIT_OUT
    MIMIQJSON -->|mimiq_to_qiskit| QISKIT_OUT
    MIMIQJSON -->|mimiq_to_cirq| CIRQ_OUT

    classDef sdk fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#00344f,font-weight:bold,rx:20px,ry:20px;
    classDef fmt fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#062b07,font-weight:bold,rx:20px,ry:20px;

    class CIRQ_IN,QISKIT_IN,CUDAQ_IN,MIMIQ_IN,CIRQ_OUT,QISKIT_OUT,CUDAQ_OUT,MIMIQ_OUT sdk;
    class CIRQJSON,QISKITJSON,CUDAQJSON,MIMIQJSON fmt;
```

### Entrypoints in `qio`

| Source SDK | Write method | Output format |
|---|---|---|
| Cirq | `QuantumProgramResult.from_cirq_result` | `CIRQ_RESULT_JSON_V1` |
| Qiskit | `QuantumProgramResult.from_qiskit_result` | `QISKIT_RESULT_JSON_V1` |
| CUDA Quantum | `QuantumProgramResult.from_cudaq_sample_result` | `CUDAQ_SAMPLE_RESULT_JSON_V1` |
| MIMIQ | `QuantumProgramResult.from_mimiq_qcsr` | `MIMIQ_QCSR_JSON_V1` |

| Target SDK | Read method | Supported input formats |
|---|---|---|
| Cirq | `QuantumProgramResult.to_cirq_result` | `CIRQ_RESULT_JSON_V1`, `QISKIT_RESULT_JSON_V1`, `MIMIQ_QCSR_JSON_V1` |
| Qiskit | `QuantumProgramResult.to_qiskit_result` | `QISKIT_RESULT_JSON_V1`, `CIRQ_RESULT_JSON_V1`, `CUDAQ_SAMPLE_RESULT_JSON_V1`, `MIMIQ_QCSR_JSON_V1` |
| CUDA Quantum | `QuantumProgramResult.to_cudaq_sample_result` | `CUDAQ_SAMPLE_RESULT_JSON_V1` |
| MIMIQ | `QuantumProgramResult.to_mimiq_qcsr` | `MIMIQ_QCSR_JSON_V1`, `CIRQ_RESULT_JSON_V1`, `QISKIT_RESULT_JSON_V1` |
