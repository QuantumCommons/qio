# Test battery — result conversion (`tests/src/result_conversion`)

This folder tests qio's result converters
(`qio/utils/conversion/program_result/*`) by crossing every SDK and every
result format through the intermediate `QuantumProgramResult`. It is the
counterpart of `tests/src/circuit_conversion`, applied to measurement results.

## Principles

- **Lengthened edges** : every conversion starts from the *SDK object* and
  makes qio's `*_to_dict` converter an explicit first step. The full chain is:
  ```
  source SDK (object) -> <sdk>_to_dict (checked) -> from_<sdk>_result_dict
      (QuantumProgramResult, checked) -> to_<target>_result (checked)
  ```
  There is **no** hand-authored intermediate dict: the format is produced by
  qio's own converters (single source of truth).
- **Input results are static and trusted** : a `ReferenceResult` describes an
  *already executed* run as a counts histogram `{bitstring: count}` plus a shot
  count, exposed identically in every SDK by the `.cirq()`, `.qiskit()`,
  `.cudaq()`, `.mimiq()` builders. It is not re-checked.
- **Bit-exact comparison** : a fixed bitstring convention (character `k` is
  the measured value of qubit `k` on the source circuit) means every converter
  only re-serializes those bitstrings, so the counts histograms must match
  **exactly**. Measurement keys (`measurement_key`) and bit order (`bit_order`)
  are SDK conventions without information loss, since the histogram is
  preserved exactly.
- **Declared information losses** : on top of the counts, each edge probes the
  target object for execution metadata (backend, job ids, date, statevector,
  MIMIQ zstates/fidelities/avggateerrors/timings/amplitudes, CUDA-Q register).
  A deviation is classified into a loss; undeclared → failure (regression),
  declared (`known_losses`) → tolerated and recorded in the session report
  (`loss_report()` + terminal summary in `conftest.py`).

## Files

| File | Role |
|---|---|
| `reference_results.py` | Static `ReferenceResult` fixtures + SDK builders (with the full execution metadata). |
| `result_testing.py` | Per-SDK checks, metadata probes, loss tracking, registries `build_edges()` / `UNSUPPORTED_CONVERSIONS`. |
| `conftest.py` | `pytest_terminal_summary`: prints the observed information losses. |
| `test_result_conversions.py` | Parametrized battery: edges × results × compressions. |

## Edge ids (annotated "type -> format -> type")

Edge ids follow the annotated form:

```
qiskit.Result -> qiskit_result_json_v1 -> qiskit.Result
cirq.Result   -> cirq_result_json_v1   -> mimiqcircuits.QCSResults
```

| SDK | Displayed type |
|---|---|
| Cirq | `cirq.Result` |
| Qiskit | `qiskit.Result` |
| CUDA-Q | `cudaq.SampleResult` |
| MIMIQ | `mimiqcircuits.QCSResults` |

| Format | Label |
|---|---|
| `CIRQ_RESULT_JSON_V1` | `cirq_result_json_v1` |
| `QISKIT_RESULT_JSON_V1` | `qiskit_result_json_v1` |
| `CUDAQ_SAMPLE_RESULT_JSON_V1` | `cudaq_sample_result_json_v1` |
| `MIMIQ_QCSR_JSON_V1` | `mimiq_qcsr_json_v1` |

Current graph (read-back): `cirq_result_json_v1` → cirq/qiskit/mimiq ;
`qiskit_result_json_v1` → qiskit/cirq/mimiq ;
`cudaq_sample_result_json_v1` → cudaq/qiskit ;
`mimiq_qcsr_json_v1` → mimiq/cirq/qiskit.
Completed by `UNSUPPORTED_CONVERSIONS` for the forbidden arcs
(e.g. `to_cirq_result on cudaq_sample_result_json_v1`).

## Flow of one edge

```
input_fn(ref) ─► <sdk>_to_dict      (check_to_dict, validates qio's dict)
             ─► from_<sdk>_result_dict (check_program_result: format + keys)
             ─► to_<target>_result  (check_<target>: exact counts + probes)
```

- Generated for the compressions `NONE` and `ZLIB_BASE64_V1`.
- `check_<sdk>` reduces the final object to its histogram (compared bit-exact
  to the reference), then runs the metadata probes of the target SDK.
- CUDA-Q exception: there is no dedicated dict; `from_cudaq_sample_result`
  serializes internally (the wire format is already exercised by the cudaq
  read edges). Register names are read via `sample_result.serialize()`,
  skipping the synthetic `__global__` register (no bitstrings) that
  `deserialize` adds.

## Losses and JSON constraints

Constraints imposed by the intermediate (plain JSON/zlib):

| Field | Constraint | Consequence |
|---|---|---|
| cirq `params` | `cirq.ParamResolver` is not JSON-serializable | always `None` → `cirq_params` |
| qiskit `date` | `datetime` is not JSON-serializable | stored as an ISO string |
| qiskit `statevector` | numpy/complex is not JSON-safe | stored as real-valued amplitudes (list of floats) |
| mimiq `amplitudes` | `bitarray` keys + complex values are not JSON-serializable | kept out of the input fixtures; reconstructed in memory (qiskit statevector -> mimiq amplitudes) and checked on the output object |

Losses declared today (each category is described by `loss_description`):
`measurement_key`, `bit_order`, `cirq_params`, `qiskit_metadata`,
`qiskit_statevector`, `mimiq_metadata`, `mimiq_amplitudes`, `cudaq_register`.
They are declared through the `known_losses` of each `add(...)` call plus
`_SOURCE_STATIC_LOSSES` (`cirq_params` auto-joined on every cirq source).
Removing a category from an edge's `known_losses` fails the corresponding
probe (turning a documented loss into a regression that forces a converter
fix).

## How to add a result

Add an entry to `REFERENCE_RESULTS` in `reference_results.py`:

```python
ReferenceResult(
    name="bell2",
    n_qubits=2,
    shots=1000,
    counts={"00": 509, "11": 491},
    description="2-qubit Bell-state sampling.",
    statevector=[_RT2, 0.0, 0.0, _RT2],
    fidelities=[0.995],
    avggateerrors=[0.002],
    zstates=["00"],
    timings={"total": 0.0123, "apply": 0.0091},
),
```

- `sum(counts) == shots` and every bitstring has length `n_qubits`
  (validated in `__post_init__`).
- The matching SDK builder is automatic; fill the execution metadata so that
  the losses stay visible.

## How to add a conversion (an edge)

An edge is a **single declarative call** in `build_edges()`, exactly like the
circuit battery: `add(edge_id, input_kind, read_fn, read_check,
known_losses=())`. The write-side steps (SDK object -> `<sdk>_to_dict` checked
-> `from_<sdk>_result_dict` wrap checked) are derived automatically from the
`input_kind` (which produces the intermediate format), and the annotated
`edge_id` is written out in full.

Example — declaring a new arc `qiskit.Result -> mimiq_qcsr_json_v1 ->
qiskit.Result`:
```python
add(
    "qiskit.Result -> mimiq_qcsr_json_v1 -> qiskit.Result",
    "qiskit",
    QuantumProgramResult.to_qiskit_result,
    check_qiskit,
    (LOSS_QISKIT_METADATA, LOSS_QISKIT_STATEVECTOR),
)
```
- `known_losses` are the losses inherent to that path: any deviation in another
  category fails the test (regression). Per-source static losses (e.g.
  `cirq_params` for cirq inputs) are auto-joined by `add()` via
  `_SOURCE_STATIC_LOSSES`, no need to repeat them.
- `add()` composes the steps (write then read) and generates the arc for every
  result × compression. Nothing else to edit.

## How to add an SDK

- **Input side**:
  - a `ReferenceResult.<sdk>()` builder in `reference_results.py`,
  - `_input_fn` + `_INPUT_FORMAT[kind] = Serialization.<fmt>` (the unique
    format this source produces),
  - `_TO_DICT`/`_FROM_DICT` (if the SDK has a dict classmethod) and a branch in
    `check_to_dict` (or the direct `from_<sdk>_result_dict` path, as for
    CUDA-Q).
- **Read-back side**: call `add(...)` with `to_<sdk>_result` and
  `check_<sdk>`; implement `check_<sdk>` (exact counts + probes) and its
  `Probe`s (otherwise no metadata is verified).
- SDKs are imported lazily inside the checks; on the `qio` side converters are
  imported inside the methods, so no SDK is required at import time.

## Running

`result_testing.py` inserts the repo root at the front of `sys.path`, so the
battery runs from `tests/` without installing `qio`:

```bash
cd tests
python -m pytest src/result_conversion/ -s --showprogress -vv
```

The `test-result-conv` target in the `tests/` Makefile does exactly this.
Heavy SDKs (`qsimcirq`/`qiskit-aer`/`quantanium`) are not required here.
