# Test battery — circuit conversion (`tests/src/circuit_conversion`)

This folder tests qio's circuit converters (`qio/utils/conversion/program/*`) by
crossing every SDK and every serialization format through the intermediate
`QuantumProgram`. It is the counterpart of `tests/src/result_conversion`,
applied to circuits.

## Principles

- **Edges "SDK -> format -> SDK"** : a conversion is an arc
  `ReferenceCircuit -> input SDK -> qio converter -> format -> qio converter ->
  target SDK`, followed by a check. Each edge is one declarative entry
  (`add(...)`) in `build_edges()`.
- **Input circuits are static and trusted** : each `ReferenceCircuit` is built
  once (literal gate lists, known `n_qubits`, expected counts for deterministic
  circuits) and exposed in every SDK by the `.cirq()`, `.qiskit()`, `.qasm2()`,
  `.qasm3()`, `.cudaq_kernel()` builders. They are not re-checked: only the
  *intermediate* and *final* objects are verified.
- **Exact-first checks / declared losses** : every check compares the converted
  objects to the reference. Any deviation is classified into a loss category;
  a loss *not declared* on the edge fails the test (regression), a *declared*
  one (`known_losses`) is tolerated and recorded in the session report
  (`loss_report()` + terminal summary in `conftest.py`).
- **Equivalence oracle** : each edge closes with an oracle that guarantees the
  circuit semantics independently of the format: `unitary` (default), `counts`
  (CUDA-Q execution), or `structural` (no numeric check, e.g. unbound-
  parameter circuits).

## Files

| File | Role |
|---|---|
| `reference_circuits.py` | Static `ReferenceCircuit` fixtures + SDK builders. |
| `circuit_testing.py` | Per-SDK/format checks, operation-chain normalization, loss tracking, oracles, registries `build_edges()` / `UNSUPPORTED_CONVERSIONS`. |
| `conftest.py` | `pytest_terminal_summary`: prints the observed information losses + the `CIRCUIT_FEATURE_COVERAGE` matrix. |
| `test_circuit_conversions.py` | Parametrized battery: edges × circuits × compressions. |

## Flow of one edge

```
input_fn(circuit) ─► write (QuantumProgram.from_<sdk>_circuit,
                            dest_format + compression)
                        ├─ intermediate checks : check_program(fmt)
                        └─ content check : check_qasm(2|3) / check_cirq_json
                   ─► read (to_<sdk>_circuit + check_<sdk>)
                   ─► oracle (unitary / counts / structural)
```

- Every edge is generated for both compressions `NONE` and `ZLIB_BASE64_V1`;
  the serialization is decompressed before analysis.
- Every *checker* reduces any object (format content or final SDK circuit) to
  the canonical `(gate name, params, qubit indices)` form and compares it to
  `expected.gates`. Names, indices and the measurement multiset must match
  exactly; rotation parameters are compared exactly, with the
  `rotation_precision` loss tolerated on the paths serialized by Cirq
  (`cirq.to_qasm` truncates angles to ~10 significant digits).
- The `unitary` oracle compares the final circuit's matrix (cirq or qiskit) to
  the reference's, up to a global phase and with bit-order reversal.
- CUDA-Q kernels cannot be introspected without executing them: the
  `add_cudaq_loop` loops verify the information *around* the leg (input
  serialization exactly, then the re-emitted QASM2 via
  `check_qasm_cudaq_documented`, gate-basis rewrites being *recorded* as the
  `cudaq_gate_decomposition` loss), and the `unitary` oracle closes every loop.

## Edge ids and conversion graph

Current ids are of the form `cirq->qasm2->cirq`, `qiskit->qasm3->cudaq`,
`qiskit->qasm2->cudaq->qasm2->cirq`, etc.

- QASM2/QASM3: `cirq`/`qiskit` → `qasm2`/`qasm3` → `cirq`/`qiskit`/`mimiq`/`cudaq`.
- CirqJSON: `cirq`/`qiskit` → (only read back as cirq).
- No-execution CUDA-Q loops:
  `{cirq,qiskit}->qasm{2,3}->cudaq->qasm2->{cirq,qiskit}`.
- *Unsupported* cases (e.g. `from_cudaq_kernel -> CIRQJSON`,
  `to_mimiq_circuit on QASM_V3`) live in `UNSUPPORTED_CONVERSIONS`: every
  `(id, callable)` entry must raise an `Exception`.

### Write-side serialization formats

| Format | Content check | Read-back |
|---|---|---|
| `QASM_V2` | `check_qasm(2)` | `to_qasm2_circuit` |
| `QASM_V3` | `check_qasm(3)` | — |
| `CIRQ_CIRCUIT_JSON_V1` | `check_cirq_json` | `to_cirq_circuit` |

Supported gate set: `h, x, y, z, s, t, sdg, tdg, rx, ry, rz, cx, cz, measure`.
The parameterized gates are `rx/ry/rz`; a symbolic parameter is encoded as the
string marker `"~<name>"` (see the `parametrized` circuit).

## How to add a circuit

1. Add an entry to `REFERENCE_CIRCUITS` in `reference_circuits.py`:
   ```python
   ReferenceCircuit(
       name="my_circuit",
       n_qubits=2,
       gates=_ops(
           ("h", (), (0,)),
           ("cx", (), (0, 1)),
           ("measure", (), (0,)),
           ("measure", (), (1,)),
       ),
       expected_counts={"00": None, "11": None},
       description="Demonstration circuit.",
   ),
   ```
2. Everything else (builders, qubit indices, counts) is derived automatically.
3. Optionally strengthen the circuit-level information: `global_phase`,
   `metadata` (qiskit), `qubit_scheme="named"/"grid"` (cirq topology),
   `oracle="structural"` and `supported_edge_ids=(...)` for circuits whose
   unitary cannot be computed or that only run on a subset of formats
   (e.g. symbolic parameters).

## How to add a conversion (an edge)

- **Regular edge**: one `add(...)` call in `build_edges()`:
  ```python
  add(
      "qiskit->qasm2->mimiq",          # id ("src->fmt->tgt" shape)
      "qiskit",                        # input_kind (INPUT_FNS)
      producer,                        # QuantumProgram.from_qiskit_circuit
      Serialization.QASM_V2,           # written format
      QuantumProgram.to_mimiq_circuit, # reader (None if no read-back)
      check_mimiq,                     # check of the target SDK
      oracle="structural",
      known_losses=_info_and(),
  )
  ```
  `add` builds the intermediate checks (`check_program(fmt)` +
  `_format_content_checks(fmt)`) automatically.
- **CUDA-Q edge**: `add_cudaq_loop(base_id, input_kind, producer, write_fmt,
  known_losses=...)` generates both terminal legs
  `...->qasm2->{cirq,qiskit}`.
- Declare losses: use `_info_and(...)` (QASM-inherent circuit-info losses) +
  `LOSS_ROTATION_PRECISION` (Cirq-serialized paths) +
  `LOSS_CUDAQ_GATE_DECOMPOSITION` (CUDA-Q paths).
- New gate in an existing circuit: add it to `_SUPPORTED_GATES` and to
  `_SINGLE_QUBIT_GATES`/`_TWO_QUBIT_GATES`/`_PARAM_GATES` as appropriate, to
  the `_CIRQ_GATE_NAMES`/`_MIMIQ_GATE_NAMES` tables and to every
  `ReferenceCircuit` builder (the four SDK builders plus QASM2/QASM3/CUDA-Q).

## How to add an SDK

- **Input side**: add `kind: lambda c: c.<builder>()` to `INPUT_FNS`.
- **Read-back side**: add a `read` branch in an `add(...)` somewhere in
  `build_edges()` with the `to_<sdk>_circuit` function (if the converter
  exists) and a `check_<sdk>`. Implement `check_<sdk>` using the
  `_<sdk>_operations()` normalization, and extend `_to_unitary` if the
  `unitary` oracle must cover it.
- qio converters are lazy, so the SDK is only imported inside the checks.

## Running

The `qio` package must be importable (it is a source checkout; the test
helpers do not install it). From the repo root:

```bash
python -m pytest tests/src/circuit_conversion/ -s --showprogress -vv
```
