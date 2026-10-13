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
  count, exposed identically in every SDK by the per-SDK adapters
  (`ResultSDK.build(reference)`). It is not re-checked.
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
| `reference_results.py` | Static `ReferenceResult` fixtures (pure data: counts + execution metadata, no SDK). |
| `result_sdks/` | One `ResultSDK` subclass per SDK (input builder, format check, probes, dict converter, wrap/read classmethods, inherent losses). |
| `result_testing.py` | Driver: loss tracking, `ResultSDK` base class, `build_edges()` / `UNSUPPORTED_CONVERSIONS` registries derived from `EDGES`. |
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
| Perceval | `pcvl.BSSamples` |
| Pulser | `pulser.backend.Results` |

| Format | Label |
|---|---|
| `CIRQ_RESULT_JSON_V1` | `cirq_result_json_v1` |
| `QISKIT_RESULT_JSON_V1` | `qiskit_result_json_v1` |
| `CUDAQ_SAMPLE_RESULT_JSON_V1` | `cudaq_sample_result_json_v1` |
| `MIMIQ_QCSR_JSON_V1` | `mimiq_qcsr_json_v1` |
| `PERCEVAL_RESULT_JSON_V1` | `perceval_result_json_v1` |
| `PULSER_RESULT_JSON_V1` | `pulser_result_json_v1` |

Current graph (read-back): `cirq_result_json_v1` → cirq/qiskit/mimiq ;
`qiskit_result_json_v1` → qiskit/cirq/mimiq ;
`cudaq_sample_result_json_v1` → cudaq/qiskit ;
`mimiq_qcsr_json_v1` → mimiq/cirq/qiskit ;
`perceval_result_json_v1` → perceval (self, the only reader today) ;
`pulser_result_json_v1` → pulser (self, the only reader today).
Completed by `UNSUPPORTED_CONVERSIONS` - the **complement** of the declared
read graph - for the forbidden arcs (e.g. `to_cirq_result on
cudaq_sample_result_json_v1`, `to_perceval_result on perceval_result_json_v1`).

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
They are declared on the adapters: `target_losses` of the arriving SDK (e.g.
`cirq_params` auto-joined to every edge departing from cirq) plus removal of a
category from an edge's `known_losses` fails the corresponding probe (turning a
documented loss into a regression that forces a converter fix).

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
- The SDK objects are built automatically by the adapters; fill the execution
  metadata so that the losses stay visible.

## How to add a conversion (an edge)

An edge is declared by adding one read target to `EDGES` in
`result_testing.py`:

```python
# result_testing.py
def _edge_rows():
    ...
    return (
        (cirq,  (cirq, qiskit, mimiq)),      # cirq_result_json_v1 -> cirq/qiskit/mimiq
        ...
    )
```

`build_edges()` expands `EDGES` into one `ConversionEdge` per target: the input
SDK produces its intermediate format (write steps: `<sdk>_to_dict` checked ->
`from_<sdk>_result_dict` wrap checked), each target becomes a read step
(`to_<target>_result` + its `check`), and `known_losses` = the target's
`target_losses` + the input's `static_source_losses`.
`UNSUPPORTED_CONVERSIONS` is then recomputed automatically as the complement of
the graph, so the negative cases never need editing by hand.

## How to add an SDK

An SDK is one class + one register line; nothing in the driver changes.

1. Create `result_sdks/<sdk>.py` with a `ResultSDK` subclass:
   - facts: `kind`, `display`, `fmt`/`fmt_label`, `dict_keys` (mandatory keys
     of the intermediate dict), `to_dict_module` (or `None` for the direct
     wrap path, as for CUDA-Q), `from_dict`/`wrap`/`read` classmethods,
     `target_losses`/`static_source_losses`, `probes`,
   - `build(reference)` stores the static SDK input object,
   - `check(result, expected)` validates the type, the bit-exact counts and the
     metadata probes.
2. Register it in the `result_sdks/__init__.py` `ALL_SDKS` tuple.
3. Add its read targets in the `EDGES` table in `result_testing.py`.

- All SDK imports stay lazy (inside `build`/`check`; the `*_to_dict` converters
  are resolved by `_LazyFn` on first call), so collection never imports an SDK.
- **Perceval (photonic) leg — separate pytest invocation**: mimiqcircuits and
  Perceval both register a `circuit.proto` in protobuf's global descriptor
  pool, so they cannot be imported in the same Python process (either order).
  The perceval leg (the `perceval.BSSamples -> perceval_result_json_v1 ->
  perceval.BSSamples` edge and every negative on/from `..._PERCEVAL...`) is
  therefore selected with `-k "perceval"` and runs in the perceval-only
  invocation; the main invocation uses `-k "not perceval"`. Both use the exact
  same node ids, which always contain `perceval` for this leg. The two
  negatives that would pull mimiqcircuits into that process
  (`to_mimiq_qcsr on PERCEVAL_RESULT_JSON_V1`,
  `to_perceval_result on MIMIQ_QCSR_JSON_V1`) are pinned last by
  `_to_negatives()`.

## Running

`result_testing.py` itself imports no SDK, and the battery needs the repo root
on `sys.path` when run from `tests/` (e.g. `PYTHONPATH=..` or installing the
wheel), so the Perceval leg can be deselected without pulling mimiqcircuits:

```bash
cd tests
PYTHONPATH=.. python -m pytest src/result_conversion/ -s --showprogress -vv -k "not perceval"
PYTHONPATH=.. python -m pytest src/result_conversion/ -s --showprogress -vv -k "perceval"
```

The `test-result-conv` target in the `tests/` Makefile runs exactly these two
invocations (mirroring the circuit battery). Heavy SDKs
(`qsimcirq`/`qiskit-aer`/`quantanium`) are not required here.
