# Test battery — circuit conversion (`tests/src/circuit_conversion`)

This folder tests qio's circuit converters (`qio/utils/conversion/program/*`) by
crossing every SDK and every serialization format through the intermediate
`QuantumProgram`. It is the counterpart of `tests/src/result_conversion`,
applied to circuits.

## Principles

- **Programming paradigms** : the battery is organized by *paradigm* - the
  gate-based qubit circuits (`GateParadigm`), the analog pulse schedules
  (`AnalogParadigm`) and the mode-based photonic circuits (`PhotonicParadigm`).
  The abstract base `Paradigm` (in `circuit_testing.py`) hosts the generic
  machinery (conversion pipeline, oracles, loss report, run filter); each
  concrete paradigm contributes its own reference circuits, edge builder,
  SDK checks, negative cases and feature-coverage rows. `circuit_testing` then
  aggregates them into the module-level registries (`build_edges`,
  `REFERENCE_CIRCUITS`, `UNSUPPORTED_CONVERSIONS`, `CIRCUIT_FEATURE_COVERAGE`).
- **Edges "SDK -> format -> SDK"** : a conversion is an arc
  `ReferenceCircuit -> input SDK -> qio converter -> format -> qio converter ->
  target SDK`, followed by a check. Each edge is one declarative entry
  (`add(...)` in a paradigm's `build_edges()`).
- **Input circuits are static and trusted** : each `ReferenceCircuit` wraps a
  single paradigm-specific `spec` (`GateSpec` / `AnalogSpec` / `PhotonicSpec`)
  described once, and is exposed in every SDK by the spec's builders
  (`.cirq()`, `.qiskit()`, `.qasm2()`, `.qasm3()`, `.cudaq_kernel()`,
  `.pulser()`, `.perceval()`). Circuits are not re-checked: only the
  *intermediate* and *final* objects are verified.
- **Exact-first checks / declared losses** : every check compares the converted
  objects to the reference. Any deviation is classified into a loss category;
  a loss *not declared* on the edge fails the test (regression), a *declared*
  one (`known_losses`) is tolerated and recorded in the session report
  (`loss_report()` + terminal summary in `conftest.py`).
- **Equivalence oracle** : each edge closes with an oracle that guarantees the
  circuit semantics independently of the format: `unitary` (default), `counts`
  (CUDA-Q execution), `structural` (no numeric check, e.g. unbound-parameter
  circuits or analog sequences) or `photonic` (exact post-selected SLOS
  distribution, registered by `PhotonicParadigm` in `EXTRA_ORACLES`).

## Files

| File | Role |
|---|---|
| `reference_circuits.py` | Specs (`GateSpec`/`AnalogSpec`/`PhotonicSpec`) + the `ReferenceCircuit` envelope and its SDK builders. |
| `circuit_testing.py` | Generic `Paradigm` base: pipeline (`convert`/`run_path`), `ConversionEdge`/`Step`, loss tracking, oracles, `Paradigm.circuit_runnable`, and the aggregated facade registries. |
| `gate_paradigm.py` | `GateParadigm`: the gate reference circuits + every gate SDK check (Cirq/Qiskit/OpenQASM/MIMIQ/CirqJSON + no-execution CUDA-Q loops) + edge builder + negatives. |
| `analog_paradigm.py` | `AnalogParadigm`: the analog circuits + `check_pulser`/`check_pulser_json` + the `pulser->pulserjson->pulser` edge. |
| `photonic_paradigm.py` | `PhotonicParadigm`: the photonic circuits + `check_perceval`/`check_perceval_json` + the `perceval->percevaljson->perceval` edge + the `"photonic"` SLOS oracle. |
| `conftest.py` | `pytest_terminal_summary`: prints the observed information losses + the `CIRCUIT_FEATURE_COVERAGE` matrix. |
| `test_circuit_conversions.py` | Parametrized battery: edges × circuits × compressions. |

## Flow of one edge

```
input_fn(circuit) ─► write (QuantumProgram.from_<sdk>_circuit,
                            dest_format + compression)
                        ├─ intermediate checks : check_program(fmt)
                        └─ content check : check_qasm(2|3) / check_cirq_json
                                           / check_pulser_json / check_perceval_json
                   ─► read (to_<sdk>_circuit + check_<sdk>)
                   ─► oracle (unitary / counts / structural / photonic)
```

- Every edge is generated for both compressions `NONE` and `ZLIB_BASE64_V1`;
  the serialization is decompressed before analysis.
- Every *gate* checker reduces any object (format content or final SDK circuit)
  to the canonical `(gate name, params, qubit indices)` form and compares it to
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
- **CUDA-Q leg gate limitations** : CUDA-Q's OpenQASM 2 printer (`getASM`)
  cannot emit a call to a sub-kernel carrying float parameters
  (`failed to emit parameters`; reproduced on the cp310 build of
  `cuda-quantum-cu12 0.12.0.post1`, i.e. under Python 3.10 as used by CI).
  `qasm3_to_cudaq` therefore inlines the standard single-qubit parameterized
  gates (`rx/ry/rz/u3`) directly on the main kernel (`rx(angle, qubit)`), which
  emits fine — this is transparent for the battery. **Controlled rotations**
  (`crx/cry/crz`) would hit the same printer bug on the CUDA-Q leg, so they are
  **not supported there** and are deliberately left out of the reference gate
  set (`_TWO_QUBIT_GATES` restricts two-qubit gates to `cx`/`cz`).

- Perceval has no unitary either: the `"photonic"` oracle simulates the final
  circuit with Perceval's SLOS backend on the spec's Fock input state (with the
  heralded CNOT's ancilla seeded), post-selects the heralds, and requires the
  exact same output distribution as the statically built reference circuit.
  `check_perceval`/`check_perceval_json` additionally assert the canonical
  serialization round-trips byte-exactly (lossless, like Pulser).

## Edge ids and conversion graph

Current ids are of the form `cirq->qasm2->cirq`, `qiskit->qasm3->cudaq`,
`qiskit->qasm2->cudaq->qasm2->cirq`, etc.

- QASM2/QASM3: `cirq`/`qiskit` → `qasm2`/`qasm3` → `cirq`/`qiskit`/`mimiq`/`cudaq`.
- CirqJSON: `cirq`/`qiskit` → (only read back as cirq).
- No-execution CUDA-Q loops:
  `{cirq,qiskit}->qasm{2,3}->cudaq->qasm2->{cirq,qiskit}`.
- Analog: `pulser->pulserjson->pulser` (analog-paradigm circuits only).
- Photonic: `perceval->percevaljson->perceval` (photonic-paradigm circuits
  only) - closed self-round-trip, like the analog one: qio only maps the
  Perceval serialization to/from `perceval.Circuit`. Photonic *experiments*
  (circuit + Fock input state + herald post-selection) round trip through
  `perceval_experiment->percevaljson->perceval_experiment` (the same archive,
  `PERCEVAL_EXPERIMENT_JSON_V1`).
- *Unsupported* cases (e.g. `from_cudaq_kernel -> CIRQJSON`,
  `to_mimiq_circuit on QASM_V3`, `to_cirq_circuit on PERCEVALJSON`) live in
  each paradigm's `unsupported_conversions` (aggregated into
  `UNSUPPORTED_CONVERSIONS`): every `(id, callable)` entry must raise an
  `Exception`.

### Write-side serialization formats

| Format | Content check | Read-back |
|---|---|---|
| `QASM_V2` | `check_qasm(2)` | `to_qasm2_circuit` |
| `QASM_V3` | `check_qasm(3)` | — |
| `CIRQ_CIRCUIT_JSON_V1` | `check_cirq_json` | `to_cirq_circuit` |
| `PULSER_SEQUENCE_JSON_V1` | `check_pulser_json` | `to_pulser_sequence` |
| `PERCEVAL_CIRCUIT_JSON_V1` | `check_perceval_json` | `to_perceval_circuit` |
| `PERCEVAL_EXPERIMENT_JSON_V1` | `check_perceval_experiment_json` | `to_perceval_experiment` |

Supported gate set: `h, x, y, z, s, t, sdg, tdg, rx, ry, rz, cx, cz, measure`.
The parameterized gates are `rx/ry/rz`; a symbolic parameter is encoded as the
string marker `"~<name>"` (see the `parametrized` circuit). Controlled rotations
(`crx/cry/crz`) are **excluded** from the set: the CUDA-Q leg cannot re-emit
them (see the *CUDA-Q leg gate limitations* note above).

## How to add a circuit

Circuits live in the paradigm file that owns them (`gate_paradigm.py`,
`analog_paradigm.py`, `photonic_paradigm.py`) as `ReferenceCircuit(<spec>)`
entries in that paradigm's `circuits` tuple.

### Gate-based circuit (`GateSpec`)

1. Add an entry to `GateParadigm.circuits` in `gate_paradigm.py`:
   ```python
   ReferenceCircuit(
       GateSpec(
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
   ),
   ```
2. Everything else (builders, qubit indices, counts) is derived automatically.
3. Optionally strengthen the circuit-level information: `global_phase`,
   `metadata` (qiskit), `qubit_scheme="named"/"grid"` (cirq topology),
   `oracle="structural"` and `supported_edge_ids=(...)` for circuits whose
   unitary cannot be computed or that only run on a subset of formats
   (e.g. symbolic parameters).

### Analog circuit (`AnalogSpec`)

Analog schedules are not gate lists: describe them with an `AnalogSpec`
(atom coordinates, channels, pulses) and let `AnalogSpec.pulser()` build the
Pulser `Sequence`:

```python
ReferenceCircuit(
    AnalogSpec(
        name="analog_global",
        register_coords=[(0, 0), (0, 10)],
        channels={"rydberg": "rydberg_global"},
        pulses=[
            PulseSpec(channel="rydberg", kind="constant",
                      duration=100, amplitude=1.0, detuning=0.1),
        ],
        measure=True,
        description="...",
    ),
),
```

Analog circuits only run on the `pulser->pulserjson->pulser` edge (the edge
paradigm filter in `ReferenceCircuit.supports_edge` does this automatically).

**Device handling**: the Pulser abstract repr (`Sequence.to_abstract_repr`) is
device-independent; `check_pulser` asserts the device `==` the reference's and
declares the `analog_device` loss as a *guard* on the pulser edge. With
pulser 1.9.1 and the reference devices the device round-trips exactly (so no
loss is recorded); if a future version drops/rewrites it, the loss would be
recorded (tolerated) and surfaced by the session report instead of failing.

### Photonic circuit (`PhotonicSpec`)

Photonic circuits are mode-based: describe the native blocks (beam splitters,
phase shifters, the *heralded CNOT*), the Fock input state (ancilla photons
included for the CNOT) and the herald post-selection with a `PhotonicSpec`:

```python
ReferenceCircuit(
    PhotonicSpec(
        name="ph_bell",
        n_qubits=2,
        gates=[
            PhotonicGate("bs", modes=(0, 1)),            # Hadamard on q0
            PhotonicGate("heralded_cnot", qubits=(0, 1)),
        ],
        input_state=(1, 0, 1, 0, 1, 1),   # |0>|0> + ancilla |1,1> on modes 4,5
        measure_qubits=(0, 1),
        heralds={4: 1, 5: 1},             # post-select both heralds to 1
        description="...",
    ),
),
```

Photonic circuits only run on the `perceval->percevaljson->perceval` edge.

## How to add a conversion (an edge)

Edges are built per paradigm, in that paradigm's `build_edges()`.

- **Regular gate edge**: one `add(...)` call in `GateParadigm.build_edges()`:
  ```python
  add(
      "qiskit->qasm2->mimiq",          # id ("src->fmt->tgt" shape)
      "qiskit",                        # input_kind (paradigm input_fns)
      producer,                        # QuantumProgram.from_qiskit_circuit
      Serialization.QASM_V2,           # written format
      QuantumProgram.to_mimiq_circuit, # reader (None if no read-back)
      check_mimiq,                     # check of the target SDK
      oracle="structural",
      known_losses=_info_and(),
  )
  ```
  `add` builds the intermediate checks (`check_program(fmt)` +
  `content_checks(fmt)`) automatically.
- **CUDA-Q edge**: `add_cudaq_loop(base_id, input_kind, producer, write_fmt,
  known_losses=...)` generates both terminal legs
  `...->qasm2->{cirq,qiskit}`.
- **Analog / photonic edges**: closed self-round-trips built directly in
  `AnalogParadigm.build_edges` / `PhotonicParadigm.build_edges` (single
  `ConversionEdge` with `oracle="structural"` / `"photonic"`).
- Declare losses: use `_info_and(...)` (QASM-inherent circuit-info losses) +
  `LOSS_ROTATION_PRECISION` (Cirq-serialized paths) +
  `LOSS_CUDAQ_GATE_DECOMPOSITION` (CUDA-Q paths).
- New gate in an existing circuit: add it to `_SUPPORTED_GATES` and to
  `_SINGLE_QUBIT_GATES`/`_TWO_QUBIT_GATES`/`_PARAM_GATES` as appropriate, to
  the `_CIRQ_GATE_NAMES`/`_MIMIQ_GATE_NAMES` tables and to every spec builder
  (the four SDK builders plus QASM2/QASM3/CUDA-Q).

## How to add a new paradigm

1. Create the paradigm file (e.g. `my_paradigm.py`) and subclass `Paradigm`:
   implement `build_edges(compression)`, set `name`, `circuits`, `input_fns`,
   `content_checks(fmt)`, `unsupported_conversions` and `feature_coverage`.
   Add a paradigm-specific oracle to `EXTRA_ORACLES` if needed.
2. Register it in `circuit_testing.py`: import the class and add an instance in
   the bottom registry section; the facade (`build_edges`,
   `REFERENCE_CIRCUITS`, `UNSUPPORTED_CONVERSIONS`,
   `CIRCUIT_FEATURE_COVERAGE`) aggregates it automatically.
3. Add the SDK to `tests/requirements.txt` (kept unpinned, like `pulser`).

## How to add an SDK to an existing paradigm

- **Input side**: add `kind: lambda c: c.<builder>()` to the paradigm's
  `input_fns` (e.g. `{"perceval": lambda c: c.perceval()}`).
- **Read-back side**: add a `read` branch in the paradigm's `build_edges()`
  with the `to_<sdk>_circuit` function (if the converter exists) and a
  `check_<sdk>`. Implement `check_<sdk>` using the `_<sdk>_operations()`
  normalization, and extend `_to_unitary` if the `unitary` oracle must cover it.
- qio converters are lazy, so the SDK is only imported inside the checks.

## Perceval / MIMIQ protobuf conflict

`mimiqcircuits` and Perceval (via `exqalibur`) both call for a proto file named
`circuit.proto` into protobuf's *global* descriptor pool. Whichever SDK is
imported second in a Python process fails with
`TypeError: duplicate file name circuit.proto` - the two cannot coexist in one
process with the pinned versions.

The photonic battery therefore runs in its **own pytest invocation**, in which
mimiq is never imported (all SDK imports in the battery are lazy, inside the
checks - see the *conventions* note above):

- the main invocation runs everything selected by `-k "not perceval"` (MIMIQ
  and the other SDKs; Perceval is never imported there);
- a second invocation runs only `-k "perceval"` tests, importing Perceval
  directly in-process.

`tests/Makefile`'s `test` / `test-circuit-conv` targets execute both
invocations; `test-photonic` runs the Perceval one alone.

## Running

The `qio` package must be importable (it is a source checkout; the test
helpers do not install it). From the repo root:

```bash
# full battery (two invocations, MIMIQ and Perceval in separate processes)
make -C tests test-circuit-conv

# only the MIMIQ/analog/gate side of the battery
python -m pytest tests/src/circuit_conversion/ -s --showprogress -vv -k "not perceval"

# only the photonic (Perceval) side - no mimiq in this process
python -m pytest tests/src/circuit_conversion/ -s --showprogress -vv -k "perceval"
```

The CUDA-Q edges require `cudaq`, which is not installable in every
environment; to run the battery without it (locally):

```bash
python -m pytest tests/src/circuit_conversion/ -s --showprogress -vv -k "not cudaq and not perceval"
python -m pytest tests/src/circuit_conversion/ -s --showprogress -vv -k "perceval"
```
