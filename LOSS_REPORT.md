# qio conversion report — losses, limitations & unsupported conversions

> Auto-generated from the conversion test batteries (`loss_report()` + static registries). Do not edit by hand — regenerate with `make -C tests report` (`report-observed` adds the run columns). Auto-refreshed in CI on every merge to `main`.

## Circuit conversions (`QuantumProgram`)

Each row = one conversion edge × one declared information loss. A loss not declared on an edge is checked `exact` (any deviation fails the test). Edges without declared losses are lossless round trips (listed after each paradigm's table).

### gate paradigm

| Edge | Loss | Expected | Actual | Explanation | Observed (run) | Expected (run) |
|---|---|---|---|---|---|---|
| `cirq->qasm2->cirq` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `cirq->qasm2->cirq` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `cirq->qasm2->cirq` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `cirq->qasm2->cirq` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `cirq->qasm2->cirq` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `cirq->qasm2->cirq` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `cirq->qasm2->cudaq->qasm2->cirq` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `cirq->qasm2->cudaq->qasm2->cirq` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `cirq->qasm2->cudaq->qasm2->cirq` | `cudaq_gate_decomposition` | same gate list | re-expressed in CUDA-Q's basis (cz -> h+cx+h, s/t -> rz/rx), one register per qubit | the end-of-chain unitary is checked instead of the operation chain | - | - |
| `cirq->qasm2->cudaq->qasm2->cirq` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `cirq->qasm2->cudaq->qasm2->cirq` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `cirq->qasm2->cudaq->qasm2->cirq` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `cirq->qasm2->cudaq->qasm2->cirq` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `cirq->qasm2->cudaq->qasm2->qiskit` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `cirq->qasm2->cudaq->qasm2->qiskit` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `cirq->qasm2->cudaq->qasm2->qiskit` | `cudaq_gate_decomposition` | same gate list | re-expressed in CUDA-Q's basis (cz -> h+cx+h, s/t -> rz/rx), one register per qubit | the end-of-chain unitary is checked instead of the operation chain | - | - |
| `cirq->qasm2->cudaq->qasm2->qiskit` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `cirq->qasm2->cudaq->qasm2->qiskit` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `cirq->qasm2->cudaq->qasm2->qiskit` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `cirq->qasm2->cudaq->qasm2->qiskit` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `cirq->qasm2->mimiq` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `cirq->qasm2->qiskit` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `cirq->qasm2->qiskit` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `cirq->qasm2->qiskit` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `cirq->qasm2->qiskit` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `cirq->qasm2->qiskit` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `cirq->qasm2->qiskit` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `cirq->qasm3->cirq` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `cirq->qasm3->cirq` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `cirq->qasm3->cirq` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `cirq->qasm3->cirq` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `cirq->qasm3->cirq` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `cirq->qasm3->cirq` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `cirq->qasm3->cudaq` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `cirq->qasm3->cudaq->qasm2->cirq` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `cirq->qasm3->cudaq->qasm2->cirq` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `cirq->qasm3->cudaq->qasm2->cirq` | `cudaq_gate_decomposition` | same gate list | re-expressed in CUDA-Q's basis (cz -> h+cx+h, s/t -> rz/rx), one register per qubit | the end-of-chain unitary is checked instead of the operation chain | - | - |
| `cirq->qasm3->cudaq->qasm2->cirq` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `cirq->qasm3->cudaq->qasm2->cirq` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `cirq->qasm3->cudaq->qasm2->cirq` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `cirq->qasm3->cudaq->qasm2->cirq` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `cirq->qasm3->cudaq->qasm2->qiskit` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `cirq->qasm3->cudaq->qasm2->qiskit` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `cirq->qasm3->cudaq->qasm2->qiskit` | `cudaq_gate_decomposition` | same gate list | re-expressed in CUDA-Q's basis (cz -> h+cx+h, s/t -> rz/rx), one register per qubit | the end-of-chain unitary is checked instead of the operation chain | - | - |
| `cirq->qasm3->cudaq->qasm2->qiskit` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `cirq->qasm3->cudaq->qasm2->qiskit` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `cirq->qasm3->cudaq->qasm2->qiskit` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `cirq->qasm3->cudaq->qasm2->qiskit` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `cirq->qasm3->mimiq` | `qasm_control_flow` | classical control flow preserved | not re-expressed (if/for/while/switch blocks) | qasm2<->qasm3 normalization is straight-line only | - | - |
| `cirq->qasm3->mimiq` | `qasm_gphase` | gphase() kept | silently dropped (leaves '// dropped') | QASM2 has no global-phase syntax (qasm3->mimiq path) | - | - |
| `cirq->qasm3->mimiq` | `qasm_stdlib_gate` | stdlib gates re-expressed in the target grammar | passed through unmapped (may not exist in target include) | normalization only renames u1/cu1 <-> p/cp | - | - |
| `cirq->qasm3->mimiq` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `cirq->qasm3->qiskit` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `cirq->qasm3->qiskit` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `cirq->qasm3->qiskit` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `cirq->qasm3->qiskit` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `cirq->qasm3->qiskit` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `cirq->qasm3->qiskit` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `cudaq->qasm3->cudaq` | `qasm_control_flow` | classical control flow preserved | not re-expressed (if/for/while/switch blocks) | qasm2<->qasm3 normalization is straight-line only | - | - |
| `cudaq->qasm3->cudaq` | `qasm_gphase` | gphase() kept | silently dropped (leaves '// dropped') | QASM2 has no global-phase syntax (qasm3->mimiq path) | - | - |
| `cudaq->qasm3->cudaq` | `qasm_stdlib_gate` | stdlib gates re-expressed in the target grammar | passed through unmapped (may not exist in target include) | normalization only renames u1/cu1 <-> p/cp | - | - |
| `mimiq->qasm2->cirq` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `mimiq->qasm2->cirq` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `mimiq->qasm2->cirq` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `mimiq->qasm2->cirq` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `mimiq->qasm2->cirq` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `mimiq->qasm2->qiskit` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `mimiq->qasm2->qiskit` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `mimiq->qasm2->qiskit` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `mimiq->qasm2->qiskit` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `mimiq->qasm2->qiskit` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `qiskit->qasm2->cirq` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `qiskit->qasm2->cirq` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `qiskit->qasm2->cirq` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `qiskit->qasm2->cirq` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `qiskit->qasm2->cirq` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `qiskit->qasm2->cudaq->qasm2->cirq` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `qiskit->qasm2->cudaq->qasm2->cirq` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `qiskit->qasm2->cudaq->qasm2->cirq` | `cudaq_gate_decomposition` | same gate list | re-expressed in CUDA-Q's basis (cz -> h+cx+h, s/t -> rz/rx), one register per qubit | the end-of-chain unitary is checked instead of the operation chain | - | - |
| `qiskit->qasm2->cudaq->qasm2->cirq` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `qiskit->qasm2->cudaq->qasm2->cirq` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `qiskit->qasm2->cudaq->qasm2->cirq` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `qiskit->qasm2->cudaq->qasm2->cirq` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `qiskit->qasm2->cudaq->qasm2->qiskit` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `qiskit->qasm2->cudaq->qasm2->qiskit` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `qiskit->qasm2->cudaq->qasm2->qiskit` | `cudaq_gate_decomposition` | same gate list | re-expressed in CUDA-Q's basis (cz -> h+cx+h, s/t -> rz/rx), one register per qubit | the end-of-chain unitary is checked instead of the operation chain | - | - |
| `qiskit->qasm2->cudaq->qasm2->qiskit` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `qiskit->qasm2->cudaq->qasm2->qiskit` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `qiskit->qasm2->cudaq->qasm2->qiskit` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `qiskit->qasm2->cudaq->qasm2->qiskit` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `qiskit->qasm2->qasm3->qiskit` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `qiskit->qasm2->qasm3->qiskit` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `qiskit->qasm2->qasm3->qiskit` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `qiskit->qasm2->qasm3->qiskit` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `qiskit->qasm2->qasm3->qiskit` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `qiskit->qasm2->qiskit` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `qiskit->qasm2->qiskit` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `qiskit->qasm2->qiskit` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `qiskit->qasm2->qiskit` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `qiskit->qasm2->qiskit` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `qiskit->qasm3->cirq` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `qiskit->qasm3->cirq` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `qiskit->qasm3->cirq` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `qiskit->qasm3->cirq` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `qiskit->qasm3->cirq` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `qiskit->qasm3->cudaq->qasm2->cirq` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `qiskit->qasm3->cudaq->qasm2->cirq` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `qiskit->qasm3->cudaq->qasm2->cirq` | `cudaq_gate_decomposition` | same gate list | re-expressed in CUDA-Q's basis (cz -> h+cx+h, s/t -> rz/rx), one register per qubit | the end-of-chain unitary is checked instead of the operation chain | - | - |
| `qiskit->qasm3->cudaq->qasm2->cirq` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `qiskit->qasm3->cudaq->qasm2->cirq` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `qiskit->qasm3->cudaq->qasm2->cirq` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `qiskit->qasm3->cudaq->qasm2->cirq` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `qiskit->qasm3->cudaq->qasm2->qiskit` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `qiskit->qasm3->cudaq->qasm2->qiskit` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `qiskit->qasm3->cudaq->qasm2->qiskit` | `cudaq_gate_decomposition` | same gate list | re-expressed in CUDA-Q's basis (cz -> h+cx+h, s/t -> rz/rx), one register per qubit | the end-of-chain unitary is checked instead of the operation chain | - | - |
| `qiskit->qasm3->cudaq->qasm2->qiskit` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `qiskit->qasm3->cudaq->qasm2->qiskit` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `qiskit->qasm3->cudaq->qasm2->qiskit` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `qiskit->qasm3->cudaq->qasm2->qiskit` | `rotation_precision` | bit-exact rotation floats | angles truncated to ~10 significant digits | Cirq's QASM export serializes rotation parameters with limited precision | - | - |
| `qiskit->qasm3->mimiq` | `qasm_control_flow` | classical control flow preserved | not re-expressed (if/for/while/switch blocks) | qasm2<->qasm3 normalization is straight-line only | - | - |
| `qiskit->qasm3->mimiq` | `qasm_gphase` | gphase() kept | silently dropped (leaves '// dropped') | QASM2 has no global-phase syntax (qasm3->mimiq path) | - | - |
| `qiskit->qasm3->mimiq` | `qasm_stdlib_gate` | stdlib gates re-expressed in the target grammar | passed through unmapped (may not exist in target include) | normalization only renames u1/cu1 <-> p/cp | - | - |
| `qiskit->qasm3->qasm2->qiskit` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `qiskit->qasm3->qasm2->qiskit` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `qiskit->qasm3->qasm2->qiskit` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `qiskit->qasm3->qasm2->qiskit` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `qiskit->qasm3->qasm2->qiskit` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |
| `qiskit->qasm3->qiskit` | `circuit_metadata` | Qiskit circuit.metadata dict | dropped | no intermediate format carries it | - | - |
| `qiskit->qasm3->qiskit` | `circuit_name` | reference circuit name | regenerated (e.g. 'circuit-42') | OpenQASM carries no circuit name; Cirq has none | - | - |
| `qiskit->qasm3->qiskit` | `global_phase` | exact global phase | lost | not expressible in OpenQASM; the unitary oracle ignores phases | - | - |
| `qiskit->qasm3->qiskit` | `measurement_mapping` | exact qubit->clbit assignment / measurement keys | reordered or merged by QASM/CUDA-Q round trips | the classical-bit count stays a hard check | - | - |
| `qiskit->qasm3->qiskit` | `qubit_topology` | named/grid qubit labels & register names | flattened to linear q[i] | OpenQASM flattens topology; only CirqJSON preserves it | - | - |

*Lossless edges:* `cirq->cirqjson->cirq`, `cirq->qir`, `cirq->qir->cudaq`, `cudaq->qir`, `cudaq->qir->cudaq`, `qiskit->qasm2->cudaq`, `qiskit->qasm2->mimiq`, `qiskit->qasm3->cudaq`, `qiskit->qir`, `qiskit->qir->cudaq`.

### analog paradigm

| Edge | Loss | Expected | Actual | Explanation | Observed (run) | Expected (run) |
|---|---|---|---|---|---|---|
| `pulser->pulserjson->pulser` | `analog_device` | exact Pulser device object | none (guard; never recorded today) | abstract repr is device-independent; guards a future pulser dropping the device | - | - |

### photonic paradigm

| Edge | Loss | Expected | Actual | Explanation | Observed (run) | Expected (run) |
|---|---|---|---|---|---|---|

*Lossless edges:* `perceval->percevaljson->perceval`, `perceval_experiment->percevaljson->perceval_experiment`.

### Unsupported conversions

Each entry must raise (`Exception`), enforced by the negative tests.

| Unsupported conversion | Paradigm | Behavior |
|---|---|---|
| `from_pulser_sequence -> QASM_V2` | analog | raises (`Exception`) |
| `from_pulser_sequence on bare-variable pulse` | analog | raises (`Exception`) |
| `to_cirq_circuit on PULSERJSON` | analog | raises (`Exception`) |
| `to_mimiq_circuit on PULSERJSON` | analog | raises (`Exception`) |
| `from_cirq_circuit -> QASM_V1` | gate | raises (`Exception`) |
| `from_cudaq_kernel -> CIRQJSON` | gate | raises (`Exception`) |
| `from_qiskit_circuit (unbound params) -> QASM_V2` | gate | raises (`Exception`) |
| `from_qiskit_circuit -> QASM_V1` | gate | raises (`Exception`) |
| `to_cudaq_kernel on CIRQJSON` | gate | raises (`Exception`) |
| `to_mimiq_circuit on CIRQJSON` | gate | raises (`Exception`) |
| `to_pulser_sequence on QASM_V2` | gate | raises (`Exception`) |
| `to_qasm2_circuit on CIRQJSON` | gate | raises (`Exception`) |
| `to_qasm3_circuit on CIRQJSON` | gate | raises (`Exception`) |
| `from_perceval_circuit -> QASM_V2` | photonic | raises (`Exception`) |
| `from_perceval_experiment -> QASM_V2` | photonic | raises (`Exception`) |
| `to_cirq_circuit on PERCEVALJSON` | photonic | raises (`Exception`) |
| `to_cirq_circuit on PERCEVAL_EXPERIMENTJSON` | photonic | raises (`Exception`) |
| `to_cudaq_kernel on PERCEVALJSON` | photonic | raises (`Exception`) |
| `to_cudaq_kernel on PERCEVAL_EXPERIMENTJSON` | photonic | raises (`Exception`) |
| `to_mimiq_circuit on PERCEVALJSON` | photonic | raises (`Exception`) |
| `to_mimiq_circuit on PERCEVAL_EXPERIMENTJSON` | photonic | raises (`Exception`) |
| `to_perceval_circuit on PERCEVAL_EXPERIMENTJSON` | photonic | raises (`Exception`) |
| `to_perceval_experiment on PERCEVALJSON` | photonic | raises (`Exception`) |
| `to_pulser_sequence on PERCEVALJSON` | photonic | raises (`Exception`) |
| `to_pulser_sequence on PERCEVAL_EXPERIMENTJSON` | photonic | raises (`Exception`) |
| `to_qasm2_circuit on PERCEVALJSON` | photonic | raises (`Exception`) |
| `to_qasm2_circuit on PERCEVAL_EXPERIMENTJSON` | photonic | raises (`Exception`) |
| `to_qiskit_circuit on PERCEVALJSON` | photonic | raises (`Exception`) |
| `to_qiskit_circuit on PERCEVAL_EXPERIMENTJSON` | photonic | raises (`Exception`) |

### Feature coverage

| Feature | Status | Note |
|---|---|---|
| qubit topology & naming | round-trip tested | named/grid Cirq qubits are preserved by CirqJSON but flattened to q[i] by OpenQASM (declared 'qubit_topology'); Qiskit/CUDA-Q have no native equivalent |
| classical control flow (dynamic circuits) | not exercised | no dynamic reference circuits; Qiskit IfElseOp/WhileOp and QASM3 control flow are outside the normalized gate set (would need new builders and checks) |
| qasm2<->qasm3 text normalization | straight-line only | the cudaq_to_qasm3 / qasm3_to_mimiq converters chain on the self-hosted text converters: literal-parameter stdgates round-trip; classical control flow, gphase (silently dropped on the QASM3->QASM2 leg) and non-mapped stdlib gates are declared 'qasm_control_flow' / 'qasm_gphase' / 'qasm_stdlib_gate' |
| circuit metadata & global phase | round-trip tested | Qiskit global_phase and metadata dict are dropped by every QASM round trip (declared 'global_phase'/'circuit_metadata'); the unitary oracle intentionally ignores phases |
| symbolic parameters | tested as unsupported | QASM3 ('input float') and CirqJSON preserve unbound parameters; QASM2 raises on unbound params and the cirq QASM importer cannot parse them, so the parametrized circuit runs on the QASM3/CirqJSON edges only |
| measurement mapping & memory slots | round-trip tested | qubit->clbit assignment and Cirq measurement keys are compared (declared 'measurement_mapping'); the classical-bit count is a hard check |
| calibrations & pulse | partial (analog) | no Qiskit Pulse calibrations / QASM3 defcal support yet, but the analog paradigm is exercised on the 'pulser->pulserjson->pulser' edge: AnalogSpec (register coords, global/local channels, constant & Blackman pulses, measurement) round trips exactly through the Pulser abstract repr |
| analog pulse parametrization | partial (expression round-trip) | variables declared on a sequence round trip symbolically when used as *expressions* ('2*x+0.5' amplitude / '0.1*x' phase on the analog_param circuit); a variable used *bare* as a pulse value cannot be serialized by pulser 1.9.1 (AbstractReprError) and is pinned as an expected unsupported conversion |
| photonic mode-based circuits | round-trip tested | native perceval circuits (dual-rail qubits, beam splitters, parametrized phase shifters and the heralded CNOT block with ancilla) round trip exactly through the perceval serialization on the 'perceval->percevaljson->perceval' edge; the 'photonic' oracle compares the exact post-selected SLOS output distribution before/after. NOTE: the intermediate QuantumProgram carries only the perceval.Circuit archive - the Fock input state, the herald condition and the post-selection are part of the PhotonicSpec and are re-applied by the oracle *outside* qio, so a full photonic processor configuration is out of the qio formats |
| photonic experiment context (input state + heralds) | round-trip tested | the perceval.Experiment archive carries the Fock input state and the output herald post-selection alongside the circuit; the 'perceval_experiment->percevaljson->perceval_experiment' edge round-trips them exactly and the shared 'photonic' oracle simulates from the experiment's own context (no re-application by the test) |

## Result conversions (`QuantumProgramResult`)

The counts histogram is always compared bit-exact; the rows below are the tolerated metadata losses.

### gate paradigm

| Edge | Loss | Expected | Actual | Explanation | Observed (run) | Expected (run) |
|---|---|---|---|---|---|---|
| `cirq.Result -> cirq_result_json_v1 -> cirq.Result` | `cirq_params` | cirq.Result params | always None | cirq.ParamResolver is not JSON-serializable by the qio pipeline | - | - |
| `cirq.Result -> cirq_result_json_v1 -> mimiqcircuits.QCSResults` | `cirq_params` | cirq.Result params | always None | cirq.ParamResolver is not JSON-serializable by the qio pipeline | - | - |
| `cirq.Result -> cirq_result_json_v1 -> mimiqcircuits.QCSResults` | `mimiq_amplitudes` | MIMIQ amplitudes | not representable through result JSON | bitarray keys + complex values are not JSON-serializable | `amplitudes` = `{}` | <details><summary>5 values</summary><br>`amplitudes` = `{'00': (0.7071067811865475+0j), '11': (0.7071067811865475+0j)}`<br>`amplitudes` = `{'000': (0.7071067811865475+0j), '111': (0.7071067811865475+0j)}`<br>`amplitudes` = `{'01': (0.7071067811865475+0j), '10': (0.7071067811865475+0j)}`<br>`amplitudes` = `{'1': (1+0j)}`<br>`amplitudes` = `{'11': (1+0j)}`</details> |
| `cirq.Result -> cirq_result_json_v1 -> mimiqcircuits.QCSResults` | `mimiq_metadata` | MIMIQ zstates/fidelities/avggateerrors/timings | dropped | not carried by mimiq_to_dict / dict_to_mimiq today | <details><summary>6 values</summary><br>`avggateerrors` = `[]`<br>`fidelities` = `[]`<br>`simulator` = `'Cirq-Simulator'`<br>`timings` = `{}`<br>`version` = `'1.5.0'`<br>`zstates` = `[]`</details> | <details><summary>20 values</summary><br>`avggateerrors` = `[0.002]`<br>`avggateerrors` = `[0.003]`<br>`avggateerrors` = `[0.01]`<br>`avggateerrors` = `[0.0]`<br>`fidelities` = `[0.98]`<br>`fidelities` = `[0.995]`<br>`fidelities` = `[0.99]`<br>`fidelities` = `[1.0]`<br>`simulator` = `'qio-test-simulator'`<br>`timings` = `{'total': 0.0031, 'apply': 0.0022}`<br>`timings` = `{'total': 0.0087, 'apply': 0.0066}`<br>`timings` = `{'total': 0.0119, 'apply': 0.0083}`<br>`timings` = `{'total': 0.0123, 'apply': 0.0091}`<br>`timings` = `{'total': 0.0231, 'apply': 0.0184}`<br>`version` = `'1.0'`<br>`zstates` = `['00']`<br>`zstates` = `['000']`<br>`zstates` = `['01']`<br>`zstates` = `['1']`<br>`zstates` = `['11']`</details> |
| `cirq.Result -> cirq_result_json_v1 -> qiskit.Result` | `cirq_params` | cirq.Result params | always None | cirq.ParamResolver is not JSON-serializable by the qio pipeline | - | - |
| `cirq.Result -> cirq_result_json_v1 -> qiskit.Result` | `qiskit_metadata` | qiskit job metadata (backend_name/version, job_id, qobj_id, date) | dropped off the self path | backend/job identifiers are not carried by every source SDK into the JSON intermediate | - | <details><summary>5 values</summary><br>`backend_name` = `'qio-test-simulator'`<br>`backend_version` = `'1.0'`<br>`date` = `'2026-01-01T00:00:00+00:00'`<br>`job_id` = `'qio-result-job'`<br>`qobj_id` = `'qio-result-qobj'`</details> |
| `cirq.Result -> cirq_result_json_v1 -> qiskit.Result` | `qiskit_statevector` | qiskit data.statevector | dropped off the self path | numpy/complex is not JSON-safe | - | <details><summary>5 values</summary><br>`statevector` = `[0.0, 0.0, 0.0, 1.0]`<br>`statevector` = `[0.0, 0.7071067811865475, 0.7071067811865475, 0.0]`<br>`statevector` = `[0.0, 1.0]`<br>`statevector` = `[0.7071067811865475, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7071067811865475]`<br>`statevector` = `[0.7071067811865475, 0.0, 0.0, 0.7071067811865475]`</details> |
| `cudaq.SampleResult -> cudaq_sample_result_json_v1 -> qiskit.Result` | `qiskit_metadata` | qiskit job metadata (backend_name/version, job_id, qobj_id, date) | dropped off the self path | backend/job identifiers are not carried by every source SDK into the JSON intermediate | - | <details><summary>5 values</summary><br>`backend_name` = `'qio-test-simulator'`<br>`backend_version` = `'1.0'`<br>`date` = `'2026-01-01T00:00:00+00:00'`<br>`job_id` = `'qio-result-job'`<br>`qobj_id` = `'qio-result-qobj'`</details> |
| `cudaq.SampleResult -> cudaq_sample_result_json_v1 -> qiskit.Result` | `qiskit_statevector` | qiskit data.statevector | dropped off the self path | numpy/complex is not JSON-safe | - | <details><summary>5 values</summary><br>`statevector` = `[0.0, 0.0, 0.0, 1.0]`<br>`statevector` = `[0.0, 0.7071067811865475, 0.7071067811865475, 0.0]`<br>`statevector` = `[0.0, 1.0]`<br>`statevector` = `[0.7071067811865475, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7071067811865475]`<br>`statevector` = `[0.7071067811865475, 0.0, 0.0, 0.7071067811865475]`</details> |
| `mimiqcircuits.QCSResults -> mimiq_qcsr_json_v1 -> mimiqcircuits.QCSResults` | `mimiq_amplitudes` | MIMIQ amplitudes | not representable through result JSON | bitarray keys + complex values are not JSON-serializable | `amplitudes` = `{}` | <details><summary>5 values</summary><br>`amplitudes` = `{'00': (0.7071067811865475+0j), '11': (0.7071067811865475+0j)}`<br>`amplitudes` = `{'000': (0.7071067811865475+0j), '111': (0.7071067811865475+0j)}`<br>`amplitudes` = `{'01': (0.7071067811865475+0j), '10': (0.7071067811865475+0j)}`<br>`amplitudes` = `{'1': (1+0j)}`<br>`amplitudes` = `{'11': (1+0j)}`</details> |
| `mimiqcircuits.QCSResults -> mimiq_qcsr_json_v1 -> mimiqcircuits.QCSResults` | `mimiq_metadata` | MIMIQ zstates/fidelities/avggateerrors/timings | dropped | not carried by mimiq_to_dict / dict_to_mimiq today | <details><summary>4 values</summary><br>`avggateerrors` = `[]`<br>`fidelities` = `[]`<br>`timings` = `{}`<br>`zstates` = `[]`</details> | <details><summary>18 values</summary><br>`avggateerrors` = `[0.002]`<br>`avggateerrors` = `[0.003]`<br>`avggateerrors` = `[0.01]`<br>`avggateerrors` = `[0.0]`<br>`fidelities` = `[0.98]`<br>`fidelities` = `[0.995]`<br>`fidelities` = `[0.99]`<br>`fidelities` = `[1.0]`<br>`timings` = `{'total': 0.0031, 'apply': 0.0022}`<br>`timings` = `{'total': 0.0087, 'apply': 0.0066}`<br>`timings` = `{'total': 0.0119, 'apply': 0.0083}`<br>`timings` = `{'total': 0.0123, 'apply': 0.0091}`<br>`timings` = `{'total': 0.0231, 'apply': 0.0184}`<br>`zstates` = `['00']`<br>`zstates` = `['000']`<br>`zstates` = `['01']`<br>`zstates` = `['1']`<br>`zstates` = `['11']`</details> |
| `mimiqcircuits.QCSResults -> mimiq_qcsr_json_v1 -> qiskit.Result` | `qiskit_metadata` | qiskit job metadata (backend_name/version, job_id, qobj_id, date) | dropped off the self path | backend/job identifiers are not carried by every source SDK into the JSON intermediate | <details><summary>12 values</summary><br>`date` = `'2026-10-30T11:48:21.063628'`<br>`date` = `'2026-10-30T11:48:21.065036'`<br>`date` = `'2026-10-30T11:48:21.066298'`<br>`date` = `'2026-10-30T11:48:21.067093'`<br>`date` = `'2026-10-30T11:48:21.068107'`<br>`date` = `'2026-10-30T11:48:21.162806'`<br>`date` = `'2026-10-30T11:48:21.164202'`<br>`date` = `'2026-10-30T11:48:21.165440'`<br>`date` = `'2026-10-30T11:48:21.166223'`<br>`date` = `'2026-10-30T11:48:21.167345'`<br>`job_id` = `'unknown'`<br>`qobj_id` = `'unknown'`</details> | <details><summary>3 values</summary><br>`date` = `'2026-01-01T00:00:00+00:00'`<br>`job_id` = `'qio-result-job'`<br>`qobj_id` = `'qio-result-qobj'`</details> |
| `mimiqcircuits.QCSResults -> mimiq_qcsr_json_v1 -> qiskit.Result` | `qiskit_statevector` | qiskit data.statevector | dropped off the self path | numpy/complex is not JSON-safe | - | <details><summary>5 values</summary><br>`statevector` = `[0.0, 0.0, 0.0, 1.0]`<br>`statevector` = `[0.0, 0.7071067811865475, 0.7071067811865475, 0.0]`<br>`statevector` = `[0.0, 1.0]`<br>`statevector` = `[0.7071067811865475, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7071067811865475]`<br>`statevector` = `[0.7071067811865475, 0.0, 0.0, 0.7071067811865475]`</details> |
| `qiskit.Result -> qiskit_result_json_v1 -> mimiqcircuits.QCSResults` | `mimiq_amplitudes` | MIMIQ amplitudes | not representable through result JSON | bitarray keys + complex values are not JSON-serializable | `amplitudes` = `{}` | <details><summary>5 values</summary><br>`amplitudes` = `{'00': (0.7071067811865475+0j), '11': (0.7071067811865475+0j)}`<br>`amplitudes` = `{'000': (0.7071067811865475+0j), '111': (0.7071067811865475+0j)}`<br>`amplitudes` = `{'01': (0.7071067811865475+0j), '10': (0.7071067811865475+0j)}`<br>`amplitudes` = `{'1': (1+0j)}`<br>`amplitudes` = `{'11': (1+0j)}`</details> |
| `qiskit.Result -> qiskit_result_json_v1 -> mimiqcircuits.QCSResults` | `mimiq_metadata` | MIMIQ zstates/fidelities/avggateerrors/timings | dropped | not carried by mimiq_to_dict / dict_to_mimiq today | <details><summary>4 values</summary><br>`avggateerrors` = `[]`<br>`fidelities` = `[]`<br>`timings` = `{}`<br>`zstates` = `[]`</details> | <details><summary>18 values</summary><br>`avggateerrors` = `[0.002]`<br>`avggateerrors` = `[0.003]`<br>`avggateerrors` = `[0.01]`<br>`avggateerrors` = `[0.0]`<br>`fidelities` = `[0.98]`<br>`fidelities` = `[0.995]`<br>`fidelities` = `[0.99]`<br>`fidelities` = `[1.0]`<br>`timings` = `{'total': 0.0031, 'apply': 0.0022}`<br>`timings` = `{'total': 0.0087, 'apply': 0.0066}`<br>`timings` = `{'total': 0.0119, 'apply': 0.0083}`<br>`timings` = `{'total': 0.0123, 'apply': 0.0091}`<br>`timings` = `{'total': 0.0231, 'apply': 0.0184}`<br>`zstates` = `['00']`<br>`zstates` = `['000']`<br>`zstates` = `['01']`<br>`zstates` = `['1']`<br>`zstates` = `['11']`</details> |
| `qiskit.Result -> qiskit_result_json_v1 -> qiskit.Result` | `qiskit_statevector` | qiskit data.statevector | dropped off the self path | numpy/complex is not JSON-safe | - | <details><summary>5 values</summary><br>`statevector` = `[0.0, 0.0, 0.0, 1.0]`<br>`statevector` = `[0.0, 0.7071067811865475, 0.7071067811865475, 0.0]`<br>`statevector` = `[0.0, 1.0]`<br>`statevector` = `[0.7071067811865475, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7071067811865475]`<br>`statevector` = `[0.7071067811865475, 0.0, 0.0, 0.7071067811865475]`</details> |

*Lossless edges:* `cudaq.SampleResult -> cudaq_sample_result_json_v1 -> cudaq.SampleResult`, `mimiqcircuits.QCSResults -> mimiq_qcsr_json_v1 -> cirq.Result`, `qiskit.Result -> qiskit_result_json_v1 -> cirq.Result`.

### analog paradigm

| Edge | Loss | Expected | Actual | Explanation | Observed (run) | Expected (run) |
|---|---|---|---|---|---|---|

*Lossless edges:* `pulser.backend.Results -> pulser_result_json_v1 -> pulser.backend.Results`.

### photonic paradigm

| Edge | Loss | Expected | Actual | Explanation | Observed (run) | Expected (run) |
|---|---|---|---|---|---|---|

*Lossless edges:* `perceval.BSSamples -> perceval_result_json_v1 -> perceval.BSSamples`, `perceval.BSSamples:BSCount -> perceval_result_json_v1 -> perceval.BSSamples:BSCount`, `perceval.BSSamples:BSDistribution -> perceval_result_json_v1 -> perceval.BSSamples:BSDistribution`.

### SDK conventions (tolerated, no information loss)

| Field | Expected | Actual | Explanation |
|---|---|---|---|
| `measurement_key` | source measurement keys (e.g. cirq 'm0'/'m1') | target SDK keys (e.g. qiskit 'm', mimiq 'result') | SDK convention; the counts histogram is key-independent |
| `bit_order` | source bitstring order | target SDK character order | SDK convention; the bitstring multiset is preserved exactly |

### Unsupported conversions

The complement of the declared read graph. Each entry must raise (`Exception`).

| Unsupported conversion | Paradigm | Behavior |
|---|---|---|
| `to_cirq_result on PULSER_RESULT_JSON_V1` | analog | raises (`Exception`) |
| `to_cudaq_sample_result on PULSER_RESULT_JSON_V1` | analog | raises (`Exception`) |
| `to_mimiq_qcsr on PULSER_RESULT_JSON_V1` | analog | raises (`Exception`) |
| `to_perceval_result on PULSER_RESULT_JSON_V1` | analog | raises (`Exception`) |
| `to_qiskit_result on PULSER_RESULT_JSON_V1` | analog | raises (`Exception`) |
| `to_cirq_result on CUDAQ_SAMPLE_RESULT_JSON_V1` | gate | raises (`Exception`) |
| `to_cudaq_sample_result on CIRQ_RESULT_JSON_V1` | gate | raises (`Exception`) |
| `to_cudaq_sample_result on MIMIQ_QCSR_JSON_V1` | gate | raises (`Exception`) |
| `to_cudaq_sample_result on QISKIT_RESULT_JSON_V1` | gate | raises (`Exception`) |
| `to_mimiq_qcsr on CUDAQ_SAMPLE_RESULT_JSON_V1` | gate | raises (`Exception`) |
| `to_perceval_result on CIRQ_RESULT_JSON_V1` | gate | raises (`Exception`) |
| `to_perceval_result on CUDAQ_SAMPLE_RESULT_JSON_V1` | gate | raises (`Exception`) |
| `to_perceval_result on MIMIQ_QCSR_JSON_V1` | gate | raises (`Exception`) |
| `to_perceval_result on QISKIT_RESULT_JSON_V1` | gate | raises (`Exception`) |
| `to_pulser_result on CIRQ_RESULT_JSON_V1` | gate | raises (`Exception`) |
| `to_pulser_result on CUDAQ_SAMPLE_RESULT_JSON_V1` | gate | raises (`Exception`) |
| `to_pulser_result on MIMIQ_QCSR_JSON_V1` | gate | raises (`Exception`) |
| `to_pulser_result on QISKIT_RESULT_JSON_V1` | gate | raises (`Exception`) |
| `to_cirq_result on PERCEVAL_RESULT_JSON_V1` | photonic | raises (`Exception`) |
| `to_cudaq_sample_result on PERCEVAL_RESULT_JSON_V1` | photonic | raises (`Exception`) |
| `to_mimiq_qcsr on PERCEVAL_RESULT_JSON_V1` | photonic | raises (`Exception`) |
| `to_pulser_result on PERCEVAL_RESULT_JSON_V1` | photonic | raises (`Exception`) |
| `to_qiskit_result on PERCEVAL_RESULT_JSON_V1` | photonic | raises (`Exception`) |
