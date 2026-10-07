# Copyright 2026 Scaleway, Aqora, Quantum Commons
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static, controlled input circuits for the conversion test battery.

Each :class:`ReferenceCircuit` is defined *once* (a literal gate list with known
``n_qubits`` and, for deterministic circuits, known measurement counts) and
exposed in every SDK representation (Cirq, Qiskit, OpenQASM 2, OpenQASM 3,
CUDA-Q kernel) through a single generator function ``get_reference_circuit``.

The gate set is restricted to gates supported by all converters
(``qio/utils/conversion/program/*``), notably ``qasm3_to_cudaq`` and
``cirq.contrib.qasm_import``. It is deliberately representative:
single/multi qubit, parameterized gates, and measurements.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Gate parameters are usually floats; a symbolic parameter is encoded as the
# string marker "~<name>" so it flows through the op-chain comparison (see the
# ``parametrized`` circuit and the ``_*_operations`` helpers in
# ``circuit_testing``).
Param = Union[float, str]
Operation = Tuple[str, Tuple[Param, ...], Tuple[int, ...]]

_SUPPORTED_GATES = {
    "h",
    "x",
    "y",
    "z",
    "s",
    "t",
    "sdg",
    "tdg",
    "rx",
    "ry",
    "rz",
    "cx",
    "cz",
    "measure",
}

_PARAM_GATES = {"rx", "ry", "rz"}
_SINGLE_QUBIT_GATES = {
    "h",
    "x",
    "y",
    "z",
    "s",
    "t",
    "sdg",
    "tdg",
}
_TWO_QUBIT_GATES = {"cx", "cz"}


def _fmt(num: float) -> str:
    if isinstance(num, str):
        # Symbolic parameter marker "~<name>": emit the bare name.
        return num[1:] if num.startswith("~") else num
    return repr(float(num))


@dataclass
class ReferenceCircuit:
    name: str
    n_qubits: int
    gates: Sequence[Operation]
    expected_counts: Optional[Dict[str, int]] = None
    description: str = field(default="")
    # Circuit-level information (see the conversion information-loss report):
    # - global_phase / metadata are only representable by the Qiskit builder.
    # - qubit_scheme selects the Cirq qubit type ("line" | "named" | "grid");
    #   QASM flattens any scheme to a linear array, CirqJSON preserves it.
    # - oracle forces which equivalence oracle the conversion path must close
    #   with, overriding the edge's default (e.g. "structural" for circuits
    #   whose unitary cannot be computed, like unbound-parameter circuits).
    # - supported_edge_ids optionally restricts the edges this circuit runs on,
    #   for features only a subset of formats can carry (symbolic parameters).
    global_phase: float = field(default=0.0)
    metadata: Optional[Dict[str, Any]] = field(default=None)
    qubit_scheme: str = field(default="line")
    oracle: Optional[str] = field(default=None)
    supported_edge_ids: Optional[Tuple[str, ...]] = field(default=None)

    def supports_edge(self, edge_id: str) -> bool:
        if self.supported_edge_ids is not None:
            return edge_id in self.supported_edge_ids
        return True

    def has_measurements(self) -> bool:
        return any(gate == "measure" for gate, _, _ in self.gates)

    def measurement_indices(self) -> List[int]:
        """Sorted qubit indices that receive a measurement."""
        return sorted(
            {i for gate, _, indices in self.gates if gate == "measure" for i in indices}
        )

    def topology_labels(self) -> Tuple[str, ...]:
        """Canonical (label, ...) the native Cirq builder produces for the
        declared ``qubit_scheme`` - used to detect topology losses."""
        if self.qubit_scheme == "line":
            return tuple(f"q{i}" for i in range(self.n_qubits))
        if self.qubit_scheme == "named":
            return tuple(chr(ord("a") + i) for i in range(self.n_qubits))
        if self.qubit_scheme == "grid":
            return tuple(f"(0,{i})" for i in range(self.n_qubits))
        raise ValueError(f"unknown qubit_scheme {self.qubit_scheme!r}")

    def _build_cirq(self) -> "cirq.Circuit":
        import cirq

        if self.qubit_scheme == "line":
            qubits = cirq.LineQubit.range(self.n_qubits)
        elif self.qubit_scheme == "named":
            labels = self.topology_labels()
            qubits = [cirq.NamedQubit(label) for label in labels]
        elif self.qubit_scheme == "grid":
            qubits = [cirq.GridQubit(0, i) for i in range(self.n_qubits)]
        else:
            raise ValueError(f"unknown qubit_scheme {self.qubit_scheme!r}")
        operations = []
        for gate, params, indices in self.gates:
            targets = [qubits[i] for i in indices]
            if gate == "measure":
                # Distinct key per measurement so Cirq exports them to
                # distinct classical registers (a shared key collapses the
                # round trip: cirq writes every measurement to one bit).
                operations.append(
                    cirq.measure(*targets, key="m" + "".join(str(i) for i in indices))
                )
            elif gate in _PARAM_GATES:
                rads = params[0]
                if isinstance(rads, str):
                    import sympy

                    rads = sympy.Symbol(rads[1:] if rads.startswith("~") else rads)
                operations.append(getattr(cirq, gate)(rads).on(*targets))
            else:
                gate_obj = {
                    "h": cirq.H,
                    "x": cirq.X,
                    "y": cirq.Y,
                    "z": cirq.Z,
                    "s": cirq.S,
                    "t": cirq.T,
                    "sdg": cirq.S**-1,
                    "tdg": cirq.T**-1,
                    "cx": cirq.CNOT,
                    "cz": cirq.CZ,
                }[gate]
                operations.append(gate_obj(*targets))
        return cirq.Circuit(operations)

    def cirq(self) -> "cirq.Circuit":
        return self._build_cirq()

    def _build_qiskit(self) -> "qiskit.QuantumCircuit":
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits, self.n_qubits, name=self.name)
        if self.global_phase:
            qc.global_phase = self.global_phase
        if self.metadata:
            qc.metadata = dict(self.metadata)
        for gate, params, indices in self.gates:
            targets = list(indices)
            if gate == "measure":
                for i in indices:
                    qc.measure(i, i)
            elif gate in _PARAM_GATES:
                param = params[0]
                if isinstance(param, str):
                    from qiskit.circuit import Parameter

                    param = Parameter(param[1:] if param.startswith("~") else param)
                getattr(qc, gate)(param, *targets)
            elif gate == "sdg":
                qc.sdg(*targets)
            elif gate == "tdg":
                qc.tdg(*targets)
            else:
                getattr(qc, gate)(*targets)
        return qc

    def qiskit(self) -> "qiskit.QuantumCircuit":
        return self._build_qiskit()

    def _build_qasm2(self) -> str:
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{self.n_qubits}];",
            f"creg c[{self.n_qubits}];",
        ]
        for gate, params, indices in self.gates:
            qargs = ", ".join(f"q[{i}]" for i in indices)
            if gate == "measure":
                for i in indices:
                    lines.append(f"measure q[{i}] -> c[{i}];")
                continue
            param_str = f"({', '.join(_fmt(p) for p in params)})" if params else ""
            lines.append(f"{gate}{param_str} {qargs};")
        return "\n".join(lines) + "\n"

    def qasm2(self) -> str:
        return self._build_qasm2()

    def _build_qasm3(self) -> str:
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{self.n_qubits}] q;",
            f"bit[{self.n_qubits}] c;",
        ]
        for gate, params, indices in self.gates:
            qargs = ", ".join(f"q[{i}]" for i in indices)
            if gate == "measure":
                for i in indices:
                    lines.append(f"c[{i}] = measure q[{i}];")
                continue
            param_str = f"({', '.join(_fmt(p) for p in params)})" if params else ""
            lines.append(f"{gate}{param_str} {qargs};")
        return "\n".join(lines) + "\n"

    def qasm3(self) -> str:
        return self._build_qasm3()

    def _build_cudaq(self):
        import cudaq

        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(self.n_qubits)
        for gate, params, indices in self.gates:
            targets = [qubits[i] for i in indices]
            if gate == "measure":
                for q in targets:
                    kernel.mz(q)
            elif gate in _PARAM_GATES:
                param = params[0]
                if isinstance(param, str):
                    raise NotImplementedError(
                        "symbolic params are not representable in a CUDA-Q kernel"
                    )
                getattr(kernel, gate)(param, *targets)
            else:
                getattr(kernel, gate)(*targets)
        return kernel

    def cudaq_kernel(self):
        return self._build_cudaq()


def _ops(*operations: Operation) -> List[Operation]:
    for gate, params, indices in operations:
        assert gate in _SUPPORTED_GATES, f"unsupported gate: {gate}"
        if gate in _PARAM_GATES:
            assert len(params) == 1, gate
        if gate in _SINGLE_QUBIT_GATES:
            assert len(indices) == 1, gate
        if gate in _TWO_QUBIT_GATES or gate == "measure":
            assert len(indices) >= 1, gate
    return list(operations)


# Representative, controlled reference circuits.
REFERENCE_CIRCUITS = [
    ReferenceCircuit(
        name="bell",
        n_qubits=2,
        gates=_ops(
            ("h", (), (0,)),
            ("cx", (), (0, 1)),
            ("measure", (), (0,)),
            ("measure", (), (1,)),
        ),
        expected_counts={"00": None, "11": None},
        description="2-qubit Bell state: h + cx + measure.",
    ),
    ReferenceCircuit(
        name="ghz",
        n_qubits=3,
        gates=_ops(
            ("h", (), (0,)),
            ("cx", (), (0, 1)),
            ("cx", (), (0, 2)),
            ("measure", (), (0,)),
            ("measure", (), (1,)),
            ("measure", (), (2,)),
        ),
        expected_counts={"000": None, "111": None},
        description="3-qubit GHZ state.",
    ),
    ReferenceCircuit(
        name="rotations",
        n_qubits=2,
        gates=_ops(
            ("rx", (0.3,), (0,)),
            ("ry", (0.5,), (1,)),
            ("cz", (), (0, 1)),
            ("rz", (0.7,), (0,)),
            ("measure", (), (0,)),
            ("measure", (), (1,)),
        ),
        description="2-qubit circuit with parameterized rotations and cz.",
    ),
    ReferenceCircuit(
        name="x_cx",
        n_qubits=2,
        gates=_ops(
            ("x", (), (0,)),
            ("cx", (), (0, 1)),
            ("measure", (), (0,)),
            ("measure", (), (1,)),
        ),
        expected_counts={"11": None},
        description="Deterministic 2-qubit circuit: x(0), cx(0, 1).",
    ),
    ReferenceCircuit(
        name="gates4",
        n_qubits=4,
        gates=_ops(
            ("h", (), (0,)),
            ("x", (), (1,)),
            ("s", (), (2,)),
            ("t", (), (3,)),
            ("cx", (), (0, 1)),
            ("cz", (), (2, 3)),
            ("measure", (), (0,)),
            ("measure", (), (1,)),
            ("measure", (), (2,)),
            ("measure", (), (3,)),
        ),
        description="4-qubit circuit mixing single- and two-qubit gates.",
    ),
    ReferenceCircuit(
        name="phased_meta",
        n_qubits=2,
        gates=_ops(
            ("h", (), (0,)),
            ("cx", (), (0, 1)),
            ("z", (), (0,)),
            ("measure", (), (0,)),
            ("measure", (), (1,)),
        ),
        expected_counts={"00": None, "11": None},
        global_phase=0.5,
        metadata={"id": 42, "ansatz": "rx"},
        description=(
            "Deterministic circuit carrying non-trivial circuit-level "
            "information: global_phase and a Qiskit metadata dict."
        ),
    ),
    ReferenceCircuit(
        name="topology",
        n_qubits=2,
        gates=_ops(
            ("h", (), (0,)),
            ("cx", (), (0, 1)),
            ("measure", (), (0,)),
            ("measure", (), (1,)),
        ),
        expected_counts={"00": None, "11": None},
        qubit_scheme="named",
        description=(
            "Bell circuit on named qubits (a, b); the named topology is only "
            "preserved by CirqJSON, QASM flattens it to a linear array."
        ),
    ),
    ReferenceCircuit(
        name="parametrized",
        n_qubits=1,
        gates=_ops(
            ("rx", ("~theta",), (0,)),
        ),
        oracle="structural",
        supported_edge_ids=(
            "qiskit->qasm3->qiskit",
            "cirq->cirqjson->cirq",
        ),
        description=(
            "Unbound parameterized circuit rx(theta). Symbolic parameters only "
            "survive QASM3 (input float) and CirqJSON; QASM2 raises on unbound "
            "params and the cirq qasm importer cannot parse them."
        ),
    ),
]

REFERENCE_CIRCUIT_INDEX = {c.name: c for c in REFERENCE_CIRCUITS}


def get_reference_circuit(name: str) -> ReferenceCircuit:
    """Statically build a controlled input circuit by name."""
    if name not in REFERENCE_CIRCUIT_INDEX:
        raise KeyError(
            f"unknown reference circuit {name!r}; available: "
            f"{sorted(REFERENCE_CIRCUIT_INDEX)}"
        )
    return REFERENCE_CIRCUIT_INDEX[name]
