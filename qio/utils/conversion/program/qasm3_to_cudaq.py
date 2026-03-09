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
# limitations under the License.from enum import Enum
import cudaq
import pyqasm

from openqasm3 import ast
from cudaq import PyKernel, QuakeValue


_ONE_QUBIT_GATES = {
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "h",
    "s",
    "t",
    "sdg",
    "tdg",
    "u3",
    "i",
    "id",
    "iden",
}

_TWO_QUBIT_GATES = {"swap"}

# 3-qubit gates exposed as first-class CUDA-Q operations
_THREE_QUBIT_GATES = {"ccx", "ccz", "cswap"}

# Gates that can be the base of a c-prefixed controlled gate
_CONTROLLABLE_BASE_GATES = _ONE_QUBIT_GATES | _TWO_QUBIT_GATES


def _parse_controlled_name(name: str) -> tuple[int, str]:
    """Detect multi-controlled gate names encoded as repeated 'c' prefixes.

    Iteratively strips leading 'c' chars and checks whether the remainder
    is a known controllable base gate.

    Examples
    --------
    "cx"    → (1, "x")
    "ccx"   → (2, "x")   ← Toffoli
    "cccx"  → (3, "x")
    "cswap" → (1, "swap") ← Fredkin
    "h"     → (0, "h")   ← not a controlled gate

    Returns
    -------
    (num_controls, base_gate_name)
        num_controls == 0 means the name is *not* a c-prefixed controlled gate.
    """
    namel = name.lower()

    for i in range(1, len(namel)):
        if namel[:i] == "c" * i and namel[i:] in _CONTROLLABLE_BASE_GATES:
            return i, namel[i:]

    return 0, namel


def _make_gate_kernel(name: str, targs: tuple[type]) -> PyKernel:
    """Returns CUDA-Q kernel for pure standard gates (no modifiers - ctrl or adj)."""

    if name in _ONE_QUBIT_GATES:
        size = 1
    elif name in _TWO_QUBIT_GATES:
        size = 2
    elif name in _THREE_QUBIT_GATES:
        size = 3
    else:
        raise Exception(f"Unsupported gate: {name}")

    kernel, *qparams = cudaq.make_kernel(*[cudaq.qubit for _ in range(size)], *targs)
    qrefs, qargs = qparams[:size], qparams[size:]

    if name in ["i", "id", "iden"]:
        return kernel

    op = getattr(kernel, name)

    if len(targs) > 0:
        op(*qargs, *qrefs)
    else:
        op(*qrefs)

    return kernel


def convert(circuit_str: str) -> PyKernel:
    """Returns a CUDA-Q kernel representing the input OpenQASM program.

    Args:
        qasm (str or ast.Program): OpenQASM program to convert to CUDA-Q kernel.

    Returns:
        kernel: CUDA-Q kernel equivalent to input OpenQASM string.
    """

    module = pyqasm.loads(circuit_str)
    module.validate()
    module.unroll()

    program = module.unrolled_ast

    kernel = cudaq.make_kernel()
    ctx = {}
    gate_kernels = {}

    def get_gate(name: str, targs: tuple[type]) -> PyKernel:
        if name in gate_kernels:
            return gate_kernels[name]

        # Fix from qbraid: make sure we use float and not integer
        targs = list(map(lambda x: float if isinstance(x, type(int)) else x, targs))
        gate_kernels[name] = _make_gate_kernel(name, targs)
        return gate_kernels[name]

    def qubit_lookup(
        qubit: ast.IndexedIdentifier | ast.Identifier,
    ) -> QuakeValue:
        assert isinstance(
            qubit, ast.IndexedIdentifier
        ), f"all identifiers should've been indexed: {qubit}"

        assert (
            len(qubit.indices) == 1
        ), f"multi-dim arrays are not supported: {qubit.indices}"

        inds = qubit.indices[0]

        assert len(inds) == 1, f"indices should've been a single integer: {inds}"
        assert isinstance(ind := inds[0], ast.IntegerLiteral)

        q = ctx[qubit.name.name][ind.value]
        return q

    for statement in program.statements:
        if isinstance(statement, ast.Include):
            if statement.filename not in {"stdgates.inc", "qelib1.inc"}:
                raise Exception(f"Custom includes are unsupported: {statement}")
        elif isinstance(statement, ast.QubitDeclaration):
            ctx[statement.qubit.name] = kernel.qalloc(statement.size.value)
        elif isinstance(statement, ast.ClassicalDeclaration):
            if statement.init_expression and isinstance(
                statement.init_expression, ast.QuantumMeasurement
            ):
                ctx[statement.identifier.name] = kernel.mz(
                    qubit_lookup(statement.init_expression.qubit)
                )
            else:
                ctx[statement.identifier.name] = None
        elif isinstance(statement, ast.QuantumMeasurementStatement):
            val = kernel.mz(qubit_lookup(statement.measure.qubit))
            if statement.target is not None:
                assert isinstance(
                    statement.target, ast.IndexedIdentifier
                ), f"identifiers should've been unrolled to indexed identifiers: {statement.target}"
                ctx[statement.target.name.name] = val
        elif isinstance(statement, ast.QuantumGate):
            name, qubits = statement.name.name, statement.qubits

            args = []
            for arg in statement.arguments:
                assert (
                    arg.value is not None
                ), f"gate arguments should've been literals: {arg}"
                args.append(arg.value)
            targs = [type(a) for a in args]

            qubit_refs = [qubit_lookup(q) for q in qubits]

            # pyqasm unrolls multiple modifiers.
            assert len(statement.modifiers) <= 1

            if len(statement.modifiers) == 1:
                mod = statement.modifiers[0]
                assert (
                    mod.modifier == ast.GateModifierName.ctrl
                ), f"non-ctrl modifiers should've be unrolled: {mod}"

                gate = get_gate(name, targs)
                kernel.control(gate, qubit_refs[0], *qubit_refs[1:])
            else:
                # ----------------------------------------------------------
                # Implicit controlled gates: cx, ccx, cccx, cswap, …
                # ----------------------------------------------------------
                num_ctrl, base_gate = _parse_controlled_name(name)

                if num_ctrl > 0:
                    # _parse_controlled_name() strips ALL leading 'c' chars,
                    # so we get the true base gate and the right control count.
                    #
                    # Examples:
                    #   cx    → control(x_kernel,   q0,       q1)
                    #   ccx   → control(x_kernel,  [q0,q1],   q2)
                    #   cccx  → control(x_kernel,  [q0,q1,q2],q3)
                    #   cswap → control(swap_kernel, q0, q1,q2)
                    # --------------------------------------------------------
                    gate = get_gate(base_gate, targs)
                    ctrl_qubits = qubit_refs[:num_ctrl]
                    target_qubits = qubit_refs[num_ctrl:]

                    # CUDA-Q accepts a single QuakeValue *or* a list of them
                    ctrl_arg = ctrl_qubits[0] if len(ctrl_qubits) == 1 else ctrl_qubits
                    kernel.control(gate, ctrl_arg, *target_qubits, *args)
                else:
                    # ----------------------------------------------------------
                    # Non-controlled gate — direct apply_call path (unchanged)
                    # ----------------------------------------------------------
                    gate = get_gate(name, targs)
                    # Fix from qbraid: make sure we use float and not integer
                    args = list(
                        map(
                            lambda x: float(x) if isinstance(x, int) else x,
                            args,
                        )
                    )
                    kernel.apply_call(gate, *qubit_refs, *args)

        else:
            raise Exception(f"Unsupported statement: {statement}")

    return kernel
