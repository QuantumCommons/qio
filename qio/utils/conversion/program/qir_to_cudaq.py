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
"""QIR (LLVM IR) to CUDA-Q kernel conversion.

Reconstructs a ``cudaq.PyKernel`` from a QIR program by "disassembling" the
flat ``__quantum__qis__*`` gate calls (thin QIR emitted by ``cirq_to_qir`` /
``qiskit_to_qir`` / ``qasm3_to_qir``) and replaying them on the CUDA-Q builder
API (``cudaq.make_kernel``). No intermediate serialization format is involved.

Scope: *thin* QIR - a straight-line entry function, qubits as constant pointer
operands (``ptr null`` / ``ptr inttoptr (i64 i to ptr)``) and rotation angles
as literal ``double`` constants. Richer/full QIR (computed SSA angles, dynamic
qubit allocation, mid-circuit control flow) raises :class:`QirToCudaqError`: it
would need a full SSA value resolver, which is out of scope for this converter.

The decoding step (:func:`decode`) only depends on ``pyqir`` and is therefore
fully testable without CUDA-Q; the CUDA-Q builder emission (:func:`convert`) is
a thin layer on top.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple, Union

import cudaq
from cudaq import PyKernel
from pyqir import Context, FloatConstant, Function, Module

_QIS_CALL = re.compile(
    r"^__quantum__qis__(?P<base>[a-z0-9]+(?:_[a-z0-9]+)*)"
    r"(?:__(?P<variant>body|adj|ctl))?$"
)

_PTR_TO_INDEX = re.compile(r"^ptr inttoptr \(i64 (\d+) to ptr\)$")

_SIMPLE_ONE_QUBIT = {"h", "x", "y", "z", "s", "t", "s_adj", "t_adj"}
_ADJOINT_MAP = {"s_adj": "sdg", "t_adj": "tdg"}
_ROTATIONS = {"rx", "ry", "rz"}
_SELF_ADJOINT = {"h", "x", "y", "z", "cnot", "cz", "swap"}


class QirToCudaqError(Exception):
    """Raised when a QIR program cannot be converted to a CUDA-Q kernel."""


def _qubit_index(operand) -> int:
    """Return the qubit index encoded by a constant QIR qubit operand."""
    text = str(operand)
    if text == "ptr null":
        return 0
    match = _PTR_TO_INDEX.fullmatch(text)
    if match:
        return int(match.group(1))
    raise QirToCudaqError(f"unsupported QIR qubit operand: {text!r}")


def _rotation_angle(operand) -> float:
    """Return the literal double value of a QIR rotation angle operand."""
    if isinstance(operand, FloatConstant):
        return float(operand.value)
    raise QirToCudaqError(f"unsupported QIR rotation angle operand: {operand!r}")


def _entry_function(module: Module) -> Function:
    """Return the module's single defined (entry) function."""
    for fn in module.functions:
        if fn.basic_blocks:\
            return fn
    raise QirToCudaqError("no defined entry function found in QIR module")


def _required_num_qubits(fn: Function) -> Optional[int]:
    """Read ``required_num_qubits`` from the entry function attributes."""
    attrs = str(fn.attributes)
    match = re.search(r"required_num_qubits[^\d]*(\d+)", attrs)
    if match:
        return int(match.group(1))
    return None

def decode(module: Module) -> Tuple[int, List[Tuple[str, tuple]]]:
    """
    Decode a thin QIR module into a list of operations.
    """
    entry = _entry_function(module)
    ops: List[Tuple[str, tuple]] = []
    max_qubit = -1

    for bb in entry.basic_blocks:
        for inst in bb.instructions:
            if not hasattr(inst, "callee"):
                continue
            call = _QIS_CALL.fullmatch(inst.callee.name)
            if call is None:
                continue
            base, variant = call.group("base"), call.group("variant")
            if variant == "adj" and base not in _SELF_ADJOINT:
                if base in ("s", "t"):
                    base += "_adj"
                else:
                    raise QirToCudaqError(
                        f"unsupported QIR adjoint operation: {inst.callee.name}"
                    )
            args: list = [a for a in inst.operands if type(a).__name__ != "Function"]

            if base in _SIMPLE_ONE_QUBIT:
                (q,) = (_qubit_index(a) for a in args)
                max_qubit = max(max_qubit, q)
                ops.append((base, (q,)))
            elif base in _ROTATIONS:
                angle, q = _rotation_angle(args[0]), _qubit_index(args[1])
                max_qubit = max(max_qubit, q)
                ops.append((base, (angle, q)))
            elif base in ("cnot", "cz"):
                c, t = (_qubit_index(a) for a in args)
                max_qubit = max(max_qubit, c, t)
                ops.append((base, (c, t)))
            elif base == "ccx":
                c1, c2, t = (_qubit_index(a) for a in args)
                max_qubit = max(max_qubit, c1, c2, t)
                ops.append((base, (c1, c2, t)))
            elif base == "swap":
                q0, q1 = (_qubit_index(a) for a in args[:2])
                max_qubit = max(max_qubit, q0, q1)
                ops.append((base, (q0, q1)))
            elif base in ("mz", "reset"):
                q = _qubit_index(args[0])
                max_qubit = max(max_qubit, q)
                ops.append((base, (q,)))
            else:
                raise QirToCudaqError(
                    f"unsupported QIR quantum operation: {inst.callee.name}"
                )

    num_qubits = _required_num_qubits(entry)
    if num_qubits is None:
        num_qubits = max(0, max_qubit + 1)
    return num_qubits, ops


def _gate_kernel(gate_kernels: dict, *names: str) -> PyKernel:
    """Return (and cache) a bare one-qubit CUDA-Q gate kernel for ``control``."""
    key = names
    if key not in gate_kernels:
        gk, q = cudaq.make_kernel(cudaq.qubit)
        for name in names:
            getattr(gk, name)(q)
        gate_kernels[key] = gk
    return gate_kernels[key]


def _emit_ops(ops: List[Tuple[str, tuple]], num_qubits: int) -> PyKernel:
    """Replay a decoded op list on a new ``cudaq.PyKernel``."""
    kernel: PyKernel = cudaq.make_kernel()
    if num_qubits > 0:
        qreg = kernel.qalloc(num_qubits)

    gate_kernels: dict = {}

    def method(base: str) -> str:
        return _ADJOINT_MAP.get(base, base)

    for base, operands in ops:
        if base in _SIMPLE_ONE_QUBIT:
            (q,) = operands
            getattr(kernel, method(base))(qreg[q])
        elif base in _ROTATIONS:
            angle, q = operands
            getattr(kernel, base)(float(angle), qreg[q])
        elif base == "cnot":
            c, t = operands
            kernel.control(_gate_kernel(gate_kernels, "x"), qreg[c], qreg[t])
        elif base == "cz":
            c, t = operands
            kernel.control(_gate_kernel(gate_kernels, "z"), qreg[c], qreg[t])
        elif base in ("ccx",):
            raise QirToCudaqError(
                "unsupported QIR operation: ccx (Toffoli) needs multi-qubit "
                "control, which the CUDA-Q PyKernel builder cannot express; "
                "decompose the Toffoli circuit before QIR emission"
            )
        elif base == "swap":
            q0, q1 = operands
            kernel.swap(qreg[q0], qreg[q1])
        elif base == "mz":
            (q,) = operands
            kernel.mz(qreg[q])
        elif base == "reset":
            (q,) = operands
            kernel.reset(qreg[q])
    return kernel


def convert(qir: Union[str, Module]) -> PyKernel:
    """Convert a thin QIR program (LLVM IR text or ``pyqir.Module``) to a
    CUDA-Q ``PyKernel``.

    Args:
        qir: QIR LLVM IR text, or a parsed ``pyqir.Module``.

    Returns:
        The equivalent CUDA-Q kernel (measurements via ``mz`` are recorded).

    Raises:
        TypeError: if ``qir`` is neither ``str`` nor ``pyqir.Module``.
        QirToCudaqError: if the module fails verification or uses QIR features
            outside the thin profile supported by this converter.
    """
    if isinstance(qir, str):
        module = Module.from_ir(Context(), qir, "qir_input")
    elif isinstance(qir, Module):
        module = qir
    else:
        raise TypeError(f"expected str or pyqir.Module, got {type(qir).__name__}")

    if module.verify() is not None:
        raise QirToCudaqError("QIR module failed verification")

    num_qubits, ops = decode(module)
    return _emit_ops(ops, num_qubits)
