import pyqasm
from cudaq import PyKernel

from .qasm3_to_cudaq import convert as qasm3_to_cudaq_convert


def convert(circuit_str: str) -> PyKernel:
    qasm_module = pyqasm.loads(circuit_str)
    obj_qasm3 = qasm_module.to_qasm3(as_str=True)
    return qasm3_to_cudaq_convert(obj_qasm3)
