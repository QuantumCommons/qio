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
import pyqasm

from cudaq import PyKernel

from .qasm3_to_cudaq import convert as qasm3_to_cudaq_convert


def convert(circuit_str: str) -> PyKernel:
    qasm_module = pyqasm.loads(circuit_str)
    obj_qasm3 = qasm_module.to_qasm3(as_str=True)
    return qasm3_to_cudaq_convert(obj_qasm3)
