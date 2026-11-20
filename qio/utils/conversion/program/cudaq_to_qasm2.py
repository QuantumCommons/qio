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
import cudaq

from cudaq import PyKernel

_TRANSLATE_FAILED = "{translation failed}"


def convert(cudaq_kernel: PyKernel) -> str:
    qasm2 = cudaq.translate(cudaq_kernel, format="openqasm2")
    if qasm2.strip() == _TRANSLATE_FAILED:
        raise RuntimeError(
            "CUDA-Q failed to lower this kernel to OpenQASM 2 "
            f"(the emitter returned {_TRANSLATE_FAILED})"
        )
    return qasm2
