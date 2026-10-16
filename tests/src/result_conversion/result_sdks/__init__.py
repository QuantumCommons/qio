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
"""Per-SDK adapters of the result conversion battery.

Each module defines exactly one :class:`~result_testing.ResultSDK` subclass
instance (``SDK``). To add a new SDK:

1. create ``result_sdks/<sdk>.py`` (input builder, format check, dict converter
   name, wrap/read classmethods, probes, inherent losses),
2. import it below and append it to ``ALL_SDKS``.

Everything else - the edges, the positive/negative cases and the loss report -
is derived automatically from the registered adapters.
"""

from . import cirq, cudaq, mimiq, perceval, pulser, qiskit

ALL_SDKS = (cirq.SDK, qiskit.SDK, cudaq.SDK, mimiq.SDK, perceval.SDK, pulser.SDK)

SDK_RESULTS = {sdk.kind: sdk for sdk in ALL_SDKS}

__all__ = ["ALL_SDKS", "SDK_RESULTS"]
