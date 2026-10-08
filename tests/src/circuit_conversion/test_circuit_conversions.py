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
"""Test battery: circuit conversions behave correctly on the *same* circuit.

The test cases are not written by hand: they are generated from the declarative
registries in ``circuit_testing`` (conversion edges) and
``reference_circuits`` (static, controlled input circuits). Adding a conversion
or a circuit to those modules automatically extends this battery.
"""

import pytest

from circuit_testing import (
    NONE,
    ZLIB,
    UNSUPPORTED_CONVERSIONS,
    build_edges,
    run_path,
)
from reference_circuits import REFERENCE_CIRCUITS

COMPRESSIONS = (NONE, ZLIB)

# Expand: edge x circuit x compression. CUDA-Q "counts" edges only support
# deterministic circuits (they have known expected_counts). A circuit can also
# restrict the edges it runs on (``supports_edge``) - e.g. symbolic-parameter
# circuits only survive QASM3/CirqJSON - and can force its oracle.
CIRCUIT_CONVERSIONS = []
for compression in COMPRESSIONS:
    for edge in build_edges(compression):
        for circuit in REFERENCE_CIRCUITS:
            if not circuit.supports_edge(edge.id):
                continue
            oracle = circuit.oracle or edge.oracle
            if oracle == "counts" and circuit.expected_counts is None:
                continue
            CIRCUIT_CONVERSIONS.append((edge, circuit, compression))

CIRCUIT_CONVERSIONS_IDS = [
    f"{edge.id}|{circuit.name}|{compression.name}"
    for edge, circuit, compression in CIRCUIT_CONVERSIONS
]


@pytest.mark.parametrize(
    "edge,circuit,compression",
    CIRCUIT_CONVERSIONS,
    ids=CIRCUIT_CONVERSIONS_IDS,
)
def test_circuit_conversion(edge, circuit, compression):
    run_path(edge, circuit)


@pytest.mark.parametrize(
    "case",
    UNSUPPORTED_CONVERSIONS,
    ids=[name for name, _ in UNSUPPORTED_CONVERSIONS],
)
def test_unsupported_conversion(case):
    _, operation = case
    with pytest.raises(Exception):
        operation()
