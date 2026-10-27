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
registries in ``circuit_testing`` (per-paradigm conversion edges and the
aggregated ``REFERENCE_CIRCUITS``). Adding a conversion or a circuit to a
paradigm module (``gate_paradigm``, ``analog_paradigm``, ``photonic_paradigm``)
automatically extends this battery.
"""

import pytest

from circuit_testing import (
    NONE,
    ZLIB,
    REFERENCE_CIRCUITS,
    UNSUPPORTED_CONVERSIONS,
    build_edges,
    circuit_runnable,
    run_path,
)

COMPRESSIONS = (NONE, ZLIB)

# Expand: edge x circuit x compression. CUDA-Q "counts" edges only support
# deterministic circuits (they have known expected_counts). A circuit can also
# restrict the edges it runs on (``supports_edge``) - e.g. symbolic-parameter
# circuits only survive QASM3/CirqJSON - and can force its oracle. The
# paradigm-agnostic run filter (``circuit_runnable``) centralizes these rules.
CIRCUIT_CONVERSIONS = []
for compression in COMPRESSIONS:
    for edge in build_edges(compression):
        for circuit in REFERENCE_CIRCUITS:
            if not circuit_runnable(edge, circuit):
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


def test_qasm_normalization_limitations():
    from circuit_testing import (
        NONE,
        LOSS_QASM_CONTROL_FLOW,
        LOSS_QASM_GATE_SET,
        LOSS_QASM_GPHASE,
        build_edges,
        loss_report,
        set_loss_context,
    )
    from gate_paradigm import check_qasm_normalization_limitations
    from qio.core import (
        QuantumProgram,
        QuantumProgramCompressionFormat,
        QuantumProgramSerializationFormat,
    )
    from reference_circuits import GateSpec, ReferenceCircuit

    edges = {e.id: e for e in build_edges(NONE)}
    probe = ReferenceCircuit(  # label used in the loss report, not registered
        GateSpec(
            name="qasm_normalization_limits_probe",
            n_qubits=1,
            gates=[("measure", (), (0,))],
        )
    )
    source = (
        "OPENQASM 3.0;\n"
        'include "stdgates.inc";\n'
        "qubit[1] q;\n"
        "bit[1] c;\n"
        "gphase(0.5);\n"
        "u3(0.1, 0.2, 0.3) q[0];\n"
        "h q[0];\n"
        "if (c[0]) x q[0];\n"
        "c[0] = measure q[0];\n"
    )
    program = QuantumProgram(
        serialization_format=QuantumProgramSerializationFormat.QASM_V3,
        compression_format=QuantumProgramCompressionFormat.NONE,
        serialization=source,
    )

    set_loss_context(edges["cirq->qasm3->mimiq"], probe)
    check_qasm_normalization_limitations(program, probe)

    entry = loss_report()["cirq->qasm3->mimiq"]
    for category in (
        LOSS_QASM_CONTROL_FLOW,
        LOSS_QASM_GPHASE,
        LOSS_QASM_GATE_SET,
    ):
        assert category in entry, category
        assert "qasm_normalization_limits_probe" in entry[category]["circuits"]
