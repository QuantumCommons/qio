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
"""Photonic programming paradigm of the conversion battery.

Concrete :class:`Paradigm` of the *photonic* circuits: the static reference
circuits (``ph_h``, ``ph_mzi``, ``ph_cnot``, ``ph_bell``) described as
:class:`PhotonicSpec` mode-based specs, the two round-trip edges
(``perceval -> percevaljson -> perceval`` for ``perceval.Circuit`` and
``perceval_experiment -> percevaljson -> perceval_experiment`` for
``perceval.Experiment`` - like the analog one, qio only maps the perceval
serialization to/from these two wrappers) and the Perceval SDK checks.
Semantic equivalence is closed by the ``"photonic"`` oracle (registered in
:data:`EXTRA_ORACLES`): the exact post-selected SLOS output distribution of
the round-tripped object must equal the reference's, computed on the native
Fock input state (with the heralded CNOT's ancilla seeded).

Perceval and mimiqcircuits both register a ``circuit.proto`` in protobuf's
global descriptor pool, so they cannot be imported in the same Python process.
The photonic battery therefore runs in its own pytest invocation (no mimiq
there): see ``tests/Makefile`` (``test`` runs two invocations) and the README.
"""

from typing import Any, Callable, Dict, Optional, Tuple

from qio.core import QuantumProgram, QuantumProgramSerializationFormat as Serialization

from reference_circuits import (
    PARADIGM_PHOTONIC,
    PhotonicGate,
    PhotonicSpec,
    ReferenceCircuit,
)

from circuit_testing import (
    Compression,
    EXTRA_ORACLES,
    Paradigm,
    _resolve_serialization,
    check_program,
    get_reference_circuit,
)


def _encode(circuit) -> str:
    from qio.utils.conversion.program.perceval_to_percevaljson import (
        convert as perceval_to_percevaljson_convert,
    )

    return perceval_to_percevaljson_convert(circuit)


def _decode(source: str):
    from qio.utils.conversion.program.percevaljson_to_perceval import (
        convert as percevaljson_to_perceval_convert,
    )

    return percevaljson_to_perceval_convert(source)


def _simulate_probs(circuit, spec: PhotonicSpec) -> Dict[Any, float]:
    """Exact post-selected output distribution (mode -> probability) of
    ``circuit`` fed with the spec's Fock input state. When ``circuit`` is a
    ``perceval.Experiment``, its own input state and heralds are used instead
    of the spec's.

    Uses Perceval's SLOS backend (exact interference simulation, no sampling
    noise) with the herald post-selection: ``set_heralds`` + 
    ``keep_heralds(False)`` drops the herald modes from the returned states, so
    the distribution is over the data modes only.
    """
    import perceval as pcvl
    from perceval import Simulator
    from perceval.backends import SLOSBackend

    simulator = Simulator(backend=SLOSBackend())
    if isinstance(circuit, pcvl.Experiment):
        simulator.set_circuit(circuit.unitary_circuit())
        heralds = circuit.heralds
        input_state = circuit.input_state
    else:
        simulator.set_circuit(circuit)
        heralds = spec.heralds
        input_state = pcvl.BasicState(list(spec.input_state))
    if heralds:
        simulator.set_heralds(dict(heralds))
    simulator.keep_heralds(False)
    probs = simulator.probs(input_state)
    return {state: float(prob) for state, prob in probs.items()}


def assert_photonic_probs(result, expected: ReferenceCircuit) -> None:
    """The ``"photonic"`` oracle: the round-tripped ``perceval.Circuit`` (or
    ``perceval.Experiment``) must produce the exact same post-selected output
    distribution as the statically built reference circuit (both simulated with
    SLOS on the spec's input state - the experiment's own input state/heralds
    for the experiment edge). Guards every step of the perceval round trips
    semantically without sampling noise."""
    import math

    spec = expected.photonic
    assert spec is not None, f"no photonic spec on {expected.name!r}"
    ref = _simulate_probs(expected.perceval(), spec)
    act = _simulate_probs(result, spec)
    assert set(act.keys()) == set(ref.keys()), (
        "output modes differ after the round trip",
        set(act.keys()),
        set(ref.keys()),
    )
    for state in ref:
        assert math.isclose(act[state], ref[state], rel_tol=0.0, abs_tol=1e-9), (
            state,
            ref[state],
            act[state],
        )


EXTRA_ORACLES["photonic"] = assert_photonic_probs


def check_perceval(result, expected: Optional[ReferenceCircuit] = None) -> None:
    """Validates a final ``perceval.Circuit`` against the reference
    :class:`PhotonicSpec`.

    The perceval serialization is lossless (``Serialization.serialize`` /
    ``deserialize``) and canonical, so - like ``check_pulser`` for Pulser - the
    strong check is exact structural equality with the *statically built*
    reference circuit (mode count + re-serialized bytes). The semantic content
    is then enforced by the ``"photonic"`` oracle on the same object.
    """
    import perceval as pcvl

    assert isinstance(result, pcvl.Circuit), f"expected perceval Circuit, got {type(result)}"
    if expected is not None:
        assert result.m == expected.n_modes, (result.m, expected.n_modes)
        assert _encode(result) == _encode(expected.perceval()), (
            "round-tripped perceval circuit differs from the reference"
        )


def check_perceval_json(result, expected: Optional[ReferenceCircuit] = None) -> None:
    """Content check of the ``PERCEVAL_CIRCUIT_JSON_V1`` intermediate: it must
    deserialize to a valid ``perceval.Circuit`` (right mode count) and
    re-serialize identically - the perceval serialization is canonical."""
    import perceval as pcvl

    source = _resolve_serialization(result)
    assert source, "empty perceval serialization"
    circuit = _decode(source)
    assert isinstance(circuit, pcvl.Circuit), type(circuit)
    if expected is not None:
        assert circuit.m == expected.n_modes, (circuit.m, expected.n_modes)
    assert _encode(circuit) == source


def check_perceval_experiment(result, expected: Optional[ReferenceCircuit] = None) -> None:
    import perceval as pcvl

    assert isinstance(result, pcvl.Experiment), f"expected perceval Experiment, got {type(result)}"
    if expected is not None:
        assert result.circuit_size == expected.n_modes, (result.circuit_size, expected.n_modes)
        assert _encode(result) == _encode(expected.perceval_experiment()), (
            "round-tripped perceval experiment differs from the reference"
        )


def check_perceval_experiment_json(result, expected: Optional[ReferenceCircuit] = None) -> None:
    import perceval as pcvl

    source = _resolve_serialization(result)
    assert source, "empty perceval serialization"
    experiment = _decode(source)
    assert isinstance(experiment, pcvl.Experiment), type(experiment)
    if expected is not None:
        assert experiment.circuit_size == expected.n_modes, (experiment.circuit_size, expected.n_modes)
    assert _encode(experiment) == source


def _ph_cnot():
    return get_reference_circuit("ph_cnot").perceval()


def _perceval_program(dest_format: Serialization) -> QuantumProgram:
    return QuantumProgram.from_perceval_circuit(_ph_cnot(), dest_format=dest_format)


def _perceval_experiment_program(dest_format: Serialization) -> QuantumProgram:
    return QuantumProgram.from_perceval_experiment(
        get_reference_circuit("ph_cnot").perceval_experiment(), dest_format=dest_format
    )


class PhotonicParadigm(Paradigm):
    """The photonic paradigm: mode-based native perceval circuits (dual-rail
    qubits, beam splitters, parametrized phase shifters and the heralded CNOT
    block with ancilla + herald post-selection), closed by the ``"photonic"``
    oracle (exact SLOS distribution)."""

    name = PARADIGM_PHOTONIC
    circuits = (
        ReferenceCircuit(
            PhotonicSpec(
                name="ph_h",
                n_qubits=1,
                gates=[
                    PhotonicGate("bs", modes=(0, 1)),
                ],
                input_state=(1, 0),
                measure_qubits=(0, 1),
                description=(
                    "Single-qubit Hadamard as a 50/50 beam splitter on modes "
                    "(0, 1): a photon in q=0 (mode 0) leaves in the balanced "
                    "superposition."
                ),
            ),
        ),
        ReferenceCircuit(
            PhotonicSpec(
                name="ph_mzi",
                n_qubits=1,
                gates=[
                    PhotonicGate("ps", modes=(0,), params=(0.2,)),
                    PhotonicGate("bs", modes=(0, 1)),
                    PhotonicGate("ps", modes=(0,), params=(0.7,)),
                    PhotonicGate("bs", modes=(0, 1)),
                ],
                input_state=(1, 0),
                measure_qubits=(0, 1),
                description=(
                    "Mach-Zehnder interferometer on modes (0, 1): PS(0.2), BS, "
                    "PS(0.7), BS; exercises parametrized phase shifters through "
                    "the serialization."
                ),
            ),
        ),
        ReferenceCircuit(
            PhotonicSpec(
                name="ph_cnot",
                n_qubits=2,
                gates=[
                    PhotonicGate("heralded_cnot", qubits=(0, 1)),
                ],
                input_state=(1, 0, 1, 0, 1, 1),
                measure_qubits=(0, 1),
                heralds={4: 1, 5: 1},
                description=(
                    "Native Knill heralded CNOT (6 modes: ctrl dual-rail 0,1; "
                    "data dual-rail 2,3; ancilla/heralds 4,5) on the |00> input "
                    "with the ancilla seeded |1,1>; post-selects on {4:1, 5:1}."
                ),
            ),
        ),
        ReferenceCircuit(
            PhotonicSpec(
                name="ph_bell",
                n_qubits=2,
                gates=[
                    PhotonicGate("bs", modes=(0, 1)),
                    PhotonicGate("heralded_cnot", qubits=(0, 1)),
                ],
                input_state=(1, 0, 1, 0, 1, 1),
                measure_qubits=(0, 1),
                heralds={4: 1, 5: 1},
                description=(
                    "Photonic Bell state: H on qubit 0 (BS on modes 0,1) then "
                    "the heralded CNOT - (|00>+|11>)/sqrt2 on the data modes "
                    "after herald post-selection."
                ),
            ),
        ),
    )
    input_fns = {
        "perceval": lambda c: c.perceval(),
        "perceval_experiment": lambda c: c.perceval_experiment(),
    }

    def content_checks(self, fmt: Serialization) -> Tuple[Callable, ...]:
        if fmt in (Serialization.PERCEVAL_CIRCUIT_JSON_V1,):
            return (check_perceval_json,)
        if fmt in (Serialization.PERCEVAL_EXPERIMENT_JSON_V1,):
            return (check_perceval_experiment_json,)
        return ()

    def build_edges(self, compression: Compression):
        # Closed self-round-trips (like the analog paradigm): qio only maps
        # the perceval serialization to/from ``perceval.Circuit`` and
        # ``perceval.Experiment``, so there is one edge per wrapper. The
        # perceval round trip is lossless at the checked level (canonical
        # serialization: the ``check_*`` content checks assert byte-exact
        # equality), and its semantics are enforced by the "photonic" oracle
        # (exact post-selected SLOS distribution).
        edges = [
            self.ConversionEdge(
                "perceval->percevaljson->perceval",
                "perceval",
                self.input_fns["perceval"],
                (
                    self.Step(
                        QuantumProgram.from_perceval_circuit,
                        {
                            "dest_format": Serialization.PERCEVAL_CIRCUIT_JSON_V1,
                            "compression_format": compression,
                        },
                        (check_program(Serialization.PERCEVAL_CIRCUIT_JSON_V1),)
                        + self.content_checks(Serialization.PERCEVAL_CIRCUIT_JSON_V1),
                    ),
                    self.Step(
                        QuantumProgram.to_perceval_circuit, {}, (check_perceval,)
                    ),
                ),
                oracle="photonic",
                known_losses=(),
                paradigm=self.name,
            ),
            self.ConversionEdge(
                "perceval_experiment->percevaljson->perceval_experiment",
                "perceval_experiment",
                self.input_fns["perceval_experiment"],
                (
                    self.Step(
                        QuantumProgram.from_perceval_experiment,
                        {
                            "dest_format": Serialization.PERCEVAL_EXPERIMENT_JSON_V1,
                            "compression_format": compression,
                        },
                        (check_program(Serialization.PERCEVAL_EXPERIMENT_JSON_V1),)
                        + self.content_checks(Serialization.PERCEVAL_EXPERIMENT_JSON_V1),
                    ),
                    self.Step(
                        QuantumProgram.to_perceval_experiment, {}, (check_perceval_experiment,)
                    ),
                ),
                oracle="photonic",
                known_losses=(),
                paradigm=self.name,
            ),
        ]
        return tuple(edges)

    unsupported_conversions = (
        (
            "from_perceval_circuit -> QASM_V2",
            lambda: QuantumProgram.from_perceval_circuit(
                _ph_cnot(), dest_format=Serialization.QASM_V2
            ),
        ),
        (
            "to_qiskit_circuit on PERCEVALJSON",
            lambda: _perceval_program(Serialization.PERCEVAL_CIRCUIT_JSON_V1).to_qiskit_circuit(),
        ),
        (
            "to_cirq_circuit on PERCEVALJSON",
            lambda: _perceval_program(Serialization.PERCEVAL_CIRCUIT_JSON_V1).to_cirq_circuit(),
        ),
        (
            "to_mimiq_circuit on PERCEVALJSON",
            lambda: _perceval_program(Serialization.PERCEVAL_CIRCUIT_JSON_V1).to_mimiq_circuit(),
        ),
        (
            "to_cudaq_kernel on PERCEVALJSON",
            lambda: _perceval_program(Serialization.PERCEVAL_CIRCUIT_JSON_V1).to_cudaq_kernel(),
        ),
        (
            "to_pulser_sequence on PERCEVALJSON",
            lambda: _perceval_program(Serialization.PERCEVAL_CIRCUIT_JSON_V1).to_pulser_sequence(),
        ),
        (
            "to_qasm2_circuit on PERCEVALJSON",
            lambda: _perceval_program(Serialization.PERCEVAL_CIRCUIT_JSON_V1).to_qasm2_circuit(),
        ),
        (
            "from_perceval_experiment -> QASM_V2",
            lambda: QuantumProgram.from_perceval_experiment(
                get_reference_circuit("ph_cnot").perceval_experiment(),
                dest_format=Serialization.QASM_V2,
            ),
        ),
        (
            "to_perceval_circuit on PERCEVAL_EXPERIMENTJSON",
            lambda: _perceval_experiment_program(
                Serialization.PERCEVAL_EXPERIMENT_JSON_V1
            ).to_perceval_circuit(),
        ),
        (
            "to_perceval_experiment on PERCEVALJSON",
            lambda: _perceval_program(Serialization.PERCEVAL_CIRCUIT_JSON_V1).to_perceval_experiment(),
        ),
        (
            "to_qiskit_circuit on PERCEVAL_EXPERIMENTJSON",
            lambda: _perceval_experiment_program(
                Serialization.PERCEVAL_EXPERIMENT_JSON_V1
            ).to_qiskit_circuit(),
        ),
        (
            "to_cirq_circuit on PERCEVAL_EXPERIMENTJSON",
            lambda: _perceval_experiment_program(
                Serialization.PERCEVAL_EXPERIMENT_JSON_V1
            ).to_cirq_circuit(),
        ),
        (
            "to_mimiq_circuit on PERCEVAL_EXPERIMENTJSON",
            lambda: _perceval_experiment_program(
                Serialization.PERCEVAL_EXPERIMENT_JSON_V1
            ).to_mimiq_circuit(),
        ),
        (
            "to_cudaq_kernel on PERCEVAL_EXPERIMENTJSON",
            lambda: _perceval_experiment_program(
                Serialization.PERCEVAL_EXPERIMENT_JSON_V1
            ).to_cudaq_kernel(),
        ),
        (
            "to_pulser_sequence on PERCEVAL_EXPERIMENTJSON",
            lambda: _perceval_experiment_program(
                Serialization.PERCEVAL_EXPERIMENT_JSON_V1
            ).to_pulser_sequence(),
        ),
        (
            "to_qasm2_circuit on PERCEVAL_EXPERIMENTJSON",
            lambda: _perceval_experiment_program(
                Serialization.PERCEVAL_EXPERIMENT_JSON_V1
            ).to_qasm2_circuit(),
        ),
    )

    feature_coverage = (
        (
            "photonic mode-based circuits",
            "round-trip tested",
            "native perceval circuits (dual-rail qubits, beam splitters, "
            "parametrized phase shifters and the heralded CNOT block with "
            "ancilla) round trip exactly through the perceval serialization on "
            "the 'perceval->percevaljson->perceval' edge; the 'photonic' "
            "oracle compares the exact post-selected SLOS output distribution "
            "before/after. NOTE: the intermediate QuantumProgram carries only "
            "the perceval.Circuit archive - the Fock input state, the herald "
            "condition and the post-selection are part of the PhotonicSpec and "
            "are re-applied by the oracle *outside* qio, so a full photonic "
            "processor configuration is out of the qio formats",
        ),
        (
            "photonic experiment context (input state + heralds)",
            "round-trip tested",
            "the perceval.Experiment archive carries the Fock input state and "
            "the output herald post-selection alongside the circuit; the "
            "'perceval_experiment->percevaljson->perceval_experiment' edge "
            "round-trips them exactly and the shared 'photonic' oracle "
            "simulates from the experiment's own context (no re-application by "
            "the test)",
        ),
    )
