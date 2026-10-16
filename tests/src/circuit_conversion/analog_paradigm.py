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
"""Analog programming paradigm of the conversion battery.

Concrete :class:`Paradigm` of the *analog* circuits: the static reference
circuits (``analog_global``, ``analog_local``) described as :class:`AnalogSpec`
pulse schedules, the single round-trip edge
(``pulser -> pulserjson -> pulser``, analogous to the photonic one in
``photonic_paradigm``) and the Pulser SDK checks.
"""

from typing import Callable, Optional, Tuple

from qio.core import QuantumProgram, QuantumProgramSerializationFormat as Serialization

from reference_circuits import (
    PARADIGM_ANALOG,
    AnalogSpec,
    PulseSpec,
    ReferenceCircuit,
)

from circuit_testing import (
    Compression,
    LOSS_ANALOG_DEVICE,
    Paradigm,
    _resolve_serialization,
    check_program,
    get_reference_circuit,
    record_loss,
)


def check_pulser(result, expected: Optional[ReferenceCircuit] = None) -> None:
    """Validates an analog Pulser ``Sequence`` against the reference
    :class:`AnalogSpec`.

    A Pulser sequence has no gate chain nor a cheaply-computable unitary, so
    the semantic invariant is Pulser's own canonical, device-independent
    serialization: ``Sequence.to_abstract_repr``. The check is therefore:
    instance, atom count, declared channels, measurement presence and device
    equality (public API), plus exact equality with the *statically built*
    reference sequence's abstract representation - the analog counterpart of
    the gate op-chain comparison, and exact because the Pulser round trip is
    lossless. The ``analog_device`` loss guards the device object only (pulser
    1.9.1 preserves it; a future version dropping it is recorded, not fatal).

    Measurement presence is compared against the reference sequence (not the
    spec boolean): on a *parametrized* sequence pulser 1.9.1 defers
    ``_measurement`` to build time (``None``) even though the measured basis is
    faithfully carried in the abstract repr - the byte-exact equality below
    covers it either way.
    """
    import pulser

    assert isinstance(result, pulser.Sequence), f"expected pulser.Sequence, got {type(result)}"
    if expected is None or expected.paradigm != PARADIGM_ANALOG:
        return
    spec = expected.analog
    assert spec is not None, f"no analog spec on {expected.name!r}"
    assert len(result.register.qubits) == expected.n_qubits, (
        len(result.register.qubits),
        expected.n_qubits,
    )
    assert set(result.declared_channels) == set(spec.channels), (
        set(result.declared_channels),
        set(spec.channels),
    )
    reference_seq = expected.pulser()
    assert bool(getattr(result, "_measurement", None)) == bool(
        getattr(reference_seq, "_measurement", None)
    ), (
        getattr(result, "_measurement", None),
        getattr(reference_seq, "_measurement", None),
    )
    if result.device != reference_seq.device:
        record_loss(
            LOSS_ANALOG_DEVICE,
            detail=(
                f"{reference_seq.device.name} -> {result.device.name} "
                f"(device not preserved by the abstract repr round trip)"
            ),
        )
    assert result.to_abstract_repr() == reference_seq.to_abstract_repr()


def check_pulser_json(result, expected: Optional[ReferenceCircuit] = None) -> None:
    """Content check of the ``PULSER_SEQUENCE_JSON_V1`` intermediate: it must
    be a valid ``Sequence`` abstract repr (parses, right atom count) and
    re-serialize identically - Pulser's abstract repr is canonical."""
    import pulser

    source = _resolve_serialization(result)
    assert source, "empty Pulser abstract repr"
    sequence = pulser.Sequence.from_abstract_repr(source)
    assert isinstance(sequence, pulser.Sequence), type(sequence)
    if expected is not None:
        assert len(sequence.register.qubits) == expected.n_qubits, (
            len(sequence.register.qubits),
            expected.n_qubits,
        )
    assert sequence.to_abstract_repr() == source


def _analog_global_pulser():
    return get_reference_circuit("analog_global").pulser()


def _pulser_program(dest_format: Serialization) -> QuantumProgram:
    return QuantumProgram.from_pulser_sequence(
        _analog_global_pulser(), dest_format=dest_format
    )


def _bare_variable_pulser() -> "pulser.Sequence":
    """A sequence using a *bare* variable as a pulse amplitude.

    Pulser 1.9.1 cannot serialize this to the abstract repr (the produced JSON
    fails the schema validation - ``Sequence.to_abstract_repr`` raises
    ``AbstractReprError``), so ``from_pulser_sequence`` on it must raise. This
    is recorded as an expected unsupported conversion so that a future pulser
    fixing the serialization makes the test fail loudly instead of silently
    widening qio's pulser support.
    """
    from pulser import Pulse, Sequence
    from pulser.devices import MockDevice
    from pulser.register import Register
    from pulser.waveforms import ConstantWaveform

    seq = Sequence(Register.from_coordinates([(0, 0), (0, 10)], prefix="q"), MockDevice)
    seq.declare_channel("rydberg", "rydberg_global")
    x = seq.declare_variable("x", 1)
    seq.add(
        Pulse(ConstantWaveform(100, x), ConstantWaveform(100, 0.0), 0.0),
        "rydberg",
    )
    seq.measure("ground-rydberg")
    return seq


class AnalogParadigm(Paradigm):
    """The analog paradigm: analog pulse schedules as Pulser ``Sequence``s,
    closed by the ``structural`` oracle (pulser has no cheap unitary - the
    exact abstract-repr equality in ``check_pulser`` is the semantic check)."""

    name = PARADIGM_ANALOG
    circuits = (
        ReferenceCircuit(
            AnalogSpec(
                name="analog_global",
                register_coords=[(0, 0), (0, 10)],
                channels={"rydberg": "rydberg_global"},
                pulses=[
                    PulseSpec(
                        channel="rydberg",
                        kind="constant",
                        duration=100,
                        amplitude=1.0,
                        detuning=0.1,
                        phase=0.0,
                    ),
                ],
                measure=True,
                description=(
                    "2-atom analog schedule on a global Rydberg channel (constant "
                    "pulse) with a measurement; round trips through "
                    "QuantumProgram.from/to_pulser_sequence."
                ),
            ),
        ),
        ReferenceCircuit(
            AnalogSpec(
                name="analog_local",
                register_coords=[(0, 0), (10, 0)],
                channels={"local": "rydberg_local"},
                pulses=[
                    PulseSpec(
                        channel="local",
                        kind="constant",
                        duration=40,
                        amplitude=0.5,
                        detuning=0.2,
                        phase=0.3,
                        target="q0",
                    ),
                    PulseSpec(
                        channel="local",
                        kind="blackman",
                        duration=60,
                        amplitude=0.8,
                        detuning=0.0,
                        phase=0.0,
                        target="q1",
                    ),
                ],
                measure=False,
                description=(
                    "2-atom analog schedule on a local Rydberg channel: one constant "
                    "pulse addressed to q0 and one Blackman pulse to q1, no "
                    "measurement."
                ),
            ),
        ),
        ReferenceCircuit(
            AnalogSpec(
                name="analog_param",
                register_coords=[(0, 0), (0, 10)],
                channels={"rydberg": "rydberg_global"},
                variables=(("x", "float"),),
                pulses=[
                    PulseSpec(
                        channel="rydberg",
                        kind="constant",
                        duration=100,
                        amplitude="2*x + 0.5",
                        phase="0.1*x",
                    ),
                ],
                measure=True,
                description=(
                    "Parametrized 2-atom analog schedule on a global Rydberg "
                    "channel: the amplitude and phase are *expressions* over the "
                    "declared variable 'x' ('2*x+0.5', '0.1*x'), which round trip "
                    "symbolically through the Pulser abstract repr."
                ),
            ),
        ),
    )
    input_fns = {
        "pulser": lambda c: c.pulser(),
    }

    def content_checks(self, fmt: Serialization) -> Tuple[Callable, ...]:
        if fmt in (Serialization.PULSER_SEQUENCE_JSON_V1,):
            return (check_pulser_json,)
        return ()

    def build_edges(self, compression: Compression):
        # The analog paradigm is a closed self-round-trip (like the photonic
        # one): the qio QuantumProgram only maps the PULSER abstract repr
        # to/from Pulser, so there is exactly one edge. There is no cheap
        # unitary for a pulser sequence, so the edge closes on the "structural"
        # oracle; semantic correctness is enforced by ``check_pulser`` (exact
        # abstract-repr equality with the reference spec) and
        # ``check_pulser_json`` on the intermediate. The Pulser round trip is
        # lossless at the checked level - except the device, preserved with
        # pulser 1.9.1 / reference devices but declared as ``analog_device`` as
        # a guard in case a future version drops it (it would then be recorded,
        # not fail).
        edges = [
            self.ConversionEdge(
                "pulser->pulserjson->pulser",
                "pulser",
                self.input_fns["pulser"],
                (
                    self.Step(
                        QuantumProgram.from_pulser_sequence,
                        {
                            "dest_format": Serialization.PULSER_SEQUENCE_JSON_V1,
                            "compression_format": compression,
                        },
                        (check_program(Serialization.PULSER_SEQUENCE_JSON_V1),)
                        + self.content_checks(Serialization.PULSER_SEQUENCE_JSON_V1),
                    ),
                    self.Step(QuantumProgram.to_pulser_sequence, {}, (check_pulser,)),
                ),
                oracle="structural",
                known_losses=(LOSS_ANALOG_DEVICE,),
                paradigm=self.name,
            ),
        ]
        return tuple(edges)

    unsupported_conversions = (
        (
            "to_mimiq_circuit on PULSERJSON",
            lambda: _pulser_program(Serialization.PULSER_SEQUENCE_JSON_V1).to_mimiq_circuit(),
        ),
        (
            "from_pulser_sequence -> QASM_V2",
            lambda: QuantumProgram.from_pulser_sequence(
                _analog_global_pulser(), dest_format=Serialization.QASM_V2
            ),
        ),
        (
            "to_cirq_circuit on PULSERJSON",
            lambda: _pulser_program(Serialization.PULSER_SEQUENCE_JSON_V1).to_cirq_circuit(),
        ),
        (
            "from_pulser_sequence on bare-variable pulse",
            lambda: QuantumProgram.from_pulser_sequence(
                _bare_variable_pulser(),
                dest_format=Serialization.PULSER_SEQUENCE_JSON_V1,
            ),
        ),
    )

    feature_coverage = (
        (
            "calibrations & pulse",
            "partial (analog)",
            "no Qiskit Pulse calibrations / QASM3 defcal support yet, but the "
            "analog paradigm is exercised on the 'pulser->pulserjson->pulser' "
            "edge: AnalogSpec (register coords, global/local channels, constant & "
            "Blackman pulses, measurement) round trips exactly through the "
            "Pulser abstract repr",
        ),
        (
            "analog pulse parametrization",
            "partial (expression round-trip)",
            "variables declared on a sequence round trip symbolically when used "
            "as *expressions* ('2*x+0.5' amplitude / '0.1*x' phase on the "
            "analog_param circuit); a variable used *bare* as a pulse value "
            "cannot be serialized by pulser 1.9.1 (AbstractReprError) and is "
            "pinned as an expected unsupported conversion",
        ),
    )
