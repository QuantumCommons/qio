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
"""Modular test helpers for the result conversion battery.

Design (mirrors ``circuit_testing`` for circuits):
- **One format checker per SDK** : ``check_cirq``, ``check_qiskit``,
  ``check_cudaq``, ``check_mimiq``. Each validates the object type and reduces
  the result to its canonical **counts histogram** (``{bitstring: count}``)
  which is compared bit-exactly against the reference specification
  (``expected.counts``) - plus the total shot count.
- **Static inputs** : every reference result is an already executed run
  (``{bitstring: count}``), exposed identically in every SDK format by
  ``reference_results``. Because all SDK fixtures are built from the *same*
  bitstring set, all existing converters must preserve the histogram exactly:
  any deviation is a bug (or a declared, inherent loss).
- **Input results are NOT checked** : they are built statically and trusted.
- **Intermediate (QuantumProgramResult) check** : ``check_program_result``
  validates the serialization format and that the (de)serialization content is
  non-empty and decompresses cleanly - the rest is validated by the read step.
- **Information-loss policy** : each :class:`ConversionEdge` declares the
  losses inherent to its path (``known_losses``). Checks are exact-first: a
  deviation that is not declared fails the test; a declared one is tolerated
  and recorded (``loss_report()``).
- **Generic driver** : ``convert`` runs one conversion with its post-checks;
  ``run_path`` runs a full edge (input -> write -> read).
- **Declarative registries** : ``build_edges()`` returns the full conversion
  graph (result format -> target SDK) for each available input kind
  (SDK object and/or serialized dict). ``UNSUPPORTED_CONVERSIONS`` hosts the
  negative cases.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from qio.core import (
    QuantumProgramResult,
    QuantumProgramResultCompressionFormat,
    QuantumProgramResultSerializationFormat,
)

from reference_results import ReferenceResult, get_reference_result

Compression = QuantumProgramResultCompressionFormat
Serialization = QuantumProgramResultSerializationFormat
NONE = Compression.NONE
ZLIB = Compression.ZLIB_BASE64_V1


# Information-loss tracking
#
# The reference results are bit-exact, so the only acceptable deviations are
# SDK *convention* differences that carry no semantic information. A category
# that is not declared on the current edge fails the test (regression); a
# declared one is tolerated and recorded in the session report.

LOSS_MEASUREMENT_KEY = "measurement_key"
LOSS_BIT_ORDER = "bit_order"

_LOSS_DESCRIPTIONS = {
    LOSS_MEASUREMENT_KEY: (
        "measurement keys are SDK-specific (cirq 'm0'/'m1' per qubit, qiskit "
        "'m', mimiq 'result'); the counts histogram is key-independent, so a "
        "renamed key carries no information loss"
    ),
    LOSS_BIT_ORDER: (
        "the bitstring order is SDK-specific (which qubit is the most "
        "significant character); the bitstring multiset - and therefore the "
        "counts histogram - is preserved exactly"
    ),
}

_LOSS_REPORT: Dict[str, Dict[str, Dict[str, Any]]] = {}
_LOSS_CTX = {"id": None, "result": None, "known": frozenset()}


def record_loss(category: str, detail: Optional[str] = None) -> None:
    """Records a declared information loss - optionally with a concrete
    ``detail`` of what was observed - or fails an undeclared one."""
    if category not in _LOSS_CTX["known"]:
        raise AssertionError(
            f"undeclared information loss {category!r} on edge "
            f"{_LOSS_CTX['id']!r} (result {_LOSS_CTX['result']!r}); declare it "
            "in the edge's known_losses or fix the converter"
        )
    entry = _LOSS_REPORT.setdefault(_LOSS_CTX["id"], {}).setdefault(
        category, {"results": set(), "observations": set()}
    )
    entry["results"].add(_LOSS_CTX["result"])
    entry["observations"].add((_LOSS_CTX["result"], detail))


def set_loss_context(edge: "ConversionEdge", reference: ReferenceResult) -> None:
    _LOSS_CTX["id"] = edge.id
    _LOSS_CTX["result"] = reference.name
    _LOSS_CTX["known"] = frozenset(edge.known_losses)


def loss_report() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Session report: edge id -> {loss category: {results, observations}}."""
    return _LOSS_REPORT


def loss_description(category: str) -> str:
    """Human-readable explanation of a loss category (used by the report)."""
    return _LOSS_DESCRIPTIONS.get(category, "")


# Shared count extraction + comparison


def _assert_counts(counts: Dict[str, int], expected: ReferenceResult) -> None:
    """Bit-exact counts-histogram check against the reference result."""
    assert dict(counts) == dict(expected.counts), (
        dict(counts),
        dict(expected.counts),
    )
    total = sum(int(v) for v in counts.values())
    assert total == expected.shots, (total, expected.shots)


def _normalize_hex_counts(raw: Dict[Any, Any], n_qubits: int) -> Dict[str, int]:
    """Normalizes possibly-hex qiskit counts to plain bitstrings."""
    out: Dict[str, int] = {}
    for bitstring, count in raw.items():
        count = int(count)
        if isinstance(bitstring, str) and bitstring.startswith("0x"):
            bitstring = format(int(bitstring, 16), f"0{n_qubits}b")
        else:
            bitstring = str(bitstring)
        out[bitstring] = out.get(bitstring, 0) + count
    return out


def _resolve_serialization(result: QuantumProgramResult):
    """Uncompresses a QuantumProgramResult serialization."""
    if result.compression_format == ZLIB:
        from qio.utils.compression import zlib_to_dict

        return zlib_to_dict(result.serialization)

    import json

    return json.loads(result.serialization)


# Format checks (one per SDK / result format)


def check_program_result(serialization_format: Serialization) -> Callable:
    """Validates a QuantumProgramResult intermediate: format matches and the
    serialization is non-empty and decompresses cleanly."""

    def _check(result, expected: Optional[ReferenceResult] = None) -> None:
        assert isinstance(result, QuantumProgramResult), type(result)
        assert result.serialization_format == serialization_format, (
            result.serialization_format,
            serialization_format,
        )
        assert bool(result.serialization), "empty serialization"
        _resolve_serialization(result)

    return _check


def _cirq_counts(result) -> Dict[str, int]:
    import collections

    keys = sorted(result.measurements.keys())
    if not keys:
        return {}
    repetitions = int(result.measurements[keys[0]].shape[0])
    samples = []
    for r in range(repetitions):
        bits = []
        for key in keys:
            bits.extend(int(v) for v in result.measurements[key][r])
        samples.append("".join(map(str, bits)))
    return dict(collections.Counter(samples))


def check_cirq(result, expected: Optional[ReferenceResult] = None) -> None:
    import cirq

    assert isinstance(result, cirq.Result), f"expected cirq.Result, got {type(result)}"
    if expected is not None:
        _assert_counts(_cirq_counts(result), expected)


def _qiskit_counts(result, expected: Optional[ReferenceResult] = None) -> Dict[str, int]:
    experiment = result.to_dict()["results"][0]
    header = experiment.get("header", {}) or {}
    raw = dict((experiment.get("data", {}) or {}).get("counts", {}) or {})
    n_qubits = header.get("n_qubits")
    if n_qubits is None and raw:
        n_qubits = max(
            (len(str(k)) for k in raw if not (isinstance(k, str) and k.startswith("0x"))),
            default=0,
        )
    n_qubits = int(n_qubits or 0)
    return _normalize_hex_counts(raw, n_qubits)


def check_qiskit(result, expected: Optional[ReferenceResult] = None) -> None:
    from qiskit.result import Result

    assert isinstance(result, Result), f"expected qiskit Result, got {type(result)}"
    if expected is not None:
        _assert_counts(_qiskit_counts(result, expected), expected)


def _parse_cudaq_serialize(data) -> Dict[str, int]:
    """Independent parser for the CUDA-Q SampleResult wire format (register
    name, then ``[value, bit_size, count]`` triplets per bitstring)."""
    stride = 0
    counts: Dict[str, int] = {}
    while stride < len(data):
        n_chars = data[stride]
        stride += 1
        stride += n_chars  # register name, already accounted for
        num_bitstrings = data[stride]
        stride += 1
        for _ in range(num_bitstrings):
            value = data[stride]
            size = data[stride + 1]
            count = data[stride + 2]
            stride += 3
            counts[format(value, f"0{size}b")] = int(count)
    return counts


def check_cudaq(result, expected: Optional[ReferenceResult] = None) -> None:
    import cudaq

    assert isinstance(result, cudaq.SampleResult), f"expected SampleResult, got {type(result)}"
    if expected is not None:
        _assert_counts(_parse_cudaq_serialize(result.serialize()), expected)


def check_mimiq(result, expected: Optional[ReferenceResult] = None) -> None:
    import mimiqcircuits

    assert isinstance(result, mimiqcircuits.QCSResults), f"expected QCSResults, got {type(result)}"
    if expected is not None:
        histogram = result.histogram()
        counts = {key.to01(): int(count) for key, count in histogram.items()}
        _assert_counts(counts, expected)


# Generic driver + pipeline


@dataclass(frozen=True)
class Step:
    fn: Callable
    kwargs: Dict[str, Any] = field(default_factory=dict)
    checks: Tuple[Callable, ...] = ()


@dataclass(frozen=True)
class ConversionEdge:
    id: str
    input_fn: Callable[[ReferenceResult], Any]
    steps: Tuple[Step, ...]
    known_losses: Tuple[str, ...] = ()  # inherent information losses of the path


def convert(result, converter, *args, checks=(), expected=None, **kwargs) -> Any:
    """Generic conversion driver.

    Args:
        result: the input result (statically built and trusted - not checked).
        converter: the conversion function to call (e.g.
            ``QuantumProgramResult.from_cirq_result``).
        checks: post-conversion verifications, each ``check(result, expected)``.
        expected: known information (a ReferenceResult) checks are validated
            against.
    """
    assert result is not None, "input result is required"
    output = converter(result, *args, **kwargs)
    for check in checks:
        check(output, expected)
    return output


def run_path(edge: ConversionEdge, reference: ReferenceResult) -> Any:
    """Runs a full conversion path (input -> steps)."""
    set_loss_context(edge, reference)
    current = edge.input_fn(reference)
    for step in edge.steps:
        current = convert(
            current, step.fn, expected=reference, checks=step.checks, **step.kwargs
        )
    return current


# Declarative registries


def _input_fn(kind: str) -> Callable[[ReferenceResult], Any]:
    """input_kind -> callable producing the untrusted-but-static SDK input."""
    return {
        "cirq": lambda r: r.cirq(),
        "cirq_dict": lambda r: r.cirq_dict(),
        "qiskit": lambda r: r.qiskit(),
        "qiskit_dict": lambda r: r.qiskit_dict(),
        "cudaq": lambda r: r.cudaq(),
        "mimiq": lambda r: r.mimiq(),
        "mimiq_dict": lambda r: r.mimiq_dict(),
    }[kind]


def _producer(kind: str, compression: Compression) -> Callable[[Any], QuantumProgramResult]:
    """input_kind -> QuantumProgramResult classmethod writing the format."""
    QPR = QuantumProgramResult
    return {
        "cirq": lambda obj: QPR.from_cirq_result(obj, compression_format=compression),
        "cirq_dict": lambda obj: QPR.from_cirq_result_dict(
            obj, compression_format=compression
        ),
        "qiskit": lambda obj: QPR.from_qiskit_result(
            obj, compression_format=compression
        ),
        "qiskit_dict": lambda obj: QPR.from_qiskit_result_dict(
            obj, compression_format=compression
        ),
        "cudaq": lambda obj: QPR.from_cudaq_sample_result(
            obj, compression_format=compression
        ),
        "mimiq": lambda obj: QPR.from_mimiq_qcsr(obj, compression_format=compression),
        "mimiq_dict": lambda obj: QPR.from_mimiq_qcsr_dict(
            obj, compression_format=compression
        ),
    }[kind]


_FORMAT_LABEL = {
    Serialization.CIRQ_RESULT_JSON_V1: "cirqjson",
    Serialization.QISKIT_RESULT_JSON_V1: "qiskitjson",
    Serialization.CUDAQ_SAMPLE_RESULT_JSON_V1: "cudaqjson",
    Serialization.MIMIQ_QCSR_JSON_V1: "mimiqjson",
}

# Per-format read edges: target SDK -> (to_<sdk>_result, format check)
_READ_EDGES = {
    Serialization.CIRQ_RESULT_JSON_V1: {
        "cirq": (QuantumProgramResult.to_cirq_result, check_cirq),
        "qiskit": (QuantumProgramResult.to_qiskit_result, check_qiskit),
        "mimiq": (QuantumProgramResult.to_mimiq_qcsr, check_mimiq),
    },
    Serialization.QISKIT_RESULT_JSON_V1: {
        "qiskit": (QuantumProgramResult.to_qiskit_result, check_qiskit),
        "cirq": (QuantumProgramResult.to_cirq_result, check_cirq),
        "mimiq": (QuantumProgramResult.to_mimiq_qcsr, check_mimiq),
    },
    Serialization.CUDAQ_SAMPLE_RESULT_JSON_V1: {
        "cudaq": (QuantumProgramResult.to_cudaq_sample_result, check_cudaq),
        "qiskit": (QuantumProgramResult.to_qiskit_result, check_qiskit),
    },
    Serialization.MIMIQ_QCSR_JSON_V1: {
        "mimiq": (QuantumProgramResult.to_mimiq_qcsr, check_mimiq),
        "cirq": (QuantumProgramResult.to_cirq_result, check_cirq),
        "qiskit": (QuantumProgramResult.to_qiskit_result, check_qiskit),
    },
}

# Input kinds available per format: the SDK object and/or the serialized dict.
_FORMAT_INPUT_KINDS = {
    Serialization.CIRQ_RESULT_JSON_V1: ("cirq", "cirq_dict"),
    Serialization.QISKIT_RESULT_JSON_V1: ("qiskit", "qiskit_dict"),
    Serialization.CUDAQ_SAMPLE_RESULT_JSON_V1: ("cudaq",),
    Serialization.MIMIQ_QCSR_JSON_V1: ("mimiq", "mimiq_dict"),
}


def build_edges(compression: Compression) -> Tuple[ConversionEdge, ...]:
    """Builds the conversion graph (result format -> target SDK) for one
    compression. Each format is exercised from every supported input kind
    (SDK object and/or serialized dict). Intermediate content checks
    (``check_program_result``) run for both ``NONE`` and ``ZLIB``."""
    edges = []
    for fmt, reads in _READ_EDGES.items():
        for input_kind in _FORMAT_INPUT_KINDS[fmt]:
            write = _producer(input_kind, compression)
            for target, (to_fn, check) in reads.items():
                steps = (
                    Step(write, {}, (check_program_result(fmt),)),
                    Step(to_fn, {}, (check,)),
                )
                edges.append(
                    ConversionEdge(
                        id=f"{input_kind}.{_FORMAT_LABEL[fmt]}->{target}",
                        input_fn=_input_fn(input_kind),
                        steps=steps,
                    )
                )
    return tuple(edges)


# Negative/edge cases: each entry is (id, callable) and must raise an Exception.


def _qpr(
    fmt: Serialization, compression: Compression = NONE
) -> QuantumProgramResult:
    """Builds a QuantumProgramResult holding a reference result in ``fmt``."""
    reference = get_reference_result("bell2")
    makers = {
        Serialization.CIRQ_RESULT_JSON_V1: lambda: QuantumProgramResult.from_cirq_result_dict(
            reference.cirq_dict(), compression_format=compression
        ),
        Serialization.QISKIT_RESULT_JSON_V1: lambda: QuantumProgramResult.from_qiskit_result_dict(
            reference.qiskit_dict(), compression_format=compression
        ),
        Serialization.CUDAQ_SAMPLE_RESULT_JSON_V1: lambda: QuantumProgramResult.from_cudaq_sample_result(
            reference.cudaq(), compression_format=compression
        ),
        Serialization.MIMIQ_QCSR_JSON_V1: lambda: QuantumProgramResult.from_mimiq_qcsr_dict(
            reference.mimiq_dict(), compression_format=compression
        ),
    }
    return makers[fmt]()


UNSUPPORTED_CONVERSIONS = (
    (
        "to_cirq_result on CUDAQ_SAMPLE_RESULT_JSON_V1",
        lambda: _qpr(Serialization.CUDAQ_SAMPLE_RESULT_JSON_V1).to_cirq_result(),
    ),
    (
        "to_mimiq_qcsr on CUDAQ_SAMPLE_RESULT_JSON_V1",
        lambda: _qpr(Serialization.CUDAQ_SAMPLE_RESULT_JSON_V1).to_mimiq_qcsr(),
    ),
    (
        "to_cudaq_sample_result on CIRQ_RESULT_JSON_V1",
        lambda: _qpr(Serialization.CIRQ_RESULT_JSON_V1).to_cudaq_sample_result(),
    ),
    (
        "to_cudaq_sample_result on QISKIT_RESULT_JSON_V1",
        lambda: _qpr(Serialization.QISKIT_RESULT_JSON_V1).to_cudaq_sample_result(),
    ),
    (
        "to_cudaq_sample_result on MIMIQ_QCSR_JSON_V1",
        lambda: _qpr(Serialization.MIMIQ_QCSR_JSON_V1).to_cudaq_sample_result(),
    ),
)
