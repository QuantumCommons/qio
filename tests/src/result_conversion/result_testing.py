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
- **Lengthened edges** : every edge is driven from the SDK **object** and makes
  the qio ``<sdk>_to_dict`` converter an explicit first step, so the full chain
  ``SDK object -> <sdk>_to_dict -> QuantumProgramResult -> to_<target>`` is
  exercised (no hand-authored intermediate dicts / second source of truth).
- **One format checker per SDK** : ``check_cirq``, ``check_qiskit``,
  ``check_cudaq``, ``check_mimiq``. Each validates the object type and reduces
  the result to its canonical **counts histogram** (``{bitstring: count}``)
  compared bit-exactly against the reference - plus the total shot count.
- **Metadata probes** : on top of the counts, target-SDK objects are probed
  for execution metadata (backend identity, job ids, date, statevector, MIMIQ
  ``zstates``/``fidelities``/``avggateerrors``/``timings``/``amplitudes``,
  CUDA-Q register name). A probe that differs from the reference records the
  corresponding **information loss**; this is what makes the loss of each
  conversion path visible instead of silently dropping fields.
- **Static inputs** : every reference result is an already executed run
  (``{bitstring: count}``), exposed identically in every SDK format by
  ``reference_results``. Input results are NOT checked - they are built
  statically and trusted.
- **Information-loss policy** : each :class:`ConversionEdge` declares the
  losses inherent to its path (``known_losses``). Checks are exact-first: a
  deviation that is not declared fails the test; a declared one is tolerated
  and recorded (``loss_report()``).
- **Generic driver** : ``convert`` runs one conversion with its post-checks;
  ``run_path`` runs a full edge (input -> write -> read).
- **Declarative registry** : ``build_edges()`` returns the full conversion
  graph as :class:`ConversionEdge` rows. Each edge is declared by a single
  ``add(...)`` call (like the circuit battery) with its annotated id
  (``qiskit.Result -> qiskit_result_json_v1 -> qiskit.Result``), the input kind
  that produces the intermediate, the target reader and its check, and the
   losses inherent to the path. ``UNSUPPORTED_CONVERSIONS`` hosts the negative
   cases.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from qio.core import (
    QuantumProgramResult,
    QuantumProgramResultCompressionFormat,
    QuantumProgramResultSerializationFormat,
)
from qio.utils.conversion.program_result import (
    cirq_to_dict,
    mimiq_to_dict,
    qiskit_to_dict,
)

from reference_results import ReferenceResult, get_reference_result

Compression = QuantumProgramResultCompressionFormat
Serialization = QuantumProgramResultSerializationFormat
NONE = Compression.NONE
ZLIB = Compression.ZLIB_BASE64_V1


# Information-loss tracking
#
# The reference results are bit-exact, so the only acceptable deviations are
# SDK *convention* differences that carry no semantic information, plus the
# metadata fields that the current converters do not (or cannot) carry. A
# category that is not declared on the current edge fails the test
# (regression); a declared one is tolerated and recorded in the session
# report.

LOSS_MEASUREMENT_KEY = "measurement_key"
LOSS_BIT_ORDER = "bit_order"
LOSS_CIRQ_PARAMS = "cirq_params"
LOSS_QISKIT_METADATA = "qiskit_metadata"
LOSS_QISKIT_STATEVECTOR = "qiskit_statevector"
LOSS_MIMIQ_METADATA = "mimiq_metadata"
LOSS_MIMIQ_AMPLITUDES = "mimiq_amplitudes"
LOSS_CUDAQ_REGISTER = "cudaq_register"

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
    LOSS_CIRQ_PARAMS: (
        "cirq.Result params cannot be stored: ResultDict._json_dict_() emits "
        "a live cirq.ParamResolver, which plain-JSON (qio pipeline) cannot "
        "serialize - params are therefore always None across every conversion"
    ),
    LOSS_QISKIT_METADATA: (
        "qiskit job metadata (backend_name/version, job_id, qobj_id, date) is "
        "dropped by the qio converters: dict_to_qiskit only forwards "
        "{results, success, header, metadata}; it survives only when a target "
        "converter re-derives it (e.g. mimiq simulator -> backend_name)"
    ),
    LOSS_QISKIT_STATEVECTOR: (
        "qiskit data.statevector is dropped by non-qiskit converters; it only "
        "survives on the qiskit->qiskit path when it stays JSON-safe "
        "(real-valued amplitudes)"
    ),
    LOSS_MIMIQ_METADATA: (
        "MIMIQ QCSResults metadata (zstates, fidelities, avggateerrors, "
        "timings) is not carried by mimiq_to_dict / dict_to_mimiq today"
    ),
    LOSS_MIMIQ_AMPLITUDES: (
        "MIMIQ amplitudes (bitarray keys, complex values) are not JSON "
        "serializable, hence not representable through the result JSON "
        "intermediate; they can only be reconstructed in memory (e.g. qiskit "
        "statevector -> mimiq amplitudes)"
    ),
    LOSS_CUDAQ_REGISTER: (
        "CUDA-Q measurement register name mapping (e.g. to the qiskit "
        "experiment header name) is SDK-specific"
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


# Metadata probes
#
# A probe extracts one field from the converted (target-SDK) object and
# compares it to the reference metadata. A mismatch is an information loss:
# recorded if declared on the edge, fatal otherwise (regression).


@dataclass(frozen=True)
class Probe:
    category: str
    field: str
    extract: Callable[[Any], Any]
    expected: Callable[[ReferenceResult], Any]
    normalize: Callable[[Any], Any] = lambda v: v


def _deep_close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(
            1.0, abs(float(a)), abs(float(b))
        )
    if isinstance(a, complex) and isinstance(b, complex):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    if isinstance(a, complex) or isinstance(b, complex):
        return _deep_close(complex(a), complex(b), tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_deep_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, set) and isinstance(b, set):
        return a == b
    return a == b


def _run_probes(probes: Sequence[Probe], result: Any, expected: ReferenceResult) -> None:
    for probe in probes:
        try:
            observed = probe.extract(result)
        except Exception:
            observed = None
        observed = probe.normalize(observed)
        expected_value = probe.normalize(probe.expected(expected))
        if not _deep_close(observed, expected_value):
            record_loss(
                probe.category,
                detail=f"{probe.field}: {observed!r} != expected {expected_value!r}",
            )


# Qiskit probes: the qiskit run metadata + statevector.


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _qiskit_statevector(result: Any) -> Optional[Sequence[float]]:
    try:
        sv = result.data(0).statevector
    except Exception:
        return None
    return list(sv) if sv is not None else None


QISKIT_PROBES = (
    Probe(LOSS_QISKIT_METADATA, "backend_name", lambda r: r.backend_name, lambda ref: ref.backend_name),
    Probe(LOSS_QISKIT_METADATA, "backend_version", lambda r: r.backend_version, lambda ref: ref.backend_version),
    Probe(LOSS_QISKIT_METADATA, "job_id", lambda r: r.job_id, lambda ref: ref.job_id),
    Probe(LOSS_QISKIT_METADATA, "qobj_id", lambda r: r.qobj_id, lambda ref: ref.qobj_id),
    Probe(LOSS_QISKIT_METADATA, "date", lambda r: r.date, lambda ref: ref.date, normalize=_iso),
    Probe(
        LOSS_QISKIT_STATEVECTOR,
        "statevector",
        _qiskit_statevector,
        lambda ref: list(ref.statevector) if ref.statevector is not None else None,
        normalize=lambda v: [float(x) for x in v] if v is not None else None,
    ),
)

# MIMIQ probes.


def _mimiq_zstates(result: Any) -> list:
    zstates = getattr(result, "zstates", None)
    if not zstates:
        return []
    return [z.to01() if hasattr(z, "to01") else str(z) for z in zstates]


def _mimiq_amplitudes(result: Any) -> Dict[str, complex]:
    amplitudes = getattr(result, "amplitudes", None) or {}
    out = {}
    for key, value in amplitudes.items():
        bitstring = key.to01() if hasattr(key, "to01") else str(key)
        out[bitstring] = complex(value)
    return out


def _norm_seq(value: Optional[Sequence]) -> Sequence:
    return [] if value is None else list(value)


def _norm_dict(value: Optional[Dict]) -> Dict:
    return {} if value is None else dict(value)


MIMIQ_PROBES = (
    Probe(LOSS_MIMIQ_METADATA, "simulator", lambda r: getattr(r, "simulator", None), lambda ref: ref.backend_name),
    Probe(LOSS_MIMIQ_METADATA, "version", lambda r: getattr(r, "version", None), lambda ref: ref.backend_version),
    Probe(LOSS_MIMIQ_METADATA, "timings", lambda r: _norm_dict(getattr(r, "timings", None)), lambda ref: dict(ref.timings or {})),
    Probe(LOSS_MIMIQ_METADATA, "zstates", _mimiq_zstates, lambda ref: list(ref.zstates or [])),
    Probe(LOSS_MIMIQ_METADATA, "fidelities", lambda r: _norm_seq(getattr(r, "fidelities", None)), lambda ref: list(ref.fidelities or [])),
    Probe(LOSS_MIMIQ_METADATA, "avggateerrors", lambda r: _norm_seq(getattr(r, "avggateerrors", None)), lambda ref: list(ref.avggateerrors or [])),
    Probe(LOSS_MIMIQ_AMPLITUDES, "amplitudes", _mimiq_amplitudes, lambda ref: ref._expected_amplitudes()),
)

# CUDA-Q probes: the measurement register name(s).


def _cudaq_register_names(result: Any) -> Optional[set]:
    try:
        data = result.serialize()
    except Exception:
        return None
    names = set()
    stride = 0
    while stride < len(data):
        n_chars = data[stride]
        stride += 1
        name = "".join(chr(data[i]) for i in range(stride, stride + n_chars))
        stride += n_chars
        num_bitstrings = data[stride]
        stride += 1
        if num_bitstrings > 0:
            # Skip the synthetic, count-less '__global__' register CUDA-Q adds
            # on deserialize.
            names.add(name)
        stride += num_bitstrings * 3
    return names


CUDAQ_PROBES = (
    Probe(LOSS_CUDAQ_REGISTER, "register", _cudaq_register_names, lambda ref: {ref.register_name}),
)


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
    """Validates a QuantumProgramResult intermediate: format matches, the
    serialization is non-empty, decompresses cleanly, and carries the
    mandatory top-level fields of its format."""

    _REQUIRED_KEYS = {
        Serialization.CIRQ_RESULT_JSON_V1: ("records",),
        Serialization.QISKIT_RESULT_JSON_V1: ("results",),
        Serialization.MIMIQ_QCSR_JSON_V1: ("histogram", "simulator", "version"),
        Serialization.CUDAQ_SAMPLE_RESULT_JSON_V1: (),
    }

    def _check(result, expected: Optional[ReferenceResult] = None) -> None:
        assert isinstance(result, QuantumProgramResult), type(result)
        assert result.serialization_format == serialization_format, (
            result.serialization_format,
            serialization_format,
        )
        assert bool(result.serialization), "empty serialization"
        content = _resolve_serialization(result)
        required = _REQUIRED_KEYS[serialization_format]
        if isinstance(content, dict):
            for key in required:
                assert key in content, f"intermediate missing {key!r}: {content.keys()}"

    return _check


def check_to_dict(input_kind: str) -> Callable:
    """Validates the dict produced by qio's ``<sdk>_to_dict`` right after the
    first step of an edge (before wrapping, so no loss can be hidden)."""

    def _check(result, expected: Optional[ReferenceResult] = None) -> None:
        assert result is not None, "to_dict produced nothing"
        if input_kind == "cirq":
            assert isinstance(result, dict) and "records" in result, (
                f"cirq dict missing 'records': {type(result)}"
            )
            record_loss(
                LOSS_CIRQ_PARAMS,
                detail="cirq Result params cannot be stored: ParamResolver is "
                "not JSON-serializable, so params never cross the format",
            )
        elif input_kind == "qiskit":
            assert isinstance(result, dict) and "results" in result, (
                f"qiskit dict missing 'results': {type(result)}"
            )
        elif input_kind == "mimiq":
            assert isinstance(result, dict)
            for key in ("simulator", "version", "histogram"):
                assert key in result, f"mimiq dict missing {key!r}: {result.keys()}"

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
        _run_probes(QISKIT_PROBES, result, expected)


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
        _run_probes(CUDAQ_PROBES, result, expected)


def check_mimiq(result, expected: Optional[ReferenceResult] = None) -> None:
    import mimiqcircuits

    assert isinstance(result, mimiqcircuits.QCSResults), f"expected QCSResults, got {type(result)}"
    if expected is not None:
        histogram = result.histogram()
        counts = {key.to01(): int(count) for key, count in histogram.items()}
        _assert_counts(counts, expected)
        _run_probes(MIMIQ_PROBES, result, expected)


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
        converter: the conversion function to call.
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
    """input_kind -> callable producing the static SDK input object."""
    return {
        "cirq": lambda r: r.cirq(),
        "qiskit": lambda r: r.qiskit(),
        "cudaq": lambda r: r.cudaq(),
        "mimiq": lambda r: r.mimiq(),
    }[kind]


_TO_DICT = {
    "cirq": cirq_to_dict.convert,
    "qiskit": qiskit_to_dict.convert,
    "mimiq": mimiq_to_dict.convert,
}

_FROM_DICT = {
    "cirq": QuantumProgramResult.from_cirq_result_dict,
    "qiskit": QuantumProgramResult.from_qiskit_result_dict,
    "mimiq": QuantumProgramResult.from_mimiq_qcsr_dict,
}


def _producer_steps(input_kind: str, fmt: Serialization, compression: Compression):
    """Write-side steps of a lengthened edge: SDK object -> ``*_to_dict``
    (checked) -> ``from_*_result_dict`` wrap (checked intermediate). CUDA-Q has
    no separate dict classmethod: ``from_cudaq_sample_result`` serializes
    internally (``_to_dict`` is ``SampleResult.serialize()`` already
    exercised by the read/cudaq edges)."""
    if input_kind == "cudaq":
        wrap = lambda obj: QuantumProgramResult.from_cudaq_sample_result(
            obj, compression_format=compression
        )
        return (Step(wrap, {}, (check_program_result(fmt),)),)

    to_dict = _TO_DICT[input_kind]
    wrap_fn = _FROM_DICT[input_kind]

    def wrap(obj):
        return wrap_fn(obj, compression_format=compression)

    return (
        Step(to_dict, {}, (check_to_dict(input_kind),)),
        Step(wrap, {}, (check_program_result(fmt),)),
    )


# One (object) input kind per format: the intermediate dicts are produced by
# qio's own ``*_to_dict`` converters, never hand-authored. The write-side steps
# of an edge are derived from the input kind via this map.
_INPUT_FORMAT = {
    "cirq": Serialization.CIRQ_RESULT_JSON_V1,
    "qiskit": Serialization.QISKIT_RESULT_JSON_V1,
    "cudaq": Serialization.CUDAQ_SAMPLE_RESULT_JSON_V1,
    "mimiq": Serialization.MIMIQ_QCSR_JSON_V1,
}

# Per-source losses that apply whatever the target (e.g. cirq params cannot be
# stored at all); auto-joined to every edge of that source.
_SOURCE_STATIC_LOSSES = {
    "cirq": (LOSS_CIRQ_PARAMS,),
}


def build_edges(compression: Compression) -> Tuple[ConversionEdge, ...]:
    """Builds the conversion graph (result format -> target SDK) for one
    compression, driven from the SDK object through qio's ``*_to_dict``.

    Every edge is declared by a single ``add(...)`` call - as in the circuit
    battery - with its annotated id (``qiskit.Result -> qiskit_result_json_v1
    -> qiskit.Result``), the ``input_kind`` that produces the intermediate, the
    ``to_<target>_result`` reader and its check, plus the losses inherent to
    that path (``known_losses``); any deviation in another category fails the
    test.
    """
    edges = []

    def add(
        edge_id: str,
        input_kind: str,
        read_fn: Callable,
        read_check: Callable,
        known_losses: Tuple[str, ...] = (),
    ) -> None:
        fmt = _INPUT_FORMAT[input_kind]
        steps = _producer_steps(input_kind, fmt, compression) + (
            Step(read_fn, {}, (read_check,)),
        )
        known = tuple(known_losses) + _SOURCE_STATIC_LOSSES.get(input_kind, ())
        edges.append(
            ConversionEdge(
                id=edge_id,
                input_fn=_input_fn(input_kind),
                steps=steps,
                known_losses=known,
            )
        )

    add(
        "cirq.Result -> cirq_result_json_v1 -> cirq.Result",
        "cirq",
        QuantumProgramResult.to_cirq_result,
        check_cirq,
    )
    add(
        "cirq.Result -> cirq_result_json_v1 -> qiskit.Result",
        "cirq",
        QuantumProgramResult.to_qiskit_result,
        check_qiskit,
        (LOSS_QISKIT_METADATA, LOSS_QISKIT_STATEVECTOR),
    )
    add(
        "cirq.Result -> cirq_result_json_v1 -> mimiqcircuits.QCSResults",
        "cirq",
        QuantumProgramResult.to_mimiq_qcsr,
        check_mimiq,
        (LOSS_MIMIQ_METADATA, LOSS_MIMIQ_AMPLITUDES),
    )
    add(
        "qiskit.Result -> qiskit_result_json_v1 -> qiskit.Result",
        "qiskit",
        QuantumProgramResult.to_qiskit_result,
        check_qiskit,
        (LOSS_QISKIT_METADATA, LOSS_QISKIT_STATEVECTOR),
    )
    add(
        "qiskit.Result -> qiskit_result_json_v1 -> cirq.Result",
        "qiskit",
        QuantumProgramResult.to_cirq_result,
        check_cirq,
    )
    add(
        "qiskit.Result -> qiskit_result_json_v1 -> mimiqcircuits.QCSResults",
        "qiskit",
        QuantumProgramResult.to_mimiq_qcsr,
        check_mimiq,
        (LOSS_MIMIQ_METADATA, LOSS_MIMIQ_AMPLITUDES),
    )
    add(
        "cudaq.SampleResult -> cudaq_sample_result_json_v1 -> cudaq.SampleResult",
        "cudaq",
        QuantumProgramResult.to_cudaq_sample_result,
        check_cudaq,
    )
    add(
        "cudaq.SampleResult -> cudaq_sample_result_json_v1 -> qiskit.Result",
        "cudaq",
        QuantumProgramResult.to_qiskit_result,
        check_qiskit,
        (LOSS_QISKIT_METADATA, LOSS_QISKIT_STATEVECTOR),
    )
    add(
        "mimiqcircuits.QCSResults -> mimiq_qcsr_json_v1 -> mimiqcircuits.QCSResults",
        "mimiq",
        QuantumProgramResult.to_mimiq_qcsr,
        check_mimiq,
        (LOSS_MIMIQ_METADATA, LOSS_MIMIQ_AMPLITUDES),
    )
    add(
        "mimiqcircuits.QCSResults -> mimiq_qcsr_json_v1 -> cirq.Result",
        "mimiq",
        QuantumProgramResult.to_cirq_result,
        check_cirq,
    )
    add(
        "mimiqcircuits.QCSResults -> mimiq_qcsr_json_v1 -> qiskit.Result",
        "mimiq",
        QuantumProgramResult.to_qiskit_result,
        check_qiskit,
        (LOSS_QISKIT_METADATA, LOSS_QISKIT_STATEVECTOR),
    )

    return tuple(edges)


# Negative/edge cases: each entry is (id, callable) and must raise an Exception.


def _qpr(
    fmt: Serialization, compression: Compression = NONE
) -> QuantumProgramResult:
    """Builds a QuantumProgramResult holding a reference result in ``fmt``."""
    reference = get_reference_result("bell2")
    makers = {
        Serialization.CIRQ_RESULT_JSON_V1: lambda: QuantumProgramResult.from_cirq_result(
            reference.cirq(), compression_format=compression
        ),
        Serialization.QISKIT_RESULT_JSON_V1: lambda: QuantumProgramResult.from_qiskit_result(
            reference.qiskit(), compression_format=compression
        ),
        Serialization.CUDAQ_SAMPLE_RESULT_JSON_V1: lambda: QuantumProgramResult.from_cudaq_sample_result(
            reference.cudaq(), compression_format=compression
        ),
        Serialization.MIMIQ_QCSR_JSON_V1: lambda: QuantumProgramResult.from_mimiq_qcsr(
            reference.mimiq(), compression_format=compression
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
