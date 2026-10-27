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
"""Driver, base class and registries of the result conversion battery.

Design (mirrors ``circuit_testing`` for circuits):
- **Per-SDK adapters** : every SDK - Cirq, Qiskit, CUDA-Q, MIMIQ, Perceval,
  Pulser - is one :class:`ResultSDK` subclass in ``result_sdks/``. An adapter
  groups everything SDK-specific in one place: the static input builder, the
  format check, the dict-converter module name, the wrap/read classmethods, the
  metadata probes and the inherent losses. Adding an SDK never touches the
  driver.
- **Lengthened edges** : every edge is driven from the SDK **object** and makes
  the qio ``<sdk>_to_dict`` converter an explicit first step, so the full chain
  ``SDK object -> <sdk>_to_dict -> QuantumProgramResult -> to_<target>`` is
  exercised (no hand-authored intermediate dicts / second source of truth).
- **Declarative graph** : :data:`EDGES` declares, per input SDK, the read
  targets it can be converted to. ``build_edges()`` expands that graph into
  :class:`ConversionEdge` rows (id ``qiskit.Result -> qiskit_result_json_v1 ->
  qiskit.Result``, input builder, write+read steps, ``known_losses`` = target
  losses + source static losses). ``UNSUPPORTED_CONVERSIONS`` (the negative
  cases) is the **complement** of the graph, derived automatically.
- **SDK-lazy collection** : this module and the adapters import no SDK at
  import time (the ``<sdk>_to_dict`` converters are pulled in lazily on first
  use). Two reasons: the battery must stay selectable by ``-k perceval``
  without importing mimiqcircuits (mimiq's and Perceval's ``circuit.proto``
  conflict in protobuf's global descriptor pool), and collection must not
  depend on any heavy SDK presence.
- **Metadata probes** : on top of the counts, target-SDK objects are probed
  for execution metadata (backend identity, job ids, date, statevector, MIMIQ
  ``zstates``/``fidelities``/``avggateerrors``/``timings``/``amplitudes``,
  CUDA-Q register name). A probe that differs from the reference records the
  corresponding **information loss**; this is what makes the loss of each
  conversion path visible instead of silently dropping fields.
- **Information-loss policy** : each :class:`ConversionEdge` declares the
  losses inherent to its path (``known_losses``). Checks are exact-first: a
  deviation that is not declared fails the test; a declared one is tolerated
  and recorded (``loss_report()``).
- **Generic driver** : ``convert`` runs one conversion with its post-checks;
  ``run_path`` runs a full edge (input -> write -> read).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

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
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
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


def _run_probes(
    probes: Sequence[Probe], result: Any, expected: ReferenceResult
) -> None:
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


# Shared count comparison


def _assert_counts(counts: Dict[str, int], expected: ReferenceResult) -> None:
    """Bit-exact counts-histogram check against the reference result."""
    assert dict(counts) == dict(expected.counts), (
        dict(counts),
        dict(expected.counts),
    )
    total = sum(int(v) for v in counts.values())
    assert total == expected.shots, (total, expected.shots)


def _resolve_serialization(result: QuantumProgramResult):
    """Uncompresses a QuantumProgramResult serialization."""
    if result.compression_format == ZLIB:
        from qio.utils.compression import zlib_to_dict

        return zlib_to_dict(result.serialization)

    import json

    return json.loads(result.serialization)


# Per-SDK adapter base class
#
# One subclass per SDK lives in ``result_sdks/`` and is registered in
# ``result_sdks``. It carries everything SDK-specific; the driver below is
# generic and only consumes the adapter interface.


class ResultSDK(ABC):
    """Base class for a result-SDK adapter.

    Class attributes hold the SDK facts; ``build``/``check`` import the SDK
    lazily (never at collection time).
    """

    kind: str  # registry key / input_kind ("cirq")
    display: str  # type written in edge ids ("cirq.Result")
    fmt: Serialization  # the unique intermediate format this source writes
    fmt_label: str  # its label in edge ids ("cirq_result_json_v1")
    dict_keys: Tuple[str, ...]  # mandatory top-level keys of the intermediate dict
    to_dict_module: Optional[str] = (
        None  # qio "<sdk>_to_dict" module; None => direct wrap
    )
    from_dict: Optional[Callable] = (
        None  # from_<sdk>_result_dict classmethod; None => direct wrap
    )
    wrap: Optional[Callable] = (
        None  # from_<sdk>_result classmethod (object -> QuantumProgramResult)
    )
    read: Optional[Callable] = (
        None  # to_<sdk>_result classmethod (QuantumProgramResult -> object)
    )
    target_losses: Tuple[str, ...] = ()  # losses inherent when arriving on this SDK
    static_source_losses: Tuple[
        str, ...
    ] = ()  # losses inherent when departing from this SDK
    source_loss: Optional[Tuple[str, Optional[str]]] = (
        None  # observed at the to_dict step
    )
    probes: Tuple[Probe, ...] = ()
    # Additional result *containers* of the same format (e.g. perceval
    # BSSamples/BSCount/BSDistribution): each ``(label, build, check)``
    # expands into one more edge of the adapter's format, sharing the same
    # write steps and mandatory dict keys. ``build(reference)`` builds the
    # container, ``check(result, expected)`` validates it (default ``build``/
    # ``check`` remain the edge for the plain adapter).
    containers: Tuple[Tuple[str, Callable, Callable], ...] = ()

    @abstractmethod
    def build(self, reference: ReferenceResult):  # pragma: no cover - interface
        """Builds the static SDK input object from a reference result."""

    @abstractmethod
    def check(self, result, expected: Optional[ReferenceResult] = None) -> None:
        """Validates the object type and, when ``expected`` is given, checks
        the counts histogram bit-exactly and runs the metadata probes."""


# Generic driver + pipeline


@dataclass(frozen=True)
class _LazyFn:
    """Callable resolving the actual converter on the *first call*, so building
    the edges at collection time never imports an SDK (the perceval legs must be
    selectable by ``-k perceval`` without pulling mimiqcircuits into the
    process)."""

    loader: Callable[[], Callable]
    _fn: Callable = field(default=None, init=False, compare=False)

    def __call__(self, *args, **kwargs):
        if self._fn is None:
            object.__setattr__(self, "_fn", self.loader())
        return self._fn(*args, **kwargs)


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


# Intermediate checks: generated from the adapter's facts.


def check_program_result(sdk: ResultSDK) -> Callable:
    """Validates a QuantumProgramResult intermediate: format matches, the
    serialization is non-empty, decompresses cleanly, and carries the
    mandatory top-level fields of its format (``sdk.dict_keys``)."""

    def _check(result, expected: Optional[ReferenceResult] = None) -> None:
        assert isinstance(result, QuantumProgramResult), type(result)
        assert result.serialization_format == sdk.fmt, (
            result.serialization_format,
            sdk.fmt,
        )
        assert bool(result.serialization), "empty serialization"
        content = _resolve_serialization(result)
        if isinstance(content, dict):
            for key in sdk.dict_keys:
                assert key in content, f"intermediate missing {key!r}: {content.keys()}"

    return _check


def check_to_dict(sdk: ResultSDK) -> Callable:
    """Validates the dict produced by qio's ``<sdk>_to_dict`` right after the
    first step of an edge (before wrapping, so no loss can be hidden); records
    the declared source loss (e.g. cirq params) if the adapter declares one."""

    def _check(result, expected: Optional[ReferenceResult] = None) -> None:
        assert result is not None, "to_dict produced nothing"
        assert isinstance(
            result, dict
        ), f"{sdk.kind} to_dict produced a non-dict: {type(result)}"
        for key in sdk.dict_keys:
            assert key in result, f"{sdk.kind} dict missing {key!r}: {result.keys()}"
        if sdk.source_loss is not None:
            category, detail = sdk.source_loss
            record_loss(category, detail=detail)

    return _check


def _producer_steps(sdk: ResultSDK, compression: Compression):
    """Write-side steps of a lengthened edge: SDK object -> ``*_to_dict``
    (checked) -> ``from_*_result_dict`` wrap (checked intermediate). CUDA-Q has
    no separate dict classmethod (``to_dict_module`` is None):
    ``from_cudaq_sample_result`` serializes internally, so the write side is
    just the wrap step."""
    if sdk.to_dict_module is None:
        # ``type(sdk)`` access keeps the qio converter a plain unbound function
        # (instance access would bind it to the adapter and break the call).
        wrap = lambda obj: type(sdk).wrap(obj, compression_format=compression)
        return (Step(wrap, {}, (check_program_result(sdk),)),)

    to_dict = _LazyFn(
        lambda sdk=sdk: import_module(
            f"qio.utils.conversion.program_result.{sdk.to_dict_module}"
        ).convert
    )
    wrap_fn = type(sdk).from_dict

    def wrap(obj):
        return wrap_fn(obj, compression_format=compression)

    return (
        Step(to_dict, {}, (check_to_dict(sdk),)),
        Step(wrap, {}, (check_program_result(sdk),)),
    )


# Negative-case machinery: ``UNSUPPORTED_CONVERSIONS`` is the complement of the
# declared read graph.


def _qpr(sdk: ResultSDK, compression: Compression = NONE) -> QuantumProgramResult:
    """Builds a QuantumProgramResult holding a reference result in ``sdk``'s
    format."""
    reference = get_reference_result("bell2")
    return type(sdk).wrap(sdk.build(reference), compression_format=compression)


def _neg_op(input_sdk: ResultSDK, target_sdk: ResultSDK) -> Callable:
    """Callable building an input ``QuantumProgramResult`` and attempting the
    forbidden ``to_<target>_result`` call (must raise)."""
    read_name = type(target_sdk).read.__name__

    def op():
        return getattr(_qpr(input_sdk), read_name)()

    return op


def _to_negatives() -> Tuple[Tuple[str, Callable], ...]:
    """Derives the negative cases: every ``(input, target)`` pair not in
    :data:`EDGES` must raise. The two cases that pull mimiqcircuits into the
    perceval-only invocation are pinned last (mimiq/Perceval ``circuit.proto``
    conflict)."""
    allowed = frozenset(
        (inp.kind, tgt.kind) for inp, targets in EDGES for tgt in targets
    )
    negatives = []
    for input_sdk in SDKS.values():
        for target_sdk in SDKS.values():
            if (input_sdk.kind, target_sdk.kind) in allowed:
                continue
            negatives.append(
                (
                    f"{target_sdk.read.__name__} on {input_sdk.fmt.name}",
                    _neg_op(input_sdk, target_sdk),
                )
            )
    # mimiq cannot be imported in a Perceval process (protobuf conflict); it
    # must therefore be the last thing tried there.
    _MIMIQ_MUST_BE_LAST = {
        "to_mimiq_qcsr on PERCEVAL_RESULT_JSON_V1",
        "to_perceval_result on MIMIQ_QCSR_JSON_V1",
    }
    negatives.sort(key=lambda item: item[0] in _MIMIQ_MUST_BE_LAST)
    return tuple(negatives)


# Registered SDKs and the read graph. Populated below from ``result_sdks`` so
# that the adapters can import the driver (base class / helpers) without a
# circular-import failure.

SDKS: Dict[str, ResultSDK] = {}


def _edge_rows() -> Tuple[Tuple[ResultSDK, Tuple[ResultSDK, ...]], ...]:
    c = SDKS["cirq"]
    q = SDKS["qiskit"]
    cq = SDKS["cudaq"]
    m = SDKS["mimiq"]
    p = SDKS["perceval"]
    pu = SDKS["pulser"]
    return (
        (c, (c, q, m)),
        (q, (q, c, m)),
        (cq, (cq, q)),
        (m, (m, c, q)),
        (p, (p,)),
        (pu, (pu,)),
    )


def build_edges(compression: Compression) -> Tuple[ConversionEdge, ...]:
    """Builds the conversion graph (result format -> target SDK) for one
    compression, driven from the SDK object through qio's ``*_to_dict``.

    Every edge is generated from :data:`EDGES`: the input SDK produces its
    intermediate format, and each declared read target expands into one edge
    whose ``known_losses`` are the target losses plus the input source's static
    losses. Any deviation in another category fails the test.
    """
    edges = []
    for input_sdk, targets in EDGES:
        steps = _producer_steps(input_sdk, compression)
        # Default edge + one edge per declared result container (same format,
        # alternate input builder / read-side check).
        rows = [(None, input_sdk.build, None)] + list(input_sdk.containers)
        for label, build, check in rows:
            suffix = "" if label is None else f":{label}"
            for target in targets:
                read_check = target.check if label is None else check
                edge_id = (
                    f"{input_sdk.display}{suffix} -> {input_sdk.fmt_label} "
                    f"-> {target.display}{suffix}"
                )
                known = tuple(target.target_losses) + tuple(
                    input_sdk.static_source_losses
                )
                edges.append(
                    ConversionEdge(
                        id=edge_id,
                        input_fn=lambda ref, b=build: b(ref),
                        steps=steps + (Step(type(target).read, {}, (read_check,)),),
                        known_losses=known,
                    )
                )
    return tuple(edges)


from result_sdks import SDK_RESULTS  # noqa: E402

SDKS = dict(SDK_RESULTS)
EDGES = _edge_rows()
UNSUPPORTED_CONVERSIONS = _to_negatives()
