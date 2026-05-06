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
import cirq
import io
import numpy as np
import collections
from bitarray import frozenbitarray
from qio.utils.conversion.program_result.dict_to_mimiq import (
    convert as dict_to_mimiq_convert,
)

from typing import Union, Sequence, Dict, Tuple, cast, List


def _unpack_bits(packed_bits: str, dtype: str, shape: Sequence[int]) -> np.ndarray:
    bits_bytes = bytes.fromhex(packed_bits)
    bits = np.unpackbits(np.frombuffer(bits_bytes, dtype=np.uint8))
    return bits[: np.prod(shape).item()].reshape(shape).astype(dtype)


def _unpack_records(
    packed_digits: str,
    binary: bool,
    dtype: Union[None, str],
    shape: Union[None, Sequence[int]],
):
    if binary:
        dtype = cast(str, dtype)
        shape = cast(Sequence[int], shape)
        return _unpack_bits(packed_digits, dtype, shape)

    buffer = io.BytesIO()
    buffer.write(bytes.fromhex(packed_digits))
    buffer.seek(0)
    digits = np.load(buffer, allow_pickle=False)
    buffer.close()

    return digits


def __measurements(records: Dict):
    measurements = {}
    for key, data in records.items():
        reps, instances, qubits = data.shape
        if instances != 1:
            raise ValueError("Cannot extract 2D measurements for repeated keys")
        measurements[key] = data.reshape((reps, qubits))

    return measurements


def _generate_memory_and_counts(
    measurements: Dict[str, np.ndarray],
) -> Tuple[List[str], Dict[str, int]]:
    """Génère l'historique des tirs (mémoire) et l'histogramme des fréquences."""
    keys = sorted(measurements.keys())
    repetitions = len(next(iter(measurements.values()))) if measurements else 0

    samples = zip(*(measurements[k] for k in keys))
    if len(keys) == 0:
        samples = [()] * repetitions

    memory = []
    for sample in samples:
        bitstring = (
            "".join(str(int(a)) for a in np.concatenate(sample)) if sample else ""
        )
        memory.append(bitstring)

    counts = dict(collections.Counter(memory))
    return memory, counts


def convert(cirq_results: dict, **kwargs) -> "mimiqcircuits.QCSResults":
    """
    Convert a serialized cirq.Result dictionary into a mimiqcircuits.QCSResults object
    by extracting and unpacking records directly.
    """

    raw_records = cirq_results.get("records", {})
    records = {key: _unpack_records(**val) for key, val in raw_records.items()}
    measurements = __measurements(records)

    memory, counts = _generate_memory_and_counts(measurements)

    cstates = [frozenbitarray(bs) for bs in memory if bs]

    pivot_dict = {
        "backend_name": "Cirq-Simulator",
        "backend_version": getattr(cirq, "__version__", "unknown"),
        "results": [{"data": {"counts": counts}}],
    }

    if cstates:
        kwargs["cstates_override"] = cstates

    return dict_to_mimiq_convert(pivot_dict, **kwargs)
