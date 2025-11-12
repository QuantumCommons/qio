# Copyright 2025 Scaleway, Aqora, Quantum Commons
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
import json

from typing import Dict, Union
from enum import Enum

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from qio.utils import dict_to_zlib, zlib_to_dict, CompressionFormat


class QuantumNoiseModelSerializationFormat(Enum):
    UNKOWN_SERIALIZATION_FORMAT = 0
    QISKIT_AER_JSON_V1 = 1
    # TODO: support Qsim noise model


@dataclass_json
@dataclass
class QuantumNoiseModel:
    compression_format: CompressionFormat
    serialization_format: QuantumNoiseModelSerializationFormat
    serialization: str

    @classmethod
    def from_json_dict(cls, data: Union[Dict, str]) -> "QuantumNoiseModel":
        return QuantumNoiseModel.schema().loads(data)

    def to_json_dict(self) -> Dict:
        return QuantumNoiseModel.schema().dumps(self)

    @classmethod
    def from_json_str(cls, str: str) -> "QuantumNoiseModel":
        data = json.loads(str)
        return cls.from_json_dict(data)

    def to_json_str(self) -> str:
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_qiskit_aer_noise_model(
        self,
        noise_model: "qiskit_aer.NoiseModel",
        compression_format: CompressionFormat = CompressionFormat.ZLIB_BASE64_V1,
    ) -> "QuantumNoiseModel":
        try:
            import numpy as np
        except ImportError:
            raise Exception("Numpy is not installed")

        try:
            from qiskit_aer.noise import NoiseModel
        except ImportError:
            raise Exception("Qiskit Aer is not installed")

        compression_format = (
            CompressionFormat.NONE
            if compression_format == CompressionFormat.UNKOWN_COMPRESSION_FORMAT
            else compression_format
        )

        def _encode_numpy_complex(obj):
            """
            Recursively traverses a structure and converts numpy arrays and
            complex numbers into a JSON-serializable format.
            """
            if isinstance(obj, np.ndarray):
                return {
                    "__ndarray__": True,
                    "data": _encode_numpy_complex(
                        obj.tolist()
                    ),  # Recursively encode data
                    "dtype": obj.dtype.name,
                    "shape": obj.shape,
                }
            elif isinstance(obj, (complex, np.complex128)):
                return {"__complex__": True, "real": obj.real, "imag": obj.imag}
            elif isinstance(obj, dict):
                return {key: _encode_numpy_complex(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_encode_numpy_complex(item) for item in obj]
            else:
                return obj

        noise_model_dict = noise_model.to_dict(False)

        apply_compression = {
            CompressionFormat.NONE: lambda d: json.dumps(d),
            CompressionFormat.ZLIB_BASE64_V1: lambda d: dict_to_zlib(
                _encode_numpy_complex(d)
            ),
        }

        serialization = apply_compression[compression_format](noise_model_dict)

        return QuantumNoiseModel(
            serialization_format=QuantumNoiseModelSerializationFormat.QISKIT_AER_JSON_V1,
            compression_format=compression_format,
            serialization=serialization,
        )

    def to_qiskit_aer_noise_model(self) -> "qiskit_aer.NoiseModel":
        try:
            import numpy as np
        except ImportError:
            raise Exception("Numpy is not installed")

        try:
            from qiskit_aer.noise import NoiseModel
        except ImportError:
            raise Exception("Qiskit Aer is not installed")

        def _custom_decode_numpy_and_complex(obj):
            """
            Recursively traverses a structure and converts dictionary representations
            back into numpy arrays and complex numbers.
            """
            if isinstance(obj, dict):
                if obj.get("__ndarray__", False):
                    # The data has been recursively decoded, so we can build the array
                    return np.array(
                        _custom_decode_numpy_and_complex(obj["data"]),
                        dtype=obj["dtype"],
                    ).reshape(obj["shape"])
                elif obj.get("__complex__", False):
                    return complex(obj["real"], obj["imag"])
                else:
                    return {
                        key: _custom_decode_numpy_and_complex(value)
                        for key, value in obj.items()
                    }
            elif isinstance(obj, list):
                return [_custom_decode_numpy_and_complex(item) for item in obj]
            else:
                return obj

        try:
            serialization = self.serialization

            apply_uncompression = {
                CompressionFormat.ZLIB_BASE64_V1: lambda s: _custom_decode_numpy_and_complex(
                    zlib_to_dict(s)
                ),
                CompressionFormat.NONE: lambda s: json.loads(s),
            }

            serialization_dict = apply_uncompression[self.compression_format](
                serialization
            )

            return NoiseModel.from_dict(serialization_dict)

        except Exception as e:
            raise Exception("unsupported serialization:", self.serialization_format, e)
