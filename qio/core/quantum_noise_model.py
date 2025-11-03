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
import zlib

from typing import Dict, Union
from enum import Enum

from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json


class QuantumNoiseModelSerializationFormat(Enum):
    UNKOWN_SERIALIZATION_FORMAT = 0
    QISKIT_AER_JSON_V1 = 1
    QISKIT_AER_ZLIB_JSON_V1 = 2


@dataclass_json
@dataclass
class QuantumNoiseModel:
    serialization_format: QuantumNoiseModelSerializationFormat
    serialization: bytes

    # @classmethod
    # def from_dict(cls, data: Union[Dict, str]) -> "QuantumNoiseModel":
    #     return QuantumNoiseModel.schema().loads(data)

    # def to_dict(self) -> Dict:
    #     return QuantumNoiseModel.schema().dumps(self)

    # @classmethod
    # def from_json(cls, st: str) -> "QuantumNoiseModel":
    #     data = json.loads(str)
    #     return cls.from_dict(data)

    # def to_json(self) -> str:
    #     return json.dumps(self.to_dict())

    @classmethod
    def from_qiskit_aer_noise_model(
        noise_model: "qiskit_aer.NoiseModel",
    ) -> "QuantumNoiseModel":
        try:
            import numpy as np
        except ImportError:
            raise Exception("Numpy is not installed")

        try:
            from qiskit_aer.noise import NoiseModel
        except ImportError:
            raise Exception("Qiskit Aer is not installed")

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

        noise_model_dict = _encode_numpy_complex(noise_model.to_dict(False))
        return QuantumNoiseModel(
            serialization_format=QuantumNoiseModelSerializationFormat.QISKIT_AER_ZLIB_JSON_V1,
            serialization=zlib.compress(json.dumps(noise_model_dict).encode()),
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

        def _json_deserialization_noise_model(noise_model_str: bytes):
            noise_model_dict = json.loads(noise_model_str)
            noise_model = NoiseModel.from_dict(noise_model_dict)

            return noise_model

        def _zlib_json_deserialization_noise_model(noise_model_raw: bytes):
            """
            Attempt to decode the noise model dictionary using the custom decoder, and if it fails, try to use the default decoder.
            This is a workaround for a bug in the Qiskit NoiseModel.from_dict method.
            """
            noise_model_str = zlib.decompress(noise_model_raw)
            noise_model_dict = json.loads(noise_model_str)
            try:
                noise_model_dict_custom_decoded = _custom_decode_numpy_and_complex(
                    noise_model_dict
                )
            except:
                return NoiseModel.from_dict(noise_model_dict)
            return NoiseModel.from_dict(noise_model_dict_custom_decoded)

        match = {
            QuantumNoiseModelSerializationFormat.QISKIT_AER_JSON_V1: _json_deserialization_noise_model,
            QuantumNoiseModelSerializationFormat.QISKIT_AER_ZLIB_JSON_V1: _zlib_json_deserialization_noise_model,
        }

        try:
            return match[self.serialization_format](self.serialization)
        except:
            raise Exception(
                "unsupported serialization format:", self.serialization_format
            )
