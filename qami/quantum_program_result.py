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

from typing import Union, Dict
from enum import Enum

from dataclasses import dataclass
from dataclasses_json import dataclass_json


class QuantumProgramResultSerializationFormat(Enum):
    UNKOWN_SERIALIZATION_FORMAT = 0
    CIRQ_RESULT_JSON_V1 = 1
    QISKIT_RESULT_JSON_V1 = 2


@dataclass_json
@dataclass
class QuantumProgramResult:
    serialization_format: QuantumProgramResultSerializationFormat
    serialization: str

    @classmethod
    def from_dict(cls, data: Union[Dict, str]) -> "QuantumProgramResult":
        data = json.loads(data) if isinstance(data, str) else data
        return QuantumProgramResult.schema().load(data)

    def to_dict(self) -> Dict:
        return QuantumProgramResult.schema().dump(self)

    @classmethod
    def from_qiskit_result(
        cls, qiskit_result: "qiskit.result.Result"
    ) -> "QuantumProgramResult":
        try:
            from qiskit.result import Result
        except ImportError:
            raise Exception("Qiskit is not installed")

        serialization = json.dumps(qiskit_result.to_dict())

        return cls(
            serialization_format=QuantumProgramResultSerializationFormat.QISKIT_RESULT_JSON_V1,
            serialization=serialization,
            frozenset=True,
        )

    @classmethod
    def from_qiskit_result_dict(
        cls, qiskit_result_dict: Union[str, Dict]
    ) -> "QuantumProgramResult":
        if isinstance(qiskit_result_dict, str):
            qiskit_result_dict = json.loads(
                qiskit_result_dict
            )  # Ensure serialization is not ill-formatted

        serialization = json.dumps(qiskit_result_dict)

        return cls(
            serialization_format=QuantumProgramResultSerializationFormat.QISKIT_RESULT_JSON_V1,
            serialization=serialization,
            frozenset=True,
        )

    def to_qiskit_result(self, **kwargs) -> "qiskit.result.Result":
        try:
            from qiskit.result import Result
        except ImportError:
            raise Exception("Qiskit is not installed")

        result_dict = json.loads(self.serialization)

        return Result.from_dict(
            {
                "results": result_dict["results"],
                "success": result_dict["success"],
                "header": result_dict.get("header"),
                "metadata": result_dict.get("metadata"),
            }
        )

    @classmethod
    def from_cirq_result_dict(
        cls, cirq_result_dict: Union[str, Dict]
    ) -> "QuantumProgramResult":
        if isinstance(cirq_result_dict, str):
            cirq_result_dict = json.loads(
                cirq_result_dict
            )  # Ensure serialization is not ill-formatted

        serialization = json.dumps(cirq_result_dict)

        return cls(
            serialization_format=QuantumProgramResultSerializationFormat.CIRQ_RESULT_JSON_V1,
            serialization=serialization,
            frozenset=True,
        )

    @classmethod
    def from_cirq_result(cls, cirq_result: "cirq.Result") -> "QuantumProgramResult":
        try:
            import cirq
        except ImportError:
            raise Exception("Cirq is not installed")

        serialization = json.dumps(cirq_result._json_dict_())

        return cls(
            serialization_format=QuantumProgramResultSerializationFormat.CIRQ_RESULT_JSON_V1,
            serialization=serialization,
            frozenset=True,
        )

    def to_cirq_result(self, **kwargs) -> "cirq.Result":
        try:
            from cirq import ResultDict
        except ImportError:
            raise Exception("Cirq is not installed")

        if (
            self.serialization_format
            != QuantumProgramResultSerializationFormat.CIRQ_RESULT_JSON_V1
        ):
            raise Exception(
                "unsupported serialization format:", self.serialization_format
            )

        result_dict = json.loads(self.serialization)
        cirq_result = ResultDict._from_json_dict_(**result_dict)

        return cirq_result
