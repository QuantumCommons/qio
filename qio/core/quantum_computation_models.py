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

from typing import List, Optional, Dict, Union

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from .quantum_program import QuantumProgram
from .quantum_noise_model import QuamtumNoiseModel


@dataclass_json
@dataclass
class ClientData:
    user_agent: str


@dataclass_json
@dataclass
class BackendData:
    name: str
    version: str


@dataclass_json
@dataclass
class QuantumComputationModel:
    programs: List[QuantumProgram]
    noise_model: Optional[QuamtumNoiseModel] = None
    client: Optional[ClientData] = None
    backend: Optional[BackendData] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "QuantumComputationModel":
        return QuantumComputationModel.schema().load(data)

    def to_dict(self) -> Dict:
        return QuantumComputationModel.schema().dump(self)

    @classmethod
    def from_json(cls, str) -> "QuantumComputationModel":
        data = json.loads(data) if isinstance(data, str) else data
        return cls.from_dict(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass_json
@dataclass
class QuantumComputationParameters:
    shots: int

    @classmethod
    def from_dict(cls, data: Union[Dict, str]) -> "QuantumComputationParameters":
        return QuantumComputationParameters.schema().load(data)

    def to_dict(self) -> Dict:
        return QuantumComputationParameters.schema().dump(self)

    @classmethod
    def from_json(cls, str) -> "QuantumComputationParameters":
        data = json.loads(data) if isinstance(data, str) else data
        return cls.from_dict(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
