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
import ast
import textwrap
import json

from enum import IntEnum
from typing import Dict

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from qio.utils.compression import zlib_to_str, str_to_zlib


class ProgramSerializationFormat(IntEnum):
    UNKNOWN_SERIALIZATION_FORMAT = 0
    NORMALIZED_PYTHON = 1


class ProgramCompressionFormat(IntEnum):
    UNKNOWN_COMPRESSION_FORMAT = 0
    NONE = 1
    ZLIB_BASE64_V1 = 2


@dataclass_json
@dataclass
class Program:
    compression_format: ProgramCompressionFormat
    serialization_format: ProgramSerializationFormat
    serialization: str

    @classmethod
    def from_json_dict(cls, data: Dict) -> "Program":
        return Program.from_dict(data)

    def to_json_dict(self) -> Dict:
        return self.to_dict()

    @classmethod
    def from_json_str(cls, data: str) -> "Program":
        while isinstance(data, str):
            data = json.loads(data)
        return cls.from_json_dict(data)

    def to_json_str(self) -> str:
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_python_source(cls, source: str) -> "Program":
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        normalized = ast.unparse(tree)

        return cls(
            compression_format=ProgramCompressionFormat.ZLIB_BASE64_V1,
            serialization_format=ProgramSerializationFormat.NORMALIZED_PYTHON,
            serialization=str_to_zlib(normalized),
        )

    def to_python_source(self) -> str:
        serialization = self.serialization

        if self.compression_format == ProgramCompressionFormat.ZLIB_BASE64_V1:
            serialization = zlib_to_str(serialization)

        return serialization
