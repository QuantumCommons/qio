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
# limitations under the License.from enum import Enum
import json
import zlib
import base64

from typing import Dict


def str_to_zlib(s: str) -> str:
    json_bytes_payload = s.encode()
    compressed_payload = zlib.compress(json_bytes_payload)
    base64_payload = base64.b64encode(compressed_payload)
    string_payload = base64_payload.decode("ascii")

    return string_payload


def dict_to_zlib(d: Dict) -> str:
    return str_to_zlib(json.dumps(d))
