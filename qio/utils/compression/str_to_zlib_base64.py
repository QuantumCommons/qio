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
