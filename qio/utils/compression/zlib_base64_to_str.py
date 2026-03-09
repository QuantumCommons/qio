import json
import zlib
import base64

from typing import Dict


def zlib_to_str(e: str) -> str:
    base64_payload = e.encode("ascii")
    compressed_payload = base64.b64decode(base64_payload)
    json_bytes_payload = zlib.decompress(compressed_payload)
    string_payload = json_bytes_payload.decode()

    return string_payload


def zlib_to_dict(e: str) -> Dict:
    s = zlib_to_str(e)
    dict = json.loads(s)

    return dict
