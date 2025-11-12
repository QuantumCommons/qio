import json
import zlib
import base64

from typing import Dict
from enum import Enum


class CompressionFormat(Enum):
    UNKOWN_COMPRESSION_FORMAT = 0
    NONE = 1
    ZLIB_BASE64_V1 = 2


def zlib_to_str(e: str) -> str:
    base64_payload = e.encode("ascii")
    compressed_payload = base64.b64decode(base64_payload)
    json_bytes_payload = zlib.decompress(compressed_payload)
    string_payload = json_bytes_payload.decode()

    return string_payload


def str_to_zlib(s: str) -> str:
    json_bytes_payload = s.encode()
    compressed_payload = zlib.compress(json_bytes_payload)
    base64_payload = base64.b64encode(compressed_payload)
    string_payload = base64_payload.decode("ascii")

    return string_payload


def dict_to_zlib(d: Dict) -> str:
    return str_to_zlib(json.dumps(d))


def zlib_to_dict(e: str) -> Dict:
    s = zlib_to_str(e)
    dict = json.loads(s)

    return dict
