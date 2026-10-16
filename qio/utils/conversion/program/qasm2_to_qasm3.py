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
from __future__ import annotations

import re

__all__ = ["convert"]

_OPENQASM_HEADER_RE = re.compile(r"^\s*OPENQASM\s+2(?:\.0)?\s*;?")
_INCLUDE_RE = re.compile(r'^\s*include\s+"([^"]+)"\s*;?')
_QREG_RE = re.compile(r"^\s*qreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;?")
_CREG_RE = re.compile(r"^\s*creg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;?")
_MEASURE_RE = re.compile(
    r"^\s*measure\s+(?P<q>\w+(?:\[\s*\d+\s*\])?)\s*->\s*"
    r"(?P<c>\w+(?:\[\s*\d+\s*\])?)\s*;?"
)

_GATE_RENAME = {"u1": "p", "cu1": "cp"}
_GATE_CALL_RE = re.compile(r"^(?P<name>\w+)")


def _strip_comments(code: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in code.splitlines())


def convert(circuit_str: str) -> str:
    """Convert an OpenQASM 2.0 program string to OpenQASM 3.0.

    Args:
        circuit_str: OpenQASM 2.0 source text.

    Returns:
        The equivalent OpenQASM 3.0 source text.

    Raises:
        TypeError: If ``circuit_str`` is not a ``str``.
    """
    if not isinstance(circuit_str, str):
        raise TypeError(
            "expected OpenQASM 2.0 source as str, got %r" % type(circuit_str).__name__
        )

    cleaned = _strip_comments(circuit_str).replace("\t", " ").strip()
    if not cleaned:
        return ""

    header: list[str] = []
    body: list[str] = []
    has_stdlib_include = False

    for statement in cleaned.split(";"):
        statement = statement.strip()
        if not statement:
            continue
        if _OPENQASM_HEADER_RE.match(statement):
            header.append("OPENQASM 3.0;")
            continue
        m = _INCLUDE_RE.match(statement)
        if m:
            inc = m.group(1).strip()
            if inc in ("qelib1.inc", "stdgates.inc"):
                header.append('include "stdgates.inc";')
                has_stdlib_include = True
            else:
                header.append('include "%s";' % inc)
            continue
        m = _QREG_RE.match(statement)
        if m:
            body.append("qubit[%s] %s;" % (m.group(2), m.group(1)))
            continue
        m = _CREG_RE.match(statement)
        if m:
            body.append("bit[%s] %s;" % (m.group(2), m.group(1)))
            continue
        m = _MEASURE_RE.match(statement)
        if m:
            body.append("%s = measure %s;" % (m.group("c"), m.group("q")))
            continue
        m = _GATE_CALL_RE.match(statement)
        if m and m.group("name") in _GATE_RENAME:
            name = m.group("name")
            statement = _GATE_RENAME[name] + statement[len(name):]
        body.append("%s;" % statement)

    if not header:
        header.append("OPENQASM 3.0;")
    if not has_stdlib_include and body:
        header.insert(1, 'include "stdgates.inc";')

    return "\n".join(header + [""] + body) if body or header else ""
