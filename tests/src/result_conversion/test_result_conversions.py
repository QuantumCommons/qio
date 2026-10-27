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
"""Test battery: result conversions behave correctly on the *same* result.

The test cases are not written by hand: they are generated from the declarative
registries in ``result_testing`` (conversion edges) and ``reference_results``
(static, already-executed reference results). Adding a conversion or a result
to those modules automatically extends this battery.
"""

import pytest

from reference_results import REFERENCE_RESULTS
from result_testing import (
    NONE,
    ZLIB,
    UNSUPPORTED_CONVERSIONS,
    build_edges,
    run_path,
)

COMPRESSIONS = (NONE, ZLIB)

# Expand: edge x result x compression, tagging each case with its paradigm.
RESULT_CONVERSIONS = []
for compression in COMPRESSIONS:
    for edge in build_edges(compression):
        for reference in REFERENCE_RESULTS:
            RESULT_CONVERSIONS.append(
                pytest.param(
                    edge,
                    reference,
                    compression,
                    id=f"{edge.id}|{reference.name}|{compression.name}",
                    marks=getattr(pytest.mark, edge.paradigm),
                )
            )


@pytest.mark.parametrize("edge,reference,compression", RESULT_CONVERSIONS)
def test_result_conversion(edge, reference, compression):
    run_path(edge, reference)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            (operation,),
            id=name,
            marks=getattr(pytest.mark, paradigm),
        )
        for name, operation, paradigm in UNSUPPORTED_CONVERSIONS
    ],
)
def test_unsupported_conversion(case):
    (operation,) = case
    with pytest.raises(Exception):
        operation()
