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

from typing import Union, Dict, Any
from qio.utils.conversion.program_result.dict_to_cirq import convert as dict_to_cirq_convert
from qio.utils.conversion.program_result.mimiq_to_dict import convert as mimiq_to_dict_convert

def convert(mimiq_data: Union["mimiqcircuits.QCSResults", Dict[str, Any]], **kwargs) -> "cirq.Result":
    """
    Convert a mimiqcircuits.QCSResults object (or its serialized dictionary) into a cirq.Result object.
    """
    if not isinstance(mimiq_data, dict):
        mimiq_data = mimiq_to_dict_convert(mimiq_data, **kwargs)
        
    return dict_to_cirq_convert(mimiq_data, **kwargs)