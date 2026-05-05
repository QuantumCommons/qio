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
import datetime
from mimiqcircuits import QCSResults

def convert(qcsr: QCSResults, **kwargs) -> dict:
    """
    Convert a mimiqcircuits.qcsresults object into a dict following qiskit.result format.
    """
    job_id = kwargs.get("job_id", "unknown")
    shots = kwargs.get("shots", 0)
    num_qubits = kwargs.get("num_qubits", 0)
    
    raw_counts = qcsr.histogram()
    counts_dict = {key.to01(): int(val) for key, val in raw_counts.items()}
    sim_name = getattr(qcsr, "simulator", "Quantanium")
    sim_version = getattr(qcsr, 'version', '1.0')
    memory_slots = len(list(counts_dict.keys())[0]) if counts_dict else 0

    experiment_result = [{
        "shots": shots,
        "success": True,
        "meas_level": 2,
        "data": {
            "counts": counts_dict
        },
        "status": "DONE",
        "header": {
            "memory_slots": memory_slots, 
            "name": "quantanium_circuit",
            "n_qubits": num_qubits
        }
    }]

    return {
        "backend_name": sim_name,
        "backend_version": sim_version,
        "qobj_id": job_id,
        "job_id": job_id,
        "success": True,
        "results": experiment_result,
        "date": datetime.datetime.now().isoformat(),
        "status": "DONE"
    }
