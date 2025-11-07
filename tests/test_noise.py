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
from qiskit_aer import noise

from qio.core import QuantumNoiseModel


class TestQiskit:
    @classmethod
    def setup_class(self):
        pass

    @classmethod
    def teardown_class(self):
        pass

    def test_aer_noise(self):
        prob_1 = 0.01  # 1-qubit gate
        prob_2 = 0.1  # 2-qubit gate

        # Depolarizing quantum errors
        error_1 = noise.depolarizing_error(prob_1, 1)
        error_2 = noise.depolarizing_error(prob_2, 2)

        # Add errors to noise model
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ["rz", "sx", "x"])
        noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

        qnoise_model = QuantumNoiseModel.from_qiskit_aer_noise_model(noise_model)

        deser_noise_model = qnoise_model.to_qiskit_aer_noise_model()

        assert deser_noise_model == noise_model
