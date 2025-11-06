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
