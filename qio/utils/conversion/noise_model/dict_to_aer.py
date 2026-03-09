from qiskit_aer.noise import NoiseModel


def convert(noise_model_dict: dict) -> NoiseModel:
    return NoiseModel.from_dict(noise_model_dict)
